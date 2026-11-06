"""
qlib 模型训练/预测封装 — LightGBM, XGBoost, CatBoost, LSTM, GRU, Transformer 等
"""
import copy
import json
import logging
from pathlib import Path
import pandas as pd

from core.adapters.qlib.config import init_qlib
from core.adapters.qlib.model_contract import TRADABLE_LABEL_MODE

logger = logging.getLogger(__name__)


def _require_tradable_label_mode(label_mode):
    if label_mode != TRADABLE_LABEL_MODE:
        raise ValueError("训练模型仅允许 tradable_open_to_open 标签")


def _build_label_expression(label_horizon, label_mode="tradable_open_to_open"):
    horizon = int(label_horizon)
    if horizon < 1:
        raise ValueError("label_horizon 必须为正整数")
    if label_mode == "close_to_close":
        return f"Ref($close, -{horizon}) / $close - 1"
    if label_mode == "tradable_open_to_open":
        return f"Ref($open, -{horizon + 1}) / Ref($open, -1) - 1"
    raise ValueError(f"不支持的 label_mode: {label_mode}")


def _purge_training_segments(train_period, valid_period, test_period, purge_gap, calendar=None):
    gap = int(purge_gap or 0)
    if gap <= 0:
        return train_period, valid_period, test_period

    boundaries = []
    if train_period and valid_period and valid_period[0]:
        boundaries.append(("train", train_period, valid_period[0]))
    if valid_period and test_period and test_period[0] and test_period[0] != valid_period[0]:
        boundaries.append(("valid", valid_period, test_period[0]))
    if not boundaries:
        return train_period, valid_period, test_period

    if calendar is None:
        try:
            from qlib.data import D
            from core.adapters.qlib.generation import qlib_generation_read_lock
            start = min(str(period[0]) for _, period, _ in boundaries if period and period[0])
            end = max(str(boundary) for _, _, boundary in boundaries)
            with qlib_generation_read_lock():
                calendar = list(D.calendar(start_time=start, end_time=end, freq="day"))
        except Exception:
            calendar = []
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values().unique()
    fallback = len(calendar) == 0
    if fallback:
        start = min(pd.Timestamp(period[0]) for _, period, _ in boundaries if period and period[0])
        end = max(pd.Timestamp(boundary) for _, _, boundary in boundaries)
        calendar = pd.date_range(start, end, freq="D")
        gap = gap * 2 + 7

    periods = {"train": train_period, "valid": valid_period}
    for name, period, boundary in boundaries:
        before = calendar[calendar < pd.Timestamp(boundary)]
        if len(before) <= gap:
            raise ValueError(f"{name} 分段在标签净化后没有足够交易日")
        safe_end = before[-gap - 1]
        if period[1] and pd.Timestamp(period[1]) > safe_end:
            periods[name] = (period[0], safe_end.strftime("%Y-%m-%d"))

    return periods["train"], periods["valid"], test_period

SUPPORTED_MODELS = {
    "lightgbm":   ("LGBModel",       "qlib.contrib.model.gbdt"),
    "xgboost":    ("XGBModel",       "qlib.contrib.model.xgboost"),
    "catboost":   ("CatBoostModel",  "qlib.contrib.model.catboost_model"),
    "linear":     ("LinearModel",    "qlib.contrib.model.linear"),
    "lstm":       ("LSTM",           "qlib.contrib.model.pytorch_lstm"),
    "gru":        ("GRU",            "qlib.contrib.model.pytorch_gru"),
    "lstm_ts":    ("LSTM_ts",        "qlib.contrib.model.pytorch_lstm_ts"),
    "gru_ts":     ("GRU_ts",         "qlib.contrib.model.pytorch_gru_ts"),
    "transformer":("Transformer",    "qlib.contrib.model.pytorch_transformer"),
    "tabnet":     ("TabNetModel",    "qlib.contrib.model.pytorch_tabnet"),
    "tcn":        ("TCN",            "qlib.contrib.model.pytorch_tcn"),
    "hist":       ("HIST",           "qlib.contrib.model.pytorch_hist"),
    "alstm":      ("ALSTM",          "qlib.contrib.model.pytorch_alstm"),
    "sfm":        ("SFM",            "qlib.contrib.model.pytorch_sfm"),
    "tcn_ts":     ("TCN_ts",         "qlib.contrib.model.pytorch_tcn_ts"),
    # RankIC 早停变体: 早停指标从 valid MSE 改为 valid 逐日截面 RankIC
    "lightgbm_rank": ("RankLGBModel", "core.adapters.qlib.rank_models"),
    "xgboost_rank":  ("RankXGBModel", "core.adapters.qlib.rank_models"),
}

# label 截面标准化方案 (learn_processors 只作用于训练/验证段, 测试段 IC 用原始 label)
# cszscore: qlib Alpha158 隐式默认 (逐日截面 z-score); csrank: 逐日截面排名, 抗厚尾 outlier
LABEL_NORM_PROCESSORS = {
    "cszscore": [
        {"class": "DropnaLabel"},
        {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
    ],
    "csrank": [
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ],
    "none": [{"class": "DropnaLabel"}],
}

DEFAULT_PARAMS = {
    "lightgbm": {
        "loss": "mse",
        "learning_rate": 0.03,
        "max_depth": 7,
        "num_leaves": 127,
        "num_threads": 8,
        "min_child_samples": 50,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.3,
        "reg_lambda": 0.5,
        "early_stopping_rounds": 50,
        "verbose_eval": -1,
    },
    "xgboost": {
        "max_depth": 6,
        "learning_rate": 0.03,
        "n_estimators": 500,
        "nthread": 4,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.3,
        "reg_lambda": 0.5,
        "early_stopping_rounds": 50,
    },
    "catboost": {
        "learning_rate": 0.03,
        "depth": 6,
        "iterations": 500,
        "thread_count": 4,
        "subsample": 0.7,
        "l2_leaf_reg": 5,
        "early_stopping_rounds": 50,
        "task_type": "CPU",
    },
    "lstm": {
        "n_epochs": 100,
        "batch_size": 2000,
        "lr": 0.001,
        "d_feat": 158,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "GPU": 0,
    },
    "gru": {
        "n_epochs": 100,
        "batch_size": 2000,
        "lr": 0.001,
        "d_feat": 158,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "GPU": 0,
    },
    "transformer": {
        "n_epochs": 100,
        "batch_size": 2000,
        "lr": 0.001,
        "d_feat": 158,
        "nhead": 2,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.0,
        "GPU": 0,
    },
}
DEFAULT_PARAMS["lightgbm_rank"] = dict(DEFAULT_PARAMS["lightgbm"])
DEFAULT_PARAMS["xgboost_rank"] = dict(DEFAULT_PARAMS["xgboost"])

# 分池超参 profile: conservative(稳定核心池, 降容量强正则) / aggressive(适度放宽)
# 在 DEFAULT_PARAMS 基础上覆盖, 显式传入的 lgb_params/xgb_params 优先级最高
PARAM_PROFILES = {
    "default": {"lightgbm": {}, "xgboost": {}, "catboost": {}},
    "conservative": {
        "lightgbm": {"max_depth": 5, "num_leaves": 63, "reg_lambda": 0.8,
                     "reg_alpha": 0.5, "min_child_samples": 100},
        "xgboost": {"max_depth": 4, "reg_lambda": 0.8, "reg_alpha": 0.5,
                    "min_child_weight": 10},
        "catboost": {"depth": 4, "l2_leaf_reg": 10, "subsample": 0.6},
    },
    "aggressive": {
        "lightgbm": {"max_depth": 8, "num_leaves": 200, "reg_lambda": 0.15,
                     "reg_alpha": 0.1, "min_child_samples": 30},
        "xgboost": {"max_depth": 7, "reg_lambda": 0.15, "reg_alpha": 0.1,
                    "n_estimators": 800},
        "catboost": {"depth": 8, "l2_leaf_reg": 2, "iterations": 800},
    },
}


def _ic_fusion_weights(lgb_ic, xgb_ic, lo=0.3, hi=0.7):
    """按 IC 计算双模型融合权重, clip 防单窗口 IC 噪声极端化。
    任一 IC 缺失/NaN/Inf 或两者均 ≤0 → 等权。返回 (w_lgb, w_xgb), 和为 1。"""
    import math
    if lgb_ic is None or xgb_ic is None:
        return 0.5, 0.5
    try:
        lgb_val = float(lgb_ic)
        xgb_val = float(xgb_ic)
    except (ValueError, TypeError):
        return 0.5, 0.5
    if math.isnan(lgb_val) or math.isinf(lgb_val):
        return 0.5, 0.5
    if math.isnan(xgb_val) or math.isinf(xgb_val):
        return 0.5, 0.5
    a, b = max(lgb_val, 0.0), max(xgb_val, 0.0)
    if a + b <= 0:
        return 0.5, 0.5
    w = min(max(a / (a + b), lo), hi)
    return w, 1.0 - w


def _read_param(params_dir, name, idx=None):
    """从 MLflow params 目录读取参数值, 可选解析为 tuple 的第 idx 个元素"""
    try:
        p = Path(params_dir) / name
        if not p.exists():
            return None
        val = p.read_bytes().decode("utf-8", errors="replace").strip()
        if idx is not None:
            import ast
            parsed = ast.literal_eval(val)
            if isinstance(parsed, (tuple, list)) and len(parsed) > idx:
                return str(parsed[idx])
            return None
        return val
    except Exception:
        return None

def _read_run_metric(run_dir, name):
    """读 MLflow file-store 指标文件最后一行的值, run_dir = mlruns/<exp>/<run>"""
    import math
    try:
        f = Path(run_dir) / "metrics" / name
        if f.exists():
            raw = f.read_bytes()
            if raw[:2] == b'\xff\xfe':
                lines = raw[2:].decode("utf-16-le", errors="replace").strip().splitlines()
            elif raw[:2] == b'\xfe\xff':
                lines = raw[2:].decode("utf-16-be", errors="replace").strip().splitlines()
            else:
                lines = raw.decode("utf-8", errors="replace").strip().splitlines()
            if lines:
                val = float(lines[-1].split()[1])
                if math.isnan(val) or math.isinf(val):
                    logger.warning(f"[Metric] {name}={val}, 退回默认值")
                    return None
                return val
    except Exception as e:
        logger.warning(f"[Metric] 读取 {run_dir}/metrics/{name} 失败: {e}")
    return None


def dual_fusion_weights_from_run(run_dir):
    """从 MLflow run 目录读双模型融合权重: 优先训练时记录的 LGB_W/XGB_W,
    回退按 LGB_IC/XGB_IC 计算, 均无 (老模型) → 等权。"""
    w = _read_run_metric(run_dir, "LGB_W")
    if w is not None and 0.0 < w < 1.0:
        return w, 1.0 - w
    return _ic_fusion_weights(_read_run_metric(run_dir, "LGB_IC"),
                              _read_run_metric(run_dir, "XGB_IC"))


def _extended_eval_metrics(ic, pred_s, label_s, label_horizon=20):
    """扩展评估指标 (测试段):
    IC_WIN_RATE_M   月度 IC 均值 >0 的月份占比
    MAX_NEG_IC_MONTHS 最大连续负 IC 月数
    ICIR_YEARLY_MIN 分年度 ICIR 最小值
    Q5_Q1_SPREAD    逐日按预测分五分位, Q5-Q1 组均 label 收益差的日均值
    LS_SHARPE       Q5-Q1 日序列夏普 (sqrt(252/label_horizon) 年化)
    IC_NONOVERLAP   非重叠 IC (每隔 label_horizon 天采样, 消除重叠收益偏差)
    ICIR_NONOVERLAP 非重叠 ICIR
    IC_DECAY_1M     滞后1月 IC vs 当期 IC 的比值 (IC 衰减率, <0.7 意味信号持续性差)
    """
    import numpy as np
    out = {}
    try:
        if ic is not None and len(ic) > 0:
            try:
                idx = pd.DatetimeIndex(ic.index)
            except Exception:
                idx = pd.DatetimeIndex(pd.to_datetime(ic.index))
            monthly = ic.groupby(idx.to_period("M")).mean()
            if len(monthly):
                out["IC_WIN_RATE_M"] = float((monthly > 0).mean())
                neg = max_neg = 0
                for v in monthly:
                    neg = neg + 1 if v < 0 else 0
                    max_neg = max(max_neg, neg)
                out["MAX_NEG_IC_MONTHS"] = float(max_neg)
            yearly_icir = []
            for _, g in ic.groupby(idx.year):
                std = g.std()
                if std and std > 0:
                    yearly_icir.append(g.mean() / std)
            if yearly_icir:
                out["ICIR_YEARLY_MIN"] = float(min(yearly_icir))
            # 非重叠 IC: 每隔 label_horizon 天采样, 消除重叠收益偏差
            if label_horizon > 1 and len(ic) > label_horizon:
                try:
                    ic_no = ic.iloc[::label_horizon]
                    if len(ic_no) >= 5:
                        out["IC_NONOVERLAP"] = float(ic_no.mean())
                        if ic_no.std() > 0:
                            out["ICIR_NONOVERLAP"] = float(ic_no.mean() / ic_no.std())
                except Exception as e:
                    logger.debug(f"[ExtEval] 非重叠IC计算失败: {e}")
            # IC 衰减: 滞后约1个月 vs 当期 (按位置对齐, 非按索引取交集)
            if len(ic) > 30:
                try:
                    lag = min(20, len(ic) // 2)
                    ic_t_vals = ic.iloc[:-lag].values
                    ic_t1_vals = ic.iloc[lag:].values
                    if len(ic_t_vals) >= 10:
                        corr = float(np.corrcoef(ic_t_vals, ic_t1_vals)[0, 1])
                        if not np.isnan(corr):
                            out["IC_AUTOCORR_1M"] = float(corr)
                        decay = max(0.0, min(1.0, (ic_t1_vals.mean() / ic_t_vals.mean())
                                        if ic_t_vals.mean() > 0 else 0.0))
                        out["IC_DECAY_1M"] = float(decay)
                except Exception as e:
                    logger.debug(f"[ExtEval] IC衰减计算失败: {e}")
    except Exception as e:
        logger.warning(f"[ExtEval] IC 时序指标计算失败: {e}")
    try:
        df = pd.DataFrame({"p": np.asarray(pred_s.values).ravel(),
                           "l": np.asarray(label_s.values).ravel()}, index=pred_s.index)
        dates = df.index.get_level_values("datetime")

        def _ls(g):
            if len(g) < 20:
                return np.nan
            q = pd.qcut(g["p"], 5, labels=False, duplicates="drop")
            gm = g["l"].groupby(q).mean()
            if len(gm) < 5:
                return np.nan
            return gm.iloc[-1] - gm.iloc[0]

        ls = df.groupby(dates).apply(_ls).dropna()
        if len(ls):
            out["Q5_Q1_SPREAD"] = float(ls.mean())
            if ls.std() > 0:
                out["LS_SHARPE"] = float(ls.mean() / ls.std() * (252 / label_horizon) ** 0.5)
    except Exception as e:
        logger.warning(f"[ExtEval] 五分位多空指标计算失败: {e}")
    return out


class QlibModelManager:
    def __init__(self):
        init_qlib()

    @classmethod
    def list_models(cls):
        return list(SUPPORTED_MODELS.keys())

    def _build_task_config(
        self, model_name, factor_set, instruments, train_period, valid_period, test_period,
        model_params=None, label_horizon=20, custom_factor_keys=None,
        enable_cross_sectional=True, enable_crossover=False, label_norm="cszscore",
        enable_winsorize=True, label_mode="tradable_open_to_open", purge_gap=None,
    ):
        """构造 qlib 任务配置字典

        Args:
            factor_set: "Alpha158", "Alpha360", 或 "WhiteBoxHandler"
            custom_factor_keys: 当factor_set="WhiteBoxHandler"时, 注入的白盒因子key列表
            enable_cross_sectional: 是否追加截面百分位排名和z-score (默认开)
            enable_crossover: 是否追加因子两两乘积交互特征 (默认关, 特征数爆炸)
            label_norm: label 截面标准化 "cszscore"|"csrank"|"none" (默认 cszscore 与 qlib 隐式默认一致)
        """
        class_name, module_path = SUPPORTED_MODELS[model_name]
        params = {**DEFAULT_PARAMS.get(model_name, {}), **(model_params or {})}

        # 联动约束: 交叉特征使特征数暴涨 (+105列), 必须配强 L2 正则防过拟合
        if enable_crossover and model_name in (
            "lightgbm", "xgboost", "lightgbm_rank", "xgboost_rank"
        ) and params.get("reg_lambda", 0) < 0.5:
            params["reg_lambda"] = 0.5
            logger.warning(f"[TaskConfig] enable_crossover=True 强制 {model_name} reg_lambda>=0.5")

        # 自定义因子集: 切换handler模块
        if factor_set in ("WhiteBoxHandler",) or factor_set.startswith("WhiteBox"):
            handler_module = "core.adapters.qlib.custom_handler"
            handler_class = "WhiteBoxHandler"
        else:
            handler_module = "qlib.contrib.data.handler"
            handler_class = factor_set

        # 计算全局时间范围: start取所有阶段最早, end取所有阶段最晚
        all_starts = [p[0] for p in [train_period, valid_period, test_period] if p[0]]
        all_ends = [p[1] for p in [train_period, valid_period, test_period] if p[1]]
        global_start = min(all_starts) if all_starts else train_period[0]
        global_end = max(all_ends) if all_ends else train_period[1]

        # 清理 segments: 两端都 None → 跳过该段; 单端 None → 用 global 边界回填
        # test 全 None 时回退到 valid: SignalRecord.generate() 需要 test segment
        if not test_period[0] and not test_period[1]:
            test_period = valid_period
        if purge_gap:
            train_period, valid_period, test_period = _purge_training_segments(
                train_period, valid_period, test_period, purge_gap)
        segments = {}
        for seg_name, period in [("train", train_period), ("valid", valid_period), ("test", test_period)]:
            s, e = period
            if not s and not e:
                continue
            segments[seg_name] = (s or global_start, e or global_end)

        shared_processors = []
        if enable_winsorize:
            shared_processors.append(
                {"class": "CSWinsorize", "module_path": "core.adapters.qlib.custom_processors", "kwargs": {}}
            )
        if enable_crossover:
            shared_processors.append(
                {"class": "FactorCrossover", "module_path": "core.adapters.qlib.custom_processors", "kwargs": {}}
            )
        if enable_cross_sectional:
            shared_processors.append(
                {"class": "CSFeatureAugment", "module_path": "core.adapters.qlib.custom_processors", "kwargs": {}}
            )

        infer_processors = [
            {"class": "ProcessInf", "kwargs": {}},
            {"class": "Fillna", "kwargs": {}},
        ]

        handler_kwargs = {
            "instruments": instruments,
            "start_time": global_start,
            "end_time": global_end,
            "fit_start_time": train_period[0],
            "fit_end_time": train_period[1],
            "label": [_build_label_expression(label_horizon, label_mode)],
            "infer_processors": infer_processors,
            "learn_processors": LABEL_NORM_PROCESSORS.get(label_norm, LABEL_NORM_PROCESSORS["cszscore"]),
            **({"factor_keys": custom_factor_keys} if custom_factor_keys else {}),
        }
        if shared_processors:
            handler_kwargs["shared_processors"] = shared_processors

        return {
            "model": {
                "class": class_name,
                "module_path": module_path,
                "kwargs": params,
            },
            "dataset": {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": handler_class,
                        "module_path": handler_module,
                        "kwargs": handler_kwargs,
                    },
                    "segments": segments,
                },
            },
        }

    def train(
        self, model_name, factor_set, instruments,
        train_period, valid_period, test_period,
        model_params=None, label_horizon=5,
        experiment_name="a_share_quant",
        custom_factor_keys=None, label_norm="none",
        enable_cross_sectional=False, enable_winsorize=True,
        label_mode="tradable_open_to_open",
        factor_selection_snapshot=None,
    ):
        """
        训练模型并返回结果

        Parameters
        ----------
        model_name : str — SUPPORTED_MODELS 中的 key
        factor_set : str — "Alpha158", "Alpha360", 或 "WhiteBoxHandler"
        instruments : str — "csi300", "csi500", "all"
        train_period : tuple — ("2018-01-01", "2021-12-31")
        valid_period : tuple — ("2022-01-01", "2022-12-31")
        test_period : tuple — ("2023-01-01", "2024-12-31")
        model_params : dict — 覆盖默认参数
        experiment_name : str — MLflow 实验名称
        custom_factor_keys : list — 白盒因子key列表(factor_set="WhiteBoxHandler"时)

        Returns
        -------
        dict with keys: model, predictions, ic, ric, metrics
        """
        _require_tradable_label_mode(label_mode)
        from qlib.utils import init_instance_by_config, flatten_dict
        from qlib.workflow import R
        from qlib.workflow.record_temp import SignalRecord, SigAnaRecord

        task_config = self._build_task_config(
            model_name, factor_set, instruments,
            train_period, valid_period, test_period, model_params,
            label_horizon=label_horizon,
            custom_factor_keys=custom_factor_keys,
            label_norm=label_norm,
            enable_cross_sectional=enable_cross_sectional,
            enable_winsorize=enable_winsorize,
            label_mode=label_mode,
            purge_gap=int(label_horizon) + 1,
        )

        result = {}

        with R.start(experiment_name=experiment_name):
            instruments_val = task_config["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"]
            if isinstance(instruments_val, list):
                log_config = copy.deepcopy(task_config)
                log_config["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"] = f"list_{len(instruments_val)}_stocks"
                R.log_params(**flatten_dict(log_config))
            else:
                R.log_params(**flatten_dict(task_config))
            R.log_params(label_mode=label_mode)
            if factor_selection_snapshot is not None:
                R.save_objects(**{"factor_selection_snapshot.json": json.dumps(
                    factor_selection_snapshot, ensure_ascii=False, sort_keys=True,
                )})

            model = init_instance_by_config(task_config["model"])
            dataset = init_instance_by_config(task_config["dataset"])

            from core.adapters.qlib.fundval import record_dataset_contract
            record_dataset_contract(dataset, task_config, label_horizon, label_mode, label_norm)
            model._feature_contract = getattr(dataset.handler, '_feature_contract', None)

            model.fit(dataset)
            R.save_objects(**{"params.pkl": model})

            recorder = R.get_recorder()
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

            sar = SigAnaRecord(recorder)
            sar.generate()

            pred = sr.load("pred.pkl")
            label = sr.load("label.pkl")

            ic = sar.load("ic.pkl")
            ric = sar.load("ric.pkl")

            ic_mean = float(ic.mean()) if ic is not None else None
            ic_std = float(ic.std()) if ic is not None and ic.std() > 0 else 0
            icir = ic_mean / ic_std if ic_std and ic_std > 0 else None

            result = {
                "model": model,
                "predictions": pred,
                "label": label,
                "ic": ic,
                "ric": ric,
                "ic_mean": ic_mean,
                "icir": icir,
                "experiment_id": str(getattr(recorder, "experiment_id", "")),
                "run_id": str(getattr(recorder, "id", "")),
            }

        return result

    def predict(self, model, dataset_config):
        """使用已训练模型生成预测"""
        from qlib.utils import init_instance_by_config

        from core.adapters.qlib.fundval import feature_contract
        from core.shared.fundamental_contract import FundamentalDataError
        keys = dataset_config.get('kwargs', {}).get('handler', {}).get('kwargs', {}).get('factor_keys', [])
        expected = feature_contract(keys)
        if expected['requires_fundval']:
            stored = getattr(model, '_feature_contract', None)
            if not isinstance(stored, dict) or any(stored.get(key) != value for key, value in expected.items()):
                raise FundamentalDataError('模型缺少兼容的FundVal输入契约，必须重新训练')

        dataset = init_instance_by_config(dataset_config)
        return model.predict(dataset)

    def train_dual_model(
        self, factor_set, instruments,
        train_period, valid_period, test_period,
        label_horizon=5, experiment_name="a_share_quant",
        custom_factor_keys=None, lgb_params=None, xgb_params=None,
        enable_cross_sectional=False, enable_crossover=False, label_norm="none",
        enable_winsorize=True,
        early_stop_metric="mse", param_profile="default",
        use_reweighter=False, with_linear=False, with_catboost=False,
        catboost_params=None,
        label_mode="tradable_open_to_open",
        factor_selection_snapshot=None,
    ):
        """双模型训练: LightGBM + XGBoost, 共用 handler/dataset, 按 IC 加权融合。
        可选追加 CatBoost (第三 GBDT) 和 ridge 线性基线。

        Parameters
        ----------
        factor_set : str — "WhiteBoxHandler" (推荐) 或其他
        instruments : str — "csi300", "csi500", "all"
        train_period : tuple — ("2018-01-01", "2021-12-31")
        valid_period : tuple — ("2022-01-01", "2022-12-31")
        test_period : tuple — ("2023-01-01", "2024-12-31")
        label_horizon : int — 标签周期
        experiment_name : str — MLflow 实验名
        custom_factor_keys : list — 白盒因子key列表
        lgb_params : dict — 覆盖 LightGBM 默认参数
        xgb_params : dict — 覆盖 XGBoost 默认参数
        enable_cross_sectional : bool — 开启截面百分位排名+zscore
        enable_crossover : bool — 开启因子两两乘积交互
        label_norm : str — label 截面标准化 "cszscore"|"csrank"|"none"
        early_stop_metric : str — 早停指标 "mse"|"rank_ic" (rank_ic 用 RankLGB/RankXGB 变体)
        param_profile : str — 超参 profile "default"|"conservative"|"aggressive"
        use_reweighter : bool — 训练样本流动性加权 + 次新剔除 (LiquidityReweighter)
        with_linear : bool — 追加 ridge 线性基线为第三模型, 份额上限 20%
        with_catboost : bool — 追加 CatBoost 为第三模型 (份额上限 30%, 与 linear 不互斥)
        catboost_params : dict — 覆盖 CatBoost 默认参数

        Returns
        -------
        dict with keys: models, predictions, label, ic, ric, ic_mean, icir,
                        lgb_ic_mean, xgb_ic_mean, lgb_weight, xgb_weight, degraded
        """
        _require_tradable_label_mode(label_mode)
        import numpy as np
        from qlib.utils import init_instance_by_config, flatten_dict
        from qlib.workflow import R
        from qlib.workflow.record_temp import SignalRecord, SigAnaRecord

        lgb_name = "lightgbm_rank" if early_stop_metric == "rank_ic" else "lightgbm"
        xgb_name = "xgboost_rank" if early_stop_metric == "rank_ic" else "xgboost"
        profile = PARAM_PROFILES.get(param_profile, PARAM_PROFILES["default"])
        lgb_params = {**profile["lightgbm"], **(lgb_params or {})}
        xgb_params = {**profile["xgboost"], **(xgb_params or {})}

        lgb_cfg = self._build_task_config(
            lgb_name, factor_set, instruments,
            train_period, valid_period, test_period,
            model_params=lgb_params, label_horizon=label_horizon,
            custom_factor_keys=custom_factor_keys,
            enable_cross_sectional=enable_cross_sectional,
            enable_crossover=enable_crossover, label_norm=label_norm,
            enable_winsorize=enable_winsorize, label_mode=label_mode,
            purge_gap=int(label_horizon) + 1)

        xgb_cfg = self._build_task_config(
            xgb_name, factor_set, instruments,
            train_period, valid_period, test_period,
            model_params=xgb_params, label_horizon=label_horizon,
            custom_factor_keys=custom_factor_keys,
            enable_cross_sectional=enable_cross_sectional,
            enable_crossover=enable_crossover, label_norm=label_norm,
            enable_winsorize=enable_winsorize, label_mode=label_mode,
            purge_gap=int(label_horizon) + 1)

        result = {}
        degraded = []

        with R.start(experiment_name=experiment_name):
            instruments_val = lgb_cfg["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"]
            if isinstance(instruments_val, list):
                log_cfg = copy.deepcopy(lgb_cfg)
                log_cfg["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"] = f"list_{len(instruments_val)}_stocks"
                R.log_params(**flatten_dict(log_cfg))
            else:
                R.log_params(**flatten_dict(lgb_cfg))
            R.log_params(dual_model="lightgbm+xgboost", label_norm=label_norm,
                         early_stop_metric=early_stop_metric, param_profile=param_profile,
                         with_linear=with_linear, with_catboost=with_catboost,
                         use_reweighter=use_reweighter, label_mode=label_mode)
            if factor_selection_snapshot is not None:
                R.save_objects(**{"factor_selection_snapshot.json": json.dumps(
                    factor_selection_snapshot, ensure_ascii=False, sort_keys=True,
                )})

            dataset = init_instance_by_config(lgb_cfg["dataset"])

            from core.adapters.qlib.fundval import record_dataset_contract
            record_dataset_contract(dataset, lgb_cfg, label_horizon, label_mode, label_norm)

            # --- 保存因子配置 (供预测时还原, MLflow params 有 1000 字符截断) ---
            if custom_factor_keys:
                handler_kwargs = lgb_cfg["dataset"]["kwargs"]["handler"]["kwargs"]
                R.save_objects(**{"factor_keys.json": json.dumps({
                    "factor_set": "WhiteBoxHandler",
                    "custom_factor_keys": custom_factor_keys,
                    "label_horizon": label_horizon,
                    "label_mode": label_mode,
                    "label_norm": label_norm,
                    "train_start": train_period[0],
                    "train_end": train_period[1],
                    "valid_start": valid_period[0],
                    "valid_end": valid_period[1],
                    "effective_train_end": lgb_cfg["dataset"]["kwargs"]["segments"]["train"][1],
                    "effective_train_start": lgb_cfg["dataset"]["kwargs"]["segments"]["train"][0],
                    "effective_valid_end": lgb_cfg["dataset"]["kwargs"]["segments"]["valid"][1],
                    "effective_valid_start": lgb_cfg["dataset"]["kwargs"]["segments"]["valid"][0],
                    "test_start": test_period[0] if test_period[0] else "",
                    "test_end": test_period[1] if test_period[1] else "",
                    "shared_processors": handler_kwargs.get("shared_processors", []),
                    "infer_processors": handler_kwargs.get("infer_processors", []),
                })})

            # --- 可选: 流动性样本加权 + 次新剔除 ---
            reweighter = None
            if use_reweighter:
                try:
                    from core.adapters.qlib.reweighter import LiquidityReweighter
                    reweighter = LiquidityReweighter(train_period[0], valid_period[1])
                except Exception as e:
                    logger.warning(f"[DualModel] Reweighter 构建失败, 等权训练: {e}")

            # --- 主线程预热 dataset: 避免并行训练时多线程竞争 prepare() ---
            # Qlib DatasetH 的首次 prepare() 会加载全部数据 + fit processors,
            # 不是线程安全的。预先在主线程调用一次, 后续调用全部走缓存。
            from qlib.data.dataset.handler import DataHandlerLP
            logger.info("[DualModel] 预加载数据 (避免并行竞争)...")
            dataset.prepare(
                ["train", "valid"],
                col_set=["feature", "label"],
                data_key=DataHandlerLP.DK_L,
            )
            logger.info("[DualModel] 数据预加载完成, 开始并行训练")

            # --- LightGBM + XGBoost 并行训练 ---
            from core.shared.infrastructure.thread_pool import thread_pool

            def _train_lgb():
                try:
                    model = init_instance_by_config(lgb_cfg["model"])
                    model.fit(dataset, reweighter=reweighter)
                    model._feature_contract = getattr(dataset.handler, '_feature_contract', None)
                    R.save_objects(**{"params_lgb.pkl": model})
                    logger.info("[DualModel] LightGBM 训练完成")
                    return model
                except Exception as e:
                    logger.error(f"[DualModel] LightGBM 训练失败: {e}")
                    return None

            def _train_xgb():
                try:
                    model = init_instance_by_config(xgb_cfg["model"])
                    model.fit(dataset, reweighter=reweighter)
                    model._feature_contract = getattr(dataset.handler, '_feature_contract', None)
                    R.save_objects(**{"params_xgb.pkl": model})
                    logger.info("[DualModel] XGBoost 训练完成")
                    return model
                except Exception as e:
                    logger.error(f"[DualModel] XGBoost 训练失败: {e}")
                    return None

            with thread_pool.default.borrow() as _tpool:
                _fut_lgb = _tpool.submit(_train_lgb)
                _fut_xgb = _tpool.submit(_train_xgb)
                lgb_model = _fut_lgb.result()
                xgb_model = _fut_xgb.result()

            if lgb_model is None:
                degraded.append("lightgbm")
            if xgb_model is None:
                degraded.append("xgboost")
            if lgb_model is None and xgb_model is None:
                raise RuntimeError("LightGBM 和 XGBoost 均训练失败")

            # --- 可选: ridge 线性基线 (第三模型, 提取稳定线性收益) ---
            linear_model = None
            pred_linear = None
            if with_linear:
                try:
                    linear_cfg = {"class": "LinearModel",
                                  "module_path": "qlib.contrib.model.linear",
                                  "kwargs": {"estimator": "ridge", "alpha": 0.05,
                                             "fit_intercept": True}}
                    linear_model = init_instance_by_config(linear_cfg)
                    linear_model.fit(dataset, reweighter=reweighter)
                    linear_model._feature_contract = getattr(dataset.handler, '_feature_contract', None)
                    R.save_objects(**{"params_linear.pkl": linear_model})
                    pred_linear = linear_model.predict(dataset)
                    logger.info("[DualModel] Linear 基线训练完成")
                except Exception as e:
                    logger.warning(f"[DualModel] Linear 基线训练失败, 双模型继续: {e}")
                    linear_model = None
                    pred_linear = None
                    degraded.append("linear")

            # --- 可选: CatBoost 第三模型 (对小数据/类别特征友好, 份额上限 25%) ---
            cat_model = None
            pred_cat = None
            if with_catboost:
                try:
                    cb_params = dict(DEFAULT_PARAMS["catboost"])
                    if catboost_params:
                        cb_params.update(catboost_params)
                    cat_cfg = {"class": "CatBoostModel",
                               "module_path": "qlib.contrib.model.catboost_model",
                               "kwargs": {**cb_params, "early_stopping_rounds": 50}}
                    cat_model = init_instance_by_config(cat_cfg)
                    cat_model.fit(dataset, reweighter=reweighter)
                    cat_model._feature_contract = getattr(dataset.handler, '_feature_contract', None)
                    R.save_objects(**{"params_catboost.pkl": cat_model})
                    pred_cat = cat_model.predict(dataset)
                    logger.info("[DualModel] CatBoost 训练完成")
                except Exception as e:
                    logger.warning(f"[DualModel] CatBoost 训练失败: {e}")
                    cat_model = None
                    pred_cat = None
                    degraded.append("catboost")

            # --- 等权融合预测 ---
            recorder = R.get_recorder()

            # --- 各模型 valid IC (作为融合权重依据, 避免 test 集选择偏差) ---
            from qlib.data.dataset.handler import DataHandlerLP
            valid_label_df = dataset.prepare("valid", col_set=["label"], data_key=DataHandlerLP.DK_R)
            valid_label_s = valid_label_df if isinstance(valid_label_df, pd.Series) else valid_label_df.iloc[:, 0]
            lgb_valid_ic_mean = None
            xgb_valid_ic_mean = None
            linear_valid_ic_mean = None
            cat_valid_ic_mean = None
            try:
                if lgb_model is not None:
                    pred_lgb_valid = lgb_model.predict(dataset, segment="valid")
                    lgb_vs = pd.Series(np.array(pred_lgb_valid).ravel(), index=valid_label_df.index)
                    lgb_valid_ic_mean = float(lgb_vs.groupby('datetime').corr(valid_label_s).mean())
                if xgb_model is not None:
                    pred_xgb_valid = xgb_model.predict(dataset, segment="valid")
                    xgb_vs = pd.Series(np.array(pred_xgb_valid).ravel(), index=valid_label_df.index)
                    xgb_valid_ic_mean = float(xgb_vs.groupby('datetime').corr(valid_label_s).mean())
                if linear_model is not None:
                    pred_lin_valid = linear_model.predict(dataset, segment="valid")
                    lin_vs = pd.Series(np.array(pred_lin_valid).ravel(), index=valid_label_df.index)
                    linear_valid_ic_mean = float(lin_vs.groupby('datetime').corr(valid_label_s).mean())
                if cat_model is not None:
                    pred_cat_valid = cat_model.predict(dataset, segment="valid")
                    cat_vs = pd.Series(np.array(pred_cat_valid).ravel(), index=valid_label_df.index)
                    cat_valid_ic_mean = float(cat_vs.groupby('datetime').corr(valid_label_s).mean())
            except Exception as e:
                logger.debug(f"[DualModel] valid IC 计算失败, 退回等权: {e}")

            if lgb_model is not None:
                sr_lgb = SignalRecord(lgb_model, dataset, recorder)
                sr_lgb.generate()
                pred_lgb = sr_lgb.load("pred.pkl")
                label = sr_lgb.load("label.pkl")
            else:
                pred_lgb = None

            if xgb_model is not None:
                sr_xgb = SignalRecord(xgb_model, dataset, recorder)
                sr_xgb.generate()
                pred_xgb = sr_xgb.load("pred.pkl")
                if pred_lgb is None:
                    label = sr_xgb.load("label.pkl")
            else:
                pred_xgb = None

            # 各模型独立 test IC (仅用于报告)
            label_s = label if isinstance(label, pd.Series) else label.iloc[:, 0]
            lgb_ic_mean = None
            xgb_ic_mean = None
            linear_ic_mean = None
            cat_ic_mean = None
            try:
                if pred_lgb is not None:
                    lgb_s = pd.Series(np.array(pred_lgb).ravel(), index=label.index)
                    lgb_ic = lgb_s.groupby('datetime').corr(label_s)
                    lgb_ic_mean = float(lgb_ic.mean())
                if pred_xgb is not None:
                    xgb_s = pd.Series(np.array(pred_xgb).ravel(), index=label.index)
                    xgb_ic = xgb_s.groupby('datetime').corr(label_s)
                    xgb_ic_mean = float(xgb_ic.mean())
                if pred_linear is not None:
                    lin_s = pd.Series(np.array(pred_linear).ravel(), index=label.index)
                    linear_ic_mean = float(lin_s.groupby('datetime').corr(label_s).mean())
                if pred_cat is not None:
                    cat_s = pd.Series(np.array(pred_cat).ravel(), index=label.index)
                    cat_ic_mean = float(cat_s.groupby('datetime').corr(label_s).mean())
            except Exception as e:
                logger.debug(f"[DualModel] 分类模型IC计算失败: {e}")

            # 按 valid IC 加权融合 (IC 缺失或均≤0 退回等权)
            # linear / catboost 各份额上限 25%, 两者共存时各上限 18% (总附加 ≤35%)
            w_lgb, w_xgb = _ic_fusion_weights(lgb_valid_ic_mean, xgb_valid_ic_mean)
            if pred_lgb is not None and pred_xgb is not None and w_lgb != 0.5:
                try:
                    from scipy.stats import spearmanr
                    corr, _ = spearmanr(np.array(pred_lgb).ravel(), np.array(pred_xgb).ravel())
                    if corr > 0.8:
                        alpha = max(0.0, (corr - 0.8) / 0.2)
                        w_lgb = w_lgb * (1 - alpha) + 0.5 * alpha
                        w_xgb = 1.0 - w_lgb
                        logger.info(f"[DualModel] 预测相关度 {corr:.3f} (>0.8), 惩罚拉回 w_lgb={w_lgb:.3f}")
                except Exception as e:
                    logger.debug(f"[DualModel] 预测相关度计算失败: {e}")
            # ── 统一 IC 竞争融合: 3+模型时按 IC 归一化, 单一模型上限 30% ──
            # 仅2模型(默认LGB+XGB)时跳过cap, 直接用 _ic_fusion_weights 结果
            w_linear = 0.0
            w_cat = 0.0
            CAP = 0.3
            # 收集有效模型及其 valid IC (竞争融合权重依据)
            entries = []
            if pred_lgb is not None and lgb_valid_ic_mean is not None and lgb_valid_ic_mean > 0:
                entries.append(("lgb", lgb_valid_ic_mean))
            if pred_xgb is not None and xgb_valid_ic_mean is not None and xgb_valid_ic_mean > 0:
                entries.append(("xgb", xgb_valid_ic_mean))
            if pred_linear is not None and linear_valid_ic_mean is not None and linear_valid_ic_mean > 0:
                entries.append(("linear", linear_valid_ic_mean))
            if pred_cat is not None and cat_valid_ic_mean is not None and cat_valid_ic_mean > 0:
                entries.append(("cat", cat_valid_ic_mean))

            has_aux = any(e[0] in ("linear", "cat") for e in entries)
            if entries and has_aux:
                total = sum(v for _, v in entries)
                raw = {k: v / total for k, v in entries}

                # 应用 30% 上限, 超出部分按其他模型 IC 比例重新分配
                for _ in range(10):  # 迭代收敛, 最多10轮
                    excess = 0.0
                    capped = 0
                    for k in list(raw.keys()):
                        if raw[k] > CAP:
                            excess += raw[k] - CAP
                            raw[k] = CAP
                            capped += 1
                    if excess < 1e-6 or capped == 0:
                        break
                    free = {k: v for k, v in raw.items() if v < CAP}
                    if not free:
                        # 所有模型都被 cap → 重新归一化使总和=1
                        if excess > 1e-6:
                            total_capped = sum(raw.values())
                            if total_capped > 0:
                                for k in raw:
                                    raw[k] = raw[k] / total_capped
                        break
                    free_total = sum(free.values())
                    for k in free:
                        raw[k] += excess * (free[k] / free_total)

                w_linear = raw.get("linear", 0.0)
                w_cat = raw.get("cat", 0.0)
                # LGB/XGB 按各自 IC 比例分剩下的
                lgb_share = raw.get("lgb", 0.0)
                xgb_share = raw.get("xgb", 0.0)
                if lgb_share + xgb_share > 0:
                    w_lgb = lgb_share / (lgb_share + xgb_share)
                    w_xgb = xgb_share / (lgb_share + xgb_share)

            if pred_lgb is not None and pred_xgb is not None:
                main_pair = w_lgb * np.array(pred_lgb).ravel() + w_xgb * np.array(pred_xgb).ravel()
                main_w = max(0.0, 1.0 - w_linear - w_cat)
                pred_arr = main_w * main_pair
                if w_linear > 0:
                    pred_arr = pred_arr + w_linear * np.array(pred_linear).ravel()
                if w_cat > 0:
                    pred_arr = pred_arr + w_cat * np.array(pred_cat).ravel()
                logger.info(f"[DualModel] 统一IC竞争融合: w_lgb={w_lgb:.3f} w_xgb={w_xgb:.3f} "
                            f"main_w={main_w:.3f} w_linear={w_linear:.3f} w_cat={w_cat:.3f} (总权重=1.0) "
                            f"(LGB_IC={lgb_ic_mean} XGB_IC={xgb_ic_mean} "
                            f"LINEAR_IC={linear_ic_mean} CAT_IC={cat_ic_mean})")
            elif pred_lgb is not None:
                pred_arr = np.array(pred_lgb).ravel()
                if w_linear > 0 or w_cat > 0:
                    logger.warning("[DualModel] 仅LGB可用, linear/cat 辅助模型结果被跳过")
            else:
                pred_arr = np.array(pred_xgb).ravel()
                if w_linear > 0 or w_cat > 0:
                    logger.warning("[DualModel] 仅XGB可用, linear/cat 辅助模型结果被跳过")

            pred = pd.DataFrame({"score": pred_arr}, index=label.index)

            # IC/ICIR 计算
            try:
                ic = pred["score"].groupby('datetime').corr(label if isinstance(label, pd.Series) else label.iloc[:, 0])
                ic_mean = float(ic.mean())
                ic_std = float(ic.std()) if ic.std() > 0 else 0
                icir = ic_mean / ic_std if ic_std > 0 else None
            except Exception as e:
                logger.debug(f"[Eval] IC/ICIR 计算失败: {e}")
                ic = ic_mean = icir = None

            # Rank IC
            ric_mean = None
            try:
                ric = pred["score"].groupby('datetime').corr(
                    label if isinstance(label, pd.Series) else label.iloc[:, 0], method='spearman')
                ric_mean = float(ric.mean())
            except Exception as e:
                logger.debug(f"[Eval] RankIC 计算失败: {e}")

            # 持久化 IC 到 MLflow metrics（list_trained_models 从文件读取）
            metrics = {}
            if ic_mean is not None:
                metrics["IC"] = ic_mean
            if ric_mean is not None:
                metrics["Rank_IC"] = ric_mean
            if icir is not None:
                metrics["ICIR"] = icir
            if lgb_ic_mean is not None:
                metrics["LGB_IC"] = lgb_ic_mean
            if xgb_ic_mean is not None:
                metrics["XGB_IC"] = xgb_ic_mean
            if lgb_valid_ic_mean is not None:
                metrics["LGB_VALID_IC"] = lgb_valid_ic_mean
            if xgb_valid_ic_mean is not None:
                metrics["XGB_VALID_IC"] = xgb_valid_ic_mean
            if pred_lgb is not None and pred_xgb is not None:
                metrics["LGB_W"] = w_lgb
                metrics["XGB_W"] = w_xgb
            if linear_ic_mean is not None:
                metrics["LINEAR_IC"] = linear_ic_mean
            if w_linear > 0:
                metrics["LINEAR_W"] = w_linear
            if cat_ic_mean is not None:
                metrics["CAT_IC"] = cat_ic_mean
            if w_cat > 0:
                metrics["CAT_W"] = w_cat
            # 扩展评估: 月度胜率/分年ICIR/五分位多空 (报告修正: 评估体系补齐)
            try:
                metrics.update(_extended_eval_metrics(ic, pred["score"], label_s, label_horizon=label_horizon))
            except Exception as e:
                logger.warning(f"[DualModel] 扩展评估指标失败: {e}")
            if metrics:
                R.log_metrics(**metrics)

            # 训练结果写入 MySQL (供模型对比和历史追踪)
            try:
                from core.data import Database
                import datetime as _dt
                if Database.is_available():
                    run_id = f"{experiment_name}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    ex = metrics  # 包含扩展评估指标
                    inst_str = instruments if isinstance(instruments, str) else f"list_{len(instruments)}_stocks"
                    Database.execute(
                        "INSERT INTO training_results "
                        "(run_id, model_type, factor_set, instruments, "
                        "train_start, train_end, valid_start, valid_end, test_start, test_end, "
                        "label_horizon, ic_mean, ric_mean, icir, "
                        "lgb_ic_mean, lgb_weight, xgb_ic_mean, xgb_weight, "
                        "linear_ic_mean, linear_weight, cat_ic_mean, cat_weight, "
                        "monthly_win_rate, annual_icir, long_short_spread, max_neg_ic_months, "
                        "custom_factor_keys, extra_metrics) "
                        "VALUES (%s,%s,%s,%s, %s,%s,%s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s,%s, %s,%s)",
                        (run_id, "dual", factor_set, inst_str,
                         str(train_period[0]), str(train_period[1]),
                         str(valid_period[0]), str(valid_period[1]),
                         str(test_period[0]) if test_period[0] else None,
                         str(test_period[1]) if test_period[1] else None,
                         label_horizon, ic_mean, ric_mean, icir,
                         lgb_ic_mean, w_lgb, xgb_ic_mean, w_xgb,
                         linear_ic_mean, w_linear if w_linear > 0 else None,
                         cat_ic_mean, w_cat if w_cat > 0 else None,
                         ex.get("IC_WIN_RATE_M"), ex.get("ICIR_YEARLY_MIN"),
                         ex.get("Q5_Q1_SPREAD"), ex.get("MAX_NEG_IC_MONTHS"),
                         json.dumps(custom_factor_keys) if custom_factor_keys else None,
                         json.dumps(ex) if ex else None),
                    )
                    logger.info(f"[DualModel] 训练结果已入库 run_id={run_id} IC={ic_mean:.4f}" if ic_mean else f"[DualModel] 训练结果已入库 run_id={run_id}")
            except Exception as e:
                logger.warning(f"[DualModel] 训练结果入库失败: {e}")

            result = {
                "models": {"lightgbm": lgb_model, "xgboost": xgb_model,
                           "linear": linear_model, "catboost": cat_model},
                "predictions": pred,
                "label": label,
                "ic": ic,
                "ic_mean": ic_mean,
                "ric_mean": ric_mean,
                "icir": icir,
                "lgb_ic_mean": lgb_ic_mean,
                "lgb_valid_ic_mean": lgb_valid_ic_mean,
                "xgb_ic_mean": xgb_ic_mean,
                "xgb_valid_ic_mean": xgb_valid_ic_mean,
                "linear_ic_mean": linear_ic_mean,
                "lgb_weight": w_lgb,
                "xgb_weight": w_xgb,
                "linear_weight": w_linear,
                "cat_ic_mean": cat_ic_mean,
                "cat_weight": w_cat,
                "degraded": degraded if degraded else None,
                "experiment_id": str(getattr(recorder, "experiment_id", "")),
                "run_id": str(getattr(recorder, "id", "")),
            }

        return result

    def train_with_optuna(
        self, factor_set, instruments,
        train_period, valid_period, test_period,
        label_horizon=5, n_trials=50, experiment_name="a_share_quant_optuna",
        custom_factor_keys=None, label_norm="none",
        model_name="lightgbm", timeout=None,
        skip_final_train=False, param_profile="default",
        label_mode="tradable_open_to_open",
    ):
        """Optuna 超参自动搜索 — 在 train/valid 上最小化 -IC, 返回最佳参数和结果。

        Parameters
        ----------
        n_trials : int — 搜索轮数 (默认50)
        timeout : int — 超时秒数 (None=不限)
        model_name : str — "lightgbm" | "xgboost"

        Returns
        -------
        dict: best_params, best_ic (valid 集 IC), study_summary
        """
        _require_tradable_label_mode(label_mode)
        import numpy as np
        from qlib.utils import init_instance_by_config
        from qlib.workflow import R

        if model_name not in ("lightgbm", "xgboost"):
            raise ValueError(f"Optuna 仅支持 lightgbm/xgboost, 收到: {model_name}")

        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna 未安装, 请执行: pip install optuna")

        logger.info(f"[Optuna] 启动 {model_name} 超参搜索, {n_trials} 轮")

        base_params = dict(DEFAULT_PARAMS[model_name])
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        )

        # 预建数据集 (数据集不依赖 trial 参数, 跨 trial 复用, 省 170s×N 轮)
        logger.info(f"[Optuna] 预初始化数据集 ({n_trials}轮复用)...")
        _base_config = self._build_task_config(
            model_name, factor_set, instruments,
            train_period, valid_period, test_period,
            model_params=base_params, label_horizon=label_horizon,
            custom_factor_keys=custom_factor_keys,
            label_norm=label_norm,
            label_mode=label_mode,
            purge_gap=int(label_horizon) + 1,
        )
        _model_class = _base_config["model"]["class"]
        _model_module = _base_config["model"]["module_path"]
        _shared_dataset = init_instance_by_config(_base_config["dataset"])
        logger.info(f"[Optuna] 数据集就绪, 开始搜索 ({n_trials} 轮)")

        def _objective(trial):
            if model_name == "lightgbm":
                params = {
                    "num_leaves": trial.suggest_int("num_leaves", 31, 255),
                    "max_depth": trial.suggest_int("max_depth", 5, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0, log=True),
                    "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                    "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                }
            else:  # xgboost
                params = {
                    "max_depth": trial.suggest_int("max_depth", 4, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                }

            merged = {**base_params, **params}
            merged["early_stopping_rounds"] = 50
            if model_name == "lightgbm":
                merged["verbose_eval"] = -1

            # 复用预建数据集, 只重建模型 (省 ~170s/轮)
            model_cfg = {"class": _model_class, "module_path": _model_module, "kwargs": merged}

            try:
                with R.start(experiment_name=experiment_name):
                    model = init_instance_by_config(model_cfg)
                    model.fit(_shared_dataset)

                    from qlib.data.dataset.handler import DataHandlerLP
                    pred_valid = model.predict(_shared_dataset, segment="valid")
                    label_valid = _shared_dataset.prepare("valid", col_set=["label"], data_key=DataHandlerLP.DK_R)

                    pred_s = pd.Series(np.array(pred_valid).ravel(), index=label_valid.index)
                    label_s = label_valid if isinstance(label_valid, pd.Series) else label_valid.iloc[:, 0]
                    ic = pred_s.groupby("datetime").corr(label_s)
                    ic_mean = float(ic.mean())

                return ic_mean
            except Exception as e:
                logger.error(f"[Optuna] trial {trial.number} 失败: {e}", exc_info=True)
                raise optuna.TrialPruned()

        study.optimize(_objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

        best_params = study.best_params
        best_ic = study.best_value
        logger.info(f"[Optuna] 完成 {len(study.trials)} 轮, best_IC={best_ic:.4f}, "
                    f"best_params={best_params}")

        if skip_final_train:
            return {
                "best_params": best_params,
                "best_ic": best_ic,
                "best_trial": study.best_trial.number,
                "n_trials": len(study.trials),
            }

        # 用最佳参数完整训练一次 (双模型: LGB+XGB)
        if model_name == "lightgbm":
            lgb_best = {**base_params, **best_params, "early_stopping_rounds": 50, "verbose_eval": -1}
            xgb_best = None
        else:
            lgb_best = None
            xgb_best = {**DEFAULT_PARAMS["xgboost"], **best_params, "early_stopping_rounds": 50}

        result = self.train_dual_model(
            factor_set=factor_set, instruments=instruments,
            train_period=train_period, valid_period=valid_period, test_period=test_period,
            label_horizon=label_horizon, experiment_name=experiment_name,
            custom_factor_keys=custom_factor_keys, label_norm=label_norm,
            lgb_params=lgb_best, xgb_params=xgb_best,
            param_profile=param_profile,
            label_mode=label_mode,
        )

        return {
            "best_params": best_params,
            "best_ic": best_ic,  # valid 集 IC (非 test)
            "ic_mean": result.get("ic_mean"),
            "ric_mean": result.get("ric_mean"),
            "icir": result.get("icir"),
            "lgb_valid_ic_mean": result.get("lgb_valid_ic_mean"),
            "xgb_valid_ic_mean": result.get("xgb_valid_ic_mean"),
            "n_trials": len(study.trials),
            "study_trials": [
                {"number": t.number, "value": t.value, "params": t.params}
                for t in study.trials if t.value is not None
            ],
        }

    def predict_dual(self, experiment_id, run_id, dataset_config=None,
                     predict_start=None, predict_end=None, instruments=None,
                     dataset=None):
        """双模型预测: 从 MLflow 加载两个模型, 等权融合。

        dataset_config 为 None 时自动从 artifacts/factor_keys.json 还原训练配置。
        predict_start/predict_end: 预测日期范围，不传则回退训练时的 test 段。
        instruments: 股票池，不传则用 "all"。
        dataset: 预建的 DatasetH，传入则跳过 init_instance_by_config（复用避免重复加载）。
        """
        import numpy as np
        from qlib.utils import init_instance_by_config
        from pathlib import Path
        import pickle, json

        from core.shared.disabled_models import is_disabled
        if is_disabled(experiment_id, run_id):
            raise RuntimeError(f"模型已停用: {experiment_id}/{run_id}")

        if instruments == "stable_history_500d":
            instruments = "a500"

        from core.shared import config as _app_config
        _base = Path(_app_config.QLIB_MLFLOW_URI.replace("file:", ""))
        if not _base.exists():
            _base = Path("mlruns")
        mlruns_dir = _base / experiment_id / run_id / "artifacts"
        params_dir = mlruns_dir.parent / "params"

        from core.adapters.qlib.fundval import require_run_contract
        require_run_contract(mlruns_dir)

        if (mlruns_dir / "params_lstm.pkl").exists():
            from core.adapters.qlib.lstm_fusion import predict_lgb_lstm_fusion
            return predict_lgb_lstm_fusion(
                experiment_id, run_id,
                predict_start=predict_start or "2025-01-01",
                predict_end=predict_end or "2026-06-30",
                instruments=instruments or "all",
                dataset=dataset,
            )

        # 自动还原训练时的数据集配置 (含 WhiteBoxHandler + factor_keys)
        if dataset_config is None:
            fk_path = mlruns_dir / "factor_keys.json"
            if fk_path.exists():
                # R.save_objects 序列化为 pickle，需先尝试 pickle 再回退 JSON
                try:
                    with open(fk_path, "rb") as f:
                        raw = pickle.load(f)
                    if isinstance(raw, str):
                        fk_info = json.loads(raw)
                    elif isinstance(raw, dict):
                        fk_info = raw
                    else:
                        fk_info = {}
                except Exception:
                    with open(fk_path, "r") as f:
                        fk_info = json.load(f)
                factor_set = fk_info.get("factor_set", "Alpha158")
                factor_keys = fk_info.get("custom_factor_keys", [])
                label_horizon = fk_info.get("label_horizon", 20)
                label_mode = fk_info.get("label_mode", "close_to_close")
            else:
                # 回退: 从 MLflow params 读 + 精选集补全 (params 可能被截断到 1000 字符)
                fk_info = {}
                factor_set = "Alpha158"
                factor_keys = []
                label_horizon = 5
                label_mode = "close_to_close"
                if params_dir.exists():
                    for pf in params_dir.iterdir():
                        try:
                            val = pf.read_bytes().decode("utf-8", errors="replace").strip()
                            if pf.name == "dataset.kwargs.handler.class":
                                factor_set = val
                            elif pf.name == "dataset.kwargs.handler.kwargs.factor_keys":
                                if val and val != "None":
                                    factor_keys = [k.strip().strip("'\"") for k in val.strip("[]").split(",") if k.strip()]
                            elif pf.name == "dataset.kwargs.handler.kwargs.label":
                                import re
                                m = re.search(r'Ref\(\$open,\s*-(\d+)\)\s*/\s*Ref\(\$open,\s*-1\)', val)
                                if m:
                                    label_horizon = int(m.group(1)) - 1
                                    label_mode = "tradable_open_to_open"
                                else:
                                    m = re.search(r'Ref\(\$close,\s*-(\d+)\)', val)
                                    if m:
                                        label_horizon = int(m.group(1))
                        except Exception as e:
                            logger.debug(f"[DualPredict] label_horizon 解析失败: {e}")
                # params 截断时从精选集补全
                if factor_set == "WhiteBoxHandler" and (not factor_keys or len(factor_keys) < 10):
                    try:
                        from core.domain.factors.registry import get_pool
                        curated = get_pool().get_curated_factors("latest")
                        if curated:
                            factor_keys = curated
                            logger.info(f"[DualPredict] 从精选集补全 factor_keys: {len(factor_keys)} 个")
                    except Exception as e:
                        logger.debug(f"[DualPredict] 精选集补全失败: {e}")
                if not factor_keys:
                    factor_set = "Alpha158"
                    logger.warning("[DualPredict] 无法获取 factor_keys, 回退 Alpha158")

            if predict_start is None:
                predict_start = "2025-01-01"
            if predict_end is None:
                predict_end = "2026-06-30"
            if instruments is None:
                instruments = "all"

            # 从 MLflow 恢复训练起止日期 (归一化区间), 避免硬编码
            train_start = fk_info.get("train_start") or _read_param(params_dir, "dataset.kwargs.segments.train", 0)
            train_end = fk_info.get("train_end") or _read_param(params_dir, "dataset.kwargs.segments.train", 1)
            if not train_start:
                train_start = "2015-01-01"
            if not train_end:
                train_end = "2024-12-31"
            # data_start: 预测起始前3年足够特征计算，避免加载10+年历史数据
            ps_dt = predict_start
            if isinstance(ps_dt, str):
                from datetime import datetime
                ps_dt = datetime.strptime(ps_dt[:10], "%Y-%m-%d")
            data_start = f"{ps_dt.year - 3}-01-01"
            # 只用 train_start 填充早期数据缺口, 不使用 train_start 扩展 data_start
            # (train_start < data_start 时会拉入不必要的历史数据, 改变 z-score 归一化窗口)

            handler_module = "core.adapters.qlib.custom_handler" if factor_set == "WhiteBoxHandler" else "qlib.contrib.data.handler"
            handler_kwargs = {
                "instruments": instruments,
                "start_time": data_start,
                "end_time": predict_end,
                "fit_start_time": train_start,
                "fit_end_time": train_end,
                "label": [_build_label_expression(label_horizon, label_mode)],
            }
            if factor_set == "WhiteBoxHandler" and factor_keys:
                handler_kwargs["factor_keys"] = factor_keys

            shared_procs = fk_info.get("shared_processors", [])
            infer_procs = fk_info.get("infer_processors", [
                {"class": "ProcessInf", "kwargs": {}},
                {"class": "Fillna", "kwargs": {}},
            ])
            if shared_procs:
                handler_kwargs["shared_processors"] = shared_procs
            handler_kwargs["infer_processors"] = infer_procs

            dataset_config = {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": {
                        "class": factor_set,
                        "module_path": handler_module,
                        "kwargs": handler_kwargs,
                    },
                    "segments": {"test": (predict_start, predict_end)},
                },
            }

        if dataset is None:
            logger.info(f"[DualPredict] 创建新 dataset {experiment_id}/{run_id}")
            from core.adapters.qlib.config import build_dataset_cached
            dataset = build_dataset_cached(dataset_config)
        else:
            logger.info(f"[DualPredict] 复用 dataset {experiment_id}/{run_id} (has_cache={hasattr(dataset, '_cached_prepare_result')})")

        from core.adapters.qlib.fundval import require_dataset_contract
        trained_keys = require_run_contract(mlruns_dir)
        require_dataset_contract(dataset, trained_keys)

        # 预先 prepare 测试数据一次并缓存到 dataset 对象上（跨 predict_dual 调用复用）
        _orig_prepare = dataset.prepare

        def _cached_prepare_wrapper(*args, **kwargs):
            if not hasattr(dataset, '_cached_prepare_result'):
                logger.info(f"[DualPredict] prepare 首次调用, 缓存 args={args[:2] if len(args)>1 else args}")
                dataset._cached_prepare_result = _orig_prepare(*args, **kwargs)
            else:
                logger.info(f"[DualPredict] prepare 命中缓存, 跳过 {len(args)} 个参数")
            return dataset._cached_prepare_result

        dataset.prepare = _cached_prepare_wrapper

        single_path = mlruns_dir / "params.pkl"
        if single_path.exists() and not (mlruns_dir / "params_lgb.pkl").exists():
            with open(single_path, "rb") as f:
                model = pickle.load(f)
            pred = model.predict(dataset)
            dataset.prepare = _orig_prepare
            logger.info(f"[ModelPredict] 单模型预测完成, {len(pred)} 条")
            return pred, dataset

        pred_series = []

        for model_file, model_name in [("params_lgb.pkl", "LightGBM"), ("params_xgb.pkl", "XGBoost")]:
            model_path = mlruns_dir / model_file
            if model_path.exists():
                try:
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                    p = model.predict(dataset)
                    pred_series.append(p)
                    logger.info(f"[DualPredict] {model_name} 预测完成, {len(p)} 条")
                except Exception as e:
                    logger.warning(f"[DualPredict] {model_name} 加载/预测失败: {e}")
            else:
                logger.warning(f"[DualPredict] {model_name} 模型文件不存在: {model_path}")

        # 恢复原始 prepare（但保留 _cached_prepare_result 供后续复用）
        dataset.prepare = _orig_prepare

        if not pred_series:
            raise RuntimeError("双模型均不可用, 无法预测")
        if len(pred_series) < 2:
            raise RuntimeError("双模型预测要求两个模型均可用，请检查模型文件是否完整")

        # 按训练时记录的 IC 权重融合 (老模型无记录退回等权)
        w_lgb, w_xgb = dual_fusion_weights_from_run(mlruns_dir.parent)
        fused = w_lgb * pred_series[0] + w_xgb * pred_series[1]

        # 可选 linear 第三模型 (训练时记录了 LINEAR_W 才参与)
        w_lin = _read_run_metric(mlruns_dir.parent, "LINEAR_W") or 0.0
        lin_path = mlruns_dir / "params_linear.pkl"
        if w_lin > 0 and lin_path.exists():
            try:
                with open(lin_path, "rb") as f:
                    lin_model = pickle.load(f)
                pred_linear = lin_model.predict(dataset)
                fused = (1 - w_lin) * fused + w_lin * pred_linear
            except Exception as e:
                logger.warning(f"[DualPredict] Linear 加载/预测失败, 双模型融合: {e}")
                w_lin = 0.0
        # 可选 CatBoost 模型 (训练时记录了 CAT_W 才参与)
        w_cat = _read_run_metric(mlruns_dir.parent, "CAT_W") or 0.0
        cat_path = mlruns_dir / "params_catboost.pkl"
        if w_cat > 0 and cat_path.exists():
            try:
                with open(cat_path, "rb") as f:
                    cat_model = pickle.load(f)
                pred_cat = cat_model.predict(dataset)
                fused = (1 - w_cat) * fused + w_cat * pred_cat
            except Exception as e:
                logger.warning(f"[DualPredict] CatBoost 加载/预测失败: {e}")
                w_cat = 0.0
        logger.info(f"[DualPredict] IC加权融合: w_lgb={w_lgb:.3f} w_xgb={w_xgb:.3f} "
                    f"w_linear={w_lin:.3f} w_cat={w_cat:.3f}")
        return fused, dataset
