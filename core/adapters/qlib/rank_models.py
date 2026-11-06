"""
RankIC 早停模型 — LGB/XGB 早停指标从验证集 MSE 改为逐日截面 RankIC (越高越好)。

qlib 原生 LGBModel 的坑:
  1. loss 只接受 {"mse","binary"}, 不能直接换 lambdarank
  2. _prepare_data 用 lgb.Dataset(x.values) 丢弃 MultiIndex, feval 拿不到 datetime 分组
  3. lightgbm 默认 l2 metric 会与 feval 一起参与早停, 必须 metric="None" 禁用
解法: 在 _prepare_data 后给 valid Dataset 附加 _day_ids/_label_rank, feval 按日
分组算 Spearman (label 预排名, 每轮只对 preds 排名后做 pearson, 等价且省一半排序)。
train 段未附加分组信息, feval 返回常数 0 且 lgb.train 识别其为训练集不参与早停。
"""
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.model.xgboost import XGBModel
from qlib.data.dataset.handler import DataHandlerLP
from qlib.workflow import R

logger = logging.getLogger(__name__)


def _daily_rank_ic(pred_values, label_rank, day_ids):
    """逐日截面 RankIC 均值: pred 逐日排名后与预排名 label 做逐日 pearson (=Spearman)。"""
    pred_rank = pd.Series(pred_values).groupby(day_ids).rank().values
    df = pd.DataFrame({"p": pred_rank, "l": label_rank, "d": day_ids})
    ic = df.groupby("d")[["p", "l"]].apply(lambda g: g["p"].corr(g["l"]))
    ic = ic.dropna()
    return float(ic.mean()) if len(ic) else 0.0


def _rank_ic_feval(preds, eval_data):
    """LightGBM feval: (name, value, is_higher_better)。is_higher_better=True 保证
    早停判定方向正确 (RankIC 越高越好), 未附加分组信息的数据集 (train) 返回常数 0。"""
    day_ids = getattr(eval_data, "_day_ids", None)
    label_rank = getattr(eval_data, "_label_rank", None)
    if day_ids is None or label_rank is None:
        return "rank_ic", 0.0, True
    return "rank_ic", _daily_rank_ic(preds, label_rank, day_ids), True


class RankLGBModel(LGBModel):
    """LightGBM, 验证集逐日截面 RankIC 早停。禁用默认 l2 metric。"""

    def __init__(self, loss="mse", early_stopping_rounds=50, num_boost_round=1000, **kwargs):
        super().__init__(loss=loss, early_stopping_rounds=early_stopping_rounds,
                         num_boost_round=num_boost_round, **kwargs)
        # 禁掉 builtin l2, 否则 l2 排在 feval 前面, first_metric_only 会盯错指标
        self.params["metric"] = "None"

    def _prepare_data(self, dataset, reweighter=None):
        ds_l = super()._prepare_data(dataset, reweighter)
        for lgb_ds, key in ds_l:
            if key == "valid":
                df = dataset.prepare("valid", col_set=["label"], data_key=DataHandlerLP.DK_L)
                day_ids = df.index.get_level_values("datetime").factorize(sort=False)[0]
                y = np.squeeze(df["label"].values)
                lgb_ds._day_ids = day_ids
                lgb_ds._label_rank = pd.Series(y).groupby(day_ids).rank().values
        return ds_l

    def fit(
        self,
        dataset,
        num_boost_round=None,
        early_stopping_rounds=None,
        verbose_eval=20,
        evals_result=None,
        reweighter=None,
        log_metrics=True,
        **kwargs,
    ):
        if evals_result is None:
            evals_result = {}
        ds_l = self._prepare_data(dataset, reweighter)
        ds, names = list(zip(*ds_l))
        early_stopping_callback = lgb.early_stopping(
            self.early_stopping_rounds if early_stopping_rounds is None else early_stopping_rounds,
            first_metric_only=True,
        )
        verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
        evals_result_callback = lgb.record_evaluation(evals_result)
        self.model = lgb.train(
            self.params,
            ds[0],
            num_boost_round=self.num_boost_round if num_boost_round is None else num_boost_round,
            valid_sets=ds,
            valid_names=names,
            feval=_rank_ic_feval,
            callbacks=[early_stopping_callback, verbose_eval_callback, evals_result_callback],
            **kwargs,
        )
        logger.info(f"[RankLGB] 早停于 {self.model.best_iteration} 轮 (valid rank_ic)")
        if log_metrics:
            for k in names:
                for key, val in evals_result[k].items():
                    name = f"{key}.{k}"
                    for epoch, m in enumerate(val):
                        R.log_metrics(**{name.replace("@", "_"): m}, step=epoch)


class RankXGBModel(XGBModel):
    """XGBoost, 验证集逐日截面 RankIC 早停 (custom_metric + maximize=True)。"""

    def fit(
        self,
        dataset,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=20,
        evals_result=None,
        reweighter=None,
        **kwargs,
    ):
        import xgboost as xgb
        from qlib.data.dataset.weight import Reweighter

        if evals_result is None:
            evals_result = {}
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
            y_train_1d, y_valid_1d = np.squeeze(y_train.values), np.squeeze(y_valid.values)
        else:
            raise ValueError("XGBoost doesn't support multi-label training")

        if reweighter is None:
            w_train = None
            w_valid = None
        elif isinstance(reweighter, Reweighter):
            w_train = reweighter.reweight(df_train)
            w_valid = reweighter.reweight(df_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        dtrain = xgb.DMatrix(x_train.values, label=y_train_1d, weight=w_train)
        dvalid = xgb.DMatrix(x_valid.values, label=y_valid_1d, weight=w_valid)

        day_ids = df_valid.index.get_level_values("datetime").factorize(sort=False)[0]
        label_rank = pd.Series(y_valid_1d).groupby(day_ids).rank().values

        def rank_ic_metric(predt, dmat):
            if dmat is not dvalid:
                return "rank_ic", 0.0
            return "rank_ic", _daily_rank_ic(predt, label_rank, day_ids)

        # 禁掉 builtin rmse, 让 rank_ic 成为 valid 唯一 (即最后一个) 早停指标
        params = dict(self._params)
        params["disable_default_eval_metric"] = 1
        self.model = xgb.train(
            params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            evals_result=evals_result,
            custom_metric=rank_ic_metric,
            maximize=True,
            **kwargs,
        )
        logger.info(f"[RankXGB] 早停于 {getattr(self.model, 'best_iteration', '?')} 轮 (valid rank_ic)")
        evals_result["train"] = list(evals_result["train"].values())[0]
        evals_result["valid"] = list(evals_result["valid"].values())[0]
