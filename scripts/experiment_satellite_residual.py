"""卫星残差模型 + 合并双模型融合 实验脚本

两个实验, 共用同一套 walk-forward 折与数据/标签口径:

Exp1 (卫星残差模型): 冻结核心 LGB, 卫星因子拟合残差 (label_rank - core_pred_rank),
    分别用 Ridge 与 浅层 LGB 拟合, 看残差模型质量(residual IC)与最终合成 RankIC。

Exp2 (合并双模型融合): 核心+卫星合并因子集, 同时训 LGB_rank + Ridge 两个模型,
    对预测秩做融合, 看融合 RankIC。

用法:
    python scripts/experiment_satellite_residual.py --role trend --folds 3
    python scripts/experiment_satellite_residual.py --role rebound --folds 3
"""
import argparse
import datetime as dt
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

from core.adapters.qlib.config import init_qlib
from core.adapters.qlib.models import PARAM_PROFILES, QlibModelManager
from core.adapters.qlib.factor_set_evaluator import (
    _make_rounds,
    _matured,
    _rank_by_date,
    _as_series,
    _select_feature_keys,
    _slice_period,
    _PreparedFactorDataset,
)


def load_sets():
    with open(os.path.join(ROOT, "core", "domain", "factors", "factor_pool_state.json"),
              encoding="utf-8") as f:
        pool = json.load(f)
    cur = pool["curated"]
    return {
        "core": list(cur["core_base"]["flat_list"]),
        "trend": list(cur["trend_satellite"]["flat_list"]),
        "rebound": list(cur["rebound_satellite"]["flat_list"]),
    }


def _daily_corr(frame):
    return frame.groupby(level="datetime").apply(
        lambda g: g["prediction"].corr(g["label"], method="spearman")
    ).dropna()


def _metrics(daily):
    if daily is None or len(daily) == 0:
        return {"rank_ic": None, "icir": None, "positive_ratio": None, "n": 0}
    daily = daily.astype(float)
    return {
        "rank_ic": float(daily.mean()),
        "icir": float(daily.mean() / daily.std()) if len(daily) > 1 and daily.std() > 0 else None,
        "positive_ratio": float((daily > 0).mean()),
        "n": int(len(daily)),
    }


def _feature_matrix(frame, keys):
    """取出因子特征矩阵 (numpy), 索引与 frame 一致。"""
    feats = _select_feature_keys(frame, keys)
    return feats, feats.values.astype(float)


def _aligned_residual(core_pred, label):
    """core_pred 与 label 对齐后, 计算 label_rank - core_pred_rank。"""
    frame = pd.concat({"prediction": core_pred, "label": label}, axis=1).dropna()
    frame["pred_rank"] = _rank_by_date(frame["prediction"])
    frame["label_rank"] = _rank_by_date(frame["label"])
    frame["residual"] = frame["label_rank"] - frame["pred_rank"]
    return frame


def _fit_lambda(resid_pred_valid, residual_valid):
    """在 valid 上做一元最小二乘, 求 final = core_rank + lam * resid_pred 的最优 lam, clip 到 [0,1]。"""
    rp = np.asarray(resid_pred_valid, float)
    rs = np.asarray(residual_valid, float)
    denom = float((rp * rp).sum())
    if denom <= 1e-12:
        return 1.0
    lam = float((rp * rs).sum()) / denom
    return float(np.clip(lam, 0.0, 1.0))


def run_experiment(role, max_folds=None):
    init_qlib()
    sets = load_sets()
    core_keys = sets["core"]
    sat_keys = sets[role]
    merged_keys = list(dict.fromkeys(core_keys + sat_keys))
    print(f"role={role} core={len(core_keys)} sat={len(sat_keys)} merged={len(merged_keys)}")

    config = {
        "start": "2022-01-01",
        "end": "2024-12-31",
        "selection_cutoff": "2025-06-30",
        "step_months": 3,
        "train_months": 60,
        "valid_months": 6,
        "test_months": 3,
        "purge_gap": 21,
        "label_horizon": 10,
        "label_mode": "tradable_open_to_open",
        "pool": "a500",
    }
    cutoff = dt.date.fromisoformat(config["selection_cutoff"])
    label_horizon = config["label_horizon"]
    rounds = [r for r in _make_rounds(config)
              if _matured(r[3][1], label_horizon) <= cutoff]
    if max_folds:
        rounds = rounds[:max_folds]
    print(f"folds to run: {len(rounds)} -> {[r[0] for r in rounds]}")

    manager = QlibModelManager()
    params = dict(PARAM_PROFILES["conservative"]["lightgbm"])
    arms = [
        "core_lgb", "merged_lgb", "merged_ridge", "merged_fusion",
        "sat_resid_ridge", "sat_resid_lgb",
    ]
    accum = {a: [] for a in arms}  # daily IC series per arm
    checkpoint_path = os.path.join(ROOT, "data", "experiments", f"exp_{role}.json")

    def _save_checkpoint(anchor_done):
        try:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({"role": role, "folds_done": anchor_done, "accum": accum}, f)
        except Exception:
            pass

    for idx, (anchor, train_period, valid_period, test_period) in enumerate(rounds, 1):
        t0 = time.perf_counter()
        print(f"[fold {idx}/{len(rounds)}] anchor={anchor}", flush=True)

        task_config = manager._build_task_config(
            "lightgbm_rank", "WhiteBoxHandler", config["pool"],
            train_period, valid_period, test_period,
            model_params=params, label_horizon=label_horizon,
            custom_factor_keys=merged_keys, label_norm="csrank",
            enable_cross_sectional=False, enable_winsorize=True,
            label_mode=config["label_mode"],
        )
        source = init_instance_by_config(task_config["dataset"])
        train_frame, valid_frame, test_frame = source.prepare(
            ["train", "valid", "test"], col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        frames = {"train": train_frame, "valid": valid_frame, "test": test_frame}
        label_test = _as_series(_PreparedFactorDataset(frames, core_keys).prepare("test", col_set="label"))

        # ---- 通用模型训练 helper ----
        def _fit_qlib(model_name, keys, segment_pred=("test",)):
            ds = _PreparedFactorDataset(frames, keys)
            m = init_instance_by_config(task_config["model"]) if model_name == "lightgbm_rank"                 else init_instance_by_config(
                    manager._build_task_config(model_name, "WhiteBoxHandler", config["pool"],
                                               train_period, valid_period, test_period,
                                               model_params=params, label_horizon=label_horizon,
                                               custom_factor_keys=keys, label_norm="csrank",
                                               enable_cross_sectional=False, enable_winsorize=True,
                                               label_mode=config["label_mode"])["model"])
            m.fit(ds, verbose_eval=0, log_metrics=False)
            out = {seg: _as_series(m.predict(ds, seg)) for seg in segment_pred}
            return out

        # ---- Exp2: 合并因子集双模型 (LGB_rank + Ridge) ----
        merged_lgb = _fit_qlib("lightgbm_rank", merged_keys, ("test",))["test"]
        # Ridge 用 sklearn, 目标 = csrank label
        X_tr, _ = _feature_matrix(train_frame, merged_keys)
        X_te, _ = _feature_matrix(test_frame, merged_keys)
        y_tr = _as_series(_PreparedFactorDataset(frames, merged_keys).prepare("train", col_set="label")).values
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, y_tr)
        merged_ridge_pred = pd.Series(ridge.predict(X_te), index=test_frame.index)

        mf = pd.concat({"prediction": 0.5 * _rank_by_date(merged_lgb) + 0.5 * _rank_by_date(merged_ridge_pred),
                        "label": label_test}, axis=1).dropna()
        accum["merged_fusion"].extend(_daily_corr(mf).tolist())

        ml = pd.concat({"prediction": merged_lgb, "label": label_test}, axis=1).dropna()
        accum["merged_lgb"].extend(_daily_corr(ml).tolist())
        mr = pd.concat({"prediction": merged_ridge_pred, "label": label_test}, axis=1).dropna()
        accum["merged_ridge"].extend(_daily_corr(mr).tolist())

        # ---- Exp1: 核心 LGB 冻结 + 卫星残差模型 ----
        core_preds = _fit_qlib("lightgbm_rank", core_keys, ("train", "valid", "test"))
        core_test = pd.concat({"prediction": core_preds["test"], "label": label_test}, axis=1).dropna()
        accum["core_lgb"].extend(_daily_corr(core_test).tolist())

        def _resid_frame(seg):
            lbl = _as_series(_PreparedFactorDataset(frames, core_keys).prepare(seg, col_set="label"))
            af = _aligned_residual(core_preds[seg], lbl)
            feats, _ = _feature_matrix(frames[seg], sat_keys)
            feats = feats.loc[af.index]
            return af, feats

        af_tr, Xs_tr = _resid_frame("train")
        af_va, Xs_va = _resid_frame("valid")
        af_te, Xs_te = _resid_frame("test")

        # 卫星残差 -> Ridge
        rr = Ridge(alpha=1.0)
        rr.fit(Xs_tr.values, af_tr["residual"].values)
        resid_pred_r_va = rr.predict(Xs_va.values)
        resid_pred_r_te = rr.predict(Xs_te.values)
        lam_r = _fit_lambda(resid_pred_r_va, af_va["residual"].values)
        final_r = pd.concat({
            "prediction": af_te["pred_rank"] + lam_r * pd.Series(resid_pred_r_te, index=af_te.index),
            "label": af_te["label"],
        }, axis=1).dropna()
        accum["sat_resid_ridge"].extend(_daily_corr(final_r).tolist())

        # 卫星残差 -> 浅层 LGB
        lr = LGBMRegressor(
            objective="regression", learning_rate=0.03, max_depth=3, num_leaves=15,
            n_estimators=300, min_child_samples=100, subsample=0.7, subsample_freq=1,
            colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0, n_jobs=4,
            random_state=42, verbose=-1,
        )
        lr.fit(
            Xs_tr.values, af_tr["residual"].values,
            eval_set=[(Xs_va.values, af_va["residual"].values)],
            callbacks=[__import__("lightgbm").early_stopping(20, verbose=False)],
        )
        resid_pred_l_va = lr.predict(Xs_va.values)
        resid_pred_l_te = lr.predict(Xs_te.values)
        lam_l = _fit_lambda(resid_pred_l_va, af_va["residual"].values)
        final_l = pd.concat({
            "prediction": af_te["pred_rank"] + lam_l * pd.Series(resid_pred_l_te, index=af_te.index),
            "label": af_te["label"],
        }, axis=1).dropna()
        accum["sat_resid_lgb"].extend(_daily_corr(final_l).tolist())

        # 残差模型自身质量: corr(resid_pred, residual) on test
        print(f"    resid model corr(test): ridge={np.corrcoef(resid_pred_r_te, af_te['residual'].values)[0,1]:.4f} "
              f"lgb={np.corrcoef(resid_pred_l_te, af_te['residual'].values)[0,1]:.4f} "
              f"lam_r={lam_r:.2f} lam_l={lam_l:.2f}  ({time.perf_counter()-t0:.1f}s)", flush=True)

        del source, train_frame, valid_frame, test_frame, frames
        gc.collect()
        _save_checkpoint(anchor)

    # ---- 汇总 ----
    result = {}
    print("\n================== 汇总 ==================")
    print(f"{'arm':22s} {'RankIC':>8s} {'ICIR':>8s} {'正率':>7s} {'n_days':>7s}")
    for arm in arms:
        daily = pd.Series(accum[arm], dtype=float)
        m = _metrics(daily)
        result[arm] = m
        print(f"{arm:22s} {m['rank_ic'] if m['rank_ic'] is not None else float('nan'):>8.4f} "
              f"{m['icir'] if m['icir'] is not None else float('nan'):>8.4f} "
              f"{m['positive_ratio']:>7.4f} {m['n']:>7d}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="trend", choices=["trend", "rebound"])
    ap.add_argument("--folds", type=int, default=None)
    args = ap.parse_args()
    run_experiment(args.role, args.folds)
