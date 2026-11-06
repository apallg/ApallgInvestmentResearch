"""
滚动时序训练 — 固定训练窗口按季度 anchor 滚动, 每轮独立 MLflow experiment。

与 core/rolling_screen.py 的 anchor 思想对齐: 每轮只用 anchor 之前数据训练+验证
(train 固定 train_years 年, valid 固定 valid_months 个月保证早停阈值稳定),
anchor 之后一个步长为 test 段 (真样本外, 与下一轮 train 无重叠)。
全部轮次 test 预测拼接为全期 OOS 评估 — 滚动训练唯一可信的总评指标。
experiment 命名 rolling_{pool}_{anchor}, _find_best_qlib_model 优先取最新 anchor 达标模型。
"""
import gc
import logging
import datetime as dt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def make_rolling_rounds(start_anchor, end_anchor, step_months=3, train_years=5, valid_months=6):
    """生成滚动轮次: [(anchor, train_period, valid_period, test_period), ...]
    各轮 test 段首尾相接不重叠。"""
    rounds = []
    cur = pd.Timestamp(start_anchor)
    endt = pd.Timestamp(end_anchor)
    while cur <= endt:
        a = cur
        valid_start = a - pd.DateOffset(months=valid_months)
        train_start = valid_start - pd.DateOffset(years=train_years)
        test_end = a + pd.DateOffset(months=step_months) - pd.Timedelta(days=1)
        rounds.append((
            a.strftime("%Y-%m-%d"),
            (train_start.strftime("%Y-%m-%d"),
             (valid_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")),
            (valid_start.strftime("%Y-%m-%d"),
             (a - pd.Timedelta(days=1)).strftime("%Y-%m-%d")),
            (a.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")),
        ))
        cur = cur + pd.DateOffset(months=step_months)
    return rounds


def rolling_train_dual(pool="csi300", pool_label=None, start_anchor="2023-01-01", end_anchor=None,
                       step_months=3, train_years=5, valid_months=6,
                       model_name="lightgbm_rank", train_dual=True,
                       factor_set="Alpha158", custom_factor_keys=None, factor_snapshots_by_anchor=None,
                       code_prefixes=None,
                       label_horizon=20, label_norm="csrank",
                       label_mode="tradable_open_to_open",
                       early_stop_metric="rank_ic", param_profile="default",
                       enable_crossover=False, use_reweighter=False, with_linear=False,
                       with_catboost=False, progress_cb=None,
                       enable_cross_sectional=False, enable_winsorize=True,
                       track_importance=False, feature_key_map=None, importance_top_n=50,
                       feature_names=None):
    """逐轮串行训练 (qlib handler 内存大, 不并行), 单轮失败记录后继续。

    factor_snapshots_by_anchor 有值时, 每轮使用训练截止日已生效的冻结因子快照。

    Returns
    -------
    {"rounds": [...], "oos": {...}, "n_planned": int, "n_trained": int}
    """
    from core.adapters.qlib.models import QlibModelManager, _extended_eval_metrics

    today = dt.date.today().strftime("%Y-%m-%d")
    if end_anchor is None:
        end_anchor = today

    plan = make_rolling_rounds(start_anchor, end_anchor, step_months, train_years, valid_months)
    manager = QlibModelManager()
    rounds = []
    oos_pred = []
    oos_label = []
    for i, (anchor, train_p, valid_p, test_p) in enumerate(plan):
        if test_p[0] > today:
            logger.info(f"[Rolling] anchor={anchor} test 段在未来, 跳过")
            continue
        test_p = (test_p[0], min(test_p[1], today))
        pool_short = pool_label or str(pool)
        if isinstance(pool_short, list):
            pool_short = f"combo_{len(pool_short)}"
        exp_name = f"rolling_{pool_short}_{anchor.replace('-', '')}"
        factor_snapshot = (factor_snapshots_by_anchor or {}).get(anchor)
        round_factor_set = (factor_snapshot or {}).get("factor_set", factor_set)
        round_factor_keys = (factor_snapshot or {}).get("custom_factor_keys", custom_factor_keys)
        logger.info(f"[Rolling] 轮次 {i + 1}/{len(plan)} anchor={anchor} "
                    f"train={train_p} valid={valid_p} test={test_p} → {exp_name}")
        if progress_cb:
            progress_cb(i, len(plan), anchor)
        try:
            if train_dual:
                result = manager.train_dual_model(
                    factor_set=round_factor_set, instruments=pool,
                    train_period=train_p, valid_period=valid_p, test_period=test_p,
                    label_horizon=label_horizon, experiment_name=exp_name,
                    label_mode=label_mode,
                    custom_factor_keys=round_factor_keys,
                    factor_selection_snapshot=factor_snapshot,
                    label_norm=label_norm, early_stop_metric=early_stop_metric,
                    param_profile=param_profile, enable_crossover=enable_crossover,
                    use_reweighter=use_reweighter, with_linear=with_linear,
                    with_catboost=with_catboost,
                    enable_cross_sectional=enable_cross_sectional,
                    enable_winsorize=enable_winsorize,
                )
            else:
                result = manager.train(
                    model_name=model_name, factor_set=round_factor_set, instruments=pool,
                    train_period=train_p, valid_period=valid_p, test_period=test_p,
                    label_horizon=label_horizon, experiment_name=exp_name,
                    label_mode=label_mode,
                    custom_factor_keys=round_factor_keys,
                    factor_selection_snapshot=factor_snapshot,
                    label_norm=label_norm,
                    enable_cross_sectional=enable_cross_sectional,
                    enable_winsorize=enable_winsorize,
                )
            round_entry = {
                "anchor": anchor, "experiment": exp_name,
                "train": train_p, "valid": valid_p, "test": test_p,
                "factor_set": round_factor_set,
                "custom_factor_keys": list(round_factor_keys or []),
                "factor_selection_cutoff": (factor_snapshot or {}).get("selection_cutoff"),
                "factor_selection_source": (factor_snapshot or {}).get("selection_source"),
                "factor_selection_effective_date": (factor_snapshot or {}).get("effective_date"),
                "factor_selection_fingerprint": (factor_snapshot or {}).get("fingerprint"),
                "ic_mean": result.get("ic_mean"), "ric_mean": result.get("ric_mean"),
                "icir": result.get("icir"),
                "lgb_weight": result.get("lgb_weight"),
                "xgb_weight": result.get("xgb_weight"),
                "degraded": result.get("degraded"),
                "experiment_id": result.get("experiment_id"),
                "run_id": result.get("run_id"),
            }
            if round_entry["experiment_id"] and round_entry["run_id"]:
                try:
                    from core.adapters.qlib.model_selection import get_model_for_factor_set
                    checked = get_model_for_factor_set(
                        round_entry["experiment_id"], round_entry["run_id"], round_factor_keys or [],
                        test_p[0], None, allow_test_window=True,
                    )
                    round_entry["artifact_fingerprint"] = checked["artifact_fingerprint"]
                    round_entry["effective_train_end"] = checked.get("train_end")
                    round_entry["effective_train_start"] = checked.get("train_start")
                    round_entry["effective_valid_end"] = checked.get("valid_end")
                    round_entry["effective_valid_start"] = checked.get("valid_start")
                except Exception as error:
                    logger.warning(f"[Rolling] 轮次artifact指纹生成失败: {error}")

            if track_importance:
                lgb_model = (result.get("models") or {}).get("lightgbm")
                if lgb_model is None:
                    lgb_model = result.get("model")
                if lgb_model is not None:
                    try:
                        # qlib LGBModel 存为 self.model; 原生 lightgbm 存为 self.booster_
                        booster = getattr(lgb_model, "model", None) or getattr(lgb_model, "booster_", None)
                        if booster is not None and hasattr(booster, "feature_importance"):
                            imp_split = booster.feature_importance(importance_type="split")
                            imp_gain = booster.feature_importance(importance_type="gain")
                            raw_names = booster.feature_name()
                            # qlib 内部列名 Column_X → 因子名 映射 (前N列=原始因子, 后N列=截面增强)
                            if feature_names:
                                display_names = [
                                    feature_names[i] if i < len(feature_names) else f"Aug_{raw_names[i]}"
                                    for i in range(len(raw_names))
                                ]
                            else:
                                display_names = raw_names
                            idx = np.argsort(imp_split)[::-1][:importance_top_n]
                            round_entry["feature_importance"] = {
                                display_names[i]: {"split": float(imp_split[i]), "gain": float(imp_gain[i])}
                                for i in idx
                            }
                        else:
                            logger.warning(f"[Rolling] anchor={anchor} 无法获取 booster (model类型: {type(lgb_model).__name__})")
                    except Exception as e:
                        logger.warning(f"[Rolling] anchor={anchor} 特征重要性提取失败: {e}")

                # 特征重要性写入 MySQL (供跨期追踪因子影响力变化)
                if round_entry.get("feature_importance") and feature_key_map:
                    try:
                        from core.data import Database
                        if Database.is_available():
                            imp_data = round_entry["feature_importance"]
                            sql = (
                                "INSERT INTO rolling_factor_importance "
                                "(job_id, round_idx, anchor_date, feature_name, factor_key, "
                                "importance_split, importance_gain, rank_split) "
                                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
                            )
                            ranked = sorted(imp_data.items(), key=lambda x: x[1]["split"], reverse=True)
                            for rank, (feat_name, imp) in enumerate(ranked):
                                fk = feature_key_map.get(feat_name)
                                try:
                                    Database.execute(sql, (
                                        exp_name, i, anchor, feat_name, fk,
                                        imp["split"], imp["gain"], rank + 1,
                                    ))
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.warning(f"[Rolling] 特征重要性入库失败: {e}")

            rounds.append(round_entry)
            if result.get("predictions") is not None and result.get("label") is not None:
                oos_pred.append(result["predictions"]["score"])
                lab = result["label"]
                oos_label.append(lab if isinstance(lab, pd.Series) else lab.iloc[:, 0])
        except Exception as e:
            import traceback
            logger.error(f"[Rolling] anchor={anchor} 训练失败: {e}\n{traceback.format_exc()}")
            rounds.append({"anchor": anchor, "experiment": exp_name, "error": str(e)})
        finally:
            gc.collect()

    oos = {}
    try:
        if oos_pred:
            pred_all = pd.concat(oos_pred)
            label_all = pd.concat(oos_label)
            ic = pred_all.groupby("datetime").corr(label_all)
            ric = pred_all.groupby("datetime").corr(label_all, method="spearman")
            ic_std = float(ic.std()) if ic.std() and ic.std() > 0 else 0
            oos = {
                "start": str(pred_all.index.get_level_values("datetime").min())[:10],
                "end": str(pred_all.index.get_level_values("datetime").max())[:10],
                "ic_mean": float(ic.mean()),
                "ric_mean": float(ric.mean()),
                "icir": float(ic.mean()) / ic_std if ic_std else None,
                "n_days": int(ic.dropna().shape[0]),
            }
            try:
                oos.update(_extended_eval_metrics(ic, pred_all, label_all, label_horizon=label_horizon))
            except Exception as e_ext:
                logger.warning(f"[Rolling] 扩展评估指标失败: {e_ext}")
            logger.info(f"[Rolling] 全期 OOS: IC={oos['ic_mean']:.4f} "
                        f"RankIC={oos['ric_mean']:.4f} ICIR={oos.get('icir')}")
    except Exception as e:
        logger.warning(f"[Rolling] OOS 拼接评估失败: {e}")

    result = {"rounds": rounds, "oos": oos, "n_planned": len(plan),
              "n_trained": len([r for r in rounds if "error" not in r])}

    if track_importance:
        valid_rounds = [r for r in rounds if "feature_importance" in r]
        if len(valid_rounds) >= 8:
            result["importance_summary"] = _accumulate_importance(
                rounds, feature_key_map, feature_names)
            result["elimination_candidates"] = _find_elimination_candidates(
                result["importance_summary"], rounds)

    return result


def _accumulate_importance(rounds, key_map=None, feature_names=None):
    valid = [r for r in rounds if "feature_importance" in r]
    if not valid:
        return {}

    # 构建 Column_X → factor_key 的位置映射 (CSFeatureAugment 会使列数膨胀)
    pos_map = {}
    if feature_names:
        for i, name in enumerate(feature_names):
            pos_map[f"Column_{i}"] = name

    all_imps = {}
    all_gains = {}
    above_median_count = {}
    above_median_gain_count = {}
    for r in valid:
        splits = [v["split"] for v in r["feature_importance"].values()]
        gains = [v.get("gain", 0.0) for v in r["feature_importance"].values()]
        med_split = np.median(splits)
        med_gain = np.median(gains)
        for name, imp in r["feature_importance"].items():
            all_imps.setdefault(name, []).append(imp["split"])
            g = imp.get("gain", 0.0)
            all_gains.setdefault(name, []).append(g)
            if imp["split"] > med_split:
                above_median_count[name] = above_median_count.get(name, 0) + 1
            if g > med_gain:
                above_median_gain_count[name] = above_median_gain_count.get(name, 0) + 1

    summary = {}
    for name, imps in all_imps.items():
        # 优先级: key_map表达式映射 > Column_X位置映射 > 原名
        key = name
        if key_map and name in key_map:
            key = key_map[name]
        elif name.startswith("Column_") and name in pos_map:
            key = pos_map[name]
        gains = all_gains[name]
        mean_split = float(np.mean(imps))
        mean_gain = float(np.mean(gains))
        wins = above_median_count.get(name, 0)
        gain_wins = above_median_gain_count.get(name, 0)
        win_rate = wins / len(imps) if imps else 0
        win_rate_gain = gain_wins / len(gains) if gains else 0
        volatility = float(np.std(imps) / (mean_split + 1e-8))
        summary[key] = {
            "mean_split": mean_split,
            "mean_gain": mean_gain,
            "win_rate": win_rate,
            "win_rate_gain": win_rate_gain,
            "volatility": volatility,
            "n_rounds": len(imps),
            "original_name": name,
        }
    return summary


def _find_elimination_candidates(summary, rounds, split_thr=10, win_rate_thr=0.30):
    if len(summary) < 10:
        return []

    candidates = []
    for key, info in summary.items():
        if info["mean_split"] <= split_thr and info["win_rate"] < win_rate_thr:
            candidates.append(key)

    max_remove = max(1, int(len(summary) * 0.15))
    candidates.sort(key=lambda k: summary[k]["mean_split"])
    return candidates[:max_remove]
