import datetime as dt
import gc
import time

from core.shared.logging import get_logger


logger = get_logger(__name__)


def _as_series(value):
    import pandas as pd
    if isinstance(value, pd.Series):
        return value.astype(float)
    if isinstance(value, pd.DataFrame):
        if "score" in value.columns:
            return value["score"].astype(float)
        return value.iloc[:, 0].astype(float)
    return pd.Series(value, dtype=float)


def _rank_by_date(series):
    return series.groupby(level="datetime").rank(pct=True)


def _matured(test_end, horizon):
    return dt.date.fromisoformat(str(test_end)[:10]) + dt.timedelta(days=int(horizon * 7 / 5) + 7)


def _make_rounds(config):
    import pandas as pd
    rounds = []
    current = pd.Timestamp(config["start"])
    end = pd.Timestamp(config["end"])
    step_months = int(config.get("step_months", 3))
    test_months = int(config.get("test_months", step_months))
    train_months = int(config.get("train_months", 60))
    valid_months = int(config.get("valid_months", 6))
    purge_days = int(int(config.get("purge_gap", 21)) * 7 / 5) + 1
    while current <= end:
        valid_start = current - pd.DateOffset(months=valid_months)
        train_start = valid_start - pd.DateOffset(months=train_months)
        train_end = valid_start - pd.Timedelta(days=purge_days)
        valid_end = current - pd.Timedelta(days=purge_days)
        test_end = current + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        rounds.append((
            current.strftime("%Y-%m-%d"),
            (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
            (valid_start.strftime("%Y-%m-%d"), valid_end.strftime("%Y-%m-%d")),
            (current.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")),
        ))
        current += pd.DateOffset(months=step_months)
    return rounds


def _feature_frame(frame):
    import pandas as pd
    if isinstance(frame.columns, pd.MultiIndex):
        return frame["feature"]
    return frame


def _label_frame(frame):
    import pandas as pd
    if isinstance(frame, pd.Series):
        return frame.to_frame("LABEL0")
    if isinstance(frame.columns, pd.MultiIndex):
        return frame["label"]
    return frame


def _select_feature_keys(frame, factor_keys):
    features = _feature_frame(frame)
    missing = [key for key in factor_keys if key not in features.columns]
    if missing:
        raise ValueError(f"联合数据集中缺少因子列: {missing[:8]}")
    return features.loc[:, list(factor_keys)]


def _slice_period(frame, period):
    import pandas as pd
    dates = pd.to_datetime(frame.index.get_level_values("datetime"))
    mask = (dates >= pd.Timestamp(period[0])) & (dates <= pd.Timestamp(period[1]))
    return frame.loc[mask]


def _daily_spearman_matrix(features, target):
    import pandas as pd
    feature_ranks = features.groupby(level="datetime", sort=False).rank()
    target_ranks = target.groupby(level="datetime", sort=False).rank()
    combined = feature_ranks.copy()
    target_column = "__residual_target__"
    combined[target_column] = target_ranks
    values = {}
    for date, frame in combined.groupby(level="datetime", sort=False):
        values[date] = frame.drop(columns=target_column).corrwith(frame[target_column])
    return pd.DataFrame.from_dict(values, orient="index")


class _PreparedFactorDataset:
    def __init__(self, frames, factor_keys):
        import pandas as pd
        self.segments = {key: key for key in frames}
        self._frames = {}
        for segment, frame in frames.items():
            features = _select_feature_keys(frame, factor_keys)
            labels = _label_frame(frame)
            self._frames[segment] = pd.concat({"feature": features, "label": labels}, axis=1)

    def prepare(self, segments, col_set=None, data_key=None, **kwargs):
        if isinstance(segments, (list, tuple)):
            return [self.prepare(segment, col_set=col_set, data_key=data_key, **kwargs) for segment in segments]
        frame = self._frames[segments]
        if col_set in (None, "__all"):
            return frame
        if isinstance(col_set, str):
            return frame[col_set]
        return frame.loc[:, list(col_set)]


class FactorSetEvaluator:
    def evaluate_factor_sets(self, arms, config, progress_cb=None, fold_cb=None,
                             completed_folds=None, cancel_cb=None,
                             state_map=None, fusion_arms=None):
        import pandas as pd
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.utils import init_instance_by_config
        from core.adapters.qlib.models import PARAM_PROFILES, QlibModelManager

        manager = QlibModelManager()
        rounds = _make_rounds(config)
        cutoff = dt.date.fromisoformat(config["selection_cutoff"])
        label_horizon = int(config.get("label_horizon", 10))
        label_mode = config.get("label_mode", "tradable_open_to_open")
        arm_entries = {arm: list(factor_keys) for arm, factor_keys in arms.items()}
        for fusion_key in (fusion_arms or {}):
            arm_entries[fusion_key] = []
        accumulators = {
            arm: {"factor_keys": factor_keys, "folds": [], "daily": []}
            for arm, factor_keys in arm_entries.items()
        }
        total = len(arms) * len(rounds)
        current = 0
        completed_folds = completed_folds or {}
        params = dict(PARAM_PROFILES["conservative"]["lightgbm"])
        for anchor, train_period, valid_period, test_period in rounds:
            matured = _matured(test_period[1], label_horizon) <= cutoff
            pending = {}
            for arm, factor_keys in arms.items():
                current += 1
                if not matured:
                    continue
                if cancel_cb and cancel_cb():
                    from core.shared.infrastructure.async_task import TaskCancelled
                    raise TaskCancelled(config.get("run_id", "factor_set_ab"))
                checkpoint = (completed_folds.get(arm) or {}).get(anchor)
                checkpoint_usable = (
                    checkpoint and isinstance(checkpoint.get("daily_ics"), list)
                    and (not state_map or isinstance(checkpoint.get("daily_ic_by_state"), dict))
                )
                if checkpoint_usable:
                    if progress_cb:
                        progress_cb(current, total, arm, anchor)
                    accumulators[arm]["folds"].append(dict(checkpoint))
                    accumulators[arm]["daily"].extend(float(value) for value in checkpoint["daily_ics"])
                    continue
                pending[arm] = (factor_keys, current)
            if not pending:
                continue
            union_keys = list(dict.fromkeys(
                key for factor_keys, _ in pending.values() for key in factor_keys
            ))
            started = time.perf_counter()
            task_config = manager._build_task_config(
                "lightgbm_rank", "WhiteBoxHandler", config.get("pool", "a500"),
                train_period, valid_period, test_period,
                model_params=params, label_horizon=label_horizon,
                custom_factor_keys=union_keys, label_norm="csrank",
                enable_cross_sectional=False, enable_winsorize=True,
                label_mode=label_mode,
            )
            source_dataset = init_instance_by_config(task_config["dataset"])
            train_frame, valid_frame, test_frame = source_dataset.prepare(
                ["train", "valid", "test"],
                col_set=["feature", "label"],
                data_key=DataHandlerLP.DK_L,
            )
            prepare_seconds = time.perf_counter() - started
            logger.info(
                "[FactorSetAB] fold=%s union=%s prepare=%.1fs arms=%s",
                anchor, len(union_keys), prepare_seconds, len(pending),
            )
            frames = {"train": train_frame, "valid": valid_frame, "test": test_frame}
            label = _as_series(_label_frame(test_frame))
            fold_predictions = {}
            for arm, (factor_keys, progress_position) in pending.items():
                if cancel_cb and cancel_cb():
                    from core.shared.infrastructure.async_task import TaskCancelled
                    raise TaskCancelled(config.get("run_id", "factor_set_ab"))
                if progress_cb:
                    progress_cb(progress_position, total, arm, anchor)
                arm_started = time.perf_counter()
                dataset = _PreparedFactorDataset(frames, factor_keys)
                model = init_instance_by_config(task_config["model"])
                model.fit(dataset, verbose_eval=0, log_metrics=False)
                prediction = _as_series(model.predict(dataset, "test"))
                fold_predictions[arm] = prediction
                frame = pd.concat({"prediction": prediction, "label": label}, axis=1).dropna()
                daily = frame.groupby(level="datetime").apply(
                    lambda group: group["prediction"].corr(group["label"], method="spearman")
                )
                daily = daily.dropna()
                daily_dates = [str(value)[:10] for value in daily.index]
                daily_ic_by_state = {}
                if state_map:
                    states = [state_map.get(value, "unknown") for value in daily_dates]
                    for state_name in ("high_vol", "low_vol"):
                        state_idx = [i for i, s in enumerate(states) if s == state_name]
                        if state_idx:
                            daily_ic_by_state[state_name] = {
                                "mean": float(daily.iloc[state_idx].mean()),
                                "n": len(state_idx),
                            }
                fold_metric = {
                    "fold_id": anchor,
                    "rank_ic": float(daily.mean()) if len(daily) else None,
                    "icir": (
                        float(daily.mean() / daily.std())
                        if len(daily) > 1 and daily.std() > 0 else None
                    ),
                    "positive_ratio": float((daily > 0).mean()) if len(daily) else 0.0,
                    "n_dates": int(len(daily)),
                    "daily_ics": [float(value) for value in daily],
                    "daily_dates": daily_dates,
                    "daily_ic_by_state": daily_ic_by_state,
                    "timing": {
                        "shared_prepare_seconds": prepare_seconds,
                        "model_seconds": time.perf_counter() - arm_started,
                    },
                }
                accumulators[arm]["folds"].append(fold_metric)
                accumulators[arm]["daily"].extend(fold_metric["daily_ics"])
                if fold_cb:
                    fold_cb(arm, fold_metric)
                del dataset, model, prediction, frame, daily
                gc.collect()
            if fusion_arms:
                for fusion_key, spec in fusion_arms.items():
                    base_pred = fold_predictions.get(spec.get("base"))
                    overlay_pred = fold_predictions.get(spec.get("overlay"))
                    if base_pred is None or overlay_pred is None:
                        continue
                    weight = float(spec.get("weight", 0.5))
                    fused = weight * _rank_by_date(base_pred) + (1.0 - weight) * _rank_by_date(overlay_pred)
                    fusion_frame = pd.concat({"prediction": fused, "label": label}, axis=1).dropna()
                    fusion_daily = fusion_frame.groupby(level="datetime").apply(
                        lambda group: group["prediction"].corr(group["label"], method="spearman")
                    )
                    fusion_daily = fusion_daily.dropna()
                    fusion_dates = [str(value)[:10] for value in fusion_daily.index]
                    fusion_ic_by_state = {}
                    if state_map:
                        states = [state_map.get(value, "unknown") for value in fusion_dates]
                        for state_name in ("high_vol", "low_vol"):
                            state_idx = [i for i, s in enumerate(states) if s == state_name]
                            if state_idx:
                                fusion_ic_by_state[state_name] = {
                                    "mean": float(fusion_daily.iloc[state_idx].mean()),
                                    "n": len(state_idx),
                                }
                    fusion_metric = {
                        "fold_id": anchor,
                        "rank_ic": float(fusion_daily.mean()) if len(fusion_daily) else None,
                        "icir": (
                            float(fusion_daily.mean() / fusion_daily.std())
                            if len(fusion_daily) > 1 and fusion_daily.std() > 0 else None
                        ),
                        "positive_ratio": float((fusion_daily > 0).mean()) if len(fusion_daily) else 0.0,
                        "n_dates": int(len(fusion_daily)),
                        "daily_ics": [float(value) for value in fusion_daily],
                        "daily_dates": fusion_dates,
                        "daily_ic_by_state": fusion_ic_by_state,
                        "timing": {
                            "shared_prepare_seconds": prepare_seconds,
                            "model_seconds": 0.0,
                            "fusion_weight": weight,
                        },
                    }
                    accumulators[fusion_key]["folds"].append(fusion_metric)
                    accumulators[fusion_key]["daily"].extend(fusion_metric["daily_ics"])
                    if fold_cb:
                        fold_cb(fusion_key, fusion_metric)
                    del fusion_frame, fusion_daily
                    gc.collect()
            del source_dataset, train_frame, valid_frame, test_frame, frames
            gc.collect()
        results = {}
        for arm, item in accumulators.items():
            series = pd.Series(item["daily"], dtype=float)
            results[arm] = {
                "factor_keys": item["factor_keys"],
                "factor_count": len(item["factor_keys"]),
                "folds": item["folds"],
                "rank_ic": float(series.mean()) if len(series) else None,
                "icir": (
                    float(series.mean() / series.std())
                    if len(series) > 1 and series.std() > 0 else None
                ),
                "positive_ratio": float((series > 0).mean()) if len(series) else 0.0,
                "n_dates": int(len(series)),
            }
        return results

    def evaluate_residual_candidates(self, core_keys, candidate_keys, config, state_map=None,
                                     progress_cb=None, fold_cb=None, completed_folds=None,
                                     cancel_cb=None):
        import numpy as np
        import pandas as pd
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.utils import init_instance_by_config
        from core.adapters.qlib.models import PARAM_PROFILES, QlibModelManager

        manager = QlibModelManager()
        rounds = _make_rounds(config)
        cutoff = dt.date.fromisoformat(config["selection_cutoff"])
        label_horizon = int(config.get("label_horizon", 10))
        label_mode = config.get("label_mode", "tradable_open_to_open")
        rows = []
        core_folds = []
        completed_folds = completed_folds or {}
        pending_rounds = []
        for index, (anchor, train_period, valid_period, test_period) in enumerate(rounds):
            if _matured(test_period[1], label_horizon) > cutoff:
                continue
            checkpoint = completed_folds.get(anchor)
            if checkpoint and isinstance(checkpoint.get("residual_rows"), list):
                core_fold = checkpoint.get("core_fold")
                if core_fold:
                    core_folds.append(core_fold)
                rows.extend(checkpoint["residual_rows"])
                if progress_cb:
                    progress_cb(index + 1, len(rounds), anchor)
                continue
            pending_rounds.append((index, anchor, train_period, valid_period, test_period))

        if not pending_rounds:
            return {"core_folds": core_folds, "residual_rows": rows}

        if cancel_cb and cancel_cb():
            from core.shared.infrastructure.async_task import TaskCancelled
            raise TaskCancelled(config.get("run_id", "factor_residual"))

        params = dict(PARAM_PROFILES["conservative"]["lightgbm"])
        global_core_period = (
            min(item[2][0] for item in pending_rounds),
            max(item[4][1] for item in pending_rounds),
        )
        global_test_period = (
            min(item[4][0] for item in pending_rounds),
            max(item[4][1] for item in pending_rounds),
        )
        started = time.perf_counter()
        if progress_cb:
            progress_cb(0, len(rounds), "shared_core_prepare")
        core_config = manager._build_task_config(
            "lightgbm_rank", "WhiteBoxHandler", config.get("pool", "a500"),
            global_core_period, global_core_period, global_core_period,
            model_params=params, label_horizon=label_horizon,
            custom_factor_keys=core_keys, label_norm="csrank",
            enable_cross_sectional=False, enable_winsorize=True,
            label_mode=label_mode,
        )
        core_config["dataset"]["kwargs"]["segments"] = {"all": global_core_period}
        core_source = init_instance_by_config(core_config["dataset"])
        core_frame = core_source.prepare(
            "all", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L,
        )
        core_prepare_seconds = time.perf_counter() - started
        del core_source
        gc.collect()

        if cancel_cb and cancel_cb():
            from core.shared.infrastructure.async_task import TaskCancelled
            raise TaskCancelled(config.get("run_id", "factor_residual"))

        started = time.perf_counter()
        if progress_cb:
            progress_cb(0, len(rounds), "shared_candidate_prepare")
        candidate_prepare_chunk_size = max(1, int(config.get("candidate_prepare_chunk_size", 16)))
        candidate_feature_parts = []
        candidate_chunk_count = (len(candidate_keys) + candidate_prepare_chunk_size - 1) // candidate_prepare_chunk_size
        for chunk_index, chunk_start in enumerate(
            range(0, len(candidate_keys), candidate_prepare_chunk_size), start=1,
        ):
            chunk_keys = candidate_keys[chunk_start:chunk_start + candidate_prepare_chunk_size]
            if progress_cb:
                progress_cb(0, len(rounds), f"shared_candidate_prepare_chunk_{chunk_index}_{candidate_chunk_count}")
            candidate_config = manager._build_task_config(
                "lightgbm_rank", "WhiteBoxHandler", config.get("pool", "a500"),
                global_test_period, global_test_period, global_test_period,
                model_params=params, label_horizon=label_horizon,
                custom_factor_keys=chunk_keys, label_norm="none",
                enable_cross_sectional=False, enable_winsorize=True,
                label_mode=label_mode,
            )
            candidate_config["dataset"]["kwargs"]["segments"] = {"all": global_test_period}
            candidate_source = init_instance_by_config(candidate_config["dataset"])
            chunk_features = candidate_source.prepare(
                "all", col_set="feature", data_key=DataHandlerLP.DK_R,
            )
            chunk_features = _feature_frame(chunk_features)
            if isinstance(chunk_features.columns, pd.MultiIndex):
                chunk_features.columns = chunk_features.columns.get_level_values(-1)
            resolved_features = {}
            for position, key in enumerate(chunk_keys):
                if key in chunk_features.columns:
                    resolved_features[key] = chunk_features[key]
                elif position < len(chunk_features.columns):
                    resolved_features[key] = chunk_features.iloc[:, position]
            candidate_feature_parts.append(pd.DataFrame(resolved_features, index=chunk_features.index))
            del candidate_source, chunk_features, resolved_features
            gc.collect()
        candidate_features = pd.concat(candidate_feature_parts, axis=1, copy=False)
        del candidate_feature_parts
        candidate_prepare_seconds = time.perf_counter() - started
        gc.collect()

        logger.info(
            "[FactorResidual] shared core=%s %.1fs candidates=%s %.1fs pending_folds=%s",
            len(core_keys), core_prepare_seconds, len(candidate_features.columns),
            candidate_prepare_seconds, len(pending_rounds),
        )

        for index, anchor, train_period, valid_period, test_period in pending_rounds:
            if cancel_cb and cancel_cb():
                from core.shared.infrastructure.async_task import TaskCancelled
                raise TaskCancelled(config.get("run_id", "factor_residual"))
            if progress_cb:
                progress_cb(index + 1, len(rounds), anchor)
            fold_started = time.perf_counter()
            frames = {
                "train": _slice_period(core_frame, train_period),
                "valid": _slice_period(core_frame, valid_period),
                "test": _slice_period(core_frame, test_period),
            }
            dataset = _PreparedFactorDataset(frames, core_keys)
            model = init_instance_by_config(core_config["model"])
            model.fit(dataset, verbose_eval=0, log_metrics=False)
            prediction = _as_series(model.predict(dataset, "test"))
            label = _as_series(dataset.prepare("test", col_set="label"))
            aligned = pd.concat({"prediction": prediction, "label": label}, axis=1).dropna()
            if aligned.empty:
                del dataset, model, prediction, label, aligned, frames
                gc.collect()
                continue
            aligned["prediction_rank"] = _rank_by_date(aligned["prediction"])
            aligned["label_rank"] = _rank_by_date(aligned["label"])
            aligned["residual"] = aligned["label_rank"] - aligned["prediction_rank"]
            daily_core_ic = aligned.groupby(level="datetime").apply(
                lambda frame: frame["prediction"].corr(frame["label"], method="spearman")
            ).dropna()
            core_fold = {
                "fold_id": anchor,
                "ic_mean": float(daily_core_ic.mean()) if len(daily_core_ic) else None,
                "icir": (
                    float(daily_core_ic.mean() / daily_core_ic.std())
                    if len(daily_core_ic) > 1 and daily_core_ic.std() > 0 else None
                ),
                "n_dates": int(len(daily_core_ic)),
            }
            core_folds.append(core_fold)

            features = _slice_period(candidate_features, test_period)
            common = aligned.index.intersection(features.index)
            if common.empty:
                del dataset, model, prediction, label, features, aligned, frames
                gc.collect()
                continue
            residual = aligned.loc[common, "residual"]
            features = features.loc[common]
            daily_matrix = _daily_spearman_matrix(features, residual)
            daily_matrix = daily_matrix.replace([np.inf, -np.inf], np.nan)
            fold_rows = []
            for date, values in daily_matrix.iterrows():
                date_key = str(date)[:10]
                state = (state_map or {}).get(date_key, "unknown")
                for key, value in values.dropna().items():
                    fold_rows.append({
                        "factor_key": key, "fold_id": anchor, "date": date_key,
                        "state": state, "residual_ic": float(value),
                    })
            rows.extend(fold_rows)
            timing = {
                "shared_core_prepare_seconds": core_prepare_seconds,
                "shared_candidate_prepare_seconds": candidate_prepare_seconds,
                "fold_seconds": time.perf_counter() - fold_started,
            }
            if fold_cb:
                fold_cb(anchor, core_fold, fold_rows, timing)
            logger.info(
                "[FactorResidual] fold=%s rows=%s elapsed=%.1fs",
                anchor, len(fold_rows), timing["fold_seconds"],
            )
            del dataset, model, prediction, label, features, aligned, frames, daily_matrix
            gc.collect()
        del core_frame, candidate_features
        gc.collect()
        return {"core_folds": core_folds, "residual_rows": rows}
