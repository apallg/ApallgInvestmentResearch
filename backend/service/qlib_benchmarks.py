import logging

logger = logging.getLogger(__name__)


def sync_benchmark_indices(bridge, dates, date_to_idx, require_current=False):
    from contextlib import nullcontext
    from pathlib import Path

    import numpy as np
    from core.adapters.qlib.generation import qlib_generation_mutation_lock
    from core.adapters.qlib.bridge import FIELDS
    from core.data.fetcher import DataFetcher
    from core.shared import config

    fetcher = DataFetcher()
    index_map = {
        "sh000300": "沪深300",
        "sh000001": "上证指数",
        "sz399001": "深证成指",
        "sz399006": "创业板指",
    }
    calendar_dates = [date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)[:10] for date in dates]
    calendar_last = calendar_dates[-1]
    pending = []

    for qlib_code, index_name in index_map.items():
        frame = fetcher.get_index_daily(index_name, days=365 * 12)
        if frame is None or frame.empty:
            raise RuntimeError(f"{index_name}指数行情为空，未写入 Qlib")

        values_by_field = {}
        date_indices = []
        for _, row in frame.iterrows():
            raw_date = row["date"]
            date_str = raw_date.strftime("%Y-%m-%d") if hasattr(raw_date, "strftime") else str(raw_date)[:10]
            day_idx = date_to_idx.get(date_str)
            if day_idx is None:
                continue
            date_indices.append(day_idx)
            for field in FIELDS:
                if field in ("vwap", "factor") or field not in frame.columns:
                    continue
                values_by_field.setdefault(field, []).append(float(row[field]))

        if not date_indices or not values_by_field.get("close"):
            raise RuntimeError(f"{index_name}指数没有与 Qlib 交易日历重叠的行情")

        first_idx, last_idx = min(date_indices), max(date_indices)
        close_values = dict(zip(date_indices, values_by_field["close"]))
        latest_close = close_values.get(last_idx)
        open_values = dict(zip(date_indices, values_by_field.get("open", [])))
        latest_open = open_values.get(last_idx)
        if require_current and (
            calendar_dates[last_idx] != calendar_last
            or latest_close is None
            or not np.isfinite(latest_close)
            or latest_close <= 0
            or latest_open is None
            or not np.isfinite(latest_open)
            or latest_open <= 0
        ):
            raise RuntimeError(f"{index_name}最新行情未完整覆盖 Qlib 日历最后一天 {calendar_last}")

        arrays = {}
        for field, values in values_by_field.items():
            full = np.full(last_idx - first_idx + 1, np.nan, dtype=np.float64)
            for day_idx, value in zip(date_indices, values):
                full[day_idx - first_idx] = value
            arrays[field] = full.astype(np.float32)

        pending.append((qlib_code, index_name, first_idx, last_idx, arrays))

    bridge_dir = Path(bridge.data_dir).resolve()
    qlib_root = Path(config.QLIB_DATA_ROOT).resolve()
    if bridge_dir == qlib_root:
        mutation = qlib_generation_mutation_lock(qlib_root, require_legacy=True)
    elif bridge_dir.is_relative_to(qlib_root):
        stage_parent = (qlib_root / "generations").resolve()
        stage_id = bridge_dir.name.removeprefix(".staging-")
        if (
            bridge_dir.parent != stage_parent
            or not bridge_dir.name.startswith(".staging-")
            or len(stage_id) != 32
            or not stage_id.isalnum()
        ):
            raise RuntimeError("基准指数只能写入Qlib候选代，拒绝修改已发布generation")
        mutation = qlib_generation_mutation_lock(qlib_root)
    else:
        mutation = nullcontext()

    with mutation:
        for qlib_code, index_name, first_idx, last_idx, arrays in pending:
            feature_dir = bridge.features_dir / qlib_code
            feature_dir.mkdir(parents=True, exist_ok=True)
            for field, values in arrays.items():
                path = feature_dir / f"{field}.day.bin"
                temp_path = path.with_suffix(path.suffix + ".tmp")
                np.hstack([np.float32(first_idx), values]).astype("<f").tofile(str(temp_path))
                temp_path.replace(path)
            factor = np.ones(last_idx - first_idx + 2, dtype=np.float32)
            factor[0] = np.float32(first_idx)
            factor_path = feature_dir / "factor.day.bin"
            factor_temp_path = factor_path.with_suffix(factor_path.suffix + ".tmp")
            factor.astype("<f").tofile(str(factor_temp_path))
            factor_temp_path.replace(factor_path)

        updated_ranges = {
            qlib_code: f"{qlib_code}\t{calendar_dates[first_idx]}\t{calendar_dates[last_idx]}"
            for qlib_code, _, first_idx, last_idx, _ in pending
        }
        instruments_path = bridge.instruments_dir / "all.txt"
        existing = instruments_path.read_text(encoding="utf-8").splitlines() if instruments_path.exists() else []
        retained = [line for line in existing if line.split("\t", 1)[0] not in updated_ranges]
        instruments_path.write_text("\n".join(retained + list(updated_ranges.values())) + "\n", encoding="utf-8")

    for qlib_code, index_name, _, _, arrays in pending:
        logger.info("基准指数 %s(%s) 已写入: %s 条", qlib_code, index_name, len(arrays.get("close", [])))
    return {"indices": list(updated_ranges), "last_date": calendar_last}

