import calendar
import datetime as dt
from bisect import bisect_left


def _as_date(value):
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _shift_months(value, months):
    value = _as_date(value)
    month_index = value.year * 12 + value.month - 1 + months
    year, month = divmod(month_index, 12)
    day = min(value.day, calendar.monthrange(year, month + 1)[1])
    return dt.date(year, month + 1, day)


def purge_dates_before_boundary(eval_dates, boundary, trading_dates, purge_gap):
    values = sorted({_as_date(value) for value in eval_dates})
    if not values or purge_gap <= 0:
        return values
    trading = sorted({_as_date(value) for value in trading_dates})
    boundary_index = bisect_left(trading, _as_date(boundary))
    positions = {value: index for index, value in enumerate(trading)}
    return [
        value for value in values
        if value in positions and positions[value] + purge_gap < boundary_index
    ]


def sample_trading_dates(trading_dates, start, end, step=1):
    if step <= 0:
        raise ValueError("step 必须为正整数")
    start = _as_date(start)
    end = _as_date(end)
    values = sorted({
        _as_date(value) for value in trading_dates
        if start <= _as_date(value) <= end
    })
    return values[::step]


def evenly_sample_dates(values, max_count=None):
    ordered = sorted({_as_date(value) for value in values})
    if max_count is None or len(ordered) <= int(max_count):
        return ordered
    count = max(1, int(max_count))
    if count == 1:
        return [ordered[-1]]
    indexes = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in dict.fromkeys(indexes)]


def build_l1_validation_dates(anchor, trading_dates, purge_gap=21, step=5):
    anchor = _as_date(anchor)
    calendar_dates = sorted({_as_date(value) for value in trading_dates})
    train_boundary = anchor - dt.timedelta(days=90)
    train = sample_trading_dates(
        calendar_dates, anchor - dt.timedelta(days=270), train_boundary, step,
    )
    valid = sample_trading_dates(calendar_dates, train_boundary, anchor, step)
    test = sample_trading_dates(
        calendar_dates, anchor, anchor + dt.timedelta(days=90), step,
    )
    return {
        "train": purge_dates_before_boundary(
            train, train_boundary, calendar_dates, purge_gap,
        ),
        "valid": purge_dates_before_boundary(
            valid, anchor, calendar_dates, purge_gap,
        ),
        "test": test,
    }


def build_walkforward_folds(train_dates, valid_dates, n_folds=5, train_months=36,
                            purge_gap=21, trading_dates=None):
    train = sorted({_as_date(value) for value in train_dates})
    valid = sorted({_as_date(value) for value in valid_dates})
    if not train or not valid or n_folds <= 0:
        return []
    chunks = []
    for index in range(n_folds):
        start = len(valid) * index // n_folds
        end = len(valid) * (index + 1) // n_folds
        if end > start:
            chunks.append(valid[start:end])
    folds = []
    all_train = train + valid
    for index, chunk in enumerate(chunks):
        valid_start = chunk[0]
        eligible = [value for value in all_train if value < valid_start]
        if purge_gap:
            calendar_dates = trading_dates or all_train
            eligible = purge_dates_before_boundary(
                eligible, valid_start, calendar_dates, purge_gap)
        window_start = _shift_months(eligible[-1], -train_months) if eligible else valid_start
        fold_train = [value for value in eligible if value >= window_start]
        if not fold_train:
            continue
        folds.append({
            "fold": index + 1,
            "train_start": fold_train[0].isoformat(),
            "train_end": fold_train[-1].isoformat(),
            "valid_start": chunk[0].isoformat(),
            "valid_end": chunk[-1].isoformat(),
            "train_dates": [value.isoformat() for value in fold_train],
            "valid_dates": [value.isoformat() for value in chunk],
            "purge_gap": purge_gap,
        })
    return folds


def aggregate_fold_metrics(fold_metrics):
    valid = [item for item in fold_metrics if item.get("oos_ic") is not None]
    if not valid:
        return {"folds": 0, "passed_folds": 0, "oos_ic": None, "icir": None, "positive_ratio": 0.0}
    daily = []
    for item in valid:
        for point in item.get("oos_ic_series") or []:
            value = point.get("ic") if isinstance(point, dict) else point
            if value is not None:
                daily.append(float(value))
    oos = daily or [float(item["oos_ic"]) for item in valid]
    passed = sum(bool(item.get("passed")) for item in valid)
    mean = sum(oos) / len(oos)
    variance = sum((value - mean) ** 2 for value in oos) / len(oos)
    std = variance ** 0.5
    return {
        "folds": len(fold_metrics),
        "passed_folds": passed,
        "oos_ic": mean,
        "icir": mean / std if std > 1e-12 else (mean if mean else 0.0),
        "positive_ratio": sum(value > 0 for value in oos) / len(oos),
        "coverage": len(daily) / sum(
            int(item.get("valid_date_count") or len(item.get("oos_ic_series") or []))
            for item in fold_metrics
        ) if daily else (len(valid) / len(fold_metrics) if fold_metrics else 0.0),
    }
