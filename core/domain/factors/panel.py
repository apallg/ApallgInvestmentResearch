from collections.abc import Mapping

import numpy as np
import pandas as pd

from core.domain.backtest.common import limit_pct_for_code


PANEL_FIELDS = (
    "open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg",
)


class ResearchSnapshotStore(Mapping):
    def __init__(self, master, eval_dates, adj_latest=None, min_turnover=10_000_000,
                 skip_limit_up=True, lookback=260, min_rows=100,
                 max_calendar_days=365, dtype="float64", fields=None, progress_cb=None):
        self._dates = tuple(
            d.isoformat() if hasattr(d, "isoformat") else str(d) for d in eval_dates
        )
        self._codes = []
        self._frames = []
        self._row_adj = []
        self._names = []
        self._dtype = dtype
        self._adj_latest = adj_latest or {}
        self._lookback = int(lookback)
        self._starts = None
        self._ends = None
        self._mapped_bytes = 0
        self._backing_mode = "memory"
        self._shared_arrays = None

        for code, raw in (master or {}).items():
            normalized = self._normalize_frame(raw, dtype, fields)
            if normalized is None or len(normalized[0]) < min_rows:
                continue
            frame, row_adj, names = normalized
            self._codes.append(str(code))
            self._frames.append(frame)
            self._row_adj.append(row_adj)
            self._names.append(names)

        self._build_windows(
            min_turnover, skip_limit_up, min_rows, max_calendar_days, progress_cb,
        )

    @classmethod
    def from_shared_arrays(cls, manifest, arrays, eval_dates, adj_latest=None,
                           min_turnover=10_000_000, skip_limit_up=True,
                           lookback=260, min_rows=100, max_calendar_days=365,
                           progress_cb=None):
        self = cls.__new__(cls)
        self._dates = tuple(
            d.isoformat() if hasattr(d, "isoformat") else str(d) for d in eval_dates
        )
        self._codes = list(manifest["codes"])
        self._frames = []
        self._row_adj = []
        self._names = []
        self._dtype = manifest["dtype"]
        self._adj_latest = adj_latest or {}
        self._lookback = int(lookback)
        self._starts = None
        self._ends = None
        self._mapped_bytes = int(manifest.get("mapped_bytes", 0))
        self._backing_mode = "memmap"
        self._shared_arrays = arrays

        values = arrays["values"]
        dates = arrays["dates"]
        adj = arrays["adj"]
        name_codes = arrays["name_codes"]
        fields = manifest["fields"]
        categories = manifest.get("names") or [""]
        offsets = manifest["offsets"]
        for index in range(len(self._codes)):
            start = int(offsets[index])
            end = int(offsets[index + 1])
            frame_index = pd.to_datetime(dates[start:end].astype("int64"), unit="D", origin="unix")
            frame = pd.DataFrame(
                values[start:end], columns=fields, index=frame_index, copy=False,
            )
            self._frames.append(frame)
            self._row_adj.append(adj[start:end])
            self._names.append(pd.Categorical.from_codes(
                name_codes[start:end], categories=categories,
            ))

        self._build_windows(
            min_turnover, skip_limit_up, min_rows, max_calendar_days, progress_cb,
        )
        return self

    def _build_windows(self, min_turnover, skip_limit_up, min_rows,
                       max_calendar_days, progress_cb):
        shape = (len(self._dates), len(self._codes))
        self._starts = np.full(shape, -1, dtype=np.int32)
        self._ends = np.full(shape, -1, dtype=np.int32)

        for di, dkey in enumerate(self._dates):
            cutoff = pd.Timestamp(dkey)
            calendar_start = cutoff - pd.Timedelta(days=max_calendar_days)
            for ci, (code, frame) in enumerate(zip(self._codes, self._frames)):
                idx = frame.index
                end = int(idx.searchsorted(cutoff, side="right"))
                first = int(idx.searchsorted(calendar_start, side="left"))
                start = max(first, end - self._lookback)
                if end - start < min_rows:
                    continue
                latest = frame.iloc[end - 1]
                close = float(latest.get("close", 0) or 0)
                volume = float(latest.get("volume", 0) or 0)
                name = str(self._names[ci][end - 1])
                if name.upper().startswith(("ST", "*ST")):
                    continue
                if close <= 0 or volume <= 0:
                    continue
                amount = frame["amount"].iloc[max(start, end - 20):end].to_numpy(dtype=float)
                if not len(amount) or np.all(np.isnan(amount)) or float(np.nanmean(amount)) < min_turnover:
                    continue
                if self._low_price(code, close, self._row_adj[ci][end - 1]):
                    continue
                if not skip_limit_up:
                    pct = float(latest.get("pct_chg", 0) or 0)
                    if pct == 0 and end - start >= 2:
                        prev_close = float(frame["close"].iloc[end - 2] or 0)
                        if prev_close > 0:
                            pct = (close / prev_close - 1) * 100
                    if pct >= limit_pct_for_code(code) * 100 - 0.5:
                        continue
                self._starts[di, ci] = start
                self._ends[di, ci] = end
            if progress_cb:
                progress_cb(di + 1, len(self._dates), dkey)

    @staticmethod
    def _normalize_frame(raw, dtype, fields):
        if raw is None or getattr(raw, "empty", True):
            return None
        df = raw
        if "trade_date" in df.columns:
            index = pd.to_datetime(df["trade_date"], errors="coerce")
        elif "date" in df.columns:
            index = pd.to_datetime(df["date"], errors="coerce")
        elif isinstance(df.index, pd.DatetimeIndex):
            index = df.index
        else:
            return None
        if not {"close", "volume", "amount"}.issubset(df.columns):
            return None
        selected = list(PANEL_FIELDS)
        if fields == "all":
            ignored = {"trade_date", "date", "name", "code", "symbol", "index", "adj_factor"}
            selected.extend(c for c in df.columns if c not in ignored and c not in selected)
        elif fields:
            selected = list(dict.fromkeys(list(PANEL_FIELDS) + list(fields)))
        data = {}
        for field in selected:
            if field in df.columns:
                values = pd.to_numeric(df[field], errors="coerce").to_numpy(dtype=dtype)
                if field in PANEL_FIELDS or np.isfinite(values).any():
                    data[field] = values
            elif field in PANEL_FIELDS:
                data[field] = np.full(len(df), np.nan, dtype=dtype)
        result = pd.DataFrame(data, index=pd.DatetimeIndex(index))
        if "adj_factor" in df.columns:
            adj = pd.Series(
                pd.to_numeric(df["adj_factor"], errors="coerce").to_numpy(dtype=dtype),
                index=result.index,
            )
        else:
            adj = pd.Series(np.nan, index=result.index, dtype=dtype)
        names = pd.Series(
            df["name"].fillna("").astype(str).to_numpy() if "name" in df.columns else "",
            index=result.index,
            dtype="object",
        )
        valid = ~result.index.isna()
        result = result[valid]
        adj = adj[valid]
        names = names[valid]
        unique = ~result.index.duplicated(keep="last")
        result = result[unique].sort_index()
        adj = adj[unique].sort_index()
        names = names[unique].sort_index()
        return result, adj.to_numpy(dtype=dtype), pd.Categorical(names)

    def _low_price(self, code, close, adj_row):
        adj_latest = float(self._adj_latest.get(code, 0) or 0)
        adj_row = float(adj_row or 0) if np.isfinite(adj_row) else 0
        if adj_latest > 0 and adj_row > 0:
            return close * adj_latest / adj_row < 2
        return close < 2

    def __len__(self):
        return len(self._dates)

    def __iter__(self):
        return iter(self._dates)

    def __getitem__(self, date_key):
        key = date_key.isoformat() if hasattr(date_key, "isoformat") else str(date_key)
        try:
            di = self._dates.index(key)
        except ValueError as exc:
            raise KeyError(key) from exc
        snapshot = {}
        for ci, code in enumerate(self._codes):
            start = int(self._starts[di, ci])
            end = int(self._ends[di, ci])
            if start < 0 or end <= start:
                continue
            frame = self._frames[ci].iloc[start:end]
            latest = frame.iloc[-1]
            snapshot[code] = (
                frame,
                str(self._names[ci][end - 1]),
                float(latest.get("close", 0) or 0),
                float(latest.get("volume", 0) or 0),
            )
        return snapshot

    @property
    def memory_bytes(self):
        if self._mapped_bytes:
            return self._mapped_bytes + self._starts.nbytes + self._ends.nbytes
        frames = sum(int(df.memory_usage(index=True, deep=True).sum()) for df in self._frames)
        names = sum(
            item.codes.nbytes + int(item.categories.memory_usage(deep=True))
            for item in self._names
        )
        return frames + names + self._starts.nbytes + self._ends.nbytes

    @property
    def code_count(self):
        return len(self._codes)

    @property
    def dtype(self):
        return self._dtype

    @property
    def backing_mode(self):
        return self._backing_mode

    def clear(self):
        self._dates = ()
        self._codes.clear()
        self._frames.clear()
        self._row_adj.clear()
        self._names.clear()
        self._shared_arrays = None
        self._starts = np.empty((0, 0), dtype=np.int32)
        self._ends = np.empty((0, 0), dtype=np.int32)


def build_point_in_time_maps(rows_by_code, eval_dates, max_age_days=15):
    result = {}
    prepared = {}
    for code, rows in (rows_by_code or {}).items():
        ordered = sorted(rows, key=lambda row: str(row.get("trade_date", "")))
        prepared[str(code)] = (
            np.array([np.datetime64(str(row["trade_date"])[:10]) for row in ordered]),
            np.array([float(row.get("total_mv", 0) or 0) for row in ordered], dtype=float),
        )
    for date in eval_dates:
        dkey = date.isoformat() if hasattr(date, "isoformat") else str(date)
        target = np.datetime64(dkey[:10])
        values = {}
        for code, (dates, market_caps) in prepared.items():
            pos = int(dates.searchsorted(target, side="right")) - 1
            age = int((target - dates[pos]) / np.timedelta64(1, "D")) if pos >= 0 else max_age_days + 1
            if pos >= 0 and age <= max_age_days and market_caps[pos] > 0:
                values[code] = float(market_caps[pos])
        result[dkey] = values
    return result
