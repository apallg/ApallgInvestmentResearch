import datetime as dt

import pandas as pd

from core.data.repos.factor_panel import factor_panel_repo
from core.data.repos.factor_panel_memmap import factor_panel_memmap_repo
from core.domain.factors.expression import batch_eval_all, parse_factor
from core.domain.factors.panel import ResearchSnapshotStore, build_point_in_time_maps
from core.domain.screening.indicators import Analyzer
from core.shared import config


class FactorPanelService:
    def __init__(self, repo=None, memmap_repo=None):
        self._repo = repo or factor_panel_repo
        self._memmap_repo = memmap_repo or factor_panel_memmap_repo

    def load_trading_dates(self, start_date, end_date):
        start = dt.date.fromisoformat(str(start_date)[:10])
        end = dt.date.fromisoformat(str(end_date)[:10])
        dates = self._repo.load_trading_dates(start, end)
        if dates:
            return dates
        from core.shared.trading_time import is_trading_day
        result = []
        current = start
        while current <= end:
            if is_trading_day(current):
                result.append(current)
            current += dt.timedelta(days=1)
        return result

    def build_mining_context(self, pool, dates, risk_neutralize=True, progress_cb=None):
        dates = sorted(dates)
        total_steps = len(dates) * (2 if risk_neutralize else 1)

        def panel_progress(done, _total, dkey):
            if progress_cb:
                progress_cb(done, total_steps, dkey)

        store, loaded_codes = self.build_snapshot_store(
            pool,
            dates,
            include_tech=False,
            progress_cb=panel_progress,
        )
        market_cap_rows = self._repo.load_market_cap_history(loaded_codes, dates[0], dates[-1])
        market_caps = build_point_in_time_maps(market_cap_rows, dates)
        extra = self._build_risk_exposures(
            store,
            progress_cb=(
                (lambda done, dkey: progress_cb(len(dates) + done, total_steps, dkey))
                if progress_cb and risk_neutralize else None
            ),
        ) if risk_neutralize else {}
        return store, market_caps, extra

    def build_snapshot_store(self, pool, dates, code_prefixes=None, include_tech=False,
        progress_cb=None):
        dates = sorted(dates)
        master, loaded_codes, adj_latest = self._load_master(
            pool, dates, code_prefixes=code_prefixes, include_tech=include_tech,
        )
        store = ResearchSnapshotStore(
            master,
            dates,
            adj_latest=adj_latest,
            min_turnover=getattr(config, "BUY_SIDE_L0_MIN_TURNOVER", 10_000_000),
            skip_limit_up=True,
            fields="all" if include_tech else None,
            dtype=getattr(config, "FACTOR_PANEL_DTYPE", "float64"),
            progress_cb=progress_cb,
        )
        return store, loaded_codes

    def build_shared_memmap(self, pool, dates, directory, code_prefixes=None,
                            include_tech=True, progress_cb=None):
        dates = sorted(dates)
        master, loaded_codes, adj_latest = self._load_master(
            pool, dates, code_prefixes=code_prefixes, include_tech=include_tech,
        )
        try:
            manifest_path, manifest = self._memmap_repo.write(
                master,
                directory,
                dtype=getattr(config, "FACTOR_PANEL_DTYPE", "float64"),
                metadata={"adj_latest": adj_latest},
                progress_cb=progress_cb,
            )
        finally:
            master.clear()
        return manifest_path, manifest, loaded_codes

    def open_shared_memmap(self, manifest_path, dates, progress_cb=None):
        manifest, arrays = self._memmap_repo.open(manifest_path)
        store = ResearchSnapshotStore.from_shared_arrays(
            manifest,
            arrays,
            sorted(dates),
            adj_latest=manifest.get("adj_latest") or {},
            min_turnover=getattr(config, "BUY_SIDE_L0_MIN_TURNOVER", 10_000_000),
            skip_limit_up=True,
            progress_cb=progress_cb,
        )
        return store, arrays, manifest

    def close_shared_memmap(self, arrays):
        self._memmap_repo.close(arrays)

    def cleanup_shared_memmap(self, directory):
        self._memmap_repo.cleanup(directory)

    def _load_master(self, pool, dates, code_prefixes=None, include_tech=False):
        dates = sorted(dates)
        codes = self._repo.load_universe(pool)
        if code_prefixes:
            prefixes = tuple(str(value) for value in code_prefixes)
            codes = [code for code in codes if code.startswith(prefixes)]
        if not codes or not dates:
            raise RuntimeError("股票池或评估日期为空")
        start = dates[0] - dt.timedelta(days=365)
        master = self._repo.load_daily_master(codes, start, dates[-1])
        if not master:
            raise RuntimeError("批量日线数据为空")
        if include_tech:
            analyzer = Analyzer()
            for code, frame in list(master.items()):
                try:
                    numeric = [field for field in (
                        "open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"
                    ) if field in frame.columns]
                    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
                    master[code] = analyzer.add_indicators(frame)
                except Exception:
                    master.pop(code, None)
        loaded_codes = list(master)
        adj_latest = self._repo.load_latest_adj_factors(loaded_codes)
        return master, loaded_codes, adj_latest

    @staticmethod
    def _build_risk_exposures(store, progress_cb=None):
        expressions = {
            "momentum": "$close / Ref($close, 20) - 1",
            "volatility": "Std($close / Ref($close, 1) - 1, 20)",
            "turnover": "$turnover",
        }
        asts = {name: parse_factor(expr) for name, expr in expressions.items()}
        result = {name: {} for name in expressions}
        for index, dkey in enumerate(store, 1):
            scores = batch_eval_all(asts, store[dkey])
            for name in expressions:
                result[name][dkey] = scores.get(name, {})
            if progress_cb:
                progress_cb(index, dkey)
        return result
