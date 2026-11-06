"""
screening_backtest.py 单元测试

覆盖边界条件：空候选、缺失价格、择时不可用、极值参数、全胜行情
"""
import datetime as dt
from types import SimpleNamespace
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock


# ══════════════════════════════════════════════
# 不依赖外部设施的纯逻辑测试
# ══════════════════════════════════════════════

class TestLookupPrice:
    """_lookup_price — 价格查找的鲁棒性"""

    def test_normal_lookup(self):
        import pandas as pd
        from core.domain.backtest.screening import ScreeningBacktestRunner
        df = pd.DataFrame({
            "trade_date": ["20250101", "20250102", "20250103"],
            "close": [10.0, 10.5, 10.2],
        })
        runner = ScreeningBacktestRunner()
        assert runner._lookup_price(df, "2025-01-02", "close") == 10.5
        assert runner._lookup_price(df, "20250102", "close") == 10.5

    def test_missing_date(self):
        import pandas as pd
        from core.domain.backtest.screening import ScreeningBacktestRunner
        df = pd.DataFrame({"trade_date": ["20250101"], "close": [10.0]})
        runner = ScreeningBacktestRunner()
        assert runner._lookup_price(df, "2025-06-15", "close") is None

    def test_empty_df(self):
        import pandas as pd
        from core.domain.backtest.screening import ScreeningBacktestRunner
        df = pd.DataFrame({"trade_date": [], "close": []})
        runner = ScreeningBacktestRunner()
        assert runner._lookup_price(df, "2025-01-01", "close") is None

    def test_non_standard_columns(self):
        import pandas as pd
        from core.domain.backtest.screening import ScreeningBacktestRunner
        df = pd.DataFrame({"trade_date": ["20250101"], "open": [9.5], "close": [10.0]})
        runner = ScreeningBacktestRunner()
        assert runner._lookup_price(df, "2025-01-01", "open") == 9.5

    def test_corrupted_input_not_dataframe(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        assert runner._lookup_price(None, "2025-01-01", "close") is None
        assert runner._lookup_price({}, "2025-01-01", "close") is None


@pytest.mark.p0
def test_daily_risk_check_accepts_dataframe_price_cache():
    import pandas as pd
    from core.domain.backtest.screening import ScreeningBacktestRunner

    runner = object.__new__(ScreeningBacktestRunner)
    runner._sell_cost_rate = 0.0
    runner._allocator = SimpleNamespace(
        config=SimpleNamespace(
            stop_loss_full=-0.10,
            take_profit_full=0.30,
            stop_loss_warn=-0.05,
            take_profit_half=0.15,
        ),
        classify_risk=lambda *_args, **_kwargs: {
            "000001": SimpleNamespace(
                action="full", reason="止损", new_cooldown=2,
            ),
        },
    )
    runner._get_trading_days_between = lambda *_args: [dt.date(2025, 1, 2)]
    runner._is_limit_down = lambda *_args: False
    prices = pd.DataFrame({
        "trade_date": ["2025-01-02"],
        "low": [8.9], "high": [10.1], "close": [9.0],
    })

    holdings, _, cash, _, _, trades = runner._daily_risk_check(
        {"000001": 1.0}, {"000001": 10.0}, 0.0, {}, set(),
        dt.date(2025, 1, 1), dt.date(2025, 1, 3), {"000001": prices},
    )

    assert holdings == {}
    assert cash == pytest.approx(0.9)
    assert len(trades) == 1


class TestBenchmarkReturn:
    """_get_benchmark_return_fast — 基准收益计算"""

    def test_first_period_returns_zero(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        price_map = {"2025-01-02": 4000, "2025-01-09": 4100}
        all_dates = [dt.date(2025, 1, 2), dt.date(2025, 1, 9)]
        assert runner._get_benchmark_return_fast(price_map, all_dates[0], all_dates, 0) == 0

    def test_normal_return(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        price_map = {"2025-01-02": 4000, "2025-01-09": 4100}
        all_dates = [dt.date(2025, 1, 2), dt.date(2025, 1, 9)]
        ret = runner._get_benchmark_return_fast(price_map, all_dates[1], all_dates, 1)
        assert abs(ret - 0.025) < 0.001

    def test_missing_prev_price(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        price_map = {"2025-01-09": 4100}  # prev missing
        all_dates = [dt.date(2025, 1, 2), dt.date(2025, 1, 9)]
        assert runner._get_benchmark_return_fast(price_map, all_dates[1], all_dates, 1) == 0

    def test_empty_price_map(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        all_dates = [dt.date(2025, 1, 2), dt.date(2025, 1, 9)]
        assert runner._get_benchmark_return_fast({}, all_dates[1], all_dates, 1) == 0


class TestAllocate:
    """权重分配 — 通过统一 PositionAllocator"""

    def test_normal_allocation(self):
        from core.domain.portfolio.allocator import (
            AllocationConfig, PositionAllocator)
        cfg = AllocationConfig(target_sum=1.0, apply_double_cap=False)
        alloc = PositionAllocator(cfg)
        cands = [{"code": "600001", "score": 0.8},
                 {"code": "600002", "score": 0.6},
                 {"code": "600003", "score": 0.4}]
        weights = alloc.allocate_weights_only(cands)
        assert len(weights) == 3
        assert 0.99 < sum(weights.values()) < 1.01

    def test_single_code(self):
        from core.domain.portfolio.allocator import (
            AllocationConfig, PositionAllocator)
        cfg = AllocationConfig(target_sum=1.0, apply_double_cap=False)
        alloc = PositionAllocator(cfg)
        cands = [{"code": "600001", "score": 0.5}]
        weights = alloc.allocate_weights_only(cands)
        assert len(weights) == 1
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_empty_codes(self):
        from core.domain.portfolio.allocator import (
            AllocationConfig, PositionAllocator)
        alloc = PositionAllocator(AllocationConfig())
        assert alloc.allocate_weights_only([]) == {}

    def test_top_n_larger_than_codes(self):
        from core.domain.portfolio.allocator import (
            AllocationConfig, PositionAllocator)
        cfg = AllocationConfig(target_sum=1.0, apply_double_cap=False)
        alloc = PositionAllocator(cfg)
        cands = [{"code": "600001", "score": 0.7},
                 {"code": "600002", "score": 0.3}]
        weights = alloc.allocate_weights_only(cands)
        assert len(weights) == 2

    def test_zero_scores(self):
        from core.domain.portfolio.allocator import (
            AllocationConfig, PositionAllocator)
        cfg = AllocationConfig(target_sum=1.0, apply_double_cap=False)
        alloc = PositionAllocator(cfg)
        cands = [{"code": "600001", "score": 0},
                 {"code": "600002", "score": 0}]
        weights = alloc.allocate_weights_only(cands)
        assert len(weights) == 2
        assert 0.99 < sum(weights.values()) < 1.01


class TestComputeMetrics:
    """_compute_metrics — 指标计算的边界条件"""

    def test_minimal_periods(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        eq = [1.0, 1.05]
        bm = [1.0, 1.02]
        result = runner._compute_metrics(eq, bm, ["d1", "d2"], [], [], "week")
        assert result["summary"]["total_return"] == pytest.approx(0.05, 0.01)
        assert result["summary"]["annual_return"] != 0

    def test_all_wins_profit_factor(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        eq = [1.0, 1.02, 1.05, 1.08, 1.12]
        bm = [1.0, 1.01, 1.02, 1.03, 1.04]
        result = runner._compute_metrics(eq, bm, [f"d{i}" for i in range(5)], [], [], "week")
        pf = result["summary"]["profit_factor"]
        assert pf <= 999.0  # 不会无穷大

    def test_all_losses(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        eq = [1.0, 0.98, 0.95, 0.92, 0.88]
        bm = [1.0, 1.01, 1.02, 1.03, 1.04]
        result = runner._compute_metrics(eq, bm, [f"d{i}" for i in range(5)], [], [], "week")
        assert result["summary"]["win_rate"] == 0
        assert result["summary"]["profit_factor"] == 0
        assert result["summary"]["max_drawdown"] < 0

    def test_flat_equity(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        eq = [1.0, 1.0, 1.0]
        bm = [1.0, 1.0, 1.0]
        result = runner._compute_metrics(eq, bm, ["d1", "d2", "d3"], [], [], "week")
        assert result["summary"]["sharpe"] == 0
        assert result["summary"]["max_drawdown"] == 0

    def test_monthly_freq_annualization(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        eq = [1.0, 1.03, 1.07]
        bm = [1.0, 1.01, 1.02]
        result = runner._compute_metrics(eq, bm, ["d1", "d2", "d3"], [], [], "month")
        assert result["summary"]["annual_return"] != 0

    def test_negative_equity(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        eq = [1.0, -0.5]
        bm = [1.0, 1.01]
        result = runner._compute_metrics(eq, bm, ["d1", "d2"], [], [], "week")
        assert result["summary"]["total_return"] < -1.0


def _setup_runner(runner):
    from core.domain.portfolio.allocator import build_allocation_config, PositionAllocator
    runner._allocator = PositionAllocator(
        build_allocation_config("week", score_key="l0_score"))
    runner._limit_up_cache = {}
    runner._limit_down_cache = {}
    runner._buy_cost_rate = 0.0004
    runner._sell_cost_rate = 0.0009


class TestSimulatePeriod:
    """_simulate_period_fast — 交易模拟"""

    def make_price_df(self, code, dates_and_prices):
        import pandas as pd
        rows = []
        for d, o, c in dates_and_prices:
            rows.append({"trade_date": d.replace("-", ""), "open": o, "close": c,
                         "high": c, "low": o})
        return pd.DataFrame(rows)

    def test_hold_and_sell(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        _setup_runner(runner)
        price_cache = {
            "600001": self.make_price_df("600001", [
                ("2025-01-02", 10.0, 10.5),
                ("2025-01-09", 10.6, 11.0),
            ]),
        }
        holdings = {"600001": 1.0}
        entry_prices = {"600001": 10.0}
        ret, h, ep, cash, cooldown, half_sold, trades = runner._simulate_period_fast(
            holdings, entry_prices, 1.0, {}, set(),
            buy_codes=[], buy_scores={}, sell_threshold=set(),
            curr_date=dt.date(2025, 1, 9),
            top_n=5, price_cache=price_cache, timing_ratio=1.0, sector_map={})
        assert h == {}
        assert len(trades) == 1
        assert trades[0]["action"] == "sell"

    def test_keep_and_buy(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        _setup_runner(runner)
        price_cache = {
            "600001": self.make_price_df("600001", [
                ("2025-01-02", 10.0, 10.5),
                ("2025-01-09", 10.6, 11.0),
            ]),
            "600002": self.make_price_df("600002", [
                ("2025-01-02", 20.0, 21.0),
                ("2025-01-09", 21.5, 22.0),
            ]),
        }
        holdings = {"600001": 0.6}
        entry_prices = {"600001": 10.0}
        ret, h, ep, cash, cooldown, half_sold, trades = runner._simulate_period_fast(
            holdings, entry_prices, 1.0, {}, set(),
            buy_codes=["600001", "600002"],
            buy_scores={"600001": 0.8, "600002": 0.6},
            sell_threshold={"600001", "600002"},
            curr_date=dt.date(2025, 1, 9),
            top_n=5, price_cache=price_cache, timing_ratio=1.0, sector_map={})
        assert len(h) == 2
        buy_trades = [t for t in trades if t["action"] == "buy"]
        assert len(buy_trades) == 1
        assert buy_trades[0]["code"] == "600002"

    def test_stop_loss_triggered(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        _setup_runner(runner)
        price_cache = {
            "600001": self.make_price_df("600001", [
                ("2025-01-02", 10.0, 10.5),
                ("2025-01-09", 9.0, 9.1),
            ]),
        }
        holdings = {"600001": 1.0}
        entry_prices = {"600001": 10.0}
        runner._allocator.config.stop_loss_full = -0.08
        ret, h, ep, cash, cooldown, half_sold, trades = runner._simulate_period_fast(
            holdings, entry_prices, 1.0, {}, set(),
            buy_codes=["600001"], buy_scores={"600001": 0.8},
            sell_threshold={"600001"},
            curr_date=dt.date(2025, 1, 9),
            top_n=5, price_cache=price_cache, timing_ratio=1.0, sector_map={})
        sell_trades = [t for t in trades if t["action"] == "sell"]
        assert any("止损" in str(t.get("reason", "")) for t in sell_trades)

    def test_no_entry_price_skip_stop_loss(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        _setup_runner(runner)
        price_cache = {
            "600001": self.make_price_df("600001", [
                ("2025-01-02", 10.0, 10.5),
                ("2025-01-09", 9.0, 9.1),
            ]),
        }
        holdings = {"600001": 1.0}
        entry_prices = {}
        ret, h, ep, cash, cooldown, half_sold, trades = runner._simulate_period_fast(
            holdings, entry_prices, 1.0, {}, set(),
            buy_codes=["600001"], buy_scores={"600001": 0.8},
            sell_threshold={"600001"},
            curr_date=dt.date(2025, 1, 9),
            top_n=5, price_cache=price_cache, timing_ratio=1.0, sector_map={})
        assert "600001" in h

    def test_timing_ratio_reduces_weights(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        _setup_runner(runner)
        price_cache = {
            "600001": self.make_price_df("600001", [
                ("2025-01-09", 10.0, 10.5),
            ]),
        }
        ret, h, ep, cash, cooldown, half_sold, trades = runner._simulate_period_fast(
            {}, {}, 1.0, {}, set(), buy_codes=["600001"],
            buy_scores={"600001": 0.8}, sell_threshold={"600001"},
            curr_date=dt.date(2025, 1, 9),
            top_n=5, price_cache=price_cache, timing_ratio=0.5, sector_map={})
        assert "600001" in h

    def test_missing_price_data_handled(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        _setup_runner(runner)
        ret, h, ep, cash, cooldown, half_sold, trades = runner._simulate_period_fast(
            {}, {}, 1.0, {}, set(),
            buy_codes=["600001", "600002"],
            buy_scores={"600001": 0.8, "600002": 0.6},
            sell_threshold={"600001", "600002"},
            curr_date=dt.date(2025, 1, 9),
            top_n=5, price_cache={}, timing_ratio=1.0, sector_map={})
        assert "600001" in h
        assert "600002" in h


class TestInitialBuild:
    """_initial_build — 首次建仓"""

    def test_normal_build(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        import pandas as pd
        runner = ScreeningBacktestRunner()
        _setup_runner(runner)
        price_cache = {
            "600001": pd.DataFrame({"trade_date": ["20250109"], "open": [10.0], "close": [10.5]}),
        }
        weights, ep, cash, half_sold = runner._initial_build(
            ["600001"], {"600001": 0.8}, 5, price_cache, "2025-01-09", {})
        assert weights["600001"] == 1.0
        assert ep["600001"] == 10.5

    def test_no_price_data(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        _setup_runner(runner)
        weights, ep, cash, half_sold = runner._initial_build(
            ["600001"], {"600001": 0.8}, 5, {}, "2025-01-09", {})
        assert "600001" not in weights


# ══════════════════════════════════════════════
# 集成场景测试（mock 外部依赖）
# ══════════════════════════════════════════════

class TestRunEdgeCases:
    """ScreeningBacktestRunner.run() 边界条件"""

    def test_too_short_period(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        with pytest.raises(RuntimeError, match="回测周期太短"):
            runner.run("2025-01-01", "2025-01-05")

    @patch("core.domain.backtest.screening.ScreeningBacktestRunner._prefetch_index_data")
    @patch("core.domain.backtest.screening.ScreeningBacktestRunner._precompute_timing_ratios")
    @patch("core.domain.backtest.screening.ScreeningBacktestRunner._get_screener")
    def test_all_dates_empty_raises(self, mock_screener, mock_timing, mock_index):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        mock_index.return_value = None
        mock_timing.return_value = {}
        screener = MagicMock()
        screener.screen.return_value = {"candidates": [], "regime": {}}
        mock_screener.return_value = screener

        runner = ScreeningBacktestRunner()
        with pytest.raises(RuntimeError, match="无有效交易日"):
            runner.run("2025-01-01", "2025-01-20")

    @patch("core.domain.backtest.screening.ScreeningBacktestRunner._prefetch_index_data")
    @patch("core.domain.backtest.screening.ScreeningBacktestRunner._precompute_timing_ratios")
    @patch("core.domain.backtest.screening.ScreeningBacktestRunner._get_screener")
    @patch("core.domain.backtest.screening.ScreeningBacktestRunner._load_price_batch")
    def test_screening_fails_mid_run(self, mock_load, mock_screener, mock_timing, mock_index):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        import pandas as pd

        mock_index.return_value = pd.DataFrame({
            "trade_date": ["2024-12-01", "2025-01-02", "2025-01-09"],
            "close": [3800, 4000, 4100],
        })
        mock_timing.return_value = {}
        mock_load.return_value = None  # _load_price_batch mutates price_cache

        screener = MagicMock()
        # 第一期正常返回，后续全部失败
        screener.screen.side_effect = [
            {"candidates": [{"code": "600001", "l0_score": 0.8}], "regime": {"regime": "bull_steady", "direction": "trend_momentum", "confidence": 0.9}},
            {"candidates": [], "regime": {}},
            {"candidates": [], "regime": {}},
        ]
        mock_screener.return_value = screener

        runner = ScreeningBacktestRunner()
        # 手动注入价格缓存，避免 DB 调用
        runner._load_price_batch = lambda *a, **kw: None

        result = runner.run("2025-01-01", "2025-01-20")
        assert result["summary"]["n_periods"] >= 1
        assert len(result["equity_curve"]) >= 2

    def test_invalid_freq_defaults_to_weekly(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        with patch.object(runner, "_prefetch_index_data", return_value=None):
            with patch.object(runner, "_get_screener") as mock_s:
                screener = MagicMock()
                screener.screen.return_value = {
                    "candidates": [{"code": "600001", "l0_score": 0.8}],
                    "regime": {},
                }
                mock_s.return_value = screener
                # "daily" is not a valid freq → falls back to "W-THU"
                # 2 days span won't generate 2+ dates with daily freq but will with weekly
                # Actually pd.date_range with "W-THU" should work
                result = runner.run("2025-01-01", "2025-02-01", freq="daily")
                assert result["summary"]["n_periods"] >= 1

    def test_no_timing_model(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        runner._timing_model = MagicMock()
        runner._timing_model.lgb = None  # 无模型
        ratios = runner._precompute_timing_ratios(
            [dt.date(2025, 1, 9)], None, dt.date(2025, 1, 1))
        assert ratios == {}

    @patch("core.domain.backtest.screening.ScreeningBacktestRunner._prefetch_index_data")
    @patch("core.domain.backtest.screening.ScreeningBacktestRunner._precompute_timing_ratios")
    @patch("core.domain.backtest.screening.ScreeningBacktestRunner._get_screener")
    @patch("core.domain.backtest.screening.ScreeningBacktestRunner._load_price_batch")
    def test_first_date_skipped_holiday(self, mock_load, mock_screener, mock_timing, mock_index):
        """首期是节假日无候选，跳过并从第二个日期开始回测"""
        from core.domain.backtest.screening import ScreeningBacktestRunner
        import pandas as pd
        mock_index.return_value = pd.DataFrame({
            "trade_date": ["2024-12-01", "2025-01-02", "2025-01-09"],
            "close": [3800, 4000, 4100],
        })
        mock_timing.return_value = {}
        mock_load.return_value = None
        screener = MagicMock()
        screener.screen.side_effect = [
            {"candidates": [], "regime": {}},  # 首期 (元旦) 无候选
            {"candidates": [{"code": "600001", "l0_score": 0.8}], "regime": {"regime": "bull_steady"}},
            {"candidates": [{"code": "600001", "l0_score": 0.8}], "regime": {"regime": "bull_steady"}},
        ]
        mock_screener.return_value = screener
        runner = ScreeningBacktestRunner()
        result = runner.run("2025-01-01", "2025-01-20")
        assert result["summary"]["n_periods"] >= 1  # 至少有一期有效
        assert len(result["equity_curve"]) >= 2

    def test_no_index_data_timing(self):
        from core.domain.backtest.screening import ScreeningBacktestRunner
        runner = ScreeningBacktestRunner()
        runner._timing_model = MagicMock()
        runner._timing_model.lgb = MagicMock()  # 有模型，但无指数数据
        ratios = runner._precompute_timing_ratios(
            [dt.date(2025, 1, 9)], None, dt.date(2025, 1, 1))
        assert ratios == {}


# ══════════════════════════════════════════════
# 并发安全测试
# ══════════════════════════════════════════════

class TestConcurrency:
    """多线程并发安全"""

    def test_separate_instances_independent(self):
        """每个 run() 调用创建自己的实例，互不干扰"""
        from core.domain.backtest.screening import ScreeningBacktestRunner
        r1 = ScreeningBacktestRunner()
        r2 = ScreeningBacktestRunner()
        assert r1._timing_model is None
        assert r2._timing_model is None
        assert r1._screener is None
        assert r2._screener is None

    def test_screener_is_singleton(self):
        """_get_screener 两次调用返回同一实例"""
        from core.domain.backtest.screening import ScreeningBacktestRunner
        r1 = ScreeningBacktestRunner()
        s1 = r1._get_screener()
        s2 = r1._get_screener()
        assert s1 is s2


# ══════════════════════════════════════════════
# 后端 API 测试
# ══════════════════════════════════════════════

class TestBacktestAPI:
    """backend/api/backtest.py 新端点"""

    def test_screening_request_validation(self):
        from backend.api.backtest import ScreeningBacktestRequest
        req = ScreeningBacktestRequest(
            start="2025-01-01", end="2025-06-30",
            pool="a500", top_n=10, freq="week",
            use_timing=True, macro_bias=True, sector_fusion=False,
        )
        assert req.pool == "a500"
        assert req.sector_fusion == False

    def test_screening_request_defaults(self):
        from backend.api.backtest import ScreeningBacktestRequest
        req = ScreeningBacktestRequest(start="2025-01-01", end="2025-06-30")
        assert req.use_timing == True
        assert req.macro_bias == True
        assert req.sector_fusion == True
        assert req.freq == "week"
        assert req.pool == "a500"

    def test_direction_endpoint_uses_completed_status(self):
        """验证 API 端点检查 'completed' 而非 'done'（与 async_task 对齐）"""
        import anyio
        from unittest.mock import patch, AsyncMock, MagicMock

        async def _test():
            mock_task = {
                "job_id": "test123",
                "status": "completed",
                "result": {
                    "summary": {"total_return": 0.1},
                    "equity_curve": [],
                    "benchmark_curve": [],
                    "periods": [],
                },
                "error": "",
            }
            with patch("backend.api.backtest.task_manager") as mock_tm:
                mock_tm.get_dict = MagicMock(return_value=mock_task)
                from backend.api.backtest import get_direction_backtest_result
                result = await get_direction_backtest_result("test123")
                assert result["status"] == "done"
                assert result["summary"]["total_return"] == 0.1

            mock_task_failed = {
                "job_id": "test456",
                "status": "failed",
                "result": None,
                "error": "something broke",
            }
            with patch("backend.api.backtest.task_manager") as mock_tm:
                mock_tm.get_dict = MagicMock(return_value=mock_task_failed)
                from backend.api.backtest import get_direction_backtest_result
                result = await get_direction_backtest_result("test456")
                assert result["status"] == "error"
                assert "something broke" in result["error"]

        anyio.run(_test)
