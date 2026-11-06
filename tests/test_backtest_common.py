"""回测引擎共享纯函数测试 (从 engine.py/screening.py 提取到 common.py)"""
import pytest
import pandas as pd

from core.domain.backtest.common import (
    lookup_price, norm_code, get_benchmark_return_fast,
    equal_weight_pool_return,
    precompute_limit_up, precompute_limit_down,
)


@pytest.mark.p0
class TestNormCode:
    def test_sh_prefix(self):
        assert norm_code("sh600000") == "600000"

    def test_sz_prefix(self):
        assert norm_code("sz000858") == "000858"

    def test_bj_suffix(self):
        assert norm_code("830799.BJ") == "830799"

    def test_uppercase(self):
        assert norm_code("SH600519") == "600519"

    def test_no_prefix(self):
        assert norm_code("600519") == "600519"

    def test_int_input(self):
        assert norm_code(600519) == "600519"


@pytest.mark.p1
class TestLookupPrice:
    def _df(self):
        # 数据库 trade_date 为紧凑格式 (YYYYMMDD), lookup_price 依赖此格式
        return pd.DataFrame({
            "trade_date": ["20250101", "20250102", "20250103"],
            "close": [10.0, 10.5, 11.0],
            "open": [9.8, 10.2, 10.8],
        })

    def test_normal_lookup(self):
        assert lookup_price(self._df(), "2025-01-02", "close") == 10.5

    def test_compact_date(self):
        assert lookup_price(self._df(), "20250102", "close") == 10.5

    def test_missing_date(self):
        assert lookup_price(self._df(), "2025-06-15", "close") is None

    def test_none_df(self):
        assert lookup_price(None, "2025-01-01", "close") is None

    def test_empty_df(self):
        assert lookup_price(pd.DataFrame(), "2025-01-01", "close") is None


@pytest.mark.p1
class TestBenchmarkReturn:
    def _dates(self):
        return [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")]

    def test_first_period_zero(self):
        price_map = {"2025-01-01": 100.0, "2025-01-02": 101.0}
        d = self._dates()
        assert get_benchmark_return_fast(price_map, d[0], d, 0) == 0

    def test_normal_return(self):
        price_map = {"2025-01-01": 100.0, "2025-01-02": 105.0}
        d = self._dates()
        assert abs(get_benchmark_return_fast(price_map, d[1], d, 1) - 0.05) < 1e-9

    def test_missing_prev_price(self):
        price_map = {"2025-01-02": 105.0}  # 缺 prev
        d = self._dates()
        assert get_benchmark_return_fast(price_map, d[1], d, 1) == 0

    def test_equal_weight_pool_uses_only_same_window_covered_members(self):
        frames = {
            "a": pd.DataFrame({"trade_date": ["20250102", "20250109"], "open": [10.0, 12.0], "close": [11.0, 12.0]}),
            "b": pd.DataFrame({"trade_date": ["20250102", "20250109"], "open": [20.0, 18.0], "close": [20.0, 18.0]}),
            "future": pd.DataFrame({"trade_date": ["20250109"], "open": [5.0], "close": [5.0]}),
        }
        result = equal_weight_pool_return(frames, "2025-01-02", "2025-01-09", "open", "close")
        assert result["eligible_count"] == 2
        assert result["covered_count"] == 2
        assert result["coverage"] == 1.0
        assert result["return"] == pytest.approx(0.05)

    def test_equal_weight_pool_respects_pit_membership_codes(self):
        frames = {
            "000001": pd.DataFrame({
                "trade_date": ["20250102", "20250109"],
                "open": [10.0, 11.0], "close": [10.0, 11.0],
            }),
            "000002": pd.DataFrame({
                "trade_date": ["20250102", "20250109"],
                "open": [10.0, 8.0], "close": [10.0, 8.0],
            }),
        }
        result = equal_weight_pool_return(
            frames, "2025-01-02", "2025-01-09", codes=["000001"],
        )
        assert result["return"] == pytest.approx(0.1)
        assert result["eligible_count"] == 1
        assert result["covered_count"] == 1


@pytest.mark.p1
class TestPrecomputeLimit:
    def _price_cache(self, code="600000"):
        df = pd.DataFrame({
            "trade_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "close": [10.0, 11.0],  # +10% = 涨停(主板阈值 9.8%)
        })
        return {code: df}

    def test_limit_up_detected(self):
        cache = self._price_cache()
        limit = {}
        precompute_limit_up(cache, limit)
        # 主板 10% 涨停, 阈值 9.8%, 涨幅 10% >= 9.8% → 涨停
        assert "600000" in limit
        assert "20250103" in limit["600000"]

    def test_no_limit_up(self):
        df = pd.DataFrame({
            "trade_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "close": [10.0, 10.5],  # +5% 未涨停
        })
        limit = {}
        precompute_limit_up({"600000": df}, limit)
        assert "600000" not in limit

    def test_limit_down_detected(self):
        df = pd.DataFrame({
            "trade_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "close": [10.0, 9.0],  # -10% = 跌停
        })
        limit = {}
        precompute_limit_down({"600000": df}, limit)
        assert "600000" in limit
        assert "20250103" in limit["600000"]

    def test_limit_up_after_long_holiday_is_detected(self):
        df = pd.DataFrame({
            "trade_date": pd.to_datetime(["2025-01-24", "2025-02-05"]),
            "open": [10.0, 11.0],
            "close": [10.0, 11.0],
        })
        limit = {}
        precompute_limit_up({"600000": df}, limit)
        assert limit["600000"]["20250205"] is True

    def test_explicit_no_limit_date_is_exempt(self):
        df = pd.DataFrame({
            "trade_date": pd.to_datetime(["2025-01-24", "2025-02-05"]),
            "open": [10.0, 11.0],
            "close": [10.0, 11.0],
        })
        limit = {}
        precompute_limit_up(
            {"600000": df},
            limit,
            no_limit_dates={"600000": {"20250205"}},
        )
        assert "600000" not in limit

    def test_point_in_time_st_status_controls_limit_threshold(self):
        df = pd.DataFrame({
            "trade_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "open": [10.0, 10.5],
            "close": [10.0, 10.5],
        })
        historical_st = {}
        precompute_limit_up(
            {"600000": df.copy()}, historical_st,
            st_codes_by_date={"20250102": set(), "20250103": {"600000"}},
        )
        historical_normal = {}
        precompute_limit_up(
            {"600000": df.copy()}, historical_normal,
            st_codes={"600000"},
            st_codes_by_date={"20250102": set(), "20250103": set()},
        )

        assert historical_st["600000"]["20250103"] is True
        assert "600000" not in historical_normal
