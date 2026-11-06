"""
调仓频率元数据 (REBALANCE_FREQ / get_freq_meta) 单一事实源测试。

回归重点: 新增 3d/4d/5d/10d/20d/3w 后, 各消费点 (engine/allocator/metrics/screening)
统一从这里取值, 确保 11 个频率的派生参数一致。
"""
import pytest

from core.shared.constants import REBALANCE_FREQ, get_freq_meta, cooldown_periods


@pytest.mark.p0
@pytest.mark.unit
class TestFreqMeta:
    def test_all_freqs_present(self):
        assert set(REBALANCE_FREQ) == {"day", "2d", "3d", "4d", "5d", "10d", "20d", "week", "biweek", "3w", "month"}

    def test_days(self):
        assert REBALANCE_FREQ["day"].days == 1
        assert REBALANCE_FREQ["2d"].days == 2
        assert REBALANCE_FREQ["3d"].days == 3
        assert REBALANCE_FREQ["4d"].days == 4
        assert REBALANCE_FREQ["5d"].days == 5
        assert REBALANCE_FREQ["10d"].days == 10
        assert REBALANCE_FREQ["20d"].days == 20
        assert REBALANCE_FREQ["week"].days == 7
        assert REBALANCE_FREQ["biweek"].days == 14
        assert REBALANCE_FREQ["3w"].days == 21
        assert REBALANCE_FREQ["month"].days == 21

    def test_pd_freq(self):
        assert REBALANCE_FREQ["3d"].pd_freq == "3B"
        assert REBALANCE_FREQ["4d"].pd_freq == "4B"
        assert REBALANCE_FREQ["3w"].pd_freq == "3W-THU"
        assert REBALANCE_FREQ["week"].pd_freq == "W-THU"
        assert REBALANCE_FREQ["month"].pd_freq == "MS"

    def test_stride(self):
        for f in ("day", "2d", "3d", "4d"):
            assert REBALANCE_FREQ[f].stride is True
        for f in ("week", "biweek", "3w", "month"):
            assert REBALANCE_FREQ[f].stride is False

    def test_cooldown(self):
        # 止损 2 周 / 止盈 1 周换算为期数: 交易日频率 ceil(10/days) / ceil(5/days)
        assert (REBALANCE_FREQ["day"].cooldown_stop, REBALANCE_FREQ["day"].cooldown_profit) == (10, 5)
        assert (REBALANCE_FREQ["2d"].cooldown_stop, REBALANCE_FREQ["2d"].cooldown_profit) == (5, 3)
        assert (REBALANCE_FREQ["3d"].cooldown_stop, REBALANCE_FREQ["3d"].cooldown_profit) == (4, 2)
        assert (REBALANCE_FREQ["4d"].cooldown_stop, REBALANCE_FREQ["4d"].cooldown_profit) == (3, 2)
        assert (REBALANCE_FREQ["5d"].cooldown_stop, REBALANCE_FREQ["5d"].cooldown_profit) == (2, 1)
        assert (REBALANCE_FREQ["10d"].cooldown_stop, REBALANCE_FREQ["10d"].cooldown_profit) == (1, 1)
        assert (REBALANCE_FREQ["20d"].cooldown_stop, REBALANCE_FREQ["20d"].cooldown_profit) == (1, 1)
        assert (REBALANCE_FREQ["week"].cooldown_stop, REBALANCE_FREQ["week"].cooldown_profit) == (2, 1)
        assert (REBALANCE_FREQ["biweek"].cooldown_stop, REBALANCE_FREQ["biweek"].cooldown_profit) == (1, 1)
        assert (REBALANCE_FREQ["3w"].cooldown_stop, REBALANCE_FREQ["3w"].cooldown_profit) == (1, 1)
        assert (REBALANCE_FREQ["month"].cooldown_stop, REBALANCE_FREQ["month"].cooldown_profit) == (1, 1)

    def test_periods_per_year(self):
        assert REBALANCE_FREQ["3d"].periods_per_year == 84
        assert REBALANCE_FREQ["4d"].periods_per_year == 63
        assert REBALANCE_FREQ["3w"].periods_per_year == 17
        assert REBALANCE_FREQ["month"].periods_per_year == 12

    def test_hold_mult(self):
        for f in ("day", "2d", "3d", "4d"):
            assert REBALANCE_FREQ[f].hold_mult == 8
        for f in ("week", "biweek", "3w", "month"):
            assert REBALANCE_FREQ[f].hold_mult == 2

    def test_unknown_freq_falls_back_to_week(self):
        assert get_freq_meta("nonsense") == REBALANCE_FREQ["week"]
        assert get_freq_meta("").days == 7


@pytest.mark.p0
@pytest.mark.unit
class TestCooldownPeriods:
    def test_default_none(self):
        assert cooldown_periods("week") == (2, 1)
        assert cooldown_periods("day") == (10, 5)
        assert cooldown_periods("3d") == (4, 2)

    def test_explicit_weeks(self):
        assert cooldown_periods("week", stop_weeks=4, profit_weeks=3) == (4, 3)
        assert cooldown_periods("day", stop_weeks=4, profit_weeks=3) == (20, 15)
        assert cooldown_periods("biweek", stop_weeks=4) == (2, 1)

    def test_min_one_period(self):
        assert cooldown_periods("biweek", stop_weeks=1) == (1, 1)
        assert cooldown_periods("3w", stop_weeks=1, profit_weeks=1) == (1, 1)
