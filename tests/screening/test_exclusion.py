import pytest
from core.domain.screening.exclusion import is_excluded, filter_codes


class TestIsExcluded:
    def test_excludes_etf_prefix_51(self):
        ok, reason = is_excluded("510050")
        assert ok is True
        assert "51" in reason

    def test_excludes_etf_prefix_15(self):
        ok, reason = is_excluded("159915")
        assert ok is True

    def test_excludes_b_share_920(self):
        ok, reason = is_excluded("920001")
        assert ok is True
        assert "920" in reason

    def test_excludes_b_share_900(self):
        ok, reason = is_excluded("900901")
        assert ok is True

    def test_excludes_b_share_200(self):
        ok, reason = is_excluded("200001")
        assert ok is True

    def test_excludes_st_name(self):
        ok, reason = is_excluded("000001", "ST康美")
        assert ok is True
        assert reason == "st_stock"

    def test_excludes_star_st(self):
        ok, reason = is_excluded("000002", "*ST信威")
        assert ok is True
        assert reason == "st_stock"

    def test_excludes_beijing_exchange(self):
        ok, reason = is_excluded("830001.BJ")
        assert ok is True
        assert reason == "beijing_exchange"

    def test_excludes_invalid_code_empty(self):
        ok, reason = is_excluded("")
        assert ok is True
        assert reason == "invalid_code"

    def test_excludes_invalid_code_short(self):
        ok, reason = is_excluded("12")
        assert ok is True
        assert reason == "invalid_code"

    def test_passes_normal_a_share_shanghai(self):
        ok, reason = is_excluded("600519")
        assert ok is False
        assert reason == ""

    def test_passes_normal_a_share_shenzhen(self):
        ok, reason = is_excluded("000001")
        assert ok is False

    def test_passes_gem_300(self):
        ok, reason = is_excluded("300750")
        assert ok is False

    def test_passes_star_market_688(self):
        ok, reason = is_excluded("688981")
        assert ok is False

    def test_passes_code_with_dot(self):
        ok, reason = is_excluded("sh.600519")
        assert ok is False
        assert reason == ""


class TestFilterCodes:
    def test_filters_mixed_list(self):
        codes = [
            ("600519", "贵州茅台"),
            ("510050", "50ETF"),
            ("000001", "ST康美"),
            ("300750", "宁德时代"),
            ("920001", "B股"),
        ]
        result = filter_codes(codes)
        assert len(result) == 2
        assert result[0] == ("600519", "贵州茅台")
        assert result[1] == ("300750", "宁德时代")

    def test_all_valid_returns_all(self):
        codes = [("600519", "茅台"), ("000001", "平安")]
        result = filter_codes(codes)
        assert len(result) == 2

    def test_all_excluded_returns_empty(self):
        codes = [("510050", "ETF"), ("159915", "ETF2")]
        result = filter_codes(codes)
        assert result == []
