"""PIT 工具层测试 — 确保财报生效日规则正确，防止前视。"""
import datetime as dt

from core.data.pit_utils import FALLBACK_LAG_DAYS, effective_date


def test_effective_date_prefers_ann_date():
    report_date = dt.date(2024, 6, 30)
    ann_date = dt.date(2024, 8, 15)
    assert effective_date(report_date, ann_date) == ann_date


def test_effective_date_falls_back_to_legal_deadline():
    report_date = dt.date(2024, 6, 30)
    expected = report_date + dt.timedelta(days=FALLBACK_LAG_DAYS[(6, 30)])
    assert effective_date(report_date, None) == expected


def test_effective_date_ignores_invalid_ann_date_before_report():
    report_date = dt.date(2024, 12, 31)
    ann_date = dt.date(2024, 10, 1)
    expected = report_date + dt.timedelta(days=FALLBACK_LAG_DAYS[(12, 31)])
    assert effective_date(report_date, ann_date) == expected


def test_effective_date_quarterly_deadlines():
    cases = [
        (dt.date(2024, 3, 31), 30),
        (dt.date(2024, 6, 30), 62),
        (dt.date(2024, 9, 30), 31),
        (dt.date(2024, 12, 31), 120),
    ]
    for report_date, lag in cases:
        eff = effective_date(report_date, None)
        assert eff == report_date + dt.timedelta(days=lag)
        assert eff > report_date
