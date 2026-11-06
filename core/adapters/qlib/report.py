"""Qlib 官方 Report 适配器 — 将 qlib 原始回测报告转成标准指标/曲线。"""
from __future__ import annotations

import pandas as pd


def build_standard_report(report_normal: pd.DataFrame | None):
    """从 qlib report_normal 生成标准风险指标和图表数据。

    report_normal 通常包含列：return, bench, cost。
    """
    if report_normal is None or len(report_normal) == 0:
        return {}

    from qlib.contrib.evaluate import risk_analysis

    result = {
        "dates": [str(x)[:10] for x in report_normal.index],
        "cumulative_return": (1 + report_normal["return"]).cumprod().round(6).tolist(),
        "benchmark_return": (1 + report_normal["bench"]).cumprod().round(6).tolist(),
        "excess_return_without_cost": (1 + (report_normal["return"] - report_normal["bench"])).cumprod().round(6).tolist(),
    }

    if "cost" in report_normal.columns:
        result["excess_return_with_cost"] = (
            1 + (report_normal["return"] - report_normal["bench"] - report_normal["cost"])
        ).cumprod().round(6).tolist()
        risk_cost = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"])
        result["risk_with_cost"] = _risk_to_dict(risk_cost)
    else:
        result["risk_with_cost"] = {}

    result["risk_without_cost"] = _risk_to_dict(risk_analysis(report_normal["return"] - report_normal["bench"]))
    result["risk"] = _risk_to_dict(risk_analysis(report_normal["return"]))
    return result


def _risk_to_dict(risk_series: pd.Series) -> dict:
    if risk_series is None:
        return {}
    return {str(k): _safe_float(v) for k, v in risk_series.items()}


def _safe_float(v):
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return round(f, 6)
    except (TypeError, ValueError):
        return None
