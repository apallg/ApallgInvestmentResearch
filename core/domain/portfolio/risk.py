"""
Barra 风格因子暴露监控 — 选股后检查组合在 10 个风格因子上的隐性集中

- 超过 ±0.3 标准差 → 告警
- 超过 ±0.5 标准差 → 约束修正
"""
import logging
import numpy as np
import pandas as pd

from core.shared import config

logger = logging.getLogger(__name__)

__all__ = ["compute_factor_exposures", "compute_portfolio_exposure",
           "check_risk", "format_risk_report", "try_correct_weights",
           "FACTOR_NAMES"]

FACTOR_NAMES = [
    "size", "momentum", "volatility", "value",
    "growth", "quality", "leverage", "liquidity",
    "beta", "nonlinear_size",
]

FACTOR_LABELS = {
    "size": "市值", "momentum": "动量", "volatility": "波动率",
    "value": "价值", "growth": "成长", "quality": "质量",
    "leverage": "杠杆", "liquidity": "流动性",
    "beta": "Beta", "nonlinear_size": "非线性市值",
}

WARNING_THRESHOLD = 0.3
DANGER_THRESHOLD = 0.5
MIN_STOCKS_FOR_CORRECTION = 3


def compute_factor_exposures(codes, snapshot, fundamentals=None, index_returns=None):
    """计算每只股票在 10 个 Barra 风格因子上的暴露 (z-score)

    Args:
        codes: 股票代码列表
        snapshot: {code: (df, name, close, volume)}
        fundamentals: {code: {roe, debt_to_assets, ...}} 可选
        index_returns: pd.Series 指数日收益, 用于 Beta 计算

    Returns:
        {code: {factor_name: zscore}}
    """
    exposures = {}
    fundamentals = fundamentals or {}

    # 批量提取数据
    raw = {}
    for code in codes:
        entry = snapshot.get(code)
        if entry is None:
            continue
        df, name, close, volume = entry[:4]
        close_val = float(close) if close else 0
        vol_val = float(volume) if volume else 0
        if close_val <= 0:
            continue

        # 从 df 计算衍生指标
        closes = df["close"].astype(float)
        rets = closes.pct_change(fill_method=None).dropna()
        if len(rets) < 20:
            continue
        rets_20 = rets.iloc[-20:]

        raw[code] = {
            "close": close_val,
            "volume": vol_val,
            "returns": rets,
            "ret_20d": rets_20,
        }

    if len(raw) < MIN_STOCKS_FOR_CORRECTION:
        return {}

    all_codes = list(raw.keys())

    # --- 1. Size: ln(market_cap) ---
    sizes = {c: np.log(max(raw[c]["close"] * raw[c]["volume"] / 100, 1e6)) for c in all_codes}

    # --- 2. Momentum: 过去12个月收益 (剔除最近1个月) ---
    momentums = {}
    for c in all_codes:
        rets = raw[c]["returns"]
        if len(rets) >= 240:
            mom_rets = rets.iloc[-240:-20]
            momentums[c] = (1 + mom_rets).prod() - 1
        elif len(rets) >= 60:
            mom_rets = rets.iloc[-60:-5]
            momentums[c] = (1 + mom_rets).prod() - 1
        else:
            momentums[c] = 0

    # --- 3. Volatility: 日收益标准差 ---
    volatilities = {c: max(raw[c]["ret_20d"].std(), 0.001) for c in all_codes}

    # --- 4. Value: B/P (账面/市值), 用 1/close 做简化代理 + ROE ---
    values = {}
    for c in all_codes:
        fin = fundamentals.get(c, {})
        roe = fin.get("roe", 0)
        bp = roe / max(raw[c]["close"], 0.01) if roe else 1.0 / max(raw[c]["close"], 1)
        values[c] = bp

    # --- 5. Growth: 利润增速 ---
    growths = {}
    for c in all_codes:
        fin = fundamentals.get(c, {})
        growths[c] = fin.get("profit_growth", 0) or fin.get("revenue_growth", 0) or 0

    # --- 6. Quality: ROE ---
    qualities = {}
    for c in all_codes:
        fin = fundamentals.get(c, {})
        qualities[c] = fin.get("roe", 0) or 0

    # --- 7. Leverage: 资产负债率 ---
    leverages = {}
    for c in all_codes:
        fin = fundamentals.get(c, {})
        leverages[c] = fin.get("debt_to_assets", 0) or 0

    # --- 8. Liquidity: 月均换手率 ---
    liquidities = {}
    for c in all_codes:
        rets = raw[c]["returns"]
        if "turnover" in raw[c]:
            liquidities[c] = raw[c]["turnover"]
        else:
            liquidities[c] = raw[c]["volume"] / 1e8

    # --- 9. Beta: 对沪深300回归 ---
    betas = {}
    if index_returns is not None and len(index_returns) >= 20:
        idx_rets = index_returns.iloc[-60:] if len(index_returns) >= 60 else index_returns
        for c in all_codes:
            stock_rets = raw[c]["returns"].iloc[-len(idx_rets):]
            if len(stock_rets) < 20:
                betas[c] = 1.0
                continue
            common_len = min(len(stock_rets), len(idx_rets))
            sr = stock_rets.iloc[-common_len:]
            ir = idx_rets.iloc[-common_len:]
            cov = np.cov(sr, ir)[0, 1]
            var = np.var(ir)
            betas[c] = cov / var if var > 0 else 1.0
    else:
        for c in all_codes:
            betas[c] = 1.0

    # --- 10. Nonlinear Size: size^3 对 size 正交 ---
    size_arr = np.array([sizes[c] for c in all_codes])
    size_cube = size_arr ** 3
    if len(size_arr) > 2:
        from numpy.linalg import lstsq
        X = np.column_stack([np.ones(len(size_arr)), size_arr])
        coeffs, _, _, _ = lstsq(X, size_cube, rcond=None)
        predicted = X @ coeffs
        nl_residuals = size_cube - predicted
    else:
        nl_residuals = size_cube

    nonlinear_sizes = {c: float(nl_residuals[i]) for i, c in enumerate(all_codes)}

    # --- Z-score 标准化 ---
    def _zscore(d, invert=False):
        vals = np.array(list(d.values()))
        mu, std = vals.mean(), vals.std(ddof=1)
        if std < 1e-8:
            return {k: 0.0 for k in d}
        s = (vals - mu) / std
        return {k: float(s[i]) * (-1 if invert else 1) for i, k in enumerate(d)}

    exposures = {}
    size_z = _zscore(sizes, invert=True)  # 小盘→正暴露
    mom_z = _zscore(momentums)
    vol_z = _zscore(volatilities)
    val_z = _zscore(values)
    gro_z = _zscore(growths)
    qua_z = _zscore(qualities)
    lev_z = _zscore(leverages, invert=True)  # 高杠杆→负暴露
    liq_z = _zscore(liquidities)
    beta_z = _zscore(betas)
    nl_z = _zscore(nonlinear_sizes)

    for c in all_codes:
        exposures[c] = {
            "size": size_z.get(c, 0),
            "momentum": mom_z.get(c, 0),
            "volatility": vol_z.get(c, 0),
            "value": val_z.get(c, 0),
            "growth": gro_z.get(c, 0),
            "quality": qua_z.get(c, 0),
            "leverage": lev_z.get(c, 0),
            "liquidity": liq_z.get(c, 0),
            "beta": beta_z.get(c, 0),
            "nonlinear_size": nl_z.get(c, 0),
        }

    return exposures


def compute_portfolio_exposure(weights, exposures):
    """组合加权因子暴露

    Args:
        weights: {code: weight}
        exposures: {code: {factor: zscore}}

    Returns:
        {factor: weighted_zscore}
    """
    result = {}
    for factor in FACTOR_NAMES:
        total = 0.0
        for code, w in weights.items():
            exp = exposures.get(code, {})
            total += w * exp.get(factor, 0)
        result[factor] = round(total, 4)
    return result


def check_risk(portfolio_exposure):
    """检查组合因子暴露是否超标

    Returns:
        {factor: {level: 'ok'|'warning'|'danger', exposure: float}}
    """
    alerts = {}
    for factor, exp in portfolio_exposure.items():
        if abs(exp) >= DANGER_THRESHOLD:
            alerts[factor] = {"level": "danger", "exposure": exp,
                             "label": FACTOR_LABELS.get(factor, factor)}
        elif abs(exp) >= WARNING_THRESHOLD:
            alerts[factor] = {"level": "warning", "exposure": exp,
                             "label": FACTOR_LABELS.get(factor, factor)}
        else:
            alerts[factor] = {"level": "ok", "exposure": exp,
                             "label": FACTOR_LABELS.get(factor, factor)}
    return alerts


def format_risk_report(alerts):
    """格式化风险报告"""
    warnings = [v for v in alerts.values() if v["level"] != "ok"]
    if not warnings:
        return None

    lines = ["[Barra] 风格因子暴露告警:"]
    for item in warnings:
        icon = "🔴" if item["level"] == "danger" else "🟡"
        lines.append(f"  {icon} {item['label']}: {item['exposure']:+.2f}σ")
    return "\n".join(lines)


def try_correct_weights(weights, exposures, top_candidates,
                        max_iter=50, step=0.02):
    """对超标因子做迭代权重修正：超标的股票逐步降权，低暴露股票补权

    Args:
        weights: {code: weight}
        exposures: {code: {factor: zscore}}
        top_candidates: 候选股列表 (需要 code 字段)
        max_iter: 最大迭代次数
        step: 每次调整步长

    Returns:
        corrected_weights, report_lines
    """
    weights = dict(weights)
    if len(weights) < MIN_STOCKS_FOR_CORRECTION:
        return weights, []

    code_list = list(weights.keys())
    lo = 0.05  # 最低权重
    hi = 0.30  # 最高权重
    reports = []

    for _ in range(max_iter):
        pf_exp = compute_portfolio_exposure(weights, exposures)
        violated = {f: e for f, e in pf_exp.items() if abs(e) >= DANGER_THRESHOLD}

        if not violated:
            break

        for factor, over_exposure in violated.items():
            # 找出对该因子暴露最高的股票
            code_exposures = [(c, exposures.get(c, {}).get(factor, 0))
                            for c in code_list if c in weights]
            code_exposures.sort(key=lambda x: -x[1])

            # 降权暴露最高的 1-2 只
            for code, exp in code_exposures[:2]:
                if (over_exposure > 0 and exp > 0) or (over_exposure < 0 and exp < 0):
                    old_w = weights[code]
                    weights[code] = max(lo, old_w - step)
                    if weights[code] < old_w:
                        reports.append(f"  降权 {code}: {old_w:.1%}→{weights[code]:.1%} "
                                     f"({FACTOR_LABELS[factor]}{exp:+.1f}σ)")

            # 补权给暴露最低的
            for code, exp in reversed(code_exposures[-2:]):
                if (over_exposure > 0 and exp < 0) or (over_exposure < 0 and exp > 0):
                    weights[code] = min(hi, weights[code] + step)

        # 归一化
        w_sum = sum(weights.values())
        if w_sum > 0:
            weights = {c: w / w_sum for c, w in weights.items()}

    return weights, reports
