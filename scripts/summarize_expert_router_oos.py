import argparse
import json
import math
from pathlib import Path


def _fmt(value, percent=False):
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "-"
    return f"{value:.2%}" if percent else f"{value:.4f}"


def _metric_rows(name, metrics):
    rows = []
    for top in ("top5", "top10"):
        item = metrics.get(top, {})
        rows.append(
            f"| {name} | {top} | {item.get('n_periods', '-')} | "
            f"{_fmt(item.get('total_return'), True)} | {_fmt(item.get('annual_return'), True)} | "
            f"{_fmt(item.get('rank_ic'))} | {_fmt(item.get('icir'))} | "
            f"{_fmt(item.get('turnover'), True)} | {_fmt(item.get('max_drawdown'), True)} | "
            f"{_fmt(item.get('positive_ratio'), True)} | {_fmt(item.get('transaction_cost'))} |"
        )
    return rows


def _window_section(window_id, window):
    heading = "### 最终 OOS 指标" if window_id == "final_oos" else f"## 开发窗口 {window_id}"
    lines = [heading, "", f"- 周期：`{window['period']}`"]
    constraints = window.get("execution_constraints", {})
    lines.append(f"- 成交约束覆盖率：`{_fmt(constraints.get('coverage'), True)}`，价格代码数：`{constraints.get('price_codes', '-')}`")
    lines.extend([
        "",
        "### 专家与基线扣费指标",
        "",
        "| 组合 | TopN | 期数 | 总收益 | 年化收益 | Rank IC | ICIR | 单期换手 | 最大回撤 | 盈利期占比 | 累计费用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    lines.extend(_metric_rows("core", window.get("core_baseline", {})))
    for mode, result in window.get("modes", {}).items():
        lines.append(f"| **{mode} / trend** |  |  |  |  |  |  |  |  |  |  |")
        lines.extend(_metric_rows(f"{mode} / trend", result.get("trend", {}).get("metrics", {})))
        lines.append(f"| **{mode} / rebound** |  |  |  |  |  |  |  |  |  |  |")
        lines.extend(_metric_rows(f"{mode} / rebound", result.get("rebound", {}).get("metrics", {})))
    lines.extend(["", "### 路由器与最终组合", "", "| 训练方式 | AUC | Brier | LogLoss | ECE | 样本数 |", "|---|---:|---:|---:|---:|---:|"])
    for mode, result in window.get("router", {}).items():
        d = result.get("diagnostics", {})
        lines.append(f"| {mode} | {_fmt(d.get('auc'))} | {_fmt(d.get('brier'))} | {_fmt(d.get('log_loss'))} | {_fmt(d.get('ece'))} | {d.get('n', '-')} |")
    lines.extend(["", "| 训练方式 / 路由 | TopN | 总收益 | 年化收益 | Rank IC | ICIR | 换手 | 最大回撤 | 盈利期占比 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for mode, result in window.get("router", {}).items():
        for route_name in ("fixed_trend", "hard_vol_switch", "soft_fusion"):
            lines.extend(
                f"| {mode} / {route_name} | {top} | {_fmt(result.get(route_name, {}).get(top, {}).get('total_return'), True)} | "
                f"{_fmt(result.get(route_name, {}).get(top, {}).get('annual_return'), True)} | "
                f"{_fmt(result.get(route_name, {}).get(top, {}).get('rank_ic'))} | "
                f"{_fmt(result.get(route_name, {}).get(top, {}).get('icir'))} | "
                f"{_fmt(result.get(route_name, {}).get(top, {}).get('turnover'), True)} | "
                f"{_fmt(result.get(route_name, {}).get(top, {}).get('max_drawdown'), True)} | "
                f"{_fmt(result.get(route_name, {}).get(top, {}).get('positive_ratio'), True)} |"
                for top in ("top5", "top10")
            )
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    lines = [
        "# 专家训练方式与市场路由 OOS 实验报告",
        "",
        f"- 实验指纹：`{data.get('experiment_fingerprint', data.get('fingerprint', '-'))}`",
        f"- 因子快照：`{data.get('factor_source', '-')}`，股票池数量：`{data.get('instrument_count', '-')}`",
        "- 约束：冻结核心/趋势/反弹因子；严格 walk-forward；标签 horizon=10；成交约束和交易成本已纳入。",
    ]
    for window_id, window in data.get("windows", {}).items():
        lines.extend(["", *_window_section(window_id, window)])
    final = data.get("final_oos")
    if final:
        lines.extend(["", "## 最终 OOS（锁定）", "", f"- 锁定信息：`{data.get('final_oos_lock', {})}`", *_window_section("final_oos", final)])
    lines.extend(["", "## 解释边界", "", "- 路由器标签来自两个专家扣费后收益差，不是人为定义的市场状态标签。", "- AUC/校准只说明专家胜负概率预测质量；最终是否采用以 Top5/Top10 扣费收益、IC/ICIR、换手、回撤及跨窗口稳定性为准。", "- 最终 OOS 虽已锁定，但测试仅含 3 个调仓期，不能单独作为生产晋级依据；需结合开发窗口稳定性和后续扩展 OOS。"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
