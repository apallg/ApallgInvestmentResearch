"""
Qlib因子表达式转换 — 白盒模板算子 → Qlib DSL表达式

Qlib表达式语法参考:
  $open, $high, $low, $close, $volume, $amount, $vwap
  Ref($close, 5)           — 5天前的收盘价
  Mean($close, 20)          — 20日均值
  Std($close, 20)           — 20日标准差
  Corr($close, $volume, 20) — 20日相关系数
  Rank($close)              — 截面排名
  Delta($close, 5)          — 5日变化
  Sum($volume, 5)           — 5日总和
  Max(), Min(), Abs(), Log()
  $close / Ref($close, 1) - 1  — 日收益率
"""

# ══════════════════════════════════════════════
# 算子映射表
# ══════════════════════════════════════════════

OPERATOR_MAP = {
    "momentum":
        "Ref($close, 1) / Ref($close, {n}) - 1",
    "volatility":
        "Std(Ref($close, 1) / Ref($close, 2) - 1, {n})",
    "correlation":
        "Corr($close, $volume, {n})",
    "price_correlation":
        "Corr(Ref($close, 1) / Ref($close, 2) - 1, Ref($volume, 1) / Ref($volume, 2) - 1, {n})",
    "turnover_diff":
        "(Mean($turnover, {n}) / Mean($turnover, {n}*2) - 1)",
    "atr":
        "Mean(Max($high - $low, Max(Abs($high - Ref($close, 1)), Abs($low - Ref($close, 1)))), {n})",
    "volume_ratio":
        "Mean($volume, {short}) / Mean($volume, {long})",
    "price_position":
        "($close - Min($low, {n})) / (Max($high, {n}) - Min($low, {n}) + 1e-8)",
    "ma_cross":
        "Mean($close, {short}) / Mean($close, {long}) - 1",
    "breakout":
        "($close / Max($high, {n}) - 1)",
    "bb_width":
        "(Mean($close, {n}) + 2*Std($close, {n}) - (Mean($close, {n}) - 2*Std($close, {n}))) / Mean($close, {n})",
    "rsi":
        "Sum(Max(Ref($close, 1) / Ref($close, 2) - 1, 0), {n}) / (Sum(Abs(Ref($close, 1) / Ref($close, 2) - 1), {n}) + 1e-8)",
}


def to_qlib_expression(template_def, params=None):
    """将白盒因子模板转为Qlib表达式字符串。

    Args:
        template_def: whitebox_templates.py中的模板dict
        params: 参数dict, 用于填充{n}等占位符

    Returns:
        Qlib表达式字符串
    """
    expr = template_def.get("qlib_expression", "")

    if not expr or expr.startswith("#"):
        # 无Qlib表达式: 尝试从operator推导
        expr = _derive_expression(template_def)

    # 填入参数
    if params:
        for k, v in params.items():
            expr = expr.replace(f"{{{k}}}", str(v))

    # 清理未填充的占位符（用默认值）
    import re
    remaining = re.findall(r'\{(\w+)\}', expr)
    for key in remaining:
        if params and key in params:
            expr = expr.replace(f"{{{key}}}", str(params[key]))
        else:
            # 从param_definitions找默认值
            for pdef in template_def.get("param_definitions", []):
                if pdef["name"] == key:
                    expr = expr.replace(f"{{{key}}}", str(pdef["default"]))
                    break
            else:
                expr = expr.replace(f"{{{key}}}", "20")

    return expr.strip()


def _derive_expression(template_def):
    """从模板operator推导Qlib表达式（无法精确推导时返回注释说明）"""
    operator = template_def.get("operator", "")
    dataflow = template_def.get("dataflow", "price_only")

    if dataflow == "price_with_db":
        return f"# 依赖外部数据库({template_def.get('label', '')})，需通过custom_handler注入"

    key = template_def.get("key", template_def.get("template_key", ""))

    # 尝试匹配已知模式
    if "PEG" in operator or "ROE" in operator:
        return "# 依赖PEG/ROE外部数据"
    if "Corr" in operator or "corr" in operator:
        return "-Corr($close / Ref($close, 1) - 1, $volume / Ref($volume, 1) - 1, {n})"
    if "turnover" in operator.lower() and "price" in operator.lower():
        return "(Mean($turnover, {n}) / Mean($turnover, {n}*2) - 1) * ($close / Mean($close, {ma_period}) - 1)"
    if "vol" in operator.lower() and "break" in operator.lower():
        return "($close / Max($high, {breakout_window}) - 1) / Mean(Abs($close - Ref($close, 1)), {n})"
    if "残差" in template_def.get("label", "") or "residual" in operator.lower():
        return "Ref($close, 1) / Ref($close, {n}) - 1"
    if "量比" in operator or "volume_ratio" in operator.lower() or "amount_ratio" in operator.lower():
        return "(Mean($amount, {n}) / Mean($amount, {n}+20) - 1)"

    # 默认: 简单动量 — Qlib 中 Ref(x, N) 为正数表示 N 天前
    n = template_def.get('default_n', 20)
    return f"$close / Ref($close, {n}) - 1"


def generate_qlib_factor_file(factor_keys, output_path=None):
    """将通过校验的因子集导出为Qlib可加载的因子定义文件。

    Args:
        factor_keys: list of template keys
        output_path: 输出文件路径
    """
    from core.domain.factors.whitebox import FACTOR_TEMPLATES

    lines = [
        '# 白盒自定义因子集 - Apallg投研',
        '# 自动生成，每行一个因子表达式',
        '# 用法: 在训练时指定 factor_set="WhiteBox_Custom"',
        '',
        'import numpy as np',
        '',
        'CUSTOM_FACTORS = {',
    ]

    for key in factor_keys:
        tmpl = FACTOR_TEMPLATES.get(key)
        if not tmpl:
            continue
        expr = to_qlib_expression(tmpl)
        lines.append(f'    "{key}": lambda df: {expr},')
        # 安全的注释（避免在f-string中出现花括号问题）
        label = tmpl.get("label", key)
        lines.append(f'    # {label}')

    lines.append('}')
    lines.append('')
    lines.append(f'FACTOR_NAMES = {factor_keys}')

    if output_path:
        import os
        path = output_path
    else:
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "data", "qlib_custom_factors.py")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path
