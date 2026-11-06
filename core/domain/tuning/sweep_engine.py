"""
参数遍历引擎 — 基于白盒模板自动生成参数变体

功能:
- 根据模板参数定义生成笛卡尔积变体
- 生成Python评分函数源码（注入buy_side.py用）
- 生成Qlib表达式字符串（模型训练用）
- 粗粒度先扫 + 精细遍历两阶段策略
"""
import itertools
import logging

logger = logging.getLogger(__name__)


class ParamSweepEngine:

    def __init__(self):
        from core.domain.factors.whitebox import FACTOR_TEMPLATES
        self.templates = FACTOR_TEMPLATES

    # ── 变体生成 ──

    def generate_variants(self, template_key, coarse=True):
        """根据模板参数定义生成所有变体。

        Args:
            template_key: 模板key
            coarse: True=粗粒度(快速初筛), False=精细遍历(通过初筛后)

        Returns:
            [{"variant_key": "wb_xxx_n20", "label": "...", "params": {...}, "template_key": "..."}, ...]
        """
        tmpl = self.templates.get(template_key)
        if not tmpl:
            raise ValueError(f"未知模板: {template_key}")

        param_grid = []
        param_names = []
        for pdef in tmpl.get("param_definitions", []):
            param_names.append(pdef["name"])
            if pdef["type"] == "int":
                values = pdef["range"]
                if coarse:
                    # 粗粒度: 只取首尾 + 中间1-2个点
                    if len(values) <= 3:
                        grid_values = values
                    else:
                        grid_values = [values[0], values[len(values)//2], values[-1]]
                else:
                    grid_values = values
                param_grid.append(grid_values)
            elif pdef["type"] == "float":
                if coarse:
                    # 粗粒度: 只取default附近
                    default = pdef["default"]
                    param_grid.append([default])
                else:
                    values = []
                    lo, hi = pdef["range"][0], pdef["range"][1]
                    step = pdef.get("step", 0.5)
                    v = lo
                    while v <= hi + 1e-9:
                        values.append(round(v, 2))
                        v += step
                    param_grid.append(values)

        if not param_grid:
            # 没有参数: single variant with defaults
            defaults = {p["name"]: p["default"] for p in tmpl.get("param_definitions", [])}
            return [self._make_variant(template_key, defaults)]

        variants = []
        seen = set()
        for combo in itertools.product(*param_grid):
            params = dict(zip(param_names, combo))
            # Coarse sweep dedup on default_n
            dedup_key = tuple(sorted(params.items()))
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            variants.append(self._make_variant(template_key, params))

        return variants

    def _make_variant(self, template_key, params):
        tmpl = self.templates[template_key]
        default_n = params.get("n", tmpl.get("default_n", 20))
        label_parts = []
        for k, v in sorted(params.items()):
            if isinstance(v, float):
                label_parts.append(f"{k}={v:.1f}")
            else:
                label_parts.append(f"{k}={v}")
        label_suffix = "_".join(label_parts)
        variant_key = f"{template_key}_{label_suffix}"
        return {
            "variant_key": variant_key,
            "template_key": template_key,
            "label": f"{tmpl['label']}({label_suffix})",
            "params": params,
            "default_n": default_n,
        }

    # ── 代码生成 ──

    def generate_scoring_code(self, variant, body_code=None):
        """根据变体参数生成完整的Python评分函数源码。

        Args:
            variant: generate_variants返回的变体dict
            body_code: 自定义函数体，None则使用模板的scoring_template

        Returns:
            (code_string, func_name)
        """
        tmpl = self.templates.get(variant["template_key"])
        if not tmpl:
            raise ValueError(f"未知模板: {variant['template_key']}")

        params = variant["params"]
        default_n = variant.get("default_n", tmpl.get("default_n", 20))

        if body_code:
            code = body_code
        elif "scoring_template" in tmpl:
            code = tmpl["scoring_template"]
            # 填入参数值
            for k, v in params.items():
                code = code.replace(f"{{{k}}}", str(v))
            code = code.replace("{default_n}", str(default_n))
            # 清理未填充的模板占位符
            for pdef in tmpl.get("param_definitions", []):
                code = code.replace(f"{{{pdef['name']}}}", str(pdef["default"]))
        else:
            # fallback: generic score function
            code = _generic_scoring_template(tmpl, variant)

        func_name = f"_score_{variant['template_key']}"
        return code, func_name

    def generate_qlib_expression(self, variant):
        """生成Qlib表达式，填入具体参数值。"""
        tmpl = self.templates.get(variant["template_key"])
        if not tmpl:
            return ""

        expr = tmpl.get("qlib_expression", "")
        params = variant["params"]

        # 填入参数值
        for k, v in params.items():
            expr = expr.replace(f"{{{k}}}", str(v))

        # 清理未填充的模板占位符
        import re
        expr = re.sub(r'\{[^}]+\}', '0', expr)

        return expr.strip()

    # ── 批量遍历 ──

    def sweep_all_variants(self, template_key, coarse=True):
        """返回模板的所有变体 + 对应的代码和Qlib表达式"""
        variants = self.generate_variants(template_key, coarse=coarse)
        results = []
        for v in variants:
            code, func_name = self.generate_scoring_code(v)
            qlib_expr = self.generate_qlib_expression(v)
            results.append({
                **v,
                "func_name": func_name,
                "code": code,
                "qlib_expression": qlib_expr,
            })
        return results

    def get_sweep_summary(self, template_keys=None):
        """获取所有模板的参数遍历概览（用于前端展示）"""
        keys = template_keys or list(self.templates.keys())
        summary = {}
        for k in keys:
            tmpl = self.templates.get(k)
            if not tmpl:
                continue
            coarse = self.generate_variants(k, coarse=True)
            fine = self.generate_variants(k, coarse=False)
            summary[k] = {
                "label": tmpl["label"],
                "category": tmpl["category"],
                "logic_desc": tmpl["logic_desc"],
                "dataflow": tmpl.get("dataflow", "price_only"),
                "data_safety": tmpl.get("data_safety", "safe"),
                "param_definitions": tmpl.get("param_definitions", []),
                "coarse_variant_count": len(coarse),
                "fine_variant_count": len(fine),
                "coarse_variants": coarse,
            }
        return summary


def _generic_scoring_template(tmpl, variant):
    """Fallback: 根据模板operator生成通用评分函数"""
    key = variant["template_key"]
    params = variant.get("params", {})
    default_n = variant.get("default_n", tmpl.get("default_n", 20))
    operator = tmpl.get("operator", "tanh(ret)")

    cols = tmpl.get("available_cols", ["close"])
    cols_str = ", ".join(f'"{c}"' for c in cols[:3])

    code = f'''def _score_{key}(df, n={default_n}):
    """{tmpl["label"]}: {operator}"""
    try:
        import math as _math
        closes = df["close"].astype(float)
        if len(closes) < n:
            return 0.0
        ret = closes.iloc[-1] / closes.iloc[-n] - 1.0
        result = _math.tanh(ret * 3.0)
        return round(float(result), 4)
    except Exception:
        return 0.0'''
    return code
