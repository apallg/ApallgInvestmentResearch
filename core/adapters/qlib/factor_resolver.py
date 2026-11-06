"""因子表达式统一解析 — 从 custom_handler._build_custom_features 提取的单因子解析逻辑。

供训练 (WhiteBoxHandler) 与评估 (PublicFactorAdapter) 共用, 消除「因子管道断裂」。
返回 (expr, skip_reason): 成功时 expr 有效、skip_reason=None; 失败时 expr=None。
"""
import logging
import re

logger = logging.getLogger(__name__)


def detect_future_shift(expr):
    """检测 qlib 表达式中的前视算子(未来函数)。

    Qlib 约定: Ref(field, -N) / Delta(field, -N) 表示 N 天后(前视)。
    因子表达式里出现负位移一律视为未来函数。返回命中列表(空=安全)。
    """
    if not expr:
        return []
    hits = []
    for fn in ("Ref", "Delta"):
        i = 0
        while True:
            i = expr.find(fn + "(", i)
            if i < 0:
                break
            depth = 0
            j = i + len(fn) + 1
            while j < len(expr):
                c = expr[j]
                if c == "(":
                    depth += 1
                elif c == ")":
                    if depth == 0:
                        break
                    depth -= 1
                j += 1
            if j >= len(expr):
                break
            inner = expr[i + len(fn) + 1:j]
            args = []
            d = 0
            cur = ""
            for c in inner:
                if c == "(":
                    d += 1
                    cur += c
                elif c == ")":
                    d -= 1
                    cur += c
                elif c == "," and d == 0:
                    args.append(cur.strip())
                    cur = ""
                else:
                    cur += c
            if cur.strip():
                args.append(cur.strip())
            if len(args) >= 2 and args[1].startswith("-"):
                shift = args[1].lstrip("-").strip()
                if re.fullmatch(r"\d+(\.\d+)?", shift):
                    hits.append(f"{fn}(..., -{shift})")
            i = j + 1
    return hits


def resolve_factor_expression(factor_key):
    """单个因子 key → 有效 Qlib 表达式。

    查找优先级 (与 _build_custom_features 一致):
      1. 因子池 entry.expr
      2. entry.template_key → FACTOR_TEMPLATES
      3. key 本身 → FACTOR_TEMPLATES
      4. "wb_" 前缀反向映射
    之后做兼容性修复 + 常量/无效表达式过滤。
    """
    from core.domain.factors.whitebox import FACTOR_TEMPLATES
    from core.adapters.qlib.factor_expression import to_qlib_expression
    # 惰性 import 避免与 custom_handler 的循环依赖
    from core.adapters.qlib.custom_handler import _fix_qlib_compat, _fix_qlib_expr

    # 因子池懒加载
    pool_entries = {}
    try:
        from core.domain.factors.registry import get_pool
        pool = get_pool()
        for cat in ("white_box", "traditional", "hybrid"):
            pool_entries.update(pool.state.get(cat, {}))
    except Exception:
        pass

    entry = pool_entries.get(factor_key, {})
    expr = entry.get("expr")

    if not expr and str(factor_key).startswith(("gp_", "llm_")):
        try:
            from core.data.repos.factor_mining import factor_mining_repo
            candidate = factor_mining_repo.find_candidate(factor_key)
            if candidate:
                expr = candidate.get("expr")
        except Exception:
            pass

    if not expr:
        tk = entry.get("template_key", "")
        tmpl = FACTOR_TEMPLATES.get(tk) or FACTOR_TEMPLATES.get(factor_key)
        if tmpl:
            expr = to_qlib_expression(tmpl)

    if not expr:
        tmpl = FACTOR_TEMPLATES.get("wb_" + factor_key)
        if tmpl:
            expr = to_qlib_expression(tmpl)

    if not expr or expr.startswith("#"):
        return None, (expr if expr else "# SKIP: 无表达式")

    # 兼容性修复: Ret/IdxMin/RowNum/Quantile/Skewness/Kurtosis
    expr = _fix_qlib_compat(expr)
    if expr.startswith("# SKIP:"):
        return None, expr

    # 嵌套 Max 在修复后再检查
    if "Max(" in expr and ",Max(" in expr:
        return None, "# SKIP: nested Max not supported in qlib"

    expr = _fix_qlib_expr(expr)
    expr = re.sub(r'Delta\(\s*(FundVal\(\s*\$\w+\s*,\s*\d+\s*\))\s*,\s*([1-9]\d*)\s*\)',
                  lambda match: f'({match[1]}-Ref({match[1]},{match[2]}))', expr)

    # 前视防护: 修复后仍残留负位移(未来函数)的一律拒绝
    future_hits = detect_future_shift(expr)
    if future_hits:
        return None, f"# SKIP: future-shift(未来函数) detected: {future_hits[:3]}"

    # 过滤求值后为纯常量/无效表达式的因子
    try:
        from qlib.data.base import Expression
        from qlib.utils import parse_field
        obj = eval(parse_field(expr))
        if not isinstance(obj, (Expression,)):
            return None, f"# SKIP: 求值为 {type(obj).__name__} 而非 Expression"
        obj.get_extended_window_size()
        obj.get_longest_back_rolling()
    except AttributeError as e:
        msg = str(e)
        if "not registered" in msg:
            pass  # 算子未注册, 留给 Qlib 原生流程处理
        else:
            return None, f"# SKIP: 嵌套标量 ({msg[:80]})"
    except Exception:
        pass  # 其他 eval 失败留给 Qlib 原生流程

    return expr, None
