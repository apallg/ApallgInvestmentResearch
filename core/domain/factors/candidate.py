import re
import hashlib

import numpy as np

from core.domain.factors.expression import parse_factor


ALLOWED_FIELDS = {
    "open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg",
}
_FIELD_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_FUTURE_RE = re.compile(r"(?:Ref|Delay)\s*\([^,]+,\s*-")
_WINDOW_RE = re.compile(r",\s*(\d+)\s*\)")


def required_fields(expr):
    return sorted({name.lower() for name in _FIELD_RE.findall(expr or "")})


def expression_complexity(expr):
    text = expr or ""
    operators = len(re.findall(r"\b(?:Ref|Delay|Mean|Std|Sum|Delta|Corr|Cov|Max|Min|Ts_Rank|Abs|Log|Sign|Rank|Scale)\b|[+*/-]", text))
    fields = len(required_fields(text))
    constants = len(re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?", text))
    depth = 0
    max_depth = 0
    for char in text:
        if char == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ")":
            depth = max(0, depth - 1)
    return operators + fields + constants + max_depth


def expression_windows(expr):
    return [int(value) for value in _WINDOW_RE.findall(str(expr or ""))]


def replace_expression_window(expr, old, new):
    text = str(expr or "")
    pattern = re.compile(rf"(,\s*){int(old)}(\s*\))")
    return pattern.sub(rf"\g<1>{int(new)}\g<2>", text, count=1)


def _ast_key(node):
    tag = node[0]
    if tag == "num":
        return (tag, round(float(node[1]), 12))
    if tag in ("field", "name"):
        return (tag, str(node[1]).lower())
    if tag == "neg":
        return (tag, _ast_key(node[1]))
    if tag == "tern":
        return (tag, _ast_key(node[1]), _ast_key(node[2]), _ast_key(node[3]))
    if tag == "bin":
        op = node[1]
        children = [_ast_key(node[2]), _ast_key(node[3])]
        if op in ("+", "*"):
            children.sort(key=repr)
        return (tag, op, *children)
    if tag == "call":
        return (tag, str(node[1]).lower(), *[_ast_key(arg) for arg in node[2]])
    return tuple(node)


def canonical_expression(expr):
    text = re.sub(r"\s+", "", str(expr or ""))
    try:
        ast, outer = parse_factor(text)
        return repr((outer, _ast_key(ast)))
    except Exception:
        return text.lower()


def validate_expression(expr):
    text = str(expr or "").strip()
    if not text:
        return {"valid": False, "reason": "表达式为空", "required_fields": []}
    fields = required_fields(text)
    unknown = sorted(set(fields) - ALLOWED_FIELDS)
    if unknown:
        return {"valid": False, "reason": f"非法字段: {', '.join(unknown)}", "required_fields": fields}
    if _FUTURE_RE.search(text):
        return {"valid": False, "reason": "表达式包含未来窗口", "required_fields": fields}
    try:
        parse_factor(text)
    except Exception as exc:
        return {"valid": False, "reason": str(exc), "required_fields": fields}
    return {"valid": True, "reason": "", "required_fields": fields}


def build_candidate(expr, source="gp", seed=None, parent_exprs=None, metrics=None, status="generated"):
    check = validate_expression(expr)
    if not check["valid"]:
        raise ValueError(check["reason"])
    return {
        "expr": str(expr).strip(),
        "source": source,
        "seed": seed,
        "parent_exprs": list(parent_exprs or []),
        "complexity": expression_complexity(expr),
        "required_fields": check["required_fields"],
        "metrics": dict(metrics or {}),
        "status": status,
    }


def candidate_key(expr, source="gp"):
    digest = hashlib.sha1(canonical_expression(expr).encode("utf-8")).hexdigest()[:10]
    prefix = "llm" if source == "llm" else "gp"
    return prefix + "_" + digest


def max_abs_correlation(candidate_scores, reference_scores, min_count=20):
    best_corr = None
    best_key = None
    compared = 0
    for key, scores in (reference_scores or {}).items():
        common = [code for code in candidate_scores if code in scores]
        if len(common) < min_count:
            continue
        left = np.asarray([candidate_scores[code] for code in common], dtype=float)
        right = np.asarray([scores[code] for code in common], dtype=float)
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.sum() < min_count:
            continue
        left = left[finite]
        right = right[finite]
        if np.std(left) < 1e-9 or np.std(right) < 1e-9:
            continue
        corr = abs(float(np.corrcoef(left, right)[0, 1]))
        if not np.isfinite(corr):
            continue
        compared += 1
        if best_corr is None or corr > best_corr:
            best_corr = corr
            best_key = key
    return best_corr, best_key, compared
