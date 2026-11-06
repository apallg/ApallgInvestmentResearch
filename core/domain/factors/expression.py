"""
轻量 qlib 表达式求值器 — 在单只股票的 OHLCV DataFrame 上对 qlib DSL 表达式求值。
用于对公开 Alpha158/101/191 因子（以表达式字符串存储）求出因子值向量，供 IC 校验。

约定（重要）:
- 正 k = 过去 k 天 (pandas shift 语义): Ref($close, 5) = 5 天前的收盘价。
  即 Ref($x, k) 映射 df[x].shift(k)。负 k 为未来, 在 point-in-time df 末尾取到 NaN, 自然无前视。
- 截面算子 Rank/Scale 只支持最外层(由 caller 跨截面处理); 嵌套截面/数组/RowNum/未知算子 → 抛 UnsupportedExpr。
- 每只股票求值取 Series 的 iloc[-1] 作为该股在 eval_date 的因子值。
"""
import re
import warnings
import numpy as np
import pandas as pd

# 滚动窗口函数对数据不足的股票产生全 NaN 切片是正常情况, 抑制 numpy 警告刷屏
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0')
warnings.filterwarnings('ignore', message='All-NaN')


class UnsupportedExpr(Exception):
    pass


_TOKEN_RE = re.compile(r"""
    \s+
  | (?P<num>\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)
  | (?P<field>\$[A-Za-z_][A-Za-z0-9_]*)
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>\*\*|<=|>=|==|!=|&&|\|\||[-+*/<>?:(),\[\]])
""", re.VERBOSE)


def _tokenize(expr):
    toks = []
    pos = 0
    for m in _TOKEN_RE.finditer(expr):
        if m.start() != pos:
            raise UnsupportedExpr(f"无法解析: ...{expr[pos:m.start()+1]}...")
        pos = m.end()
        if m.lastgroup == "num":
            toks.append(("num", float(m.group("num"))))
        elif m.lastgroup == "field":
            toks.append(("field", m.group("field")[1:].lower()))
        elif m.lastgroup == "name":
            name = m.group("name")
            if name in ("and", "or"):
                toks.append(("op", name))
            else:
                toks.append(("name", name))
        elif m.lastgroup == "op":
            toks.append(("op", m.group("op")))
    if pos != len(expr):
        raise UnsupportedExpr(f"尾部无法解析: {expr[pos:]}")
    toks.append(("eof", None))
    return toks


# ── 递归下降 parser → AST(嵌套 tuple) ──
class _Parser:
    def __init__(self, toks):
        self.toks = toks
        self.i = 0

    def _peek(self):
        return self.toks[self.i]

    def _next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def _expect_op(self, op):
        t = self._next()
        if t != ("op", op):
            raise UnsupportedExpr(f"期望 '{op}', 得到 {t}")

    def parse(self):
        node = self._ternary()
        if self._peek()[0] != "eof":
            raise UnsupportedExpr(f"多余 token: {self._peek()}")
        return node

    def _ternary(self):
        cond = self._or()
        if self._peek() == ("op", "?"):
            self._next()
            a = self._ternary()
            self._expect_op(":")
            b = self._ternary()
            return ("tern", cond, a, b)
        return cond

    def _binary(self, next_fn, ops, opmap=None):
        node = next_fn()
        while self._peek()[0] == "op" and self._peek()[1] in ops:
            op = self._next()[1]
            if opmap:
                op = opmap[op]
            node = ("bin", op, node, next_fn())
        return node

    def _or(self):
        return self._binary(self._and, ("||", "or"), {"||": "or", "or": "or"})

    def _and(self):
        return self._binary(self._cmp, ("&&", "and"), {"&&": "and", "and": "and"})

    def _cmp(self):
        return self._binary(self._add, ("<", ">", "<=", ">=", "==", "!="))

    def _add(self):
        return self._binary(self._mul, ("+", "-"))

    def _mul(self):
        return self._binary(self._pow, ("*", "/"))

    def _pow(self):
        node = self._unary()
        if self._peek() == ("op", "**"):
            self._next()
            return ("bin", "**", node, self._pow())
        return node

    def _unary(self):
        if self._peek() == ("op", "-"):
            self._next()
            return ("neg", self._unary())
        if self._peek() == ("op", "+"):
            self._next()
            return self._unary()
        return self._primary()

    def _primary(self):
        t = self._peek()
        if t == ("op", "("):
            self._next()
            node = self._ternary()
            self._expect_op(")")
            return node
        if t == ("op", "["):
            raise UnsupportedExpr("不支持数组语法 [...]")
        if t[0] == "num":
            self._next()
            return ("num", t[1])
        if t[0] == "field":
            self._next()
            return ("field", t[1])
        if t[0] == "name":
            name = self._next()[1]
            if self._peek() == ("op", "("):
                self._next()
                args = []
                if self._peek() != ("op", ")"):
                    args.append(self._ternary())
                    while self._peek() == ("op", ","):
                        self._next()
                        args.append(self._ternary())
                self._expect_op(")")
                if name == "If" and len(args) == 3:
                    return ("tern", args[0], args[1], args[2])
                return ("call", name, args)
            return ("name", name)  # 裸标识符 (Ret 等)
        raise UnsupportedExpr(f"无法解析 token: {t}")


def _extract_outer_rank(ast):
    """从 AST 中提取最外层 Rank/Scale 算子。
    支持 Rank(expr), -Rank(expr), expr * Rank(expr) 等简单嵌套。
    返回 (inner_ast, outer_op) 或 (ast, None)。
    """
    if ast[0] == "call" and ast[1] in ("Rank", "Scale") and len(ast[2]) == 1:
        return ast[2][0], ("rank" if ast[1] == "Rank" else "scale")
    if ast[0] == "neg":
        inner, op = _extract_outer_rank(ast[1])
        if op:
            return ("neg", inner), op
    if ast[0] == "bin" and ast[2] in ("*", "/"):
        left, lop = _extract_outer_rank(ast[2])
        if lop:
            return ("bin", ast[2], ast[3], left), lop
        right, rop = _extract_outer_rank(ast[3])
        if rop:
            return ("bin", ast[2], left, right), rop
    return ast, None


def parse_factor(expr):
    """解析表达式; 若最外层是 Rank/Scale 则剥离并返回 outer_op, 内层交给单股求值。
    内层若含嵌套截面算子(Rank/Scale)或不支持算子 → 抛 UnsupportedExpr(回退综合分)。
    返回 (inner_ast, outer_op)  outer_op ∈ {None,'rank','scale'}。"""
    ast = _Parser(_tokenize(expr)).parse()
    inner, outer_op = _extract_outer_rank(ast)
    _validate(inner)
    return inner, outer_op


_SUPPORTED_FUNCS = {
    "Ref", "Delay", "Mean", "Std", "Sum", "Delta", "Skewness", "Kurtosis", "Quantile",
    "Corr", "Cov", "Ts_Rank", "TsRank", "Ts_ArgMax", "IdxMax", "Ts_ArgMin", "IdxMin",
    "Slope", "Rsquare", "RegBeta", "Abs", "Log", "Sign", "SignedPower",
    "Max", "Min", "Greater", "Less", "FundVal", "DayOfWeek",
    "Rank", "Scale",
}


def _validate(node):
    """内层 AST 只允许单股可算的时序/数学算子; 截面 Rank/Scale 或未知算子 → 抛异常。"""
    tag = node[0]
    if tag in ("num", "field"):
        return
    if tag == "name":
        if node[1] != "Ret":
            raise UnsupportedExpr(f"不支持标识符: {node[1]}")
        return
    if tag == "neg":
        _validate(node[1])
        return
    if tag == "tern":
        _validate(node[1])
        _validate(node[2])
        _validate(node[3])
        return
    if tag == "bin":
        _validate(node[2])
        _validate(node[3])
        return
    if tag == "call":
        if node[1] not in _SUPPORTED_FUNCS:
            raise UnsupportedExpr(f"嵌套截面/不支持算子: {node[1]}")
        for a in node[2]:
            _validate(a)
        return
    raise UnsupportedExpr(f"未知节点: {tag}")


# ── 求值 ──
_FIELDS = {"open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg"}


def _to_float_arr(s):
    """Series 或 DataFrame → float64 numpy array (拷贝)。"""
    if isinstance(s, pd.DataFrame):
        return s.to_numpy(dtype=float, copy=True)
    return np.asarray(s, dtype=float)


def _wrap_series(a, index):
    """numpy array → pandas Series with index。"""
    return pd.Series(a, index=index)


# ── numpy 向量化滚动算子 (sliding_window_view, 10-50× faster than pandas .rolling()) ──

def _make_simple_rolling_fn(np_fn, **kwargs):
    def _rolling(a, n):
        out = np.full(len(a), np.nan)
        if len(a) < n:
            return out
        if a.ndim != 1:
            a = a.ravel()
        from numpy.lib.stride_tricks import sliding_window_view
        win = sliding_window_view(a, n)
        with np.errstate(invalid='ignore'):
            out[n - 1:] = np_fn(win, axis=1, **kwargs)
        return out
    return _rolling

_np_rolling_mean = _make_simple_rolling_fn(np.nanmean)
_np_rolling_std = _make_simple_rolling_fn(np.nanstd, ddof=1)
_np_rolling_sum = _make_simple_rolling_fn(np.nansum)
_np_rolling_max = _make_simple_rolling_fn(np.nanmax)
_np_rolling_min = _make_simple_rolling_fn(np.nanmin)


def _np_rolling_skew(a, n):
    """滚动偏度: Fisher-Pearson standardized moment (与 pandas 一致)。"""
    out = np.full(len(a), np.nan)
    if len(a) < n or n < 3:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(a, n)
    mu = np.nanmean(win, axis=1, keepdims=True)
    d = win - mu
    m2 = np.nansum(d ** 2, axis=1) / n   # biased
    m3 = np.nansum(d ** 3, axis=1) / n   # biased
    valid = m2 > 1e-15
    adj = np.sqrt(n * (n - 1)) / (n - 2)
    with np.errstate(invalid='ignore'):
        res = np.where(valid, adj * m3 / (m2 ** 1.5), np.nan)
    out[n - 1:] = res
    return out


def _np_rolling_kurt(a, n):
    """滚动峰度: 超额峰度 (与 pandas kurtosis() 一致)。"""
    out = np.full(len(a), np.nan)
    if len(a) < n or n < 4:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(a, n)
    mu = np.nanmean(win, axis=1, keepdims=True)
    d = win - mu
    m2 = np.nansum(d ** 2, axis=1) / n   # biased
    m4 = np.nansum(d ** 4, axis=1) / n   # biased
    valid = m2 > 1e-15
    r = m4 / (m2 ** 2)
    adj = ((n + 1) * (r - 3) + 6) * (n - 1) / ((n - 2) * (n - 3))
    res = np.where(valid, adj, np.nan)
    out[n - 1:] = res
    return out


def _np_rolling_corr(a, b, n):
    """滚动 Pearson 相关系数。"""
    out = np.full(len(a), np.nan)
    if len(a) < n:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    wa = sliding_window_view(a, n)
    wb = sliding_window_view(b, n)
    with np.errstate(invalid='ignore', divide='ignore'):
        ma = np.nanmean(wa, axis=1, keepdims=True)
        mb = np.nanmean(wb, axis=1, keepdims=True)
        da = wa - ma
        db = wb - mb
        num = np.nansum(da * db, axis=1)
        den = np.sqrt(np.nansum(da ** 2, axis=1) * np.nansum(db ** 2, axis=1))
        valid = den > 1e-15
        res = np.where(valid, num / den, 0.0)
    out[n - 1:] = res
    return out


def _np_rolling_cov(a, b, n):
    """滚动协方差。"""
    out = np.full(len(a), np.nan)
    if len(a) < n:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    wa = sliding_window_view(a, n)
    wb = sliding_window_view(b, n)
    ma = np.nanmean(wa, axis=1, keepdims=True)
    mb = np.nanmean(wb, axis=1, keepdims=True)
    res_arr = np.nansum((wa - ma) * (wb - mb), axis=1) / (n - 1)
    out[n - 1:] = res_arr
    return out


def _np_rolling_quantile(a, n, q):
    """滚动分位数。"""
    out = np.full(len(a), np.nan)
    if len(a) < n:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(a, n)
    out[n - 1:] = np.nanquantile(win, q, axis=1)
    return out


def _np_diff(a, k):
    """k 阶差分 (向量化)。"""
    out = np.full(len(a), np.nan)
    if len(a) <= k:
        return out
    out[k:] = a[k:] - a[:-k]
    return out


def _np_shift(a, k):
    """正 k → 过去 (右移, shift(k)); 负 k → 未来 (左移, shift(-k), 自然 NaN)。"""
    out = np.full(len(a), np.nan)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def _rolling_ols(y, x, n, out):
    """滚动 OLS: out='slope'|'rsq'|'beta' (beta=cov(x,y)/var(x))。y,x 为 Series。"""
    yv, xv = np.asarray(y, float), np.asarray(x, float)
    res = np.full(len(yv), np.nan)
    for i in range(n - 1, len(yv)):
        ys, xs = yv[i - n + 1:i + 1], xv[i - n + 1:i + 1]
        if np.isnan(ys).any() or np.isnan(xs).any():
            continue
        xm, ym = xs.mean(), ys.mean()
        vx = ((xs - xm) ** 2).sum()
        if vx <= 1e-12:
            continue
        beta = ((xs - xm) * (ys - ym)).sum() / vx
        if out in ("beta", "slope"):
            res[i] = beta
        else:  # rsq
            pred = ym + beta * (xs - xm)
            ss_tot = ((ys - ym) ** 2).sum()
            res[i] = 1 - ((ys - pred) ** 2).sum() / ss_tot if ss_tot > 1e-12 else np.nan
    return pd.Series(res, index=y.index)


def _to_bool(v):
    if isinstance(v, pd.Series):
        return v.astype(bool)
    return bool(v)


def _const(node):
    if node[0] == "num":
        return node[1]
    if node[0] == "neg":
        return -_const(node[1])
    if node[0] == "bin":
        op = node[1]
        left = _const(node[2])
        right = _const(node[3])
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "**":
            return left ** right
    raise UnsupportedExpr("参数必须是常数")


def _sw_apply(v, n, fn):
    """对 Series 或 DataFrame(逐列) 应用 numpy 滑窗函数 fn(1d_array, n)->1d_array。"""
    if isinstance(v, pd.DataFrame):
        out = np.column_stack([fn(v[c].to_numpy(dtype=float), n) for c in v.columns])
        return pd.DataFrame(out, index=v.index, columns=v.columns)
    s = pd.Series(v)
    return pd.Series(fn(s.to_numpy(dtype=float), n), index=s.index)


def _win(node):
    value = float(_const(node))
    rounded = round(value)
    if rounded == 0 or abs(value - rounded) > 1e-6:
        raise UnsupportedExpr(f"窗口或位移必须是非零整数: {value}")
    return int(rounded)


def _rolling_window(node):
    try:
        value = float(_const(node))
    except Exception:
        return None
    rounded = round(value)
    if rounded < 2 or abs(value - rounded) > 1e-6:
        return None
    return int(rounded)


def _sw_ts_rank(a, n):
    """滑窗百分位排名 (每窗末元素在窗内的 pct rank), numpy 向量化。a: 1D array。"""
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(len(a), np.nan)
    if len(a) < n:
        return out
    win = sliding_window_view(a, n)  # (len-n+1, n)
    last = win[:, -1:]
    valid = ~np.isnan(win).any(axis=1)
    # pandas rank(pct=True).iloc[-1]: 末元素含并列的平均序号 / n
    less = (win < last).sum(axis=1)
    equal = (win == last).sum(axis=1)
    avg_rank = less + (equal + 1) / 2.0  # 并列取平均序号
    res = np.where(valid, avg_rank / n, np.nan)
    out[n - 1:] = res
    return out


def _sw_arg(a, n, want_max):
    """滑窗 argmax/argmin (窗内位置索引 0..n-1), numpy 向量化。"""
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(len(a), np.nan)
    if len(a) < n:
        return out
    win = sliding_window_view(a, n)
    valid = ~np.isnan(win).any(axis=1)
    idx = (np.argmax(win, axis=1) if want_max else np.argmin(win, axis=1)).astype(float)
    out[n - 1:] = np.where(valid, idx, np.nan)
    return out


class _Ev:
    def __init__(self, df, code=None):
        self.df = df
        self.idx = df.index
        self.code = code
        self._col_cache = {}

    def _series(self, col):
        s = self._col_cache.get(col)
        if s is None:
            s = pd.to_numeric(self.df[col], errors="coerce").astype(float)
            self._col_cache[col] = s
        return s

    def ev(self, node):
        tag = node[0]
        if tag == "num":
            return node[1]
        if tag == "name":
            if node[1] == "Ret":
                c = self._series("close")
                return c / c.shift(1) - 1
            raise UnsupportedExpr(f"未知标识符: {node[1]}")
        if tag == "field":
            f = node[1]
            if f == "vwap":
                vol = self._series("volume").replace(0, np.nan)
                return self._series("amount") / vol
            if f in _FIELDS:
                return self._series(f)
            raise UnsupportedExpr(f"未知字段: ${f}")
        if tag == "neg":
            return -self.ev(node[1])
        if tag == "tern":
            cond = _to_bool(self.ev(node[1]))
            a, b = self.ev(node[2]), self.ev(node[3])
            return pd.Series(np.where(np.asarray(cond), a, b), index=self.idx)
        if tag == "bin":
            return self._binop(node[1], node[2], node[3])
        if tag == "call":
            return self._call(node[1], node[2])
        raise UnsupportedExpr(f"未知节点: {tag}")

    def _binop(self, op, ln, rn):
        l, r = self.ev(ln), self.ev(rn)
        if op == "+":
            return l + r
        if op == "-":
            return l - r
        if op == "*":
            return l * r
        if op == "/":
            return l / r
        if op == "**":
            return l ** r
        if op in ("<", ">", "<=", ">=", "==", "!="):
            res = {"<": l < r, ">": l > r, "<=": l <= r, ">=": l >= r,
                   "==": l == r, "!=": l != r}[op]
            return res.astype(float) if isinstance(res, pd.Series) else float(res)
        if op == "and":
            res = _to_bool(l) & _to_bool(r)
            return res.astype(float) if isinstance(res, pd.Series) else float(res)
        if op == "or":
            res = _to_bool(l) | _to_bool(r)
            return res.astype(float) if isinstance(res, pd.Series) else float(res)
        raise UnsupportedExpr(f"未知算子: {op}")

    def _call(self, name, args):
        if name in ("Rank", "Scale"):
            raise UnsupportedExpr("嵌套截面算子 Rank/Scale 不支持")
        if name == "DayOfWeek":
            values = pd.DatetimeIndex(pd.to_datetime(self.idx)).dayofweek + 1
            return pd.Series(values, index=self.idx, dtype=float)
        if name == "RowNum":
            raise UnsupportedExpr(f"不支持 {name}")

        if name in ("Ref", "Delay"):
            x = self.ev(args[0]); k = _win(args[1])
            if isinstance(x, pd.Series):
                # WorldQuant 语义: Ref(expr, -N) = N 天前 (滞后), 与 qlib Ref(expr, N) 一致。
                # 本引擎按 pandas shift (正=过去) 求值, 故对负号取 abs 统一到"过去 |k| 日"。
                return _wrap_series(_np_shift(_to_float_arr(x), abs(k)), self.idx)
            return x
        if name == "Mean":
            x = self.ev(args[0])
            if isinstance(x, pd.Series):
                return _wrap_series(_np_rolling_mean(_to_float_arr(x), _win(args[1])), self.idx)
            return x
        if name == "Std":
            x = self.ev(args[0])
            if isinstance(x, pd.Series):
                return _wrap_series(_np_rolling_std(_to_float_arr(x), _win(args[1])), self.idx)
            return x
        if name == "Sum":
            x = self.ev(args[0])
            if isinstance(x, pd.Series):
                return _wrap_series(_np_rolling_sum(_to_float_arr(x), _win(args[1])), self.idx)
            return x
        if name == "Delta":
            x = self.ev(args[0])
            if isinstance(x, pd.Series):
                return _wrap_series(_np_diff(_to_float_arr(x), _win(args[1])), self.idx)
            return x
        if name == "Skewness":
            x = self.ev(args[0])
            if isinstance(x, pd.Series):
                return _wrap_series(_np_rolling_skew(_to_float_arr(x), _win(args[1])), self.idx)
            return x
        if name == "Kurtosis":
            x = self.ev(args[0])
            if isinstance(x, pd.Series):
                return _wrap_series(_np_rolling_kurt(_to_float_arr(x), _win(args[1])), self.idx)
            return x
        if name == "Quantile":
            x = self.ev(args[0])
            if isinstance(x, pd.Series):
                return _wrap_series(_np_rolling_quantile(_to_float_arr(x), _win(args[1]), float(_const(args[2]))), self.idx)
            return x
        if name == "Corr":
            x, y = self.ev(args[0]), self.ev(args[1])
            if isinstance(x, pd.Series):
                return _wrap_series(_np_rolling_corr(_to_float_arr(x), _to_float_arr(y), _win(args[2])), self.idx)
            return x
        if name == "Cov":
            x, y = self.ev(args[0]), self.ev(args[1])
            if isinstance(x, pd.Series):
                return _wrap_series(_np_rolling_cov(_to_float_arr(x), _to_float_arr(y), _win(args[2])), self.idx)
            return x
        if name in ("Ts_Rank", "TsRank"):
            return _sw_apply(self.ev(args[0]), _win(args[1]), _sw_ts_rank)
        if name in ("Ts_ArgMax", "IdxMax"):
            return _sw_apply(self.ev(args[0]), _win(args[1]), lambda a, n: _sw_arg(a, n, True))
        if name in ("Ts_ArgMin", "IdxMin"):
            return _sw_apply(self.ev(args[0]), _win(args[1]), lambda a, n: _sw_arg(a, n, False))
        if name in ("Slope", "Rsquare", "RegBeta"):
            x = self.ev(args[0])
            if name == "RegBeta":
                return _rolling_ols(x, self.ev(args[1]), _win(args[2]), "beta")
            n = _win(args[1])
            t = pd.Series(np.arange(len(x), dtype=float), index=self.idx)
            return _rolling_ols(x, t, n, "slope" if name == "Slope" else "rsq")
        if name == "Abs":
            v = self.ev(args[0])
            return v.abs() if isinstance(v, pd.Series) else abs(v)
        if name == "Log":
            v = self.ev(args[0])
            return np.log(v.where(v > 0)) if isinstance(v, pd.Series) else np.log(v)
        if name == "Sign":
            v = self.ev(args[0])
            return np.sign(v)
        if name == "SignedPower":
            v = self.ev(args[0]); e = float(_const(args[1]))
            return np.sign(v) * (np.abs(v) ** e)
        if name in ("Max", "Min", "Greater", "Less"):
            window = _rolling_window(args[1]) if len(args) == 2 and name in ("Max", "Min") else None
            if window is not None:
                x = self.ev(args[0])
                if isinstance(x, pd.Series):
                    fn = _np_rolling_max if name == "Max" else _np_rolling_min
                    return _wrap_series(fn(_to_float_arr(x), window), self.idx)
                return x
            l, r = self.ev(args[0]), self.ev(args[1])
            big = name in ("Max", "Greater")
            return pd.Series(np.maximum(l, r) if big else np.minimum(l, r), index=self.idx)
        if name == "FundVal":
            return self._eval_fundval(args)
        raise UnsupportedExpr(f"未知函数: {name}")

    def _eval_fundval(self, args):
        field_id = int(_const(args[1]))
        from core.data.fundamental import read_fundamental
        from core.shared.fundamental_contract import fundamental_dates
        dates = fundamental_dates(self.idx)
        values = read_fundamental(field_id, self.code, dates)
        return pd.Series(values.to_numpy(), index=self.idx)



def eval_on_df(inner_ast, df, code=None):
    """在单只股票 df 上求 inner_ast 的最后一个有效值 (float); 无效返回 nan。"""
    if df is None or len(df) < 2:
        return float("nan")
    v = _Ev(df, code=code).ev(inner_ast)
    if not isinstance(v, pd.Series):
        return float(v)
    v = v.replace([np.inf, -np.inf], np.nan).dropna()
    return float(v.iloc[-1]) if len(v) else float("nan")


# ══════════════════════════════════════════════
# 面板批量求值 — 同交易日历的多股拼成宽表, 一次向量化 (逐列 C 运算)
# 语义与 _Ev 逐股版严格一致; Corr/Cov/Slope/RegBeta 用列安全滚动和公式
# ══════════════════════════════════════════════

_PANEL_FIELDS = ("open", "high", "low", "close", "volume", "amount", "turnover", "pct_chg")
# 扩展字段：从 DataFrame 实际列动态获取，不再硬编码。因子引用 MA5/MACD/RSI14 等
# 技术指标字段时也能走快速面板路径，避免逐股回退。
_NON_NUMERIC_COLS = {"date", "trade_date", "name", "code", "symbol", "index"}


class _PanelEv:
    def __init__(self, fields, index, codes):
        self.fields = fields      # {field: 宽表 DataFrame(index=日期, columns=codes)}
        self.index = index
        self.codes = codes

    def _series(self, col):
        if col == "vwap":
            vol = self.fields["volume"].replace(0, np.nan)
            return self.fields["amount"] / vol
        return self.fields[col]

    def _arr(self, v):
        return v.values if isinstance(v, pd.DataFrame) else v

    def _wrap(self, arr):
        return pd.DataFrame(arr, index=self.index, columns=self.codes)

    def _roll_cov(self, x, y, n):
        sx = x.rolling(n).sum(); sy = y.rolling(n).sum()
        sxy = (x * y).rolling(n).sum()
        return (sxy - sx * sy / n) / (n - 1)

    def _roll_corr(self, x, y, n):
        return self._roll_cov(x, y, n) / (x.rolling(n).std() * y.rolling(n).std())

    def _time_df(self):
        t = np.arange(len(self.index), dtype=float)
        return self._wrap(np.repeat(t[:, None], len(self.codes), axis=1))

    def ev(self, node):
        tag = node[0]
        if tag == "num":
            return node[1]
        if tag == "name":
            if node[1] == "Ret":
                c = self._series("close")
                return c / c.shift(1) - 1
            raise UnsupportedExpr(f"未知标识符: {node[1]}")
        if tag == "field":
            f = node[1]
            if f == "vwap" or f in _FIELDS:
                return self._series(f)
            raise UnsupportedExpr(f"未知字段: ${f}")
        if tag == "neg":
            return -self.ev(node[1])
        if tag == "tern":
            cond = self.ev(node[1]); a = self.ev(node[2]); b = self.ev(node[3])
            cond_a = self._arr(cond)
            cond_b = cond_a.astype(bool) if isinstance(cond_a, np.ndarray) else bool(cond_a)
            return self._wrap(np.where(cond_b, self._arr(a), self._arr(b)))
        if tag == "bin":
            return self._binop(node[1], node[2], node[3])
        if tag == "call":
            return self._call(node[1], node[2])
        raise UnsupportedExpr(f"未知节点: {tag}")

    def _binop(self, op, ln, rn):
        l, r = self.ev(ln), self.ev(rn)
        if op == "+":
            return l + r
        if op == "-":
            return l - r
        if op == "*":
            return l * r
        if op == "/":
            return l / r
        if op == "**":
            return l ** r
        if op in ("<", ">", "<=", ">=", "==", "!="):
            res = {"<": l < r, ">": l > r, "<=": l <= r, ">=": l >= r,
                   "==": l == r, "!=": l != r}[op]
            return res.astype(float) if isinstance(res, pd.DataFrame) else float(res)
        if op == "and":
            lb = l.astype(bool) if isinstance(l, pd.DataFrame) else bool(l)
            rb = r.astype(bool) if isinstance(r, pd.DataFrame) else bool(r)
            res = lb & rb
            return res.astype(float) if isinstance(res, pd.DataFrame) else float(res)
        if op == "or":
            lb = l.astype(bool) if isinstance(l, pd.DataFrame) else bool(l)
            rb = r.astype(bool) if isinstance(r, pd.DataFrame) else bool(r)
            res = lb | rb
            return res.astype(float) if isinstance(res, pd.DataFrame) else float(res)
        raise UnsupportedExpr(f"未知算子: {op}")

    def _call(self, name, args):
        if name == "Rank":
            x = self.ev(args[0])
            if isinstance(x, pd.DataFrame):
                return x.rank(axis=1, pct=True)
            return x
        if name == "Scale":
            x = self.ev(args[0])
            if isinstance(x, pd.DataFrame):
                mu = x.mean(axis=1); sd = x.std(axis=1)
                sd = sd.where(sd > 1e-12, 1.0)
                return x.sub(mu, axis=0).div(sd, axis=0)
            return x
        if name == "DayOfWeek":
            values = pd.DatetimeIndex(pd.to_datetime(self.index)).dayofweek.to_numpy(dtype=float) + 1
            return self._wrap(np.repeat(values[:, None], len(self.codes), axis=1))
        if name == "RowNum":
            raise UnsupportedExpr(f"不支持 {name}")

        if name in ("Ref", "Delay"):
            x = self.ev(args[0]); k = _win(args[1])
            if isinstance(x, pd.DataFrame):
                # WorldQuant 语义: Ref(expr, -N) = N 天前, 对负号取 abs 统一到"过去 |k| 日"。
                return self._wrap(np.column_stack([_np_shift(_to_float_arr(x[c]), abs(k)) for c in x.columns]))
            return x
        if name == "Mean":
            x = self.ev(args[0])
            if isinstance(x, pd.DataFrame):
                return self._wrap(np.column_stack([_np_rolling_mean(_to_float_arr(x[c]), _win(args[1])) for c in x.columns]))
            return x
        if name == "Std":
            x = self.ev(args[0])
            if isinstance(x, pd.DataFrame):
                return self._wrap(np.column_stack([_np_rolling_std(_to_float_arr(x[c]), _win(args[1])) for c in x.columns]))
            return x
        if name == "Sum":
            x = self.ev(args[0])
            if isinstance(x, pd.DataFrame):
                return self._wrap(np.column_stack([_np_rolling_sum(_to_float_arr(x[c]), _win(args[1])) for c in x.columns]))
            return x
        if name == "Delta":
            x = self.ev(args[0])
            if isinstance(x, pd.DataFrame):
                return self._wrap(np.column_stack([_np_diff(_to_float_arr(x[c]), _win(args[1])) for c in x.columns]))
            return x
        if name == "Skewness":
            x = self.ev(args[0])
            if isinstance(x, pd.DataFrame):
                return self._wrap(np.column_stack([_np_rolling_skew(_to_float_arr(x[c]), _win(args[1])) for c in x.columns]))
            return x
        if name == "Kurtosis":
            x = self.ev(args[0])
            if isinstance(x, pd.DataFrame):
                return self._wrap(np.column_stack([_np_rolling_kurt(_to_float_arr(x[c]), _win(args[1])) for c in x.columns]))
            return x
        if name == "Quantile":
            x = self.ev(args[0])
            if isinstance(x, pd.DataFrame):
                return self._wrap(np.column_stack([_np_rolling_quantile(_to_float_arr(x[c]), _win(args[1]), float(_const(args[2]))) for c in x.columns]))
            return x
        if name == "Corr":
            x, y = self.ev(args[0]), self.ev(args[1])
            if isinstance(x, pd.DataFrame):
                return self._wrap(np.column_stack([_np_rolling_corr(_to_float_arr(x[c]), _to_float_arr(y[c]), _win(args[2])) for c in x.columns]))
            return x
        if name == "Cov":
            x, y = self.ev(args[0]), self.ev(args[1])
            if isinstance(x, pd.DataFrame):
                return self._wrap(np.column_stack([_np_rolling_cov(_to_float_arr(x[c]), _to_float_arr(y[c]), _win(args[2])) for c in x.columns]))
            return x
        if name in ("Ts_Rank", "TsRank"):
            return _sw_apply(self.ev(args[0]), _win(args[1]), _sw_ts_rank)
        if name in ("Ts_ArgMax", "IdxMax"):
            return _sw_apply(self.ev(args[0]), _win(args[1]), lambda a, n: _sw_arg(a, n, True))
        if name in ("Ts_ArgMin", "IdxMin"):
            return _sw_apply(self.ev(args[0]), _win(args[1]), lambda a, n: _sw_arg(a, n, False))
        if name in ("Slope", "Rsquare", "RegBeta"):
            x = self.ev(args[0])
            if name == "RegBeta":
                b = self.ev(args[1]); n = _win(args[2])
                return self._roll_cov(b, x, n) / (b.rolling(n).var())
            n = _win(args[1])
            t = self._time_df()
            if name == "Slope":
                return self._roll_cov(t, x, n) / (t.rolling(n).var())
            r = self._roll_corr(t, x, n)
            return r * r
        if name == "Abs":
            v = self.ev(args[0])
            return v.abs() if isinstance(v, pd.DataFrame) else abs(v)
        if name == "Log":
            v = self.ev(args[0])
            return np.log(v.where(v > 0)) if isinstance(v, pd.DataFrame) else np.log(v)
        if name == "Sign":
            return np.sign(self.ev(args[0]))
        if name == "SignedPower":
            v = self.ev(args[0]); e = float(_const(args[1]))
            return np.sign(v) * (np.abs(v) ** e)
        if name in ("Max", "Min", "Greater", "Less"):
            window = _rolling_window(args[1]) if len(args) == 2 and name in ("Max", "Min") else None
            if window is not None:
                x = self.ev(args[0])
                if isinstance(x, pd.DataFrame):
                    fn = _np_rolling_max if name == "Max" else _np_rolling_min
                    return self._wrap(np.column_stack([fn(_to_float_arr(x[c]), window) for c in x.columns]))
                return x
            l, r = self.ev(args[0]), self.ev(args[1])
            big = name in ("Max", "Greater")
            return self._wrap(np.maximum(self._arr(l), self._arr(r)) if big
                              else np.minimum(self._arr(l), self._arr(r)))
        if name == "FundVal":
            return self._eval_fundval(args)
        raise UnsupportedExpr(f"未知函数: {name}")

    def _eval_fundval(self, args):
        field_id = int(_const(args[1]))
        from core.data.fundamental import read_fundamental
        from core.shared.fundamental_contract import fundamental_dates
        dates = fundamental_dates(self.index)
        data = np.column_stack([read_fundamental(field_id, code, dates).to_numpy() for code in self.codes])
        return self._wrap(data)



def _build_panel(group):
    """从 {code: df} 构建面板字典: 字段→DataFrame(code×date), 只做一次供批量求值复用。"""
    import numpy as np
    codes = list(group)
    first = group[codes[0]]
    index = first.index
    available = {c for c in first.columns if c not in _NON_NUMERIC_COLS
                 and first[c].dtype.kind in ('f', 'i')}
    available.update(_PANEL_FIELDS)
    fields = {}
    for f in available:
        if f in first.columns:
            try:
                arrs = [grp[f].to_numpy(dtype=float, na_value=np.nan) for _, grp in group.items()]
                fields[f] = pd.DataFrame(np.column_stack(arrs), index=index, columns=codes)
            except Exception:
                try:
                    fields[f] = pd.DataFrame(
                        {c: pd.to_numeric(group[c][f], errors="coerce").astype(float) for c in codes},
                        index=index, columns=codes)
                except Exception:
                    pass
    return codes, index, fields


def eval_on_panel(inner_ast, group):
    """group: {code: 单股 df} 且各 df index 完全相同 (调用方按 index 签名分组保证)。
    拼成字段宽表一次向量化, 返回 Series(index=code) 每股最后有效因子值。语义同逐股 eval_on_df。"""
    import numpy as np
    codes, index, fields = _build_panel(group)
    v = _PanelEv(fields, index, codes).ev(inner_ast)
    if not isinstance(v, pd.DataFrame):
        return pd.Series(float(v), index=codes)
    v = v.replace([np.inf, -np.inf], np.nan)
    return v.ffill().iloc[-1]


def evaluate(expr, df):
    """便捷: 解析 + 单股求值 (忽略最外层截面算子, 供单元测试时序因子)。"""
    inner, _ = parse_factor(expr)
    return eval_on_df(inner, df)


def batch_eval_all(ast_map, snapshot):
    """批量求值: 对 snapshot 一次性求所有 AST, 返回 {factor_sig: {code: score}}。

    ast_map: {factor_sig: (inner_ast, outer_op)}
    snapshot: {code: (df, name, close, volume)} 或 {code: df}

    优化: 分组只做一次, 每组内所有 AST 共享面板, 避免逐因子重复分组+逐股回退。
    """
    import math
    import numpy as np
    from collections import defaultdict

    # 按 index 签名分组 (只做一次)
    groups = defaultdict(dict)
    for code, tup in snapshot.items():
        df = tup[0] if isinstance(tup, (list, tuple)) else tup
        if df is None or len(df) < 2:
            continue
        sig = (len(df.index), df.index[0], df.index[-1])
        groups[sig][code] = df

    # 结果: {factor_sig: {code: score}}
    result = {sig: {} for sig in ast_map}

    for grp in groups.values():
        codes, idx, fields = _build_panel(grp)
        panel_ev = _PanelEv(fields, idx, codes)
        for factor_sig, (inner_ast, outer_op) in ast_map.items():
            try:
                v = panel_ev.ev(inner_ast)
                if isinstance(v, pd.DataFrame):
                    v = v.replace([np.inf, -np.inf], np.nan).ffill().iloc[-1]
                else:
                    v = pd.Series(float(v), index=codes)
                for code in codes:
                    fv = float(v.get(code, np.nan))
                    if math.isfinite(fv):
                        result[factor_sig][code] = fv
            except Exception:
                for code, df in grp.items():
                    try:
                        fv = eval_on_df(inner_ast, df)
                        if isinstance(fv, float) and math.isfinite(fv):
                            result[factor_sig][code] = fv
                    except Exception:
                        pass

    # 最外层截面算子 (Rank/Scale)
    for factor_sig, (_, outer_op) in ast_map.items():
        vals = result[factor_sig]
        if outer_op and len(vals) >= 2:
            codes = list(vals)
            arr = np.array([vals[c] for c in codes], float)
            if outer_op == "rank":
                ranks = arr.argsort().argsort() / max(len(arr) - 1, 1)
                result[factor_sig] = {c: float(r) for c, r in zip(codes, ranks)}
            else:  # scale
                mu, sd = arr.mean(), arr.std()
                result[factor_sig] = {
                    c: float((x - mu) / sd) if sd > 1e-12 else 0.0
                    for c, x in zip(codes, arr)
                }

    return result
