"""
自定义Qlib DataHandler — 注入白盒因子表达式

用法:
  训练时指定 factor_set="WhiteBoxHandler"
  module_path 指向此模块

WhiteBoxHandler 继承 Alpha158, 重写 get_feature_config() 将白盒因子的 Qlib
表达式追加到特征列表中, 使其参与 Qlib 的特征计算和数据加载。
"""
import hashlib
import json

# 因子校验结果缓存: key=factor_keys_hash → set(good_factor_keys)
# 首次 binary search 后缓存, 后续轮直接复用, 避免每轮 40+ 分钟的重复校验
# LRU 限制 256 条, 防长期运行内存泄漏
import collections
_good_keys_cache = collections.OrderedDict()
_bad_keys_cache = collections.OrderedDict()
_CACHE_MAX_SIZE = 256


def _cache_put(cache, key, value):
    if len(cache) >= _CACHE_MAX_SIZE:
        cache.popitem(last=False)  # FIFO 淘汰
    cache[key] = value

import pandas as pd
from qlib.data.ops import Rolling, Operators

_ts_rank_registered = False


def _register_ts_rank():
    """延迟注册 Ts_Rank 到 Qlib 运算符表 (必须在 qlib.init() 之后调用)"""
    global _ts_rank_registered
    if _ts_rank_registered:
        return

    class Ts_Rank(Rolling):
        """滑窗百分位排名 — 每个时点返回当日值在过去 N 日中的百分位 (0~1)。

        Ts_Rank($close, 5) 表示当日收盘价在前 5 天中的百分位排名。
        """

        def __init__(self, feature, N):
            super().__init__(feature, N, "mean")

        def _load_internal(self, instrument, start_index, end_index, *args):
            series = self.feature.load(instrument, start_index, end_index, *args)
            if self.N == 0:
                series = series.expanding(min_periods=1).apply(
                    lambda x: (x.iloc[-1] >= x.iloc[:-1]).mean(), raw=False
                )
            else:
                series = series.rolling(self.N, min_periods=1).apply(
                    lambda x: (x.iloc[-1] >= x.iloc[:-1]).mean(), raw=False
                )
            return series

    # 注册到 Qlib 运算符表，使 Qlib 原生表达式引擎可以识别 Ts_Rank
    Operators.register([Ts_Rank])
    _ts_rank_registered = True


_fundval_registered = False
_FUND_CACHE_LOADED = False


def _ensure_fund_cache():
    """懒加载基本面数据缓存 (从 Parquet 或 MySQL)"""
    global _FUND_CACHE_LOADED
    from core.data.fundamental import preload_cache
    from core.shared.fundamental_contract import FundamentalDataError
    _FUND_CACHE_LOADED = bool(preload_cache())
    if not _FUND_CACHE_LOADED:
        raise FundamentalDataError('基本面缓存加载失败')


def _register_fundval():
    """延迟注册 FundVal 到 Qlib 运算符表 (必须在 qlib.init() 之后调用)"""
    global _fundval_registered
    from core.adapters.qlib.fundval import FundVal
    if _fundval_registered:
        try:
            if Operators.FundVal is FundVal:
                return
        except AttributeError:
            pass

    Operators.register([FundVal])
    _fundval_registered = True


_dayofweek_registered = False


def _register_dayofweek():
    global _dayofweek_registered
    if _dayofweek_registered:
        return

    import numpy as np

    class DayOfWeek(Rolling):
        def __init__(self, feature=None):
            if feature is None:
                from qlib.data.base import Feature
                feature = Feature("close")
            super().__init__(feature, 1, "mean")

        def _load_internal(self, instrument, start_index, end_index, *args):
            import pandas as pd
            from qlib.data.data import Cal

            series = self.feature.load(instrument, start_index, end_index, *args)
            if series.empty:
                return series

            freq = args[0] if args else 'day'
            cal = Cal.calendar(freq=freq)
            n = len(series)
            weekdays = [cal[start_index + i].dayofweek + 1 for i in range(n)]
            return pd.Series(weekdays, index=series.index)

    Operators.register([DayOfWeek])
    _dayofweek_registered = True


_const_registered = False


def _register_const():
    global _const_registered
    if _const_registered:
        return

    import math

    class Const(Rolling):
        def __init__(self, feature, N):
            self._value = math.log(int(N))
            super().__init__(feature, 1, "mean")

        def _load_internal(self, instrument, start_index, end_index, *args):
            series = self.feature.load(instrument, start_index, end_index, *args)
            if series.empty:
                return series
            return pd.Series(self._value, index=series.index)

    Operators.register([Const])
    _const_registered = True


_sign_op_registered = False


def _register_sign_op():
    """延迟注册 SignOp 到 Qlib 运算符表 (必须在 qlib.init() 之后调用)"""
    global _sign_op_registered
    if _sign_op_registered:
        return

    import numpy as np

    class SignOp(Rolling):
        """元素级符号算子: SignOp(x) → 1 if x>0, -1 if x<0, 0 if x==0"""

        def __init__(self, feature):
            super().__init__(feature, 1, "mean")

        def _load_internal(self, instrument, start_index, end_index, *args):
            series = self.feature.load(instrument, start_index, end_index, *args)
            return np.sign(series)

    Operators.register([SignOp])
    _sign_op_registered = True


_extra_ops_registered = False


def _register_extra_ops():
    """注册自定义算子到 Qlib (暂时禁用, 排查 runtime 冲突)"""
    global _extra_ops_registered
    _extra_ops_registered = True  # 跳过注册但不重复执行


def _ols_slope(arr):
    """NumPy OLS slope: y ~ x where x=[0..n-1]"""
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = arr.mean()
    num = ((x - x_mean) * (arr - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    return num / den if den != 0 else 0.0


def _ols_rsquare(arr):
    """NumPy OLS R²"""
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = arr.mean()
    ss_xy = ((x - x_mean) * (arr - y_mean)).sum()
    ss_xx = ((x - x_mean) ** 2).sum()
    ss_yy = ((arr - y_mean) ** 2).sum()
    if ss_xx == 0 or ss_yy == 0:
        return 0.0
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    ss_res = ((arr - y_pred) ** 2).sum()
    return 1.0 - ss_res / ss_yy if ss_yy != 0 else 0.0


class WhiteBoxHandler:
    """白盒因子DataHandler — 在Alpha158基础上追加自定义因子。

    Qlib内部通过 init_instance_by_config 动态加载, 需要:
      - 类名作为 factor_set
      - module_path 作为此文件路径

    关键: get_feature_config() 在 Alpha158.__init__() 中被调用,
    然后 data_loader 据此配置创建, 最后 setup_data() 加载数据。
    所以只要在 get_feature_config() 中返回包含自定义因子的完整配置即可。
    """

    def __new__(cls, instruments=None, start_time=None, end_time=None,
                fit_start_time=None, fit_end_time=None,
                factor_keys=None, **kwargs):
        from qlib.contrib.data.handler import Alpha158
        from qlib.contrib.data.loader import Alpha158DL
        import logging, traceback
        logger = logging.getLogger(__name__)

        # 强制注册自定义算子 (Qlib init 后 Operator 表可能被清空)
        import core.adapters.qlib.custom_handler as _ch
        _ch._ts_rank_registered = False
        _ch._sign_op_registered = False
        _ch._dayofweek_registered = False
        _ch._const_registered = False
        _ch._fundval_registered = False
        _register_ts_rank()
        _register_sign_op()
        _register_dayofweek()
        _register_const()
        _register_fundval()

        custom_keys = factor_keys or []
        from core.adapters.qlib.fundval import current_feature_contract
        from core.shared.fundamental_contract import FundamentalDataError
        contract = current_feature_contract(custom_keys)
        if custom_keys:
            processors = list(kwargs.get('infer_processors') or [])
            processors.insert(0, {'class': 'FundamentalQuality',
                                 'module_path': 'core.adapters.qlib.fundval',
                                 'kwargs': {'factor_keys': list(custom_keys),
                                            'allow_extra': any(isinstance(p, dict) and p.get('class') in ('CSFeatureAugment', 'FactorCrossover') for p in kwargs.get('shared_processors', []))}})
            kwargs['infer_processors'] = processors
        cache_key = hashlib.md5(json.dumps(contract, sort_keys=True).encode()).hexdigest() if custom_keys else ""

        # 缓存命中: 跳过 binary search, 直接用上次验证通过的因子
        if cache_key and cache_key in _good_keys_cache:
            good_keys = _good_keys_cache[cache_key]
            extra_fields, extra_names = _build_custom_features(good_keys)
            logger.info(f"[WhiteBoxHandler] 缓存命中: {len(good_keys)}/{len(custom_keys)} 个好因子直接复用")
        else:
            extra_fields, extra_names = _build_custom_features(custom_keys)

        _original_gfc = Alpha158.get_feature_config

        def _patched_gfc(self):
            if extra_fields:
                return extra_fields, extra_names
            conf = {
                "kbar": {},
                "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
                "rolling": {},
            }
            return Alpha158DL.get_feature_config(conf)

        try:
            Alpha158.get_feature_config = _patched_gfc
            handler = Alpha158.__new__(Alpha158)
            handler.__init__(
                instruments=instruments,
                start_time=start_time,
                end_time=end_time,
                fit_start_time=fit_start_time,
                fit_end_time=fit_end_time,
                **kwargs,
            )
            # 首次成功: 缓存所有因子 (此时 custom_keys 都是好因子, 无 binary search)
            if cache_key and cache_key not in _good_keys_cache:
                _cache_put(_good_keys_cache, cache_key, list(custom_keys))
            if current_feature_contract(custom_keys) != contract:
                raise FundamentalDataError('构建特征期间基本面缓存或交易日历发生变化')
            handler._feature_contract = contract
            return handler
        except FundamentalDataError:
            raise
        except Exception:
            if custom_keys:
                raise
            logger.warning(f"[WhiteBoxHandler] 全量因子初始化失败: {traceback.format_exc()[-300:]}")

            # 快速模式: 标记所有因子为已验证，跳过逐个二分搜索
            if cache_key and cache_key not in _good_keys_cache:
                _cache_put(_good_keys_cache, cache_key, list(custom_keys))
                logger.info(f"[WhiteBoxHandler] fast-mode: {len(custom_keys)} 因子直接标记为已验证, 跳过二分搜索")

            # 尝试用所有因子重建handler
            good_keys = _good_keys_cache.get(cache_key, list(custom_keys)) if cache_key else list(custom_keys)
            extra_fields, extra_names = _build_custom_features(good_keys)
            _patched_gfc = lambda self: (extra_fields, extra_names)

            try:
                Alpha158.get_feature_config = _patched_gfc
                handler = Alpha158.__new__(Alpha158)
                handler.__init__(
                    instruments=instruments,
                    start_time=start_time,
                    end_time=end_time,
                    fit_start_time=fit_start_time,
                    fit_end_time=fit_end_time,
                    **kwargs,
                )
                logger.info(f"[WhiteBoxHandler] 二次尝试成功 ({len(good_keys)} 因子)")
                return handler
            except Exception:
                logger.error(f"[WhiteBoxHandler] 二次尝试也失败, 回退到 Alpha158 原生特征: {traceback.format_exc()[-200:]}")
                # 最后回退: 还原原始 Alpha158 特征配置，不使用任何自定义因子
                Alpha158.get_feature_config = _original_gfc
                handler = Alpha158.__new__(Alpha158)
                handler.__init__(
                    instruments=instruments,
                    start_time=start_time,
                    end_time=end_time,
                    fit_start_time=fit_start_time,
                    fit_end_time=fit_end_time,
                    **kwargs,
                )
                return handler
        finally:
            Alpha158.get_feature_config = _original_gfc


def _fix_regbeta(expr):
    """RegBeta(y, x, N) → Corr(y,x,N)*Std(y,N)/(Std(x,N)+1e-8).
    用括号计数解析参数, 处理y/x含嵌套函数调用的情况."""
    while "RegBeta(" in expr:
        idx = expr.find("RegBeta(")
        start = idx + len("RegBeta(")
        depth = 1
        end = start
        while end < len(expr) and depth > 0:
            if expr[end] == "(":
                depth += 1
            elif expr[end] == ")":
                depth -= 1
            end += 1
        inner = expr[start:end-1]  # RegBeta的参数部分

        # 按顶层逗号分割为3个参数
        args = []
        d = 0
        last = 0
        for i, c in enumerate(inner):
            if c == "(":
                d += 1
            elif c == ")":
                d -= 1
            elif c == "," and d == 0:
                args.append(inner[last:i].strip())
                last = i + 1
        args.append(inner[last:].strip())

        if len(args) != 3:
            break

        y, x, N = args
        replacement = f"Corr({y},{x},{N})*Std({y},{N})/(Std({x},{N})+1e-8)"
        expr = expr[:idx] + replacement + expr[end:]

    return expr


def _fix_qlib_compat(expr):
    """修复 Qlib 不兼容的表达式 — 算子/变量名/一元负号。

    Qlib 原生支持: Quantile, IdxMin, IdxMax
    需改名: Skewness→Skew, Kurtosis→Kurt
    需展开: Ret→$close/Ref($close, 1)-1, RowNum→Ref项
    需改写: ?:→If(), SignedPower→*Abs(), 浮点N→整数
    不支持: 一元负号 -expr → 0-expr, $turnover→跳过
    """
    import re

    # ═══ WorldQuant → Qlib Ref 方向统一 ═══
    # WorldQuant: Ref(expr, -N) = N天前 (滞后)
    # Qlib:       Ref(expr, -N) = N天后 (前视)
    # 递归从内向外翻转, 处理任意嵌套: Ref(Abs(Ref(x,-2)),-1) → Ref(Abs(Ref(x,2)),1)
    def _flip_ref_signs(e):
        i = 0
        result = []
        while i < len(e):
            m = re.search(r'Ref\(', e[i:])
            if not m:
                result.append(e[i:])
                break
            start = i + m.start()
            paren_start = i + m.end() - 1
            result.append(e[i:start])
            depth = 1
            j = paren_start + 1
            while j < len(e) and depth > 0:
                if e[j] == '(': depth += 1
                elif e[j] == ')': depth -= 1
                j += 1
            inner = e[paren_start+1:j-1]
            inner_processed = _flip_ref_signs(inner)
            d = 0
            last_comma = -1
            for k, c in enumerate(inner_processed):
                if c == '(': d += 1
                elif c == ')': d -= 1
                elif c == ',' and d == 0:
                    last_comma = k
            if last_comma >= 0:
                first = inner_processed[:last_comma].strip()
                second = inner_processed[last_comma+1:].strip()
                if second.startswith('-'):
                    result.append(f'Ref({first}, {second[1:]})')
                else:
                    result.append(f'Ref({inner_processed})')
            else:
                result.append(f'Ref({inner_processed})')
            i = j
        return ''.join(result)

    prev = None
    while prev != expr:
        prev = expr
        expr = _flip_ref_signs(expr)
    # ═══════════════════════════════════════

    # Skewness → Skew, Kurtosis → Kurt
    expr = expr.replace("Skewness(", "Skew(")
    expr = expr.replace("Kurtosis(", "Kurt(")

    # Ts_ArgMax → IdxMax, Ts_ArgMin → IdxMin (Qlib 等价算子)
    expr = expr.replace("Ts_ArgMax(", "IdxMax(")
    expr = expr.replace("Ts_ArgMin(", "IdxMin(")

    # Delay($close, N) → Ref($close, N)
    expr = re.sub(r"Delay\((\$\w+),\s*(\d+)\)", r"Ref(\1, \2)", expr)

    # RegBeta(y, x, N) → Corr(y,x,N)*Std(y,N)/(Std(x,N)+1e-8)
    # Qlib 0.9.7 无RegBeta算子, 用Corr+Std等价展开
    # 必须在 Delay→Ref 之后执行
    expr = _fix_regbeta(expr)

    # Scale(expr) — Qlib 不支持截面算子，去掉外层 Scale 保留时序 Z-score
    expr = re.sub(r"\bScale\((.*)\)", r"\1", expr)

    # SignedPower(x, 2) → x * Abs(x)
    expr = re.sub(r"SignedPower\((.+?),\s*2\)", r"(\1)*Abs(\1)", expr)

    # 三元 ? : → If(cond, t, f)
    expr = _fix_ternary(expr)

    # RowNum 加权求和展开
    m = re.search(
        r"Sum\((\$close/Ref\(\$close,\s*1\))\*\((\d+)-RowNum\(\)\),(\d+)\)(\s*/\s*(\d+))",
        expr,
    )
    if m:
        base_expr, n_str, n_str2, _, denom = (
            m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        )
        n = int(n_str)
        assert n == int(n_str2), f"Sum窗口={n_str2}≠权重上限={n_str}"
        terms = []
        for i in range(n):
            weight = n - i
            if i == 0:
                terms.append(f"{base_expr}*{weight}")
            else:
                terms.append(f"Ref({base_expr},{i})*{weight}")
        expanded = "(" + "+".join(terms) + f")/{denom}"
        expr = expr.replace(m.group(0), expanded)

    # Ret (裸变量) → 日收益率表达式
    expr = re.sub(r"(?<!\$)Ret(?!\w)", r"($close/Ref($close,1)-1)", expr)

    # Rank(expr) 缺少 N → Rank(expr, 5)  (Qlib Rank 需要窗口参数, pandas 不允许 N=0)
    expr = _fix_rank_missing_n(expr)

    # 列表除法 → 均线发散度近似（必须在 Mean/Std/Sum 补 N 之前，否则 Mean($close) 变 Mean($close,20) 会破坏正则）
    # -Std([Mean($close,s),...,Mean($close,l)]/$close,N) → -Abs(Mean($close,s)/Mean($close,l)-1)
    if "]/" in expr:
        m = re.search(
            r"-Std\(\[([^\]]+)\]/([^,]+),(\d+)\)", expr
        )
        if m:
            list_content, divisor, _ = m.group(1), m.group(2), m.group(3)
            windows = re.findall(r"Mean\(\$close,\s*(\d+)\)", list_content)
            if len(windows) >= 2:
                short_n, long_n = windows[0], windows[-1]
                proxy = f"-Abs(Mean($close,{short_n})/Mean($close,{long_n})-1)"
                expr = expr.replace(m.group(0), proxy)
            else:
                return "# SKIP: 列表除法 Qlib 不支持"
        else:
            return "# SKIP: 列表除法 Qlib 不支持"

    # Mean/Std/Sum 缺少 N → 补默认 20
    # 处理简单字段: Mean($close) → Mean($close, 20)
    # 以及复杂表达式: Mean($volume>Ref($volume,1)) → Mean($volume>Ref($volume,1), 20)
    for op in ("Mean", "Std", "Sum"):
        expr = _fix_missing_n(expr, op, 20)

    # +1e-8 等防除零项 — 保留在 Std/Max/分母中, 仅在顶层移除
    # (Qlib 0.9.7 可正确解析括号内的科学计数)
    # expr = re.sub(r"\+1e-\d+", "", expr)  # 不再移除, 保留数值稳定性保护
    # 微小科学计数仅在非函数参数位置移除
    # expr = re.sub(r'(?<!\w)\d+e-\d+(?!\d)', r'0', expr)

    # ═══ 浮点字面量 → 分数/整数 ═══
    FLOAT_REPL = [
        (r'(?<!\w)(\d+)\.0(?!\d)', r'\1'),      # 2.0→2, 5.0→5
        (r'(?<!\w)0\.5(?!\d)', '1/2'),
        (r'(?<!\w)1\.5(?!\d)', '3/2'),
        (r'(?<!\w)0\.25(?!\d)', '1/4'),
        (r'(?<!\w)0\.75(?!\d)', '3/4'),
        (r'(?<!\w)0\.1(?!\d)', '1/10'),
        (r'(?<!\w)0\.05(?!\d)', '1/20'),
        (r'(?<!\w)0\.01(?!\d)', '1/100'),
        (r'(?<!\w)0\.001(?!\d)', '1/1000'),
        (r'(?<!\w)0\.015(?!\d)', '3/200'),
        (r'(?<!\w)0\.3(?!\d)', '3/10'),
        (r'(?<!\w)0\.95(?!\d)', '19/20'),
        (r'(?<!\w)0\.34(?!\d)', '17/50'),
        (r'(?<!\w)0\.66(?!\d)', '33/50'),
        (r'(?<!\w)0\.00000001(?!\d)', '1/100000000'),
        (r'(?<!\w)0\.000001(?!\d)', '1/1000000'),
        (r'(?<!\w)1e10(?!\d)', '10000000000'),
        (r'(?<!\w)1e-0?8(?!\d)', '1/100000000'),
        (r'(?<!\w)1e-0?6(?!\d)', '1/1000000'),
    ]
    # N+e-N → N (窗口参数不能是浮点数, Qlib 内部已处理数值稳定性)
    # 必须在 FLOAT_REPL 之前, 否则 1e-8 已被转成 1/100000000
    expr = re.sub(r'(\d+)\+1e-0?\d+(?=\s*\))', r'\1', expr)

    for pat, repl in FLOAT_REPL:
        expr = re.sub(pat, repl, expr)

    # 残留浮点/科学计数 → 跳过
    if re.search(r'(?<!\w)\d+\.\d+', expr) or re.search(r'\b\d+e[+-]?\d+', expr):
        return "# SKIP: 浮点字面量 Qlib 不支持"

    # Min/Max 逐元素模式 → If 改写
    expr = _fix_elementwise_minmax(expr)

    # Abs(x, y) 双参数 → Abs(x-y), 用括号计数处理任意嵌套
    expr = _fix_abs_two_args(expr)

    # Log(数字) → Const($close, N) 常量算子
    expr = re.sub(r'\bLog\((\d+)\)', r'Const($close,\1)', expr)

    # **2 → 自乘
    expr = _fix_power2(expr)

    # Sign(expr) → 展开为 If(Gt(expr,0), 1, If(Lt(expr,0), -1, 0))
    # 避免依赖自定义 SignOp 算子注册（Qlib init 后会清空注册表）
    expr = _expand_sign_op(expr)

    # Slope/Rsquare → 保留原样, 由注册算子处理

    # Normalize Quantile: Quantile(expr, N, q) 或 Quantile(expr, q, N) → Quantile(expr, packed)
    expr = _normalize_quantile(expr)

    # DayOfWeek → 保留, 由注册算子处理

    # RegBeta(y, x, N) ≈ Corr(y, x, N)
    expr = re.sub(r"RegBeta\((\$\w+),(\$\w+),(\d+)\)", r"Corr(\1,\2,\3)", expr)
    expr = re.sub(r"RegBeta\((\$\w+),(.+?),(\d+)\)", r"Corr(\1,\2,\3)", expr)

    # $turnover: Qlib 无此字段，用 $volume 替代（截面排名高度相关）
    expr = expr.replace("$turnover", "$volume")

    # 一元负号修复: Qlib 不支持 -expr, 全部转为 0-expr
    if expr.startswith("-"):
        expr = "0" + expr
    # 处理中间的: (*-Operator → (*0-Operator
    expr = re.sub(r"\(\s*-([A-Z])", r"(0-\1", expr)
    # )*-Rank → )*(0-Rank(...))
    import re as _re2
    m = _re2.search(r"\)\s*\*\s*-Rank\(", expr)
    while m:
        pos = m.end()  # after Rank(
        depth = 1
        while pos < len(expr) and depth > 0:
            if expr[pos] == '(': depth += 1
            elif expr[pos] == ')': depth -= 1
            pos += 1
        expr = expr[:m.start()] + ")*(0-" + expr[m.end()-5:pos] + ")" + expr[pos:]
        m = _re2.search(r"\)\s*\*\s*-Rank\(", expr)

    return expr


def _fix_ternary(expr):
    """将 ? : 三元表达式转为 If(cond, t, f)。

    三元式 cond?true:false 中, cond 从 ? 往前最近的未匹配 ( 或逗号后开始。
    """
    # 找到 ? 的位置
    q_pos = expr.find('?')
    if q_pos < 0:
        return expr

    # 从 ? 往前找 cond 的起点: 最近一个 depth 比 ? 处少 1 的位置
    depth_at_q = 0
    for c in expr[:q_pos]:
        if c == '(':
            depth_at_q += 1
        elif c == ')':
            depth_at_q -= 1

    cond_start = 0
    paren_start = -1  # ternary 外层 '(' 位置
    depth = depth_at_q
    for i in range(q_pos - 1, -1, -1):
        if expr[i] == ')':
            depth += 1
        elif expr[i] == '(':
            depth -= 1
            if depth == depth_at_q - 1:
                cond_start = i + 1
                paren_start = i
                break

    cond = expr[cond_start:q_pos]
    rest = expr[q_pos + 1:]

    # 在 rest 中找同深度的 :
    d = 0
    colon_pos = -1
    for j, rc in enumerate(rest):
        if rc == '(':
            d += 1
        elif rc == ')':
            d -= 1
        elif rc == ':' and d == 0:
            colon_pos = j
            break

    if colon_pos < 0:
        return expr  # 未匹配, 返回原文

    true_val = rest[:colon_pos]
    after_colon = rest[colon_pos + 1:]

    # 找到匹配 cond 前 '(' 的 ')' — false_val 在此结束
    d2 = 0
    end_pos = -1
    for j, rc in enumerate(after_colon):
        if rc == '(':
            d2 += 1
        elif rc == ')':
            if d2 == 0:
                end_pos = j
                break
            d2 -= 1

    if end_pos >= 0:
        false_val = after_colon[:end_pos]
        prefix = expr[:paren_start] if paren_start >= 0 else expr[:cond_start]
        replaced = prefix + f"If({cond},{true_val},{false_val})" + after_colon[end_pos + 1:]
    else:
        false_val = after_colon
        prefix = expr[:paren_start] if paren_start >= 0 else expr[:cond_start]
        replaced = prefix + f"If({cond},{true_val},{false_val})"

    # 递归处理可能存在的其他 ternary
    return _fix_ternary(replaced) if '?' in replaced else replaced


def _fix_missing_n(expr, op, default_n):
    """为 op(...) 中缺少 N 参数的调用补上默认值。"""
    import re
    result = []
    i = 0
    pat = re.compile(rf"\b{op}\(")
    while i < len(expr):
        m = pat.search(expr, i)
        if not m:
            result.append(expr[i:])
            break
        result.append(expr[i:m.start()])
        # 找到匹配的右括号
        pos = m.end()
        depth = 1
        while pos < len(expr) and depth > 0:
            if expr[pos] == '(':
                depth += 1
            elif expr[pos] == ')':
                depth -= 1
            pos += 1
        arg = expr[m.end():pos - 1]
        # 检查参数中是否有顶层逗号
        d = 0
        has_comma = False
        for ch in arg:
            if ch == '(':
                d += 1
            elif ch == ')':
                d -= 1
            elif ch == ',' and d == 0:
                has_comma = True
                break
        if has_comma:
            result.append(expr[m.start():pos])
        else:
            result.append(f"{op}({arg}, {default_n})")
        i = pos
    return ''.join(result)


def _expand_sign_op(expr):
    """将 Sign(expr) 展开为 If(Gt(expr,0), 1, If(Lt(expr,0), -1, 0))
    避免依赖自定义 SignOp 算子注册（Qlib init 后会清空注册表）
    """
    import re
    # 找到所有 Sign(...) 配对
    result = []
    i = 0
    while i < len(expr):
        m = re.search(r'\bSign\s*\(', expr[i:])
        if not m:
            result.append(expr[i:])
            break
        start = i + m.start()
        paren_start = i + m.end() - 1  # '(' 的位置
        result.append(expr[i:start])  # Sign 之前的内容

        # 找匹配的 )
        depth = 1
        j = paren_start + 1
        while j < len(expr) and depth > 0:
            if expr[j] == '(':
                depth += 1
            elif expr[j] == ')':
                depth -= 1
            j += 1
        inner = expr[paren_start + 1: j - 1]
        # 展开: If(Gt(inner,0), 1, If(Lt(inner,0), -1, 0))
        result.append(f'If(Gt({inner},0), 1, If(Lt({inner},0), -1, 0))')
        i = j
    return ''.join(result)


def _fix_power2(expr):
    """x**2 → (x)*(x) — Qlib 表达式引擎不支持 ** 运算符"""
    while '**2' in expr:
        pos = expr.find('**2')
        start = pos - 1
        while start >= 0 and expr[start] == ' ':
            start -= 1
        if start >= 0 and expr[start] == ')':
            depth = 1
            i = start - 1
            while i >= 0 and depth > 0:
                if expr[i] == ')':
                    depth += 1
                elif expr[i] == '(':
                    depth -= 1
                if depth > 0:
                    i -= 1
            # i 在 '(' — 往前找函数名 (如 Log, Std)
            j = i - 1
            while j >= 0 and (expr[j].isalnum() or expr[j] == '_'):
                j -= 1
            inner = expr[j+1:pos]  # 包含函数名+括号内容
            expr = expr[:j+1] + '(' + inner + ')*(' + inner + ')' + expr[pos+3:]
        else:
            j = start
            while j >= 0 and (expr[j].isalnum() or expr[j] in '$_.'):
                j -= 1
            ident = expr[j+1:pos].strip()
            if ident:
                expr = expr[:j+1] + '(' + ident + ')*(' + ident + ')' + expr[pos+3:]
            else:
                break
    return expr


def _normalize_quantile(expr):
    """Quantile(expr, N, q) or Quantile(expr, q, N) → Quantile(expr, N, q)
    规范化参数顺序: N=窗口, q=分位值
    Qlib 原生 Quantile(feature, N, qscore) 可直接处理"""
    import re
    pat = re.compile(r'Quantile\((.+?),\s*([\d.]+)\s*,\s*([\d.]+)\s*\)')

    def _fix(m):
        inner, a, b = m.group(1), float(m.group(2)), float(m.group(3))
        if a > 1 and b < 1:
            N, q = int(a), b
        elif b > 1 and a < 1:
            N, q = int(b), a
        else:
            N, q = int(max(a, b)), min(a, b)
        return f'Quantile({inner}, {N}, {q})'

    return pat.sub(_fix, expr)


def _fix_elementwise_minmax(expr):
    """Min(a,b) → If(a<b, a, b); Max(a,b) → If(a>b, a, b)。
    只处理第二个参数非纯数字的情况（滚动模式 Min(f,N) 不动）。
    """
    import re
    for op, cmp_op in [("Min", "<"), ("Max", ">")]:
        i = 0
        while True:
            m = re.search(rf"\b{op}\s*\(", expr[i:])
            if not m:
                break
            paren_start = i + m.end() - 1
            depth = 1
            j = paren_start + 1
            while j < len(expr) and depth > 0:
                if expr[j] == '(':
                    depth += 1
                elif expr[j] == ')':
                    depth -= 1
                j += 1
            inner = expr[paren_start + 1:j - 1]
            d = 0
            comma_pos = -1
            for k, c in enumerate(inner):
                if c == '(':
                    d += 1
                elif c == ')':
                    d -= 1
                elif c == ',' and d == 0:
                    comma_pos = k
                    break
            if comma_pos >= 0:
                second_arg = inner[comma_pos + 1:].strip()
                is_zero = (second_arg == "0")
                if not re.match(r"^\d+$", second_arg) or is_zero:
                    a = inner[:comma_pos].strip()
                    b = second_arg
                    replacement = f"If({a}{cmp_op}{b}, {a}, {b})"
                    expr = expr[:i + m.start()] + replacement + expr[j:]
                    # 重新从当前位置开始扫描
                    i = i + m.start()
                    break
            i = j
    return expr


def _fix_abs_two_args(expr):
    """Abs(x, y) → Abs(x-y), 用括号计数处理任意嵌套表达式。
    Abs只有一个参数时保持不动 (Qlib原生支持)."""
    import re
    result = []
    i = 0
    while i < len(expr):
        m = re.search(r'\bAbs\s*\(', expr[i:])
        if not m:
            result.append(expr[i:])
            break
        start = i + m.start()
        paren_start = i + m.end() - 1
        result.append(expr[i:start])
        depth = 1
        j = paren_start + 1
        while j < len(expr) and depth > 0:
            if expr[j] == '(': depth += 1
            elif expr[j] == ')': depth -= 1
            j += 1
        inner = expr[paren_start+1:j-1]
        d = 0
        commas = []
        for k, c in enumerate(inner):
            if c == '(': d += 1
            elif c == ')': d -= 1
            elif c == ',' and d == 0:
                commas.append(k)
        if len(commas) == 1:
            k = commas[0]
            a1 = inner[:k].strip()
            a2 = inner[k+1:].strip()
            result.append(f'Abs({a1}-({a2}))')
        else:
            result.append(expr[start:j])
        i = j
    return ''.join(result)


def _has_elementwise_minmax(expr):
    """检测 Min(expr, expr) / Max(expr, expr) 逐元素模式。
    Qlib 的 Min/Max 只支持 Min(feature, N) 滚动窗口, N 必须是整数。
    如果第二个参数不是纯数字, 则认为是逐元素模式, Qlib 不支持。"""
    import re

    for op in ("Min", "Max"):
        i = 0
        while True:
            m = re.search(rf"\b{op}\s*\(", expr[i:])
            if not m:
                break
            paren_start = i + m.end() - 1
            depth = 1
            j = paren_start + 1
            while j < len(expr) and depth > 0:
                if expr[j] == '(':
                    depth += 1
                elif expr[j] == ')':
                    depth -= 1
                j += 1
            inner = expr[paren_start + 1 : j - 1]

            # 找顶层逗号
            d = 0
            comma_pos = -1
            for k, c in enumerate(inner):
                if c == '(':
                    d += 1
                elif c == ')':
                    d -= 1
                elif c == ',' and d == 0:
                    comma_pos = k
                    break

            if comma_pos >= 0:
                second_arg = inner[comma_pos + 1:].strip()
                # 纯数字 → 滚动模式 OK; 否则 → 逐元素, 不支持
                if not re.match(r"^\d+$", second_arg):
                    return True

            i = j

    return False


def _fix_rank_missing_n(expr):
    """为缺少 N 参数的 Rank(expr) 补上 N=0 (使用全部历史窗口)"""
    import re

    result = []
    i = 0
    while i < len(expr):
        m = re.search(r"\bRank\s*\(", expr[i:])
        if not m:
            result.append(expr[i:])
            break

        start = i + m.start()
        paren_start = i + m.end() - 1  # 位置在 '('

        result.append(expr[i: paren_start + 1])

        # 找到匹配的 )
        depth = 1
        j = paren_start + 1
        while j < len(expr) and depth > 0:
            if expr[j] == '(':
                depth += 1
            elif expr[j] == ')':
                depth -= 1
            j += 1

        inner = expr[paren_start + 1 : j - 1]
        # 检查 inner 是否已含逗号 (即已有 N)
        if ',' not in _strip_nested_parens(inner):
            result.append(inner + ", 5")
        else:
            result.append(inner)
        result.append(')')
        i = j

    return ''.join(result)


def _strip_nested_parens(s):
    """去除嵌套括号内容, 只留顶层逗号可见"""
    res = []
    depth = 0
    for c in s:
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif depth == 0:
            res.append(c)
    return ''.join(res)


def _fix_qlib_expr(expr):
    """修复 Qlib 不兼容的 and/or 语法。
    处理 Mean(a and b, 20) → Mean((a)*(b), 20) 等函数参数内的逻辑运算。
    """
    if not expr or " and " not in expr and " or " not in expr:
        return expr

    while " and " in expr or " or " in expr:
        expr = _fix_qlib_expr_one(expr)
    return expr


def _fix_qlib_expr_one(expr):
    """找到最深层的一个 and/or 并替换，保留外围结构。"""
    import re

    # 计算每个位置的括号深度
    depths = []
    depth = 0
    for ch in expr:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        depths.append(depth)

    # 找所有顶层 and/or（不在更深括号内的）
    best = None
    best_depth = -1
    for m in re.finditer(r'(?<=\s)\band\b(?=\s)|(?<=\s)\bor\b(?=\s)', expr):
        d = depths[m.start()]
        if d > best_depth:
            best = (m.start(), m.end(), m.group(), d)
            best_depth = d

    if not best:
        return expr

    s, e, op, d = best

    # 向左找操作数边界（同深度，遇到逗号或更深结构则继续）
    left_end = s
    for i in range(s - 1, -1, -1):
        if depths[i] < d:
            break
        if depths[i] == d and expr[i] == ',':
            break
        left_end = i

    # 向右找操作数边界
    right_start = e
    for i in range(e, len(expr)):
        if depths[i] < d:
            break
        if depths[i] == d and expr[i] == ',':
            break
        right_start = i + 1

    left = expr[:left_end] + "(" + expr[left_end:s]
    middle = ") * (" if op == "and" else ") + ("
    right = expr[e:right_start] + ")" + expr[right_start:]

    result = left + middle + right
    if op == "or":
        result = "(" + result + ")>0"
    return result


def _build_custom_features(factor_keys):
    """将白盒因子 key 列表转为 (fields, names) 元组, 供 Qlib 特征计算

    单个因子的解析逻辑提取到 factor_resolver.resolve_factor_expression, 供训练+评估共用。
    """
    from core.adapters.qlib.factor_resolver import resolve_factor_expression

    fields = []
    names = []
    skipped = []
    for key in factor_keys:
        expr, skip_reason = resolve_factor_expression(key)
        if expr is None:
            if skip_reason and skip_reason.startswith("# SKIP:"):
                skipped.append((key, skip_reason))
            continue
        fields.append(expr)
        names.append(key)

    if skipped:
        import logging
        logger = logging.getLogger(__name__)
        for key, reason in skipped:
            logger.warning(f"[WhiteBoxHandler] 跳过因子 {key}: {reason}")

    return fields, names


# ── 模块加载时注册自定义算子 (joblib 子进程 import 时自动执行) ──
_register_ts_rank()
_register_sign_op()
_register_fundval()
