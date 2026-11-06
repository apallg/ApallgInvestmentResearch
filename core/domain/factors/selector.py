"""
因子风格精选器 — 8大类分组 + 类内去冗余 + 综合排名 + 弹性配额 + 跨类去重

将 Alpha158/101/191 三套因子的原生分类归一化到统一 8 大类，
按风格方向做有意义的因子收束，输出 30-40 个精选因子供 LGB/XGB 双模型训练。
"""
import logging
import datetime
import numpy as np
import json
import os

logger = logging.getLogger(__name__)

L5_CORR_THRESHOLD = 0.75   # 类内聚类阈值 (与 overfitting_checker 一致)
CROSS_CORR_THRESHOLD = 0.70  # 跨类去重阈值 (比类内宽松)
MAX_CROSS_DEDUP_ITERS = 10   # 跨类递补迭代上限
MIN_ICIR_THRESHOLD = 0.01    # 精选最低 ICIR 门槛 (0.05→0.01, 降低以保证类别分散, 多期中位数已过滤噪音)
UNEVAL_ICIR_SENTINEL = 0.001  # 未评估因子占位 ICIR (排序沉底, 不被门槛过滤)

SCORE_WEIGHTS = {
    "icir": 0.35,
    "l2": 0.25,
    "l3": 0.20,
    "l4": 0.20,
}

UNIFIED_CATEGORIES = [
    "价格动量", "反转效应", "波动率特征", "成交量特征",
    "趋势强度", "流动性",   "技术形态",   "高阶特征",
    "规模",     "估值",     "盈利",       "成长",
    "杠杆",     "现金流",   "运营",       "股息",
]

CATEGORY_MAP_158 = {
    "价格动量": "价格动量",
    "价格位置": "趋势强度",
    "波动率":   "波动率特征",
    "成交量":   "成交量特征",
    "相关性":   "高阶特征",    # corr_hl/corr_pv20 是量价相关性, 非趋势
    "趋势强度": "趋势强度",
    "技术指标": "技术形态",    # WR/KDJ/RSI 等超买超卖指标
}

CATEGORY_MAP_101 = {
    "反转":   "反转效应",
    "动量":   "价格动量",
    "波动率": "波动率特征",
    "量价":   "成交量特征",
    "均值回归": "反转效应",
    "突破":   "趋势强度",
    "流动性": "流动性",
    "趋势":   "趋势强度",
    "特殊":   "高阶特征",
}


def _classify_by_expr(expr):
    """根据表达式内容推断因子类别 (处理 pdf_extracted 等未归类的因子)。"""
    import re
    if not expr:
        return None
    # FundVal($close, N): N 决定类别
    m = re.search(r'FundVal\([^,]+,\s*(\d+)\)', expr)
    if m:
        fid = int(m.group(1))
        fid_map = {1: "规模", 2: "规模", 3: "估值", 4: "估值", 5: "盈利",
                   6: "成长", 7: "成长", 8: "杠杆", 9: "流动性", 10: "盈利",
                   11: "盈利", 12: "盈利", 13: "盈利", 14: "杠杆", 15: "成长",
                   16: "运营", 17: "杠杆", 18: "盈利", 19: "估值", 20: "估值",
                   21: "现金流", 22: "成长", 23: "成长", 24: "成长", 25: "成长", 26: "估值"}
        return fid_map.get(fid)
    # $close / Ref($close, N) - 1: 价格动量 (N>=10) 或 反转效应 (N<10)
    m = re.search(r'/\s*Ref\(\$close,\s*(\d+)\)', expr)
    if m:
        n = int(m.group(1))
        return "价格动量" if n >= 10 else "反转效应"
    # $volume 相关 → 成交量特征
    if '$volume' in expr:
        return "成交量特征"
    # $high/$low 相关 → 波动率特征或技术形态
    if '$high' in expr or '$low' in expr:
        return "波动率特征"
    # $open 和 $close 组合 → 反转效应 (隔夜收益类)
    if '$open' in expr:
        return "反转效应"
    return None


# 英文 native category → 统一大类 (补充: 基本面类走表达式推断)
_EN_CATEGORY_MAP = {
    "momentum": "价格动量", "price_volume": "成交量特征",
    "size": "规模", "flow": "流动性", "volatility": "波动率特征",
    "technical": "技术形态", "growth": "成长", "leverage": "杠杆",
    "profitability": "盈利", "cash_flow": "现金流", "dividend": "股息",
    "分红": "股息",
}
# fundamental 因子 key-based 推断 (FundVal 的 fid 是占位值，key 前缀更可靠)
_KEY_CATEGORY_HINTS = {
    # 估值
    "pe": "估值", "pb": "估值", "ps": "估值", "ep": "估值", "bp": "估值",
    "eps": "估值", "bps": "估值",
    # 盈利
    "roe": "盈利", "roa": "盈利", "gpm": "盈利", "grossprofitmargin": "盈利",
    "netprofitmargin": "盈利", "npm": "盈利",
    # 现金流 (FCFP/OCFP/NCFP 是各种现金流/市值比，FundVal fid=6 会被误判为"成长")
    "fcfp": "现金流", "ocfp": "现金流", "ncfp": "现金流",
    "cfps": "现金流", "ocf": "现金流", "cf": "现金流", "ocftosales": "现金流",
    # 运营
    "assetturnover": "运营", "op": "运营", "op_margin": "运营",
    # 股息 (DP=股息率，FundVal fid=1 会被误判为"规模")
    "dp": "股息", "dividend": "股息", "dyr": "股息", "payout": "股息",
    # 成长
    "sue": "成长", "fsue": "成长", "revenue_yoy": "成长", "sales": "成长",
    "net_profit_yoy": "成长", "sur": "成长", "yoy": "成长",
    # 杠杆
    "debt_to_assets": "杠杆", "dar": "杠杆", "quick_ratio": "杠杆",
    # 规模
    "total_mv": "规模", "circ_mv": "规模",
    # 流动性
    "turnover_rate": "流动性",
}


# Alpha158/Alpha191 key-pattern-based 分类 (因子池注册时缺 source/category 时的回退)
def _classify_a158_key(key):
    """按 Alpha158 因子 key 推断类别"""
    k = key.lower()
    # 技术形态: MA/MACD/KDJ/WR/RSI/BOLL/CCI/DMI
    if any(w in k for w in ['ma5', 'ma10', 'ma20', 'ma60', 'macd', 'kdj', 'wr', 'rsi', 'bb_', 'boll', 'cci', 'dmi']):
        return "技术形态"
    # 成交量特征
    if any(w in k for w in ['amt', 'vol', 'turn']):
        return "成交量特征"
    # 波动率特征: STD/VSTD/ATR
    if any(w in k for w in ['std', 'vstd', 'atr']):
        return "波动率特征"
    # 趋势强度: ROC/MOM/RSI/High/Low位置
    if any(w in k for w in ['roc', 'mom', 'resi', 'high', 'low']):
        return "趋势强度"
    if any(w in k for w in ['corr']):
        return "高阶特征"
    return None

def _classify_a101_key(key):
    """按 Alpha101 因子 key 推断类别 — 基于 WorldQuant 101 Formulaic Alphas 论文分类"""
    k = key.lower()
    # 反转效应
    if any(w in k for w in ['a028', 'a005', 'a008', 'meanrev', 'reversal', 'short_rev', 'a004']):
        return "反转效应"
    # 波动率特征
    if any(w in k for w in ['a035', 'a053', 'a054', 'a078', 'lowvol', 'volregime', 'a032', 'a027']):
        return "波动率特征"
    # 趋势强度/突破
    if any(w in k for w in ['a046', 'a034', 'a065', 'breakout', 'newhigh', 'range_pos', 'a029', 'a047', 'a055', 'a059',
                             'a060', 'a082', 'a084', 'a088', 'a092', 'a087']):
        return "趋势强度"
    # 成交量特征 (大部分a101因子含volume)
    if any(w in k for w in ['a007', 'a009', 'a011', 'a019', 'a023', 'a029', 'a031', 'a038', 'a040', 'a041',
                             'a043', 'a067', 'a068', 'a071', 'a073', 'a074', 'a081', 'liq', 'volume_pct',
                             'a083', 'a095']):
        return "成交量特征"
    # 价格动量
    if any(w in k for w in ['a033', 'a052', 'a070', 'accel', 'momentum', 'a001', 'a012', 'a018', 'a030',
                             'a020', 'a050', 'a097', 'a079']):
        return "价格动量"
    # 价量相关性/高阶特征 (Corr, RegBeta)
    if any(w in k for w in ['a002', 'a003', 'a006', 'a010', 'a014', 'a015', 'a016', 'a017', 'a021',
                             'a022', 'a024', 'a025', 'a026', 'a036', 'a037', 'a039', 'a042', 'a044',
                             'a045', 'a048', 'a049', 'a051', 'a056', 'a057', 'a058', 'a061', 'a062',
                             'a063', 'a064', 'a066', 'a069', 'a072', 'a075', 'a076', 'a077', 'a080',
                             'a085', 'a086', 'a089', 'a090', 'a091', 'a093', 'a094', 'a096', 'a098',
                             'a099', 'a100', 'a101', 'skew', 'vpdiv', 'vpdiv2', 'open_gap',
                             'range', 'trend', 'turnover']):
        return "高阶特征"
    return None

def _classify_a191_key(key):
    """按 Alpha191 因子 key 推断类别 — 基于 WorldQuant 191 Alpha 论文分类"""
    k = key.lower()
    # 成交量特征
    if any(w in k for w in ['amt_', 'liq_', 'trd_']):
        return "成交量特征"
    # 波动率特征
    if any(w in k for w in ['vol_']):
        return "波动率特征"
    # 价格动量
    if any(w in k for w in ['mom_']):
        return "价格动量"
    # 技术形态/Pattern (pat_ / hof_)
    if any(w in k for w in ['pat_']):
        return "技术形态"
    # 反转效应
    if any(w in k for w in ['rev_']):
        return "反转效应"
    # 高阶特征 (hof_ goes here since they're mostly statistical)
    if any(w in k for w in ['hof_']):
        return "高阶特征"
    return None

# 因子 key 前缀 → 隐含 source (池注册时缺 source 字段的回退)
_KEY_SOURCE_HINTS = {
    "a158_": "Alpha158", "a101_": "Alpha101", "a191_": "Alpha191",
}


def get_unified_category(factor_key, source=None, native_category=None, expr=None):
    # 0. 回退缺失的 source (从 key 前缀推断)
    if not source:
        for prefix, src in _KEY_SOURCE_HINTS.items():
            if factor_key.lower().startswith(prefix):
                source = src
                break
    # 1. 显式 source 映射 (key-based 回退已在映射函数内)
    if source == "Alpha158":
        if native_category and native_category in CATEGORY_MAP_158:
            return CATEGORY_MAP_158[native_category]
        return _classify_a158_key(factor_key) or "高阶特征"
    if source == "Alpha101":
        if native_category and native_category in CATEGORY_MAP_101:
            return CATEGORY_MAP_101[native_category]
        cat = _classify_a101_key(factor_key)
        if cat: return cat
    if source == "Alpha191":
        if native_category and native_category in CATEGORY_MAP_158:
            return CATEGORY_MAP_158[native_category]
        cat = _classify_a191_key(factor_key)
        if cat: return cat

    # 2. native category 映射优先于 key 前缀启发式 (避免 key 中偶然匹配覆盖显式分类)
    if native_category and native_category in _EN_CATEGORY_MAP:
        return _EN_CATEGORY_MAP[native_category]
    if native_category and native_category in UNIFIED_CATEGORIES:
        return native_category

    # 3. key 前缀启发式 (仅在 native_category 缺失/无效时回退)
    key_lower = factor_key.lower().replace("_n", "").replace("_q", "")
    key_base = key_lower.split("_")[0]
    for prefix_len in range(len(key_lower.split("_")[0]), 1, -1):
        candidate = key_lower[:prefix_len].rstrip("_")
        if candidate in _KEY_CATEGORY_HINTS:
            return _KEY_CATEGORY_HINTS[candidate]
    if key_base in _KEY_CATEGORY_HINTS:
        return _KEY_CATEGORY_HINTS[key_base]
    for part in key_lower.split("_")[1:]:
        for pl in range(len(part), 1, -1):
            candidate = part[:pl].rstrip("_")
            if candidate in _KEY_CATEGORY_HINTS:
                return _KEY_CATEGORY_HINTS[candidate]

    # 4. FundVal 表达式 (仅作为 key 未命中时的回退)
    if expr and 'FundVal' in expr:
        cat = _classify_by_expr(expr)
        if cat:
            return cat

    # 5. 通用表达式推断 (非 FundVal)
    if expr and 'FundVal' not in expr:
        cat = _classify_by_expr(expr)
        if cat:
            return cat

    return "高阶特征"


def group_by_category(factor_keys, adapter=None, pool_active=None):
    groups = {c: [] for c in UNIFIED_CATEGORIES}
    for key in factor_keys:
        native_cat = None
        source = None
        expr = None
        if adapter:
            info = adapter.all_factors.get(key, {})
            native_cat = info.get("category", "")
            source = info.get("source", "")
        # adapter 里没有的，从因子池取
        if not source and pool_active:
            pool_info = pool_active.get(key, {})
            expr = pool_info.get("expr", "")
            # 优先用 pool_active 中的 native category，被 get_active_factors 覆盖时用 expr 分类
            native_cat_from_pool = pool_info.get("category", "")
            if native_cat_from_pool and native_cat_from_pool not in ("white_box", "traditional", "hybrid"):
                native_cat = native_cat_from_pool
            elif not native_cat:
                native_cat = native_cat_from_pool
            source = pool_info.get("source", "")
        unified = get_unified_category(key, source, native_cat, expr)
        if unified not in groups:
            groups[unified] = []
        groups[unified].append(key)
    return {c: keys for c, keys in groups.items() if keys}


def _compute_factor_scores_vector(factor_key, variant, date_snapshots, eval_dates,
                                  screener, checker, pool, code_prefixes, industry_map):
    """计算单个因子在多个日期的截面得分向量，返回 {date_key: (codes, scores)}。"""
    from core.domain.factors.market_cap import get_market_cap_map
    from core.domain.factors.neutralize import cross_sectional_neutralize

    iso, iso_sig, use_expr, inner_ast, outer_op = checker._resolve_eval(variant)

    result = {}
    for ed in eval_dates:
        dk = ed.isoformat() if hasattr(ed, 'isoformat') else str(ed)
        snap = date_snapshots.get(dk)
        if snap is None:
            continue
        try:
            cs = checker._factor_codes_scores(
                iso, iso_sig, use_expr, inner_ast, outer_op,
                ed, dk, pool, code_prefixes, date_snapshots, screener)
            if cs is None:
                continue
            codes, raw_scores = cs
            if len(codes) < 20:
                continue
            mktcap_map = get_market_cap_map(codes, ed)
            resid = cross_sectional_neutralize(codes, raw_scores, mktcap_map, industry_map)
            result[dk] = (codes, resid)
        except Exception:
            continue
    return result


def dedupe_within_category(keys, variant_map, date_snapshots, screener, checker,
                           pool, code_prefixes, industry_map, eval_dates):
    """类内层次聚类去重: |ρ|>L5_CORR_THRESHOLD 归为一簇, 每簇保留 ICIR 最高因子。

    返回 (kept_keys, icir_map) — icir_map 复用 vecs 计算过程, 避免二次 _factor_codes_scores。
    """
    if len(keys) <= 5:
        # 小集合直接返回, ICIR 轻量计算
        if date_snapshots:
            icir_map = _compute_icir_map(list(keys), variant_map, date_snapshots, screener,
                                         checker, pool, code_prefixes, industry_map, eval_dates)
        else:
            icir_map = {}
        return list(keys), icir_map

    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from core.domain.factors.market_cap import get_market_cap_map
    from core.domain.factors.neutralize import cross_sectional_neutralize
    from scipy.stats import spearmanr

    n = len(keys)
    key_list = sorted(keys)

    # 计算 pairwise 平均绝对相关矩阵 + 同时收集 IC (一趟完成)
    corr_sum = np.zeros((n, n))
    corr_cnt = np.zeros((n, n))
    key_ics = {k: [] for k in key_list}

    for dk, snap in date_snapshots.items():
        ed = datetime.date.fromisoformat(dk[:10])
        vecs = {}
        for key in key_list:
            v = variant_map.get(key)
            if v is None:
                continue
            try:
                iso, iso_sig, use_expr, inner_ast, outer_op = checker._resolve_eval(v)
                cs = checker._factor_codes_scores(
                    iso, iso_sig, use_expr, inner_ast, outer_op,
                    ed, dk, pool, code_prefixes, {dk: snap}, screener)
                if cs is None:
                    continue
                codes, scores = cs
                if len(codes) >= 20:
                    vecs[key] = (codes, scores)
            except Exception:
                continue

        # 计算此日期的 forward returns + 截面 IC (复用 vecs, 不重算 _factor_codes_scores)
        if vecs:
            codes_all = sorted(set(c for _, (c_list, _) in vecs.items() for c in c_list))
            fwd = checker._batch_forward_returns(codes_all, 20, ed)
            fwd_map = {c: r for c, r in zip(codes_all, fwd)} if fwd is not None else {}
            mktcap_map = get_market_cap_map(codes_all, ed) if codes_all else {}

            for key, (codes, scores) in vecs.items():
                if len(codes) < 20:
                    continue
                resid = cross_sectional_neutralize(codes, scores, mktcap_map, industry_map)
                if not resid or len(resid) < 10:
                    continue
                xs = [resid.get(c, 0) for c in codes if c in fwd_map]
                ys = [fwd_map[c] for c in codes if c in resid and c in fwd_map]
                if len(xs) >= 10:
                    corr, _ = spearmanr(xs, ys)
                    if corr is not None and not np.isnan(corr):
                        key_ics[key].append(float(corr))

        for i, ki in enumerate(key_list):
            if ki not in vecs:
                continue
            ci, si = vecs[ki]
            for j, kj in enumerate(key_list):
                if j <= i or kj not in vecs:
                    continue
                cj, sj = vecs[kj]
                common = sorted(set(ci) & set(cj))
                if len(common) < 20:
                    continue
                ai = np.array([si[ci.index(c)] for c in common])
                aj = np.array([sj[cj.index(c)] for c in common])
                if np.std(ai) < 1e-9 or np.std(aj) < 1e-9:
                    continue
                r = float(np.corrcoef(ai, aj)[0, 1])
                if not np.isnan(r):
                    corr_sum[i, j] += r
                    corr_sum[j, i] += r
                    corr_cnt[i, j] += 1
                    corr_cnt[j, i] += 1

    mask = corr_cnt > 0
    corr_avg = np.zeros((n, n))
    corr_avg[mask] = np.abs(corr_sum[mask] / corr_cnt[mask])
    np.fill_diagonal(corr_avg, 1.0)

    # 层次聚类
    dist = 1.0 - corr_avg
    dist = np.nan_to_num(dist, nan=1.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')
    th = 1.0 - L5_CORR_THRESHOLD
    clusters = fcluster(Z, th, criterion='distance')

    # 从一趟计算中得出 ICIR (abs mean / std), 替代独立的 _compute_icir_map 调用
    icir_map = {}
    for k, ics in key_ics.items():
        if len(ics) >= 3:
            m = np.mean(ics)
            s = np.std(ics)
            icir_map[k] = abs(m) / max(s, 0.001)
        elif ics:
            icir_map[k] = abs(ics[0])
        else:
            icir_map[k] = 0.0

    # 每簇保留 ICIR 最高者
    cluster_best = {}
    for i, key in enumerate(key_list):
        cid = int(clusters[i])
        cur_best = cluster_best.get(cid)
        if cur_best is None or icir_map.get(key, 0) > icir_map.get(cur_best, 0):
            cluster_best[cid] = key

    kept = sorted(cluster_best.values())
    if len(kept) < len(keys):
        logger.debug(f"[StyleSelector] 类内去重: {len(keys)} → {len(kept)} "
                     f"(聚类{len(cluster_best)}组)")
    return kept, icir_map


def _compute_icir_map(keys, variant_map, date_snapshots, screener, checker,
                      pool, code_prefixes, industry_map, eval_dates):
    """批量计算一组因子的 ICIR (中性化截面 IC 的 |mean|/std)。"""
    from core.domain.factors.market_cap import get_market_cap_map
    from core.domain.factors.neutralize import cross_sectional_neutralize
    from scipy.stats import spearmanr

    key_ics = {k: [] for k in keys}

    for dk, snap in date_snapshots.items():
        ed = datetime.date.fromisoformat(dk[:10])
        codes_all = sorted({c for s in snap.values() for c in [s[0].iloc[0].get("code", "")] if c} or
                           list(snap.keys()))
        fwd = checker._batch_forward_returns(codes_all, 20, ed)
        fwd_map = {c: r for c, r in zip(codes_all, fwd)}
        mktcap_map = get_market_cap_map(codes_all, ed) if codes_all else {}

        for key in keys:
            v = variant_map.get(key)
            if v is None:
                continue
            try:
                iso, iso_sig, use_expr, inner_ast, outer_op = checker._resolve_eval(v)
                cs = checker._factor_codes_scores(
                    iso, iso_sig, use_expr, inner_ast, outer_op,
                    ed, dk, pool, code_prefixes, {dk: snap}, screener)
                if cs is None:
                    continue
                codes, scores = cs
                if len(codes) < 20:
                    continue
                resid = cross_sectional_neutralize(codes, scores, mktcap_map, industry_map)
                if not resid or len(resid) < 10:
                    continue
                xs = [resid.get(c, 0) for c in codes if c in fwd_map]
                ys = [fwd_map[c] for c in codes if c in resid and c in fwd_map]
                if len(xs) >= 10:
                    corr, _ = spearmanr(xs, ys)
                    if corr is not None and not np.isnan(corr):
                        key_ics[key].append(float(corr))
            except Exception:
                continue

    result = {}
    for k, ics in key_ics.items():
        if len(ics) >= 3:
            m = np.mean(ics)
            s = np.std(ics)
            result[k] = abs(m) / max(s, 0.001)
        elif ics:
            result[k] = abs(ics[0])
        else:
            result[k] = 0.0
    return result


def compute_factor_icir_single(factor_key, variant, date_snapshots, screener, checker,
                                pool, code_prefixes, industry_map, eval_dates):
    """计算单个因子的 ICIR (供 OverfittingChecker 代理调用)。"""
    icir_map = _compute_icir_map(
        [factor_key], {factor_key: variant}, date_snapshots, screener,
        checker, pool, code_prefixes, industry_map, eval_dates)
    return icir_map.get(factor_key, 0.0)


def rank_within_category(keys, validation_results, icir_map):
    """类内综合排名 (min-max 归一化 + 加权)。

    validation_results: batch_validate() 的返回值 {key: {layers, all_pass, ...}}
    icir_map: {key: float} 预计算的 ICIR 值
    返回按 composite 降序排列的 key 列表。
    """
    if len(keys) <= 1:
        return [(k, 0.0, {}, {"icir": icir_map.get(k, 0)}) for k in keys]

    raw = {}
    for key in keys:
        r = validation_results.get(key, {})
        layers = {l["id"]: l for l in r.get("layers", [])}
        raw[key] = {
            "icir": icir_map.get(key, 0),
            "l2": layers.get(2, {}).get("score", 0),
            "l3": layers.get(3, {}).get("score", 0),
            "l4": layers.get(4, {}).get("score", 0),
        }

    # min-max 归一化 (同类内)
    normed = {}
    for metric in ["icir", "l2", "l3", "l4"]:
        vals = [raw[k][metric] for k in keys]
        vmin, vmax = min(vals), max(vals)
        for k in keys:
            if k not in normed:
                normed[k] = {}
            if vmax > vmin:
                normed[k][metric] = (raw[k][metric] - vmin) / (vmax - vmin)
            else:
                normed[k][metric] = 0.5

    scored = []
    for key in keys:
        composite = (SCORE_WEIGHTS["icir"] * normed[key]["icir"] +
                     SCORE_WEIGHTS["l2"]   * normed[key]["l2"] +
                     SCORE_WEIGHTS["l3"]   * normed[key]["l3"] +
                     SCORE_WEIGHTS["l4"]   * normed[key]["l4"])
        scored.append((key, composite, normed[key], raw[key]))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def allocate_quotas(category_groups, scored_by_category, target_lower=25, target_upper=35):
    """弹性配额分配。

    基础: 每类 min(len_keys, 3)
    硬上限: 每类最多 5 个 (防单一风格通吃)
    剩余名额按大类平均 ICIR 降序分配, 每类最多+1
    单类合格因子<3 → 全保留 (兜底)
    """
    MAX_PER_CATEGORY = 5

    cat_avg_icir = {}
    for cat, scored in scored_by_category.items():
        if scored:
            cat_avg_icir[cat] = np.mean([s[3]["icir"] for s in scored])
        else:
            cat_avg_icir[cat] = 0.0

    base = {}
    for cat, keys in category_groups.items():
        base[cat] = min(len(keys), 3)

    total_base = sum(base.values())
    remaining = max(0, target_upper - total_base)

    sorted_cats = sorted(cat_avg_icir, key=lambda c: cat_avg_icir[c], reverse=True)

    quotas = dict(base)
    for cat in sorted_cats:
        if remaining <= 0 or quotas[cat] >= MAX_PER_CATEGORY:
            continue
        max_extra = min(1, len(category_groups[cat]) - quotas[cat])
        give = min(max_extra, remaining)
        quotas[cat] += give
        remaining -= give

    # 硬上限约束
    for cat in quotas:
        quotas[cat] = min(quotas[cat], MAX_PER_CATEGORY)

    # 兜底: 单类合格因子 < 3 个时全保留
    for cat in category_groups:
        if len(category_groups[cat]) < 3:
            quotas[cat] = len(category_groups[cat])

    # 再次检查总量, 不足从高 ICIR 类补 (不超硬上限)
    total = sum(quotas.values())
    if total < target_lower:
        need = target_lower - total
        for cat in sorted_cats:
            if need <= 0:
                break
            can_add = min(len(category_groups[cat]), MAX_PER_CATEGORY) - quotas[cat]
            give = min(can_add, need)
            if give > 0:
                quotas[cat] += give
                need -= give

    return quotas


def select_top_per_category(scored_by_category, quotas):
    """按配额从每类选取 top-N 因子。低于 MIN_ICIR_THRESHOLD 的不入选。
    基本面类门槛降为 0.03 (财务数据季频, IC天然低于技术因子)。
    每类至少保留 1 个 (如果该类有因子)。
    """
    LOW_ICIR_CATEGORIES = {"基本面"}

    selected = {}
    for cat, scored in scored_by_category.items():
        n = quotas.get(cat, min(len(scored), 4))
        thr = 0.03 if cat in LOW_ICIR_CATEGORIES else MIN_ICIR_THRESHOLD
        qualified = [(k, c, nm, r) for k, c, nm, r in scored
                     if r.get("icir", 0) >= thr]
        # 兜底: 如该类有因子但无合格者, 至少保留 ICIR 最高的 1 个
        if not qualified and scored:
            qualified = [scored[0]]
        selected[cat] = qualified[:n]
    return selected


def cross_category_dedup(selected_by_category, icir_map, date_snapshots, screener,
                         checker, pool, code_prefixes, industry_map, eval_dates,
                         variant_map, max_iter=MAX_CROSS_DEDUP_ITERS):
    """跨类相关性终检: 仅过滤正向高相关 (ρ > 0.7), 保留负相关对冲因子。

    selected_by_category: {cat: [(key, composite, normed, raw), ...]}
    返回去重后的 {cat: [(key, ...), ...]}
    """
    # 构建 flat 列表
    all_selected = {}
    for cat, items in selected_by_category.items():
        for item in items:
            all_selected[item[0]] = cat

    if len(all_selected) <= 1:
        return selected_by_category

    # 计算所有选中因子的截面得分向量
    all_keys = list(all_selected)
    score_vectors = _compute_score_vectors_for_keys(
        all_keys, variant_map, date_snapshots, screener, checker,
        pool, code_prefixes, eval_dates)

    for iteration in range(max_iter):
        conflicts = []
        n = len(all_keys)
        for i in range(n):
            for j in range(i + 1, n):
                ki, kj = all_keys[i], all_keys[j]
                ci, cj = all_selected[ki], all_selected[kj]
                if ci == cj:
                    continue
                vi, vj = score_vectors.get(ki), score_vectors.get(kj)
                if vi is None or vj is None:
                    continue
                common = sorted(set(vi[0]) & set(vj[0]))
                if len(common) < 20:
                    continue
                si = np.array([vi[1][vi[0].index(c)] for c in common])
                sj = np.array([vj[1][vj[0].index(c)] for c in common])
                if np.std(si) < 1e-9 or np.std(sj) < 1e-9:
                    continue
                from scipy.stats import spearmanr
                rho, _ = spearmanr(si, sj)
                if rho is not None and not np.isnan(rho) and rho > CROSS_CORR_THRESHOLD:
                    loser = ki if icir_map.get(ki, 0) < icir_map.get(kj, 0) else kj
                    conflicts.append((loser, ci))

        if not conflicts:
            break

        # 去重冲突
        losers_by_cat = {}
        for loser, cat in set(conflicts):
            losers_by_cat.setdefault(cat, []).append(loser)

        for cat, losers in losers_by_cat.items():
            cat_items = selected_by_category.get(cat, [])
            new_items = [item for item in cat_items if item[0] not in set(losers)]
            selected_by_category[cat] = new_items

        # 重新计算 flat 列表 (得分向量只减不增, 过滤即可)
        all_selected = {}
        for cat, items in selected_by_category.items():
            for item in items:
                all_selected[item[0]] = cat
        all_keys = list(all_selected)
        score_vectors = {k: v for k, v in score_vectors.items() if k in all_selected}

    if iteration >= max_iter - 1:
        logger.warning(f"[StyleSelector] 跨类去重达到迭代上限 {max_iter}, 强制退出")

    return selected_by_category


def _compute_score_vectors_for_keys(keys, variant_map, date_snapshots, screener,
                                     checker, pool, code_prefixes, eval_dates):
    """取第一个代表日计算所有因子得分向量。"""
    result = {}
    if not date_snapshots:
        return result
    dk = list(date_snapshots.keys())[0]
    snap = date_snapshots[dk]
    ed = datetime.date.fromisoformat(dk[:10])

    for key in keys:
        v = variant_map.get(key)
        if v is None:
            continue
        try:
            iso, iso_sig, use_expr, inner_ast, outer_op = checker._resolve_eval(v)
            cs = checker._factor_codes_scores(
                iso, iso_sig, use_expr, inner_ast, outer_op,
                ed, dk, pool, code_prefixes, {dk: snap}, screener)
            if cs is not None and len(cs[0]) >= 20:
                result[key] = cs
        except Exception:
            continue
    return result


def build_backup_pool(scored_by_category, selected_by_category):
    """构建每类的候补池: 全量排名中未被选中的因子。"""
    backup = {}
    for cat, scored in scored_by_category.items():
        selected_keys = {item[0] for item in selected_by_category.get(cat, [])}
        backup[cat] = [item for item in scored if item[0] not in selected_keys]
    return backup


def generate_report(selected_by_category, icir_map, category_groups, eliminated,
                    max_cross_corr, curation_name):
    """生成精选集 json 报表。"""
    from core.shared.persistence import save_json, today_iso, now_iso
    per_category = {}
    total_selected = 0
    for cat in UNIFIED_CATEGORIES:
        items = selected_by_category.get(cat, [])
        total_selected += len(items)
        original = len(category_groups.get(cat, []))
        per_category[cat] = {
            "input": original,
            "selected": len(items),
            "factors": [item[0] for item in items],
            "avg_icir": round(float(np.mean([item[3]["icir"] for item in items])), 4) if items else 0,
        }

    report = {
        "name": curation_name,
        "timestamp": now_iso(),
        "input_count": sum(len(v) for v in category_groups.values()),
        "output_count": total_selected,
        "per_category": per_category,
        "max_cross_corr": round(float(max_cross_corr), 4),
        "eliminated": eliminated,
    }

    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"curation_{curation_name}_{today_iso()}.json")
    try:
        save_json(report, log_path)
        logger.info(f"[StyleSelector] 报表已保存: {log_path}")
    except Exception as e:
        logger.warning(f"[StyleSelector] 报表保存失败: {e}")

    try:
        from core.data import CurationLogRepo
        CurationLogRepo.save(curation_name, report)
    except Exception as e:
        logger.debug(f"[StyleSelector] MySQL 报表保存失败: {e}")

    return report


def _load_scores_from_db(factor_keys, anchor_date):
    """从 factor_validation_scores 表批量加载已有校验结果。

    返回 {factor_key: {icir, l2, l3, l4, all_pass, passed_count, direction}}
    未命中返回空 dict。
    """
    if not factor_keys or not anchor_date:
        return {}
    try:
        from core.data import Database
        if not Database.is_available():
            return {}
    except Exception:
        return {}

    result = {}
    try:
        placeholders = ','.join(['%s'] * len(factor_keys))
        rows = Database.fetchall(
            f"SELECT factor_key, icir, l2_score, l3_score, l4_score, "
            f"all_pass, passed_count, direction "
            f"FROM factor_validation_scores "
            f"WHERE anchor_date=%s AND factor_key IN ({placeholders})",
            [anchor_date.isoformat() if hasattr(anchor_date, 'isoformat') else str(anchor_date)]
            + list(factor_keys))
        for r in rows:
            result[r["factor_key"]] = {
                "icir": r["icir"] or 0,
                "l2": r["l2_score"] or 0,
                "l3": r["l3_score"] or 0,
                "l4": r["l4_score"] or 0,
                "all_pass": bool(r["all_pass"]),
                "passed_count": r["passed_count"] or 0,
                "direction": r["direction"] or "long",
            }
        if result:
            logger.info(f"[StyleSelector] DB命中 {len(result)}/{len(factor_keys)} 因子 (anchor={anchor_date})")
    except Exception as e:
        logger.warning(f"[StyleSelector] DB查询失败: {e}")

    return result


def _save_scores_to_db(factor_keys, anchor_date, scores_map):
    """将因子校验分数写入 factor_validation_scores 表。"""
    if not factor_keys or not anchor_date:
        return
    try:
        from core.data import Database
        if not Database.is_available():
            return
        anchor_str = anchor_date.isoformat() if hasattr(anchor_date, 'isoformat') else str(anchor_date)
        sql = (
            "INSERT INTO factor_validation_scores "
            "(factor_key, anchor_date, icir, ic_mean, ic_std, n_ic_dates, daily_ics) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE "
            "icir=VALUES(icir), ic_mean=VALUES(ic_mean), ic_std=VALUES(ic_std), "
            "n_ic_dates=VALUES(n_ic_dates), daily_ics=VALUES(daily_ics)"
        )
        params_list = []
        for key in factor_keys:
            info = scores_map.get(key, {})
            icir = info.get("icir", 0)
            ic_mean = info.get("ic_mean", 0)
            ic_std = info.get("ic_std", 0)
            n_ic = info.get("n_ic_dates", 0)
            daily = info.get("daily_ics")
            daily_json = json.dumps(daily) if daily is not None else None
            params_list.append((key, anchor_str, icir, ic_mean, ic_std, n_ic, daily_json))
        try:
            Database.executemany(sql, params_list)
        except Exception as e:
            logger.debug(f"[StyleSelector] 批量保存失败: {e}")
        logger.info(f"[StyleSelector] 已写入 {len(factor_keys)} 条 ICIR 到 factor_validation_scores (anchor={anchor_str})")
    except Exception as e:
        logger.warning(f"[StyleSelector] 保存校验分数失败: {e}")


# ═══ 策略方向预设 ═══
DIRECTION_PRESETS = {
    "trend_momentum": {
        "label": "趋势动量",
        "categories": ["价格动量", "趋势强度", "成长", "规模", "高阶特征", "成交量特征"],
        "desc": "追涨不抄底，牛市/上涨趋势中动量+成长因子主导",
        "regimes": ["trending_up", "bull_steady"],
        "target_n": 50,
    },
    "value_defense": {
        "label": "价值防御",
        "categories": ["估值", "盈利", "现金流", "杠杆", "股息", "运营"],
        "desc": "熊市避险，便宜+质量扛跌",
        "regimes": ["trending_down", "bear_steady"],
        "target_n": 50,
    },
    "oversold_rebound": {
        "label": "超卖反弹",
        "categories": ["波动率特征", "流动性", "成交量特征", "反转效应"],
        "desc": "高波动中识别超卖+恐慌放量+聪明钱抄底，博短期反弹",
        "regimes": ["volatile"],
        "target_n": 21,
    },
    "volume_breakout": {
        "label": "量价突破",
        "categories": ["成交量特征", "流动性", "趋势强度", "价格动量", "技术形态"],
        "desc": "横盘等放量突破+量价背离+资金异动",
        "regimes": ["ranging"],
        "target_n": 48,
    },
}


def _compute_regime_icir(factor_keys, regime_dates, min_dates=5, as_of=None):
    """按指定交易日集合计算各因子的 regime-specific ICIR (多期中位数)。

    从 factor_validation_scores.daily_ics 中提取给定交易日的 IC，
    按 anchor_date 分批计算每期的 ICIR，取中位数作为最终分数。
    相比单期快照，多期中位数对 regime 标签噪声更鲁棒。

    Args:
        factor_keys: 因子 key 列表
        regime_dates: 目标市场状态下的交易日集合 (set of str 'YYYY-MM-DD')
        min_dates: 该状态下最少 IC 观测天数 (单期内)
        as_of: 时序截断点 (可选) — 只用 anchor_date <= as_of 的快照, 无前视

    Returns:
        {factor_key: {icir, ic_mean, n_dates, n_periods}}
    """
    if not factor_keys or not regime_dates:
        return {}

    from core.data import Database
    if not Database.is_available():
        return {}

    try:
        placeholders = ','.join(['%s'] * len(factor_keys))
        sql = (f"SELECT factor_key, daily_ics, anchor_date FROM factor_validation_scores "
               f"WHERE daily_ics IS NOT NULL AND factor_key IN ({placeholders})")
        params = list(factor_keys)
        if as_of is not None:
            sql += " AND anchor_date <= %s"
            params.append(str(as_of)[:10])
        sql += " ORDER BY anchor_date DESC LIMIT 8000"
        rows = Database.fetchall(sql, params)
    except Exception as e:
        logger.warning(f"[RegimeICIR] DB查询失败: {e}")
        return {}

    # 按 (factor_key, anchor_date) 分组，每期独立计算 ICIR
    from collections import defaultdict
    period_icirs = defaultdict(list)
    seen_periods = defaultdict(set)

    for r in rows:
        key = r["factor_key"]
        anchor = str(r.get("anchor_date", ""))[:10]
        daily_ics = r["daily_ics"]
        if not daily_ics:
            continue
        try:
            pairs = json.loads(daily_ics) if isinstance(daily_ics, str) else daily_ics
        except Exception:
            continue

        regime_ics = [float(p[1]) for p in pairs if len(p) >= 2 and str(p[0]) in regime_dates]
        if len(regime_ics) < min_dates:
            continue

        ic_mean = sum(regime_ics) / len(regime_ics)
        ic_std = (sum((x - ic_mean) ** 2 for x in regime_ics) / len(regime_ics)) ** 0.5
        icir = ic_mean / ic_std if ic_std > 0 else 0

        # 同一 anchor_date 只取最新一条
        if anchor not in seen_periods[key]:
            seen_periods[key].add(anchor)
            period_icirs[key].append((icir, len(regime_ics)))

    # 取中位数 ICIR (最少2期才有意义)
    result = {}
    for key, icirs in period_icirs.items():
        if len(icirs) < 1:
            continue
        icir_values = sorted([v[0] for v in icirs])
        median_icir = icir_values[len(icir_values) // 2]
        total_dates = sum(v[1] for v in icirs)
        result[key] = {
            "icir": round(median_icir, 4),
            "ic_mean": round(np.mean([v[0] for v in icirs]), 4),
            "n_dates": total_dates,
            "n_periods": len(icirs),
        }

    n_multi = sum(1 for v in result.values() if v.get("n_periods", 0) >= 2)
    logger.info(f"[RegimeICIR] {len(result)}/{len(factor_keys)} 因子, "
                f"其中 {n_multi} 个有>=2期观测 (中位数评分)")
    return result


def _load_regime_dates_for_direction(direction_name, as_of=None):
    """加载某策略方向对应市场状态的所有交易日集合。

    从 market_regime_history 表查询该方向覆盖的 regime 值对应的所有 trade_date。
    as_of 给定时只取 <= as_of 的交易日 (时序无前视)。
    """
    from core.domain.regime.weights import get_direction_regimes
    target_regimes = get_direction_regimes(direction_name)
    if not target_regimes:
        return set()

    from core.data import Database
    if not Database.is_available():
        return set()

    try:
        placeholders = ','.join(['%s'] * len(target_regimes))
        sql = f"SELECT trade_date FROM market_regime_history " \
              f"WHERE regime IN ({placeholders})"
        params = list(target_regimes)
        if as_of is not None:
            sql += " AND trade_date <= %s"
            params.append(str(as_of)[:10])
        sql += " ORDER BY trade_date"
        rows = Database.fetchall(sql, params)
        regime_dates = {str(r["trade_date"]) for r in rows}
        logger.info(f"[RegimeICIR] {direction_name} → {target_regimes}: {len(regime_dates)} 个交易日"
                    + (f" (as_of={str(as_of)[:10]})" if as_of is not None else ""))
        return regime_dates
    except Exception as e:
        logger.warning(f"[RegimeICIR] 加载regime日期失败: {e}")
        return set()


def _build_direction_icir_map(direction_name, selected_keys, regime_aware, as_of=None):
    """构建方向因子 ICIR 映射: regime ICIR 优先, 无数据回退全局 ICIR。

    返回 (icir_map, regime_icir_map, regime_dates, icir_source)。
    未评估因子给 UNEVAL_ICIR_SENTINEL 占位, 排序时沉底。
    as_of 给定时, regime 交易日与校验快照均截断到 <= as_of (无前视)。
    """
    regime_icir_map = {}
    regime_dates = set()
    if regime_aware:
        regime_dates = _load_regime_dates_for_direction(direction_name, as_of=as_of)
        if regime_dates:
            regime_icir_map = _compute_regime_icir(selected_keys, regime_dates, min_dates=2, as_of=as_of)

    _snapshot_fail = 0
    icir_map = {}

    # 先加载全局 ICIR 作为回退 (分批查询避免 IN 子句过长)
    global_icir_map = {}
    if not regime_icir_map:
        logger.info(f"[Direction] {direction_name}: regime ICIR 为空, 尝试全局 ICIR")
    if len(regime_icir_map) < len(selected_keys) * 0.5:
        try:
            from core.data import Database
            if Database.is_available():
                BATCH = 200
                for i in range(0, len(selected_keys), BATCH):
                    batch = selected_keys[i:i + BATCH]
                    ph = ','.join(['%s'] * len(batch))
                    sql = (f"SELECT factor_key, icir, n_ic_dates FROM factor_validation_scores "
                           f"WHERE icir IS NOT NULL AND icir > 0 AND n_ic_dates > 0 "
                           f"AND factor_key IN ({ph})")
                    params = list(batch)
                    if as_of is not None:
                        sql += " AND anchor_date <= %s"
                        params.append(str(as_of)[:10])
                    sql += " ORDER BY anchor_date DESC"
                    rows = Database.fetchall(sql, params)
                    for r in rows:
                        k = r["factor_key"]
                        kl = k.lower()
                        if kl not in global_icir_map:
                            global_icir_map[kl] = float(r["icir"])
                logger.info(f"[Direction] {direction_name}: 全局 ICIR 命中 {len(global_icir_map)}/{len(selected_keys)}")
        except Exception as e:
            logger.debug(f"[Direction] 全局 ICIR 查询失败: {e}")

    for key in selected_keys:
        if key in regime_icir_map:
            info = regime_icir_map[key]
            if info.get("n_dates", 0) > 0:
                icir_map[key] = info["icir"]
            else:
                _snapshot_fail += 1
    for key in selected_keys:
        if key not in icir_map:
            kl = key.lower()
            if kl in global_icir_map and global_icir_map[kl] > 0:
                icir_map[key] = global_icir_map[kl]
            else:
                icir_map[key] = UNEVAL_ICIR_SENTINEL

    n_regime = sum(1 for k in selected_keys if k in regime_icir_map)
    n_global = sum(1 for k in selected_keys if k not in regime_icir_map and k.lower() in global_icir_map)
    n_ok = n_regime - _snapshot_fail
    icir_source = f"regime({n_ok})" + (f"+global({n_global})" if n_global else "") + f"/{len(selected_keys)}"
    logger.info(f"[Direction] {direction_name}: ICIR source = {icir_source}")

    return icir_map, regime_icir_map, regime_dates, icir_source


def _load_layer_scores_from_db(factor_keys, as_of=None):
    """从 factor_validation_scores 加载 L2/L3/L4 层分数与 max_corr, 跨锚点取中位数。

    返回 {key: {"layers": [{"id": 2/3/4, "score"}], "max_corr": float}},
    供 rank_within_category 综合评分 + 类内去冗余复用。L2/L3 已删除 → score 恒 0。
    """
    if not factor_keys:
        return {}
    try:
        from core.data import Database
        if not Database.is_available():
            return {}
    except Exception:
        return {}

    try:
        placeholders = ','.join(['%s'] * len(factor_keys))
        sql = (f"SELECT factor_key, l2_score, l3_score, l4_score, max_corr "
               f"FROM factor_validation_scores WHERE factor_key IN ({placeholders})")
        params = list(factor_keys)
        if as_of is not None:
            sql += " AND anchor_date <= %s"
            params.append(str(as_of)[:10])
        rows = Database.fetchall(sql, params)
    except Exception as e:
        logger.warning(f"[LayerScore] DB查询失败: {e}")
        return {}

    from collections import defaultdict
    acc = defaultdict(lambda: {"l2": [], "l3": [], "l4": [], "mc": []})
    for r in rows:
        k = r["factor_key"]
        acc[k]["l2"].append(r.get("l2_score") or 0)
        acc[k]["l3"].append(r.get("l3_score") or 0)
        acc[k]["l4"].append(r.get("l4_score") or 0)
        acc[k]["mc"].append(r.get("max_corr") or 0)

    def _med(xs):
        return sorted(xs)[len(xs) // 2] if xs else 0

    result = {}
    for k, v in acc.items():
        result[k] = {
            "layers": [
                {"id": 2, "score": _med(v["l2"])},
                {"id": 3, "score": _med(v["l3"])},
                {"id": 4, "score": _med(v["l4"])},
            ],
            "max_corr": _med(v["mc"]),
        }
    return result


def _dedupe_by_expression(keys):
    """按解析后表达式去重: 相同表达式的多个 key 只保留注册顺序靠前的第一个。

    _n 扩展因子(YOYROE_n1/4/8/12 等)的表达式定义重复, 校验 daily_ics 相同,
    排序时会扎堆污染精选。此函数在精选候选层剔除重复表达式因子。
    """
    from core.adapters.qlib.factor_resolver import resolve_factor_expression
    seen = set()
    result = []
    for k in keys:
        expr, _skip = resolve_factor_expression(k)
        if not expr:
            result.append(k)
            continue
        if expr in seen:
            continue
        seen.add(expr)
        result.append(k)
    return result


def curate_direction(direction_name, active_keys, adapter, validation_results,
                     target_n=None, anchor_date=None, save=True, regime_aware=True,
                     pool_active=None, method="uniform", as_of=None):
    """按策略方向精选因子。

    Args:
        method: "uniform" — 每类均匀配额, 类内综合评分(ICIR+L2/L3/L4)+去冗余
                "icir"   — 按 DB 历史 ICIR 排名选取 (旧方式, 同样综合评分)
        as_of: 时序截断点 (可选 'YYYY-MM-DD') — 只用 <= as_of 的 regime 交易日与校验快照
    返回 (name, flat_list, count)。"""
    preset = DIRECTION_PRESETS.get(direction_name)
    if preset is None:
        raise ValueError(f"未知策略方向: {direction_name}, 可选: {list(DIRECTION_PRESETS)}")
    if target_n is None:
        target_n = preset.get("target_n", 50)

    active_keys = _dedupe_by_expression(active_keys)
    category_groups = group_by_category(active_keys, adapter, pool_active=pool_active)

    # ── uniform 模式: 每类均匀配额, 类内按注册/插入顺序取前 N, 不做数据排序/去冗余 ──
    if method == "uniform":
        cats_present = [c for c in preset["categories"] if c in category_groups]
        if not cats_present:
            return direction_name, [], 0
        per_cat = max(5, target_n // len(cats_present))

        taken = []
        for cat in cats_present:
            keys = category_groups.get(cat, [])  # group_by_category 保持注册/插入顺序
            taken.extend(keys[:per_cat])
        taken = list(dict.fromkeys(taken))[:target_n]

        if save:
            from core.domain.factors.registry import get_pool
            get_pool().save_curated_set(
                name=direction_name,
                factors_by_category={preset["label"]: [(k, 0.0, {}, {}) for k in taken]},
                config={
                    "description": preset["desc"],
                    "direction": direction_name,
                    "categories": preset["categories"],
                    "target_n": target_n,
                    "method": "uniform",
                },
                stats={"total": len(taken), "method": "uniform",
                       "note": "每类均匀配额, 类内按注册/插入顺序, 不做数据排序/去冗余"},
            )
        dist = {c: len([k for k in taken if k in category_groups.get(c, [])]) for c in cats_present}
        logger.info(f"[Direction] {direction_name}: uniform {len(cats_present)}类 → {len(taken)}因子, 分布={dist}")
        return direction_name, taken, len(taken)

    # ── icir 模式: 原有 DB ICIR 排序逻辑 ──
    selected_keys = []
    for cat in preset["categories"]:
        if cat in category_groups:
            selected_keys.extend(category_groups[cat])
    selected_keys = list(dict.fromkeys(selected_keys))  # 去重保序

    if not selected_keys:
        logger.warning(f"[Direction] {direction_name}: 无匹配因子")
        return direction_name, [], 0

    # ── regime-aware ICIR: regime ICIR 优先, 无数据回退全局 ICIR ──
    icir_map, regime_icir_map, regime_dates, icir_source = _build_direction_icir_map(
        direction_name, selected_keys, regime_aware, as_of=as_of)
    layer_scores = _load_layer_scores_from_db(selected_keys, as_of=as_of)

    # 每类分配配额: 基础 min(类内因子数, target_n//len(cats)), 余数给大类和ICIR高的类
    cats = preset["categories"]
    n_cats = len([c for c in cats if c in category_groups])
    if n_cats == 0:
        return direction_name, [], 0
    base_per_cat = max(3, target_n // n_cats)
    remainder = target_n - base_per_cat * n_cats

    taken = []
    cat_remainder = sorted(
        [(c, len(category_groups.get(c, [])), np.mean([icir_map.get(k,0) for k in category_groups.get(c,[])]) or 0)
         for c in cats if c in category_groups],
        key=lambda x: (x[1], x[2]), reverse=True)
    for cat, _, _ in cat_remainder:
        keys_in_cat = category_groups.get(cat, [])
        evaluated = [k for k in keys_in_cat
                     if icir_map.get(k, 0) > UNEVAL_ICIR_SENTINEL and icir_map.get(k, 0) >= MIN_ICIR_THRESHOLD]
        unevaluated = [k for k in keys_in_cat if icir_map.get(k, 0) == UNEVAL_ICIR_SENTINEL]
        # 类内综合评分 (ICIR + L2/L3/L4) + 去冗余 (max_corr > L5 阈值)
        scored = rank_within_category(evaluated, layer_scores, icir_map)
        ranked = []
        for k, _, _, _ in scored:
            if ranked and layer_scores.get(k, {}).get("max_corr", 0) > L5_CORR_THRESHOLD:
                continue
            ranked.append(k)
        n_uneval = min(3, len(unevaluated))
        ranked.extend(unevaluated[:n_uneval])
        n_take = base_per_cat + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        taken.extend(ranked[:n_take])

    # 不足 target_n 时从全局补充 (只用已评估且达标的)
    if len(taken) < target_n:
        already = set(taken)
        extras = [k for k in sorted(selected_keys, key=lambda k: icir_map.get(k,0), reverse=True)
                  if k not in already and icir_map.get(k, 0) >= MIN_ICIR_THRESHOLD]
        taken.extend(extras[:target_n - len(taken)])

    taken = taken[:target_n]

    # 补充: 快照不可算但Qlib训练可用的因子, 每类最多追加5个 (不看ICIR, 纯补宽度)
    _extra = 0
    for cat, _, _ in cat_remainder:
        keys_in_cat = category_groups.get(cat, [])
        already = set(taken)
        extra_uneval = [k for k in keys_in_cat
                        if k not in already and icir_map.get(k, 0) == UNEVAL_ICIR_SENTINEL]
        if extra_uneval:
            take_extra = min(5, len(extra_uneval))
            taken.extend(extra_uneval[:take_extra])
            _extra += take_extra
    if _extra:
        logger.info(f"[Direction] {direction_name}: 追加 {_extra} 个快照不可算因子 (Qlib训练可用)")
    logger.info(f"[Direction] {direction_name}: {len(selected_keys)} → {len(taken)} "
                f"(类内平衡, {n_cats}类, ICIR={icir_source})")

    if save:
        from core.domain.factors.registry import get_pool
        name = direction_name
        factors_by_category = {preset["label"]: [(k, 0.0, {}, {"icir": icir_map.get(k, 0)}) for k in taken]}
        config = {
            "description": preset["desc"],
            "direction": direction_name,
            "categories": preset["categories"],
            "target_n": target_n,
            "regime_aware": regime_aware,
            "icir_source": icir_source,
        }
        avg_icir = sum(icir_map.get(k, 0) for k in taken) / max(len(taken), 1)
        n_regime_factors = sum(1 for k in taken if k in regime_icir_map)
        stats = {
            "total": len(taken),
            "avg_icir": avg_icir,
            "n_regime_icir": n_regime_factors,
            "n_regime_dates": len(regime_dates),
        }
        get_pool().save_curated_set(name, factors_by_category, config, stats)
        logger.info(f"[Direction] 已保存精选集 '{name}': {len(taken)} 因子 "
                    f"(regime: {n_regime_factors}, avg_icir={avg_icir:.3f})")

    return direction_name, taken, len(taken)


def get_direction_factor_keys(direction_name, as_of=None, target_n=30):
    """获取某策略方向精选后的因子 key 列表，可直接作为 Qlib custom_factor_keys。
    从因子池中读取已保存的精选集；若无则返回空列表。

    as_of: 时序截断点 (可选) — 给定则要求精选集创建日期 <= as_of, 用于回测复现历史精选集。
    """
    try:
        from core.domain.factors.registry import get_pool
        pool = get_pool()
        curated = pool.state.get("curated", {}).get(direction_name)
        if curated and curated.get("flat_list"):
            created = curated.get("created", "")
            if as_of is not None and created and created[:10] > str(as_of)[:10]:
                logger.info(f"[Direction] {direction_name}: 精选集创建于 {created[:10]} "
                            f"> as_of={str(as_of)[:10]}, 视为不可用")
                return []
            logger.info(f"[Direction] {direction_name}: 加载已保存精选集 "
                        f"({len(curated['flat_list'])} 因子, {created[:10] or '?'})")
            return curated["flat_list"]
    except Exception as e:
        logger.warning(f"[Direction] 加载精选集 {direction_name} 失败: {e}")
    return []


def get_direction_categories(direction_name):
    """返回某策略方向对应的大类列表"""
    preset = DIRECTION_PRESETS.get(direction_name, {})
    return preset.get("categories", [])


def curate(active_keys, adapter, validation_results=None, date_snapshots=None,
           screener=None, checker=None, pool="all", code_prefixes=None,
           target_range=(30, 40), curation_name=None, anchor_date=None,
           method="category"):
    """主入口: 风格精选流水线。

    参数:
        active_keys: 活跃因子 key 列表
        adapter: PublicFactorAdapter 实例
        validation_results: batch_validate() 返回值 {key: {...}}, None=从DB加载
        date_snapshots: {dk: snapshot} 预计算快照 (DB模式时仅需1个代表日)
        screener: BuySideScreener 实例
        checker: OverfittingChecker 实例
        pool: 股票池
        code_prefixes: 代码前缀
        target_range: (lower, upper) 目标因子数范围
        curation_name: 精选集名称
        anchor_date: DB查询锚点日期, None=跳过DB直接全量计算
        method: "simple"=按ICIR全局排名取前N | "category"=按大类配额

    返回:
        (selected_by_category, report, icir_map)
    """
    if curation_name is None:
        from core.shared.persistence import today_iso
        curation_name = f"auto_{today_iso()}"

    if screener is None:
        from core.domain.screening._screener import get_screener
        screener = get_screener()
    if checker is None:
        from core.domain.factors.validation import OverfittingChecker
        checker = OverfittingChecker()

    industry_map = checker._get_industry_map()

    if date_snapshots:
        eval_dates = sorted(date_snapshots.keys())
        eval_dates = [datetime.date.fromisoformat(d[:10]) for d in eval_dates]
    else:
        eval_dates = []

    # 构建 variant_map
    variant_map = {}
    for key in active_keys:
        v = adapter.to_variant(key)
        if v:
            variant_map[key] = v

    # Step 0: 优先从 DB 加载已有校验结果
    if validation_results is None and anchor_date is not None:
        db_scores = _load_scores_from_db(active_keys, anchor_date)
        db_hits = set(db_scores)
        missed = [k for k in active_keys if k not in db_hits]

        if missed and eval_dates:
            logger.info(f"[StyleSelector] DB命中 {len(db_hits)}, 需补校验 {len(missed)}")
            missed_results = adapter.batch_validate(
                factor_keys=missed, eval_weeks=12, pool=pool,
                max_workers=4, mode="fast", dedupe=False,
                anchor_date=anchor_date)
            validation_results = {**db_scores, **missed_results}
        elif missed:
            # 无快照无法补校验, 用DB命中部分继续
            logger.info(f"[StyleSelector] DB命中 {len(db_hits)}, {len(missed)} 无数据跳过 (无快照)")
            validation_results = db_scores
            # 从 active_keys 中移除未命中项
            active_keys = list(db_hits)
        else:
            logger.info(f"[StyleSelector] 全部 {len(db_hits)} 因子命中 DB, 跳过 batch_validate")
            validation_results = db_scores
    elif validation_results is None:
        raise ValueError("validation_results 和 anchor_date 至少提供一个")

    logger.info(f"[StyleSelector] 开始精选: {len(active_keys)} 个因子 → 目标 {target_range[0]}-{target_range[1]} method={method}")

    # ── simple 模式: 全局排名, 不分大类 ──
    if method == "simple":
        # 收集 ICIR + L2/L3/L4 综合评分
        icir_map = {}
        has_valid_icir = False
        for key in active_keys:
            vr = validation_results.get(key, {})
            icir_map[key] = vr.get("icir", 0)
            if icir_map[key] > 0.001:
                has_valid_icir = True

        # 若 ICIR 全为 0, 用 L2+L3+L4 综合分兜底, 门槛降为 0
        effective_threshold = MIN_ICIR_THRESHOLD if has_valid_icir else 0.0
        if not has_valid_icir:
            logger.warning("[StyleSelector] ICIR 全为0, 改用 L2+L3+L4 综合分排名, 不限门槛")
            scored = []
            for key in active_keys:
                vr = validation_results.get(key, {})
                # L2/L3/L4 归一化后加权 (同 rank_within_category 逻辑)
                composite = (0.35 * (vr.get("l2", 0) + vr.get("l3", 0) + vr.get("l4", 0)) / 3)
                scored.append((key, composite))
            scored.sort(key=lambda x: x[1], reverse=True)
        else:
            scored = [(k, icir_map[k]) for k in active_keys
                      if icir_map[k] >= effective_threshold]
            scored.sort(key=lambda x: x[1], reverse=True)

        n_take = min(target_range[1], len(scored))
        selected_keys = [k for k, _ in scored[:n_take]]

        logger.info(f"[StyleSelector] simple: 阈值={effective_threshold}, "
                     f"合格 {len(scored)} 个, 取前 {n_take}")

        selected_by_category = {"精选": [(k, 0.0, {}, {"icir": icir_map.get(k, 0)}) for k in selected_keys]}
        report = {
            "input_count": len(active_keys),
            "output_count": len(selected_keys),
            "per_category": {"精选": len(selected_keys)},
            "max_cross_corr": 0.0,
            "quota_mode": "global_rank",
        }
        return selected_by_category, report, icir_map

    # ── category 模式: 大类配额 ──

    # Step 1: 按统一大类分组
    category_groups = group_by_category(active_keys, adapter)
    logger.info(f"[StyleSelector] Step1 分组: { {c: len(v) for c, v in category_groups.items()} }")

    # Step 2: 类内去冗余 (层次聚类, 需快照)
    # dedupe_within_category 返回 (kept, icir_map), 一趟完成, 避免二次 _factor_codes_scores
    after_dedupe = {}
    dedupe_icir = {}  # 从类内去重收集的 ICIR
    if date_snapshots:
        for cat, keys in category_groups.items():
            deduped, cat_icir = dedupe_within_category(
                keys, variant_map, date_snapshots, screener, checker,
                pool, code_prefixes, industry_map, eval_dates)
            after_dedupe[cat] = deduped
            dedupe_icir.update(cat_icir)
    else:
        after_dedupe = category_groups  # DB模式无快照, 跳过聚类

    # Step 3: ICIR 从 dedupe 阶段已预计算, 无需独立 _compute_icir_map
    all_keys_deduped = []
    for keys in after_dedupe.values():
        all_keys_deduped.extend(keys)

    icir_map = {}
    for key in all_keys_deduped:
        if key in dedupe_icir:
            icir_map[key] = dedupe_icir[key]
        else:
            vr = validation_results.get(key, {})
            icir_map[key] = vr.get("icir", 0)

    scored_by_category = {}
    for cat, keys in after_dedupe.items():
        scored = rank_within_category(keys, validation_results, icir_map)
        scored_by_category[cat] = scored

    # Step 3: 弹性配额 + 选取
    quotas = allocate_quotas(category_groups, scored_by_category, target_range[0], target_range[1])
    logger.info(f"[StyleSelector] 配额: {quotas}")

    selected_by_category = select_top_per_category(scored_by_category, quotas)
    backup_pool = build_backup_pool(scored_by_category, selected_by_category)

    # Step 4: 跨类去重 (仅正向)
    selected_by_category = cross_category_dedup(
        selected_by_category, icir_map, date_snapshots, screener,
        checker, pool, code_prefixes, industry_map, eval_dates,
        variant_map, max_iter=MAX_CROSS_DEDUP_ITERS)
    # 重新注入 backup
    for cat in selected_by_category:
        selected_keys = {item[0] for item in selected_by_category[cat]}
        # 补充可能因递补而减少的
        quota = quotas.get(cat, 4)
        while len(selected_by_category[cat]) < quota and backup_pool.get(cat):
            next_up = backup_pool[cat].pop(0)
            if next_up[0] not in selected_keys:
                selected_by_category[cat].append(next_up)
                selected_keys.add(next_up[0])

    # Step 5: 计算最终指标
    max_cross = compute_max_cross_corr(
        selected_by_category, date_snapshots, screener, checker,
        pool, code_prefixes, variant_map, eval_dates)

    # 汇总淘汰信息
    final_keys = set()
    for items in selected_by_category.values():
        for item in items:
            final_keys.add(item[0])
    eliminated = [k for k in active_keys if k not in final_keys]

    total = sum(len(v) for v in selected_by_category.values())
    logger.info(f"[StyleSelector] 精选完成: {len(active_keys)} → {total} "
                f"(跨类最大相关 {max_cross:.3f})")

    # Step 6: 生成报表
    report = generate_report(selected_by_category, icir_map, category_groups,
                             eliminated, max_cross, curation_name)

    return selected_by_category, report, icir_map


def compute_max_cross_corr(selected_by_category, date_snapshots, screener, checker,
                           pool, code_prefixes, variant_map, eval_dates):
    """计算选中因子间的最大跨类相关系数。"""
    all_selected = {}
    for cat, items in selected_by_category.items():
        for item in items:
            all_selected[item[0]] = cat

    keys = list(all_selected)
    if len(keys) <= 1:
        return 0.0

    score_vectors = _compute_score_vectors_for_keys(
        keys, variant_map, date_snapshots, screener, checker,
        pool, code_prefixes, eval_dates)

    max_corr = 0.0
    from scipy.stats import spearmanr
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            if all_selected[ki] == all_selected[kj]:
                continue
            vi, vj = score_vectors.get(ki), score_vectors.get(kj)
            if vi is None or vj is None:
                continue
            common = sorted(set(vi[0]) & set(vj[0]))
            if len(common) < 20:
                continue
            si = np.array([vi[1][vi[0].index(c)] for c in common])
            sj = np.array([vj[1][vj[0].index(c)] for c in common])
            if np.std(si) < 1e-9 or np.std(sj) < 1e-9:
                continue
            rho, _ = spearmanr(si, sj)
            if rho is not None and not np.isnan(rho):
                max_corr = max(max_corr, rho)

    return max_corr

