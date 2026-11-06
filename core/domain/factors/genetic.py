"""
遗传编程因子挖掘引擎 — 表达式树进化 + qlib 表达式序列化 + 截面 IC 适应度。

基因型: 前缀 list, 如 ['mul', '$close', 'Mean', '$close', 20]
序列化: 树 → qlib 中缀表达式字符串(供 expression.py parse/eval 复用)
进化: 锦标赛选择 + 子树交叉 + 子树变异 + 精英保留(自写, 不依赖 gplearn)
适应度: 五因子中性化后的截面 RankIC 均值(数据由上层注入, L3 纯计算)

约束: 函数集只使用 expression.py _SUPPORTED_FUNCS 已支持的算子,
保证生成的表达式 100% 可被现有求值器解析。
"""
import math
import warnings

import numpy as np

from core.domain.factors.archive import expression_fingerprint
from core.domain.factors.candidate import expression_complexity
from core.domain.factors.fitness import compute_training_fitness, summarize_ic_series

# 常数截面求 spearman 会刷 ConstantInputWarning, 遗传进化中大量表达式产生常数, 静音
warnings.filterwarnings("ignore", message="An input array is constant")

__all__ = [
    "FIELDS", "WINDOWS", "FUNCTIONS",
    "expr_from_tree", "random_tree",
    "subtree_mutation", "subtree_crossover", "tournament_select",
    "GeneticFactorMiner", "build_mining_context",
]

# ── 终端与函数集 ──

FIELDS = ["$close", "$open", "$high", "$low", "$volume", "$amount", "$turnover"]
SEMANTIC_TERMINALS = [
    "($close / (Ref($close, 5) + 1e-8) - 1)",
    "($close / (Ref($close, 20) + 1e-8) - 1)",
    "($open / (Ref($close, 1) + 1e-8) - 1)",
    "(($high - $low) / (Ref($close, 1) + 1e-8))",
    "($volume / (Mean($volume, 20) + 1e-8))",
    "($amount / (Mean($amount, 20) + 1e-8))",
    "($turnover / (Mean($turnover, 20) + 1e-8))",
]
TERMINALS = FIELDS + SEMANTIC_TERMINALS
WINDOWS = [3, 5, 10, 20, 60]


def _fmt_bin(op):
    def _f(a, b):
        return f"({a} {op} {b})"
    return _f


def _fmt_win(name):
    def _f(a, w):
        return f"{name}({a}, {w})"
    return _f


# name -> (arity, window_args, serializer)
# arity: 参数个数(不含窗口); window_args: 尾部窗口参数个数(0 或 1)
FUNCTIONS = {
    "add":    (2, 0, _fmt_bin("+")),
    "sub":    (2, 0, _fmt_bin("-")),
    "mul":    (2, 0, _fmt_bin("*")),
    "div":    (2, 0, lambda a, b: f"({a} / ({b} + 1e-8))"),
    "abs":    (1, 0, lambda a: f"Abs({a})"),
    "log":    (1, 0, lambda a: f"Log({a})"),
    "sign":   (1, 0, lambda a: f"Sign({a})"),
    "neg":    (1, 0, lambda a: f"(-{a})"),
    "mean":   (1, 1, _fmt_win("Mean")),
    "std":    (1, 1, _fmt_win("Std")),
    "delta":  (1, 1, _fmt_win("Delta")),
    "sum":    (1, 1, _fmt_win("Sum")),
    "tsrank": (1, 1, _fmt_win("Ts_Rank")),
    "corr":   (2, 1, lambda a, b, w: f"Corr({a}, {b}, {w})"),
    "rank":   (1, 0, lambda a: f"Rank({a})"),
}


# ── 序列化 ──

def expr_from_tree(node):
    """前缀表达式树 → qlib 中缀表达式字符串。

    叶子: 字段字符串(如 '$close'); 窗口参数: int(原样传递)。
    """
    if isinstance(node, str):
        return node
    name = node[0]
    _arity, _win, fmt = FUNCTIONS[name]
    args = [expr_from_tree(a) if isinstance(a, list) else a for a in node[1:]]
    return fmt(*args)


# ── 随机生成 ──

def random_tree(rng, depth=0, max_depth=4):
    """随机生成一棵表达式树。深度耗尽或按概率返回字段叶子。
    rank 截面算子只作根 (嵌套 Rank 在 expression.py 中不可求值)。"""
    if depth >= max_depth or (depth > 0 and rng.random() < 0.35):
        return rng.choice(TERMINALS)
    names = [n for n in FUNCTIONS if n != "rank"] if depth > 0 else list(FUNCTIONS)
    name = rng.choice(names)
    arity, window_args, _ = FUNCTIONS[name]
    node = [name]
    for _ in range(arity):
        node.append(random_tree(rng, depth + 1, max_depth))
    for _ in range(window_args):
        node.append(int(rng.choice(WINDOWS)))
    return node


# ── 子树操作 ──

def _clone(node):
    if isinstance(node, list):
        return [_clone(x) for x in node]
    return node


def _tree_depth(node):
    if not isinstance(node, list):
        return 0
    children = [child for child in node[1:] if isinstance(child, (list, str))]
    return 1 + max((_tree_depth(child) for child in children), default=0)


def _iter_subtrees(node, path=()):
    """DFS 产出所有可替换节点 (list 子树 / str 字段), 跳过 int 窗口参数。
    窗口参数 (int) 不可作为交叉/变异替换点, 否则会破坏树结构。"""
    yield path, node
    if isinstance(node, list):
        for i in range(1, len(node)):
            if isinstance(node[i], (list, str)):
                yield from _iter_subtrees(node[i], path + (i,))


def _set_path(tree, path, value):
    node = tree
    for i in path[:-1]:
        node = node[i]
    node[path[-1]] = value


def subtree_mutation(tree, rng, max_depth=4):
    """子树变异: 随机选一个子节点, 替换为同深度的随机新树。"""
    t = _clone(tree)
    paths = list(_iter_subtrees(t))
    path, _node = paths[rng.integers(len(paths))]
    new = random_tree(rng, depth=len(path), max_depth=max_depth)
    if path == ():
        return new
    _set_path(t, path, new)
    return t


def subtree_crossover(t1, t2, rng):
    """子树交叉: 交换两棵树的随机子节点。"""
    a, b = _clone(t1), _clone(t2)
    pa = list(_iter_subtrees(a))
    pb = list(_iter_subtrees(b))
    path_a, node_a = pa[rng.integers(len(pa))]
    path_b, node_b = pb[rng.integers(len(pb))]
    if path_a == () and path_b == ():
        return a, b
    if path_a == ():
        a = node_b
    else:
        _set_path(a, path_a, node_b)
    if path_b == ():
        b = node_a
    else:
        _set_path(b, path_b, node_a)
    return a, b


def tournament_select(pop, fitness, rng, k=3):
    """锦标赛选择: 随机 k 个个体中取适应度最高者。"""
    idxs = rng.choice(len(pop), k, replace=False)
    return pop[max(idxs, key=lambda i: fitness[i])]


# ── 进化引擎 ──

class GeneticFactorMiner:
    """遗传编程因子挖掘: 进化表达式树, 适应度 = 五因子中性化截面 IC 均值。

    数据全部由上层注入 (L3 纯度, 不直接碰 DB/文件):
      snapshots:    {date_iso: {code: df}}           多日期快照
      mktcap_maps:  {date_iso: {code: total_mv}}     市值映射
      industry_map: {code: industry}                 行业映射
      extra_factors:{name: {date_iso: {code: value}}} 附加中性化因子(动量/换手/波动)
    """

    def __init__(self, snapshots, forward_days=20, mktcap_maps=None,
                 industry_map=None, extra_factors=None):
        self.snapshots = snapshots
        self.forward_days = forward_days
        self.mktcap_maps = mktcap_maps or {}
        self.industry_map = industry_map or {}
        self.extra_factors = extra_factors or {}

    def _batch_ic(self, exprs, dates, progress_cb=None, return_metrics=False,
                  include_series=False):
        """批量计算每个表达式在 dates 上的中性化截面 IC 均值(带符号)。

        对每个日期只做一次 batch_eval_all(分组+面板构建一次, 所有表达式共享),
        避免逐表达式重复分组。返回与 exprs 等长的列表 (不可求值为 nan)。
        """
        from core.domain.factors.expression import parse_factor, batch_eval_all
        from core.domain.factors.neutralize import cross_sectional_neutralize
        from core.domain.factors.forward_returns import get_forward_cache
        from scipy.stats import spearmanr

        asts = {}
        for i, e in enumerate(exprs):
            try:
                asts[i] = parse_factor(e)
            except Exception:
                asts[i] = None

        forward_cache = get_forward_cache()
        ic_values = [[] for _ in exprs]
        ic_dates = [[] for _ in exprs]

        for date in dates:
            snapshot = self.snapshots.get(date)
            if not snapshot:
                continue
            valid = {i: a for i, a in asts.items() if a is not None}
            if not valid:
                continue
            try:
                scores_map = batch_eval_all(valid, snapshot)
            except Exception:
                continue

            mktcap_map = self.mktcap_maps.get(date)
            extra = {name: fmap.get(date, {}) for name, fmap in self.extra_factors.items()}
            for i in valid:
                sm = scores_map.get(i)
                if not sm or len(sm) < 20:
                    continue
                codes = list(sm)
                scores = [sm[c] for c in codes]
                resid = cross_sectional_neutralize(
                    codes, scores, mktcap_map, self.industry_map, extra)
                fwd = forward_cache.get_list(codes, date, self.forward_days)
                xs = [resid[c] for c in codes]
                if len(xs) < 10:
                    continue
                try:
                    ic, _ = spearmanr(xs, fwd)
                except Exception:
                    continue
                if ic is not None and math.isfinite(ic):
                    ic_values[i].append(float(ic))
                    ic_dates[i].append(str(date))
            if progress_cb:
                progress_cb(date, len(dates))

        if return_metrics:
            metrics = [summarize_ic_series(values, total_count=len(dates)) for values in ic_values]
            if include_series:
                for item, values, dates_for_values in zip(metrics, ic_values, ic_dates):
                    item["series"] = [
                        {"date": date, "ic": value}
                        for date, value in zip(dates_for_values, values)
                    ]
            return metrics
        return [sum(values) / len(values) if values else float("nan")
                for values in ic_values]

    def mine(self, population=300, generations=30, top_n=10, train_dates=None,
             valid_dates=None, seed=None, max_depth=4, tournament_k=3,
             crossover_prob=0.7, mutation_prob=0.3, elite=2, progress_cb=None,
             stage_cb=None, ic_progress_cb=None, excluded_fingerprints=None,
             resume_state=None, checkpoint_cb=None):
        """进化主循环。返回 [{expr, train_ic, valid_ic}] 按验证 IC 降序。

        progress_cb(gen, total, best_fitness): 每代结束时回调, 供异步任务更新进度。
        """
        rng = np.random.default_rng(seed)
        dates = list(train_dates) if train_dates else list(self.snapshots.keys())
        if not dates:
            return []
        excluded = set(excluded_fingerprints or ())
        if resume_state is None:
            archive = {}
            pop = [random_tree(rng, max_depth=max_depth) for _ in range(population)]
            start_generation = 0
        else:
            if resume_state.get("version") != 1:
                raise ValueError("遗传挖矿检查点版本不支持")
            start_generation = int(resume_state.get("next_generation", -1))
            pop = resume_state.get("population")
            archive_entries = resume_state.get("archive")
            rng_state = resume_state.get("rng_state")
            if not 0 <= start_generation <= generations:
                raise ValueError("遗传挖矿检查点代数无效")
            if not isinstance(pop, list) or len(pop) != population:
                raise ValueError("遗传挖矿检查点种群与当前配置不匹配")
            if not isinstance(archive_entries, list) or not isinstance(rng_state, dict):
                raise ValueError("遗传挖矿检查点内容不完整")
            archive = dict(archive_entries)
            rng.bit_generator.state = rng_state

        for gen in range(start_generation, generations):
            if stage_cb:
                stage_cb("train_filter", gen, generations)
            exprs = [expr_from_tree(t) for t in pop]
            train_metrics = self._batch_ic(
                exprs, dates, progress_cb=ic_progress_cb, return_metrics=True)
            fitness = []
            for expr, metrics in zip(exprs, train_metrics):
                fingerprint = expression_fingerprint(expr)
                score = (
                    compute_training_fitness(
                        metrics,
                        complexity=expression_complexity(expr),
                        max_complexity=max_depth * 10,
                    )
                    if metrics.get("count") and fingerprint not in excluded
                    else -1.0
                )
                fitness.append(score)
                if score < 0:
                    continue
                current = archive.get(fingerprint)
                if current is None or score > current["fitness"]:
                    archive[fingerprint] = {
                        "expr": expr, "metrics": metrics, "fitness": score,
                        "generation": gen + 1, "expr_fingerprint": fingerprint,
                    }

            if progress_cb:
                best = max(fitness) if fitness else -1.0
                progress_cb(gen + 1, generations, best)
            if stage_cb:
                stage_cb("generate", gen + 1, generations)

            order = np.argsort(fitness)[::-1]
            next_pop = [pop[i] for i in order[:elite]]
            while len(next_pop) < population:
                if rng.random() < crossover_prob:
                    p1 = tournament_select(pop, fitness, rng, tournament_k)
                    p2 = tournament_select(pop, fitness, rng, tournament_k)
                    c1, c2 = subtree_crossover(p1, p2, rng)
                    next_pop.append(c1 if _tree_depth(c1) <= max_depth else _clone(p1))
                    if len(next_pop) < population:
                        next_pop.append(c2 if _tree_depth(c2) <= max_depth else _clone(p2))
                else:
                    p = tournament_select(pop, fitness, rng, tournament_k)
                    mutated = subtree_mutation(p, rng, max_depth)
                    next_pop.append(mutated if _tree_depth(mutated) <= max_depth else _clone(p))
            pop = next_pop[:population]
            if checkpoint_cb:
                checkpoint_cb({
                    "version": 1,
                    "next_generation": gen + 1,
                    "population": pop,
                    "archive": list(archive.items()),
                    "rng_state": rng.bit_generator.state,
                })

        # 最终评估: 去重后取训练 IC 靠前的, 再在验证区间算样本外 IC
        exprs = [expr_from_tree(t) for t in pop]
        train_metrics = self._batch_ic(
            exprs, dates, progress_cb=ic_progress_cb, return_metrics=True)
        seen = dict(archive)
        for e, metrics in zip(exprs, train_metrics):
            fingerprint = expression_fingerprint(e)
            if not metrics.get("count") or fingerprint in excluded:
                continue
            score = compute_training_fitness(
                metrics, complexity=expression_complexity(e), max_complexity=max_depth * 10)
            current = seen.get(fingerprint)
            if current is None or score > current["fitness"]:
                seen[fingerprint] = {
                    "expr": e, "metrics": metrics, "fitness": score,
                    "generation": generations, "expr_fingerprint": fingerprint,
                }
        ranked = sorted(seen.values(), key=lambda item: item["fitness"], reverse=True)

        valid_dates = list(valid_dates) if valid_dates else []
        candidates = [item["expr"] for item in ranked[:top_n * 3]]
        valid_ics = self._batch_ic(candidates, valid_dates, progress_cb=ic_progress_cb) if valid_dates else [float("nan")] * len(candidates)

        result = []
        ranked_by_expr = {item["expr"]: item for item in ranked}
        for expr, valid_ic in zip(candidates, valid_ics):
            train_info = ranked_by_expr[expr]
            result.append({
                "expr": expr,
                "train_ic": train_info["metrics"].get("mean"),
                "train_metrics": train_info["metrics"],
                "train_fitness": train_info["fitness"],
                "valid_ic": valid_ic,
                "generation": train_info.get("generation"),
                "expr_fingerprint": train_info.get("expr_fingerprint"),
            })
        result.sort(
            key=lambda r: r["valid_ic"] if math.isfinite(r["valid_ic"]) else float("-inf"),
            reverse=True,
        )
        return result[:top_n]


def build_mining_context(screener, pool, dates, risk_neutralize=True, progress_cb=None):
    """预取遗传挖矿所需数据: 多日期 snapshot + 市值 + 风格因子截面。

    screener: BuySideScreener 实例 (依赖注入, 保持 L3 纯度不直接取数)
    risk_neutralize: True 时计算动量/波动/换手三因子截面 (五因子中性化用)
    progress_cb(done, total, dkey): 每日期预取完回调

    返回 (snapshots, mktcap_maps, extra_factors):
      snapshots:    {date_iso: {code: df}}
      mktcap_maps:  {date_iso: {code: total_mv}}
      extra_factors:{name: {date_iso: {code: value}}}
    """
    from core.domain.factors.market_cap import get_market_cap_map
    from core.domain.factors.expression import parse_factor, batch_eval_all

    snapshots = {}
    mktcap_maps = {}
    extra_factors = {"momentum": {}, "volatility": {}, "turnover": {}}
    risk_exprs = risk_asts = None
    if risk_neutralize:
        risk_exprs = {
            "momentum": "$close / Ref($close, 20) - 1",
            "volatility": "Std($close / Ref($close, 1) - 1, 20)",
            "turnover": "$turnover",
        }
        risk_asts = {name: parse_factor(e) for name, e in risk_exprs.items()}

    total = len(dates)
    for i, date in enumerate(dates):
        dkey = date.isoformat() if hasattr(date, "isoformat") else str(date)
        try:
            snapshot = screener.precompute_snapshot(
                pool=pool, eval_date=date, skip_limit_up=True, skip_tech=True, fast=True)
        except Exception:
            snapshot = None
        if snapshot:
            snapshots[dkey] = snapshot
            codes = list(snapshot.keys())
            mktcap_maps[dkey] = get_market_cap_map(codes, date)
            if risk_asts:
                try:
                    scores = batch_eval_all(risk_asts, snapshot)
                    for name in risk_exprs:
                        extra_factors[name][dkey] = scores.get(name, {})
                except Exception:
                    pass
        if progress_cb:
            progress_cb(i + 1, total, dkey)

    return snapshots, mktcap_maps, extra_factors if risk_neutralize else {}
