import numpy as np
import pytest

from core.domain.factors.genetic import (
    expr_from_tree,
    random_tree,
    subtree_mutation,
    subtree_crossover,
    tournament_select,
    GeneticFactorMiner,
)
from tests.factories import make_kline_df


def _parse_ok(expr):
    from core.domain.factors.expression import parse_factor
    try:
        parse_factor(expr)
        return True
    except Exception:
        return False


def test_if_operator_is_supported_as_elementwise_ternary():
    from core.domain.factors.expression import parse_factor

    ast, outer = parse_factor("If($close>1,$close,1)")

    assert ast[0] == "tern"
    assert outer is None


def _make_snapshots(n_dates=2, n_codes=8, n_rows=80):
    snapshots = {}
    for d in range(n_dates):
        date = f"2024-01-{(d + 1):02d}"
        snap = {}
        for c in range(n_codes):
            code = f"sh60{c:04d}"
            prices = 10.0 * (1.0 + np.linspace(0.0, 0.05 * (c + 1), n_rows))
            snap[code] = make_kline_df(prices=prices.tolist(), n_rows=n_rows)
        snapshots[date] = snap
    return snapshots


class _FakeForwardCache:
    def get_list(self, codes, date, days):
        return [(sum(ord(ch) for ch in c) % 100 - 50) / 1000.0 for c in codes]


class TestExprFromTree:
    def test_field_leaf(self):
        assert expr_from_tree("$close") == "$close"

    def test_binary(self):
        assert expr_from_tree(["add", "$close", "$open"]) == "($close + $open)"
        assert expr_from_tree(["sub", "$close", 1]) == "($close - 1)"

    def test_div_guards_zero(self):
        assert expr_from_tree(["div", "$close", "$open"]) == "($close / ($open + 1e-8))"

    def test_window_fn(self):
        assert expr_from_tree(["mean", "$close", 20]) == "Mean($close, 20)"
        assert expr_from_tree(["tsrank", "$close", 5]) == "Ts_Rank($close, 5)"

    def test_corr(self):
        assert expr_from_tree(["corr", "$close", "$volume", 20]) == "Corr($close, $volume, 20)"

    def test_nested(self):
        tree = ["mul", ["mean", "$close", 20], ["sub", "$volume", 1]]
        assert expr_from_tree(tree) == "(Mean($close, 20) * ($volume - 1))"

    def test_rank_root(self):
        assert expr_from_tree(["rank", ["std", "$close", 20]]) == "Rank(Std($close, 20))"


class TestRandomTree:
    def test_serializable_and_parseable(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            t = random_tree(rng, max_depth=4)
            expr = expr_from_tree(t)
            assert isinstance(expr, str)
            assert _parse_ok(expr)

    def test_rank_only_at_root(self):
        rng = np.random.default_rng(2)
        for _ in range(200):
            t = random_tree(rng, max_depth=4)
            if isinstance(t, list) and t[0] == "rank":
                assert len(t) == 2  # rank 单参数
            # 递归检查非根节点不含 rank
            stack = [t] if isinstance(t, list) else []
            while stack:
                node = stack.pop()
                for ch in node[1:]:
                    if isinstance(ch, list):
                        assert ch[0] != "rank"
                        stack.append(ch)


class TestGeneticOperators:
    def test_mutation_returns_valid_tree(self):
        rng = np.random.default_rng(3)
        t = ["mul", ["mean", "$close", 20], "$volume"]
        for _ in range(100):
            m = subtree_mutation(t, rng, max_depth=4)
            assert isinstance(m, (list, str))
            expr = expr_from_tree(m)
            assert isinstance(expr, str)

    def test_crossover_returns_valid_trees(self):
        rng = np.random.default_rng(4)
        a = ["mean", "$close", 20]
        b = ["corr", "$close", "$volume", 20]
        for _ in range(100):
            c1, c2 = subtree_crossover(a, b, rng)
            assert isinstance(expr_from_tree(c1), str)
            assert isinstance(expr_from_tree(c2), str)

    def test_mutation_root_can_replace_whole_tree(self):
        rng = np.random.default_rng(5)
        t = ["mean", "$close", 20]
        # 允许变异根节点; 只要不抛异常即可
        for _ in range(50):
            subtree_mutation(t, rng, max_depth=3)


class TestTournament:
    def test_picks_best(self):
        rng = np.random.default_rng(6)
        pop = ["a", "b", "c", "d"]
        fitness = [0.1, 0.9, 0.5, 0.2]
        # 反复选择, 最高适应度者应出现
        picked = [tournament_select(pop, fitness, rng, k=3) for _ in range(100)]
        assert "b" in picked


class TestNeutralizeExtra:
    def test_extra_factor_explains_scores(self):
        from core.domain.factors.neutralize import cross_sectional_neutralize
        codes = [f"c{i}" for i in range(30)]
        scores = [float(i) for i in range(30)]
        extra = {"mom": {c: float(i) for i, c in enumerate(codes)}}
        resid = cross_sectional_neutralize(codes, scores, {}, {}, extra)
        assert max(abs(r) for r in resid.values()) < 1e-6

    def test_backward_compatible_no_extra(self):
        from core.domain.factors.neutralize import cross_sectional_neutralize
        codes = [f"c{i}" for i in range(30)]
        scores = [float(i) for i in range(30)]
        # extra_factors 缺省(None)与显式传 None 结果一致 (向后兼容)
        a = cross_sectional_neutralize(codes, scores, {}, {})
        b = cross_sectional_neutralize(codes, scores, {}, {}, None)
        assert a == b


class TestBatchIc:
    def test_returns_finite_values(self, monkeypatch):
        from core.domain.factors import forward_returns as fr
        monkeypatch.setattr(fr, "get_forward_cache", lambda: _FakeForwardCache())
        miner = GeneticFactorMiner(_make_snapshots(), forward_days=20)
        exprs = ["Mean($close, 20)", "$close / $open - 1", "Corr($close, $volume, 20)"]
        ics = miner._batch_ic(exprs, ["2024-01-01", "2024-01-02"])
        assert len(ics) == 3
        assert all(isinstance(v, float) for v in ics)

    def test_unparseable_expr_is_nan(self, monkeypatch):
        from core.domain.factors import forward_returns as fr
        monkeypatch.setattr(fr, "get_forward_cache", lambda: _FakeForwardCache())
        miner = GeneticFactorMiner(_make_snapshots())
        ics = miner._batch_ic(["Rank(Rank($close))"], ["2024-01-01"])
        assert np.isnan(ics[0])


class TestMine:
    def test_runs_end_to_end(self, monkeypatch):
        from core.domain.factors import forward_returns as fr
        monkeypatch.setattr(fr, "get_forward_cache", lambda: _FakeForwardCache())
        miner = GeneticFactorMiner(_make_snapshots(), forward_days=20)
        result = miner.mine(
            population=20, generations=3, top_n=5,
            train_dates=["2024-01-01"], valid_dates=["2024-01-02"],
            seed=7,
        )
        assert isinstance(result, list)
        assert len(result) <= 5
        for r in result:
            assert "expr" in r and "train_ic" in r and "valid_ic" in r
            assert "train_fitness" in r and "train_metrics" in r
            assert "generation" in r and "expr_fingerprint" in r
            assert _parse_ok(r["expr"])

    def test_excluded_fingerprint_is_not_returned(self, monkeypatch):
        from core.domain.factors.archive import expression_fingerprint
        miner = GeneticFactorMiner(_make_snapshots(), forward_days=20)
        def fake_batch(exprs, dates, progress_cb=None, return_metrics=False):
            if return_metrics:
                return [{
                    "mean": 0.05, "icir": 0.5, "positive_ratio": 0.7,
                    "coverage": 1.0, "count": len(dates),
                } for _ in exprs]
            return [0.05 for _ in exprs]
        monkeypatch.setattr(miner, "_batch_ic", fake_batch)
        first = miner.mine(
            population=20, generations=2, top_n=5,
            train_dates=["2024-01-01"], valid_dates=["2024-01-02"], seed=11,
        )
        assert first
        excluded = {expression_fingerprint(first[0]["expr"])}
        second = miner.mine(
            population=20, generations=2, top_n=20,
            train_dates=["2024-01-01"], valid_dates=["2024-01-02"], seed=11,
            excluded_fingerprints=excluded,
        )
        assert all(expression_fingerprint(item["expr"]) not in excluded for item in second)
