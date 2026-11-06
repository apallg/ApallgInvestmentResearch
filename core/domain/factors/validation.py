"""
五层反过拟合校验器 — 按开销从小到大排序，逐层早剪枝淘汰伪因子

执行顺序 (开销升序):
  L5: 去冗余 (命名冲突检查, 纯内存O(1))
  L1: Walk-Forward (三期IC, 3次screen)
  L4: 参数鲁棒性 (窗口±2, 3次screen)
  L7: 置换检验 (Shuffle重采样, 50+次screen, 仅硬门全过才执行)
  L8: 边际贡献 (增量IC贡献, 1次screen)

早剪枝规则:
  - 硬门(L5/L1)不通过 → 后续全部跳过, 不再浪费算力
  - mode="fast": 仅跑 L5→L1 (低成本二层), 批量海选用
  - mode="full": 完整五层 (默认)
  - L2(多行情)/L3(IC衰减)/L6(换手率) 已删除: 与L1冗余, IC≈0时连锁失败
"""

__all__ = ["OverfittingChecker", "PackedExpressionScores"]

from collections.abc import Mapping
import logging
import time
import numpy as np
import pandas as pd
import datetime

from core.shared.persistence import today_date

logger = logging.getLogger(__name__)

LAYER_DEFS = [
    {"id": 5, "key": "deredundancy", "name": "去冗余",        "cost": "O(1)"},
    {"id": 1, "key": "walkforward",  "name": "Walk-Forward",  "cost": "低"},
    {"id": 4, "key": "param_robust", "name": "参数鲁棒性",     "cost": "高"},
    {"id": 7, "key": "shuffle",      "name": "置换检验",       "cost": "极高"},
    {"id": 8, "key": "marginal",     "name": "边际贡献",       "cost": "中"},
]


class PackedExpressionScores(Mapping):
    def __init__(self, signatures, offsets, codes, code_indices, scores):
        self._signatures = tuple(signatures)
        self._index = {key: index for index, key in enumerate(self._signatures)}
        self._offsets = np.asarray(offsets, dtype=np.int64)
        self._codes = np.asarray(codes, dtype=object)
        self._code_indices = np.asarray(code_indices)
        self._scores = np.asarray(scores, dtype=np.float64)
        self._overlay = {}

    @classmethod
    def from_mapping(cls, values):
        if isinstance(values, cls):
            return values
        signatures = []
        offsets = [0]
        code_table = []
        code_positions = {}
        index_parts = []
        score_parts = []
        for signature, pair in (values or {}).items():
            codes, scores = pair
            codes = list(codes)
            score_array = np.asarray(scores, dtype=np.float64)
            if len(codes) != len(score_array):
                raise ValueError(f"{signature} codes/scores 长度不一致")
            positions = []
            for code in codes:
                code = str(code)
                position = code_positions.get(code)
                if position is None:
                    position = len(code_table)
                    code_positions[code] = position
                    code_table.append(code)
                positions.append(position)
            signatures.append(signature)
            index_parts.append(np.asarray(positions, dtype=np.uint32))
            score_parts.append(score_array)
            offsets.append(offsets[-1] + len(score_array))
        code_dtype = np.uint16 if len(code_table) <= np.iinfo(np.uint16).max else np.uint32
        code_indices = (
            np.concatenate(index_parts).astype(code_dtype, copy=False)
            if index_parts else np.empty(0, dtype=code_dtype)
        )
        packed_scores = (
            np.concatenate(score_parts).astype(np.float64, copy=False)
            if score_parts else np.empty(0, dtype=np.float64)
        )
        return cls(signatures, offsets, code_table, code_indices, packed_scores)

    def __getitem__(self, signature):
        if signature in self._overlay:
            return self._overlay[signature]
        index = self._index[signature]
        start = int(self._offsets[index])
        end = int(self._offsets[index + 1])
        return self._codes[self._code_indices[start:end]], self._scores[start:end]

    def __setitem__(self, signature, value):
        self._overlay[signature] = value

    def __iter__(self):
        yield from self._signatures
        for signature in self._overlay:
            if signature not in self._index:
                yield signature

    def __len__(self):
        return len(self._signatures) + sum(
            1 for signature in self._overlay if signature not in self._index
        )

    @property
    def memory_bytes(self):
        return (
            self._offsets.nbytes + self._codes.nbytes
            + self._code_indices.nbytes + self._scores.nbytes
        )

# 快速模式止步于此层 (id)
FAST_MODE_STOP_AT = 1

# L5 相关性阈值 (聚类去重复用)
L5_CORR_THRESHOLD = 0.75


class OverfittingChecker:

    def __init__(self):
        self._screen_cache = {}
        self._neutral_cache = {}
        self._resid_cache = {}
        self._industry_map = None
        self._injected_scores = None
        self._expr_variants = None  # {iso_sig: (inner_ast, outer_op)} 批量预计算用
        self._date_expr_scores = {}  # {date_key: {iso_sig: (codes, scores)}}
        self._td_cache = {}  # {(start_iso, end_iso): [date]} 交易日缓存
        self._eval_dates_cache = {}  # {(weeks, step, anchor_iso): [date]} 评估日期缓存
        self._dates_between_cache = {}  # {(start_iso, end_iso, step): [date]} 区间日期缓存

    def _get_industry_map(self):
        if self._industry_map is None:
            try:
                from core.data.fetcher import DataFetcher
                self._industry_map = DataFetcher().get_industry_map() or {}
            except Exception:
                self._industry_map = {}
        return self._industry_map

    def set_expr_variants(self, variants):
        self._expr_variants = {}
        for v in variants:
            iso, iso_sig, use_expr, inner_ast, outer_op = self._resolve_eval(v, quiet=True)
            if use_expr:
                self._expr_variants[iso_sig] = (inner_ast, outer_op)

    def _ensure_date_expr_scores(self, date_key, eval_date, pool, code_prefixes, precomputed, screener,
                                 preloaded_scores=None):
        if date_key in self._date_expr_scores:
            return
        # 如果提供了预计算结果, 直接注入 — 跳过 batch_eval_all
        if preloaded_scores is not None and date_key in preloaded_scores:
            self._date_expr_scores[date_key] = preloaded_scores[date_key]
            return
        # preloaded 模式下日期缺失 → 直接置空, 不回退到实时计算 (调用方已明确表示用预计算数据)
        if preloaded_scores is not None:
            self._date_expr_scores[date_key] = {}
            return
        snapshot = self._snapshot_for(precomputed, date_key, eval_date, pool, code_prefixes, screener)
        if snapshot is None or not self._expr_variants:
            self._date_expr_scores[date_key] = {}
            return
        from core.domain.factors.expression import batch_eval_all
        raw = batch_eval_all(self._expr_variants, snapshot)
        date_scores = {}
        for iso_sig, vals in raw.items():
            if len(vals) >= 20:
                codes = list(vals)
                scores = [vals[c] for c in codes]
                date_scores[iso_sig] = (codes, scores)
        self._date_expr_scores[date_key] = date_scores

    def save_precomputed_scores(self, dirpath):
        """将 _date_expr_scores 按日期存为独立 pickle 文件, 供子进程按需加载。"""
        import os
        from core.shared.persistence import save_pickle
        os.makedirs(dirpath, exist_ok=True)
        count = 0
        for dk, scores in self._date_expr_scores.items():
            if not scores:
                continue
            fname = os.path.join(dirpath, f"{dk}.pkl")
            save_pickle(scores, fname)
            count += 1
        logger.info(f"[Precompute] 写入 {count} 个日期因子面板 → {dirpath}")
        return count

    @staticmethod
    def load_precomputed_scores(dirpath, date_keys):
        """从磁盘加载指定日期的因子面板到 dict。返回 {date_key: {iso_sig: (codes, scores)}}。"""
        import os
        from core.shared.persistence import load_pickle
        loaded = {}
        for dk in date_keys:
            fname = os.path.join(dirpath, f"{dk}.pkl")
            if os.path.exists(fname):
                data = load_pickle(fname)
                if data is not None:
                    loaded[dk] = data
        return loaded

    # ══════════════════════════════════════════════
    # 主入口
    # ══════════════════════════════════════════════

    def check_all_layers(self, variant, eval_weeks=18, pool="all", top_k=10, code_prefixes=None,
                          progress_cb=None, n_shuffles=500, skip_layers=None, snapshots=None,
                          mode="full", anchor_date=None, l1_dates=None, eval_dates=None,
                          label_horizon=20, max_eval_dates=8):
        """主入口。mode: "fast"=仅低成本二层, "full"=完整五层 (默认)。
        anchor_date: 时序滚动锚点, 只用该日及之前的数据 (None=今天)。

        硬门: L5(去冗余)/L1(Walk-Forward) — 不通过直接淘汰, 后续全跳过。
        软门: L4(参数鲁棒)/L7(置换检验)/L8(边际贡献) — 记录结果, 不淘汰。
        L7 闸门: L5+L1 未全过 → 跳过置换检验。
        """
        results = []
        self._last_shuffle = None
        self._last_deredundancy = None
        self._last_walkforward = None
        self._last_daily_ics = []
        context = {"variant": variant, "eval_weeks": eval_weeks, "pool": pool, "top_k": top_k,
                   "code_prefixes": code_prefixes, "n_shuffles": n_shuffles, "snapshots": snapshots,
                   "anchor_date": anchor_date, "l1_dates": l1_dates,
                   "eval_dates": list(eval_dates or []),
                   "label_horizon": int(label_horizon),
                   "max_eval_dates": max_eval_dates}
        skip_set = set(skip_layers or [])
        fast_stop_ids = {1, 5}  # L5+L1 = 低成本二层

        early_terminated = False
        hard_gate_ids = {5, 1}  # 硬门: 不通过即淘汰

        self._timing_counter = getattr(self, '_timing_counter', 0) + 1
        t0 = time.time()
        layer_times = {}
        for layer_def in LAYER_DEFS:
            lid = layer_def["id"]
            t_layer = time.time()

            # 快速模式: 只跑低成本二层
            if mode == "fast" and lid not in fast_stop_ids:
                results.append({**layer_def, "pass": True, "status": "skipped", "score": 1.0, "detail": "已跳过(快速模式)"})
                if progress_cb:
                    progress_cb({"current": lid, "total": len(LAYER_DEFS), "name": layer_def["name"], "layers": results})
                continue

            # L7/L8 闸门: 硬门未全过 → 跳过置换检验/边际贡献
            if lid in (7, 8):
                hard_pass = all(r["pass"] for r in results if r["id"] in hard_gate_ids)
                if not hard_pass:
                    results.append({**layer_def, "pass": True, "status": "skipped", "score": 0.5, "detail": "已跳过(硬门未全过)"})
                    if progress_cb:
                        progress_cb({"current": lid, "total": len(LAYER_DEFS), "name": layer_def["name"], "layers": results})
                    continue

            # 早剪枝: 硬门失败 → 后续全跳过
            if early_terminated:
                results.append({**layer_def, "pass": True, "status": "skipped", "score": 0.0, "detail": "已跳过(硬门已淘汰)"})
                if progress_cb:
                    progress_cb({"current": lid, "total": len(LAYER_DEFS), "name": layer_def["name"], "layers": results})
                continue

            if lid in skip_set:
                results.append({**layer_def, "pass": True, "status": "skipped", "score": 1.0, "detail": "已跳过(指定跳过)"})
                if progress_cb:
                    progress_cb({"current": lid, "total": len(LAYER_DEFS), "name": layer_def["name"], "layers": results})
                continue

            try:
                fn = getattr(self, f"_check_{layer_def['key']}", None)
                if fn is None:
                    results.append({**layer_def, "pass": True, "status": "skipped", "score": 1.0, "detail": "跳过(未实现)"})
                else:
                    passed, score, detail = fn(context)
                    layer_times[lid] = time.time() - t_layer
                    results.append({**layer_def, "pass": passed, "status": "passed" if passed else "failed", "score": score, "detail": detail})
                    if not passed and lid in hard_gate_ids:
                        early_terminated = True
                        logger.info(f"[OverfittingCheck] L{lid} 淘汰(硬门), 后续层跳过 (因子: {variant.get('variant_key', '?')})")
                    elif not passed:
                        logger.info(f"[OverfittingCheck] L{lid} 未通过(软门, 不淘汰) (因子: {variant.get('variant_key', '?')})")
            except Exception as e:
                logger.warning(f"[OverfittingCheck] L{lid} 异常: {e}")
                results.append({**layer_def, "pass": False, "status": "failed", "score": 0.0, "detail": str(e)})
                if lid in hard_gate_ids:
                    early_terminated = True

            if progress_cb:
                progress_cb({"current": lid, "total": len(LAYER_DEFS), "name": layer_def["name"], "layers": results})

        total_t = time.time() - t0
        if self._timing_counter % 3 == 0:
            key = variant.get('variant_key') or variant.get('template_key', '?')
            parts = [f"{lid}={layer_times.get(lid, 0):.1f}s" for lid in [5, 1, 4, 7, 8] if lid in layer_times]
            logger.info(f"[Timing][#{self._timing_counter}] {key}: total={total_t:.1f}s {' | '.join(parts)}")

        passed_count = sum(1 for r in results if r.get("status") == "passed")
        hard_pass_count = sum(
            1 for r in results
            if r["id"] in hard_gate_ids and (
                r.get("status") == "passed"
                or (r.get("status") == "skipped" and r["id"] in skip_set)
            )
        )
        soft_detail = {r["id"]: {"pass": r["pass"], "status": r.get("status"), "detail": r.get("detail", "")}
                       for r in results if r["id"] not in hard_gate_ids}
        all_pass = hard_pass_count == len(hard_gate_ids)
        # 因子方向
        direction = "long"
        for r in results:
            if r["id"] == 1 and "反向" in str(r.get("detail", "")):
                direction = "short"
                break
        # 附加: L1 测试段逐日IC + IC衰减曲线 (共用 test_dates, 缓存命中≈免费)
        daily_ics = list(getattr(self, "_last_daily_ics", None) or [])
        ic_decay = None
        test_dates = None
        if anchor_date is not None and mode == "full":
            try:
                test_dates = list((l1_dates or {}).get("test") or [])
                if not test_dates:
                    import datetime as dt
                    test_dates = self._dates_between(anchor_date, anchor_date + dt.timedelta(days=90), step=5)
                if len(test_dates) >= 3 and daily_ics:
                    ic_decay = {}
                    for fwd in (1, 5, 10, 20):
                        ic_decay[str(fwd)] = round(
                            self._calc_rank_ic(variant, test_dates, context, forward_days=fwd), 4,
                        )
            except Exception as exc:
                logger.warning("IC衰减计算失败: %s", exc)

        # 附加: L7 置换检验结构化统计 (从 _check_shuffle 的 side-effect 读取)
        shuffle_stats = getattr(self, "_last_shuffle", None)

        correlation = getattr(self, "_last_deredundancy", None) or {}
        max_corr = correlation.get("max_corr")
        max_corr_with = correlation.get("max_corr_with")

        return {
            "layers": results,
            "all_pass": all_pass,
            "passed_count": passed_count,
            "hard_pass_count": hard_pass_count,
            "soft_detail": soft_detail,
            "total_layers": len(results),
            "mode": mode,
            "direction": direction,
            "early_terminated": early_terminated,
            "daily_ics": daily_ics,
            "ic_decay": ic_decay,
            "walkforward_metrics": getattr(self, "_last_walkforward", None),
            "shuffle_stats": shuffle_stats,
            "max_corr": max_corr,
            "max_corr_with": max_corr_with,
            "correlation_evidence": correlation,
        }

    # ══════════════════════════════════════════════
    # L1: 滚动Walk-Forward
    # ══════════════════════════════════════════════

    def _check_walkforward(self, ctx):
        """训练/验证/测试三期IC方向一致性。

        anchor_date 模式 (时序滚动, 严格样本外):
          训练窗 [anchor-9月, anchor-3月] → 验证窗 [anchor-3月, anchor] → 测试窗 [anchor, anchor+3月]
          测试段用 anchor 之后的 forward return, 真正外推。
        无 anchor 回退: 最近窗口 n//3 三等分 (旧逻辑)。

        要求: |测试段IC|>0.02。不要求三段方向一致(窗口短IC噪声大)。
        """
        variant = ctx["variant"]
        eval_weeks = ctx["eval_weeks"]
        anchor = ctx.get("anchor_date")
        self._last_daily_ics = []

        try:
            test_daily_ics = []
            supplied = ctx.get("l1_dates") or {}
            if supplied:
                train_dates = list(supplied.get("train") or [])
                valid_dates = list(supplied.get("valid") or [])
                test_dates = list(supplied.get("test") or [])
                if min(len(train_dates), len(valid_dates), len(test_dates)) < 3:
                    return False, 0.0, f"滚动窗数据点不足(train={len(train_dates)},valid={len(valid_dates)},test={len(test_dates)})"
                train_ic = self._calc_rank_ic(variant, train_dates, ctx)
                valid_ic = self._calc_rank_ic(variant, valid_dates, ctx)
                test_ic, test_daily_ics = self._calc_rank_ic(
                    variant, test_dates, ctx, return_series=True, return_dates=True,
                )
            elif anchor is not None:
                import datetime as dt
                from core.domain.factors.walkforward import purge_dates_before_boundary
                calendar_dates = self._dates_between(
                    anchor - dt.timedelta(days=270), anchor + dt.timedelta(days=90), step=1)
                train_dates = self._dates_between(anchor - dt.timedelta(days=270),
                                                  anchor - dt.timedelta(days=90), step=5)
                valid_dates = self._dates_between(anchor - dt.timedelta(days=90), anchor, step=5)
                test_dates = self._dates_between(anchor, anchor + dt.timedelta(days=90), step=5)
                train_dates = purge_dates_before_boundary(
                    train_dates, anchor - dt.timedelta(days=90), calendar_dates, 21)
                valid_dates = purge_dates_before_boundary(
                    valid_dates, anchor, calendar_dates, 21)
                if min(len(train_dates), len(valid_dates), len(test_dates)) < 3:
                    return False, 0.0, f"滚动窗数据点不足(train={len(train_dates)},valid={len(valid_dates)},test={len(test_dates)})"
                train_ic = self._calc_rank_ic(variant, train_dates, ctx)
                valid_ic = self._calc_rank_ic(variant, valid_dates, ctx)
                test_ic, test_daily_ics = self._calc_rank_ic(
                    variant, test_dates, ctx, return_series=True, return_dates=True,
                )
            else:
                eval_dates = self._get_eval_dates(weeks=eval_weeks, step=5, anchor_date=None)
                if len(eval_dates) < 9:
                    return True, 0.5, f"数据点不足({len(eval_dates)}), 跳过WF"
                n = len(eval_dates)
                train_end = n // 3
                valid_end = n * 2 // 3
                train_ic = self._calc_rank_ic(variant, eval_dates[:train_end], ctx)
                valid_ic = self._calc_rank_ic(variant, eval_dates[train_end:valid_end], ctx)
                test_dates = eval_dates[valid_end:]
                test_ic, test_daily_ics = self._calc_rank_ic(
                    variant, test_dates, ctx, return_series=True, return_dates=True,
                )

            direction_sign = -1 if train_ic < 0 else 1
            direction = "反向(short)" if direction_sign < 0 else "正向(long)"
            oriented_valid_ic = valid_ic * direction_sign
            oriented_test_ic = test_ic * direction_sign
            test_valid = oriented_test_ic > 0.02

            passed = test_valid
            score = min(1.0, max(0.0, (abs(train_ic) + oriented_valid_ic + oriented_test_ic) / 3 / 0.05))
            self._last_daily_ics = test_daily_ics
            self._last_walkforward = {
                "train_ic": float(train_ic),
                "valid_ic": float(valid_ic),
                "test_ic": float(test_ic),
                "direction_sign": direction_sign,
                "oriented_valid_ic": float(oriented_valid_ic),
                "oriented_test_ic": float(oriented_test_ic),
                "passed": bool(passed),
                "train_count": len(train_dates),
                "valid_count": len(valid_dates),
                "test_count": len(test_dates),
            }

            return passed, score, (
                f"Train IC={train_ic:.4f}, Valid IC={valid_ic:.4f}, Test IC={test_ic:.4f} | "
                f"方向={direction}, 定向Valid IC={oriented_valid_ic:.4f}, "
                f"定向Test IC={oriented_test_ic:.4f}, 样本外达标={test_valid}"
            )
        except Exception as e:
            self._last_daily_ics = []
            self._last_walkforward = {"error": str(e), "passed": False}
            return False, 0.0, f"L1异常: {e}"

    # ══════════════════════════════════════════════
    # L2: 多行情分段
    # ══════════════════════════════════════════════

    def _check_regime(self, ctx):
        """四状态(趋势涨/跌/震荡/高波动)分别计算IC。

        要求: 至少3/4状态IC>0, 且最差状态IC>-0.01
        """
        variant = ctx["variant"]
        eval_weeks = ctx["eval_weeks"]

        try:
            eval_dates = self._get_eval_dates(weeks=eval_weeks, step=5, anchor_date=ctx.get("anchor_date"))
            if len(eval_dates) < 12:
                return True, 0.5, f"数据点不足({len(eval_dates)}), 跳过市态分段"

            # 按市场状态分组
            regime_dates = self._classify_regimes(eval_dates)
            regime_ics = {}
            regime_hit_rates = {}

            for regime, dates in regime_dates.items():
                if len(dates) < 3:
                    regime_ics[regime] = None
                    continue
                ic = self._calc_rank_ic(variant, dates, ctx)
                hr = self._calc_hit_rate(variant, dates, ctx)
                regime_ics[regime] = ic
                regime_hit_rates[regime] = hr

            valid_regimes = {k: v for k, v in regime_ics.items() if v is not None}
            if not valid_regimes:
                return True, 0.5, "无足够市态数据"

            positive_regimes = sum(1 for ic in valid_regimes.values() if ic > 0)
            negative_regimes = sum(1 for ic in valid_regimes.values() if ic < 0)
            # 主方向: 多数状态 IC 的符号 (双向: 反向因子的主方向为负)
            dir_sign = 1 if positive_regimes >= negative_regimes else -1
            aligned_regimes = sum(1 for ic in valid_regimes.values() if (1 if ic > 0 else -1) == dir_sign)
            # 最差状态在主方向上的表现 (主方向为负时, min(ic*dir) 检查最反向的偏离)
            worst_aligned = min(ic * dir_sign for ic in valid_regimes.values())

            # 多数状态与主方向一致 (门槛随有效市态数缩放, 避免样本少时硬门槛3永远不达标),
            # 且最差状态在主方向上不显著反向
            nvr = len(valid_regimes)
            threshold = nvr if nvr <= 2 else nvr - 1  # 1-2个:需全同向; 3+:允许1个背离
            passed = aligned_regimes >= threshold and worst_aligned > -0.01
            detail_parts = [f"{r} IC={ic:.4f}" for r, ic in valid_regimes.items()]
            detail_parts.append(f"主方向={'正' if dir_sign > 0 else '反'}, 同向={aligned_regimes}/{len(valid_regimes)}, 最差同向IC={worst_aligned:.4f}")
            score = min(1.0, max(0.0, (aligned_regimes / max(len(valid_regimes), 1)) * 0.8 + worst_aligned * 10))

            return passed, score, " | ".join(detail_parts)
        except Exception as e:
            return False, 0.0, f"L2异常: {e}"

    # ══════════════════════════════════════════════
    # L4: 参数鲁棒性
    # ══════════════════════════════════════════════

    def _check_param_robust(self, ctx):
        """窗口±2变化后IC波动<30%。

        对主要参数n, 测试n-2和n+2的IC是否稳定。
        """
        variant = ctx["variant"]
        params = variant.get("params", {})

        n = params.get("n")
        expr_mode = False
        if n is None and variant.get("expr"):
            from core.domain.factors.candidate import expression_windows
            windows = expression_windows(variant["expr"])
            if windows:
                n = windows[0]
                expr_mode = True
        if n is None:
            return True, 0.8, "无窗口参数, 跳过鲁棒性检验"

        try:
            eval_dates = self._evidence_dates(ctx)

            base_variant = dict(variant)
            base_ic = self._calc_rank_ic(base_variant, eval_dates, ctx)

            ics = [base_ic]

            for delta in [-2, +2]:
                test_n = n + delta
                if test_n < 3:
                    continue
                test_variant = dict(variant)
                if expr_mode:
                    from core.domain.factors.candidate import replace_expression_window
                    test_variant["expr"] = replace_expression_window(variant["expr"], n, test_n)
                else:
                    test_params = dict(params)
                    test_params["n"] = test_n
                    test_variant["params"] = test_params
                test_ic = self._calc_rank_ic(test_variant, eval_dates, ctx)
                ics.append(test_ic)

            ic_mean = np.mean(ics)
            ic_std = np.std(ics)
            cv = ic_std / max(abs(ic_mean), 0.001)

            passed = bool(cv < 0.25)
            detail = (f"n={n} IC={base_ic:.4f}, n-2 IC={ics[1] if len(ics)>1 else 'N/A'}, "
                      f"n+2 IC={ics[2] if len(ics)>2 else 'N/A'} | CV={cv:.2%}")

            score = max(0.0, 1.0 - cv)

            return passed, score, detail
        except Exception as e:
            return False, 0.0, f"L4异常: {e}"

    # ══════════════════════════════════════════════
    # L5: 去冗余
    # ══════════════════════════════════════════════

    def _check_deredundancy(self, ctx):
        """相关性去冗余: 候选因子与池内活跃因子在同一截面上的 Pearson 相关性 > 0.8 → 淘汰。
        仅覆盖能映射到 screen 子分项的预定义大类因子; 无法映射(如 white_box 模板变体)时回退命名冲突检查。"""
        try:
            from core.domain.factors.registry import get_pool
            pool = get_pool()
            active = pool.get_active_factors()

            if not active:
                self._last_deredundancy = {
                    "max_corr": 0.0, "max_corr_with": None, "compared_count": 0,
                }
                return True, 1.0, "因子池为空, 无冗余风险"

            variant = ctx["variant"]
            template_key = variant.get("template_key", "")
            variant_key = variant.get("variant_key", "")

            # 排除候选自身: 因子已在池中(如复检、滚动写回后重跑)时不与自己比, 否则误判同名冲突
            self_keys = {variant_key, template_key}
            active = {k: v for k, v in active.items() if k not in self_keys}
            if not active:
                self._last_deredundancy = {
                    "max_corr": 0.0, "max_corr_with": None, "compared_count": 0,
                }
                return True, 1.0, "因子池中仅候选自身, 无冗余风险"

            FIELD = {
                "tech": "tech_score", "momentum": "momentum_score", "volatility": "volatility_score",
                "size_turnover": "size_turnover_score", "fundamental": "fundamental_score",
                "sentiment": "sentiment_score", "flow": "flow_score", "qlib": "qlib_score",
                "north_flow": "north_flow_score",
                # 白盒因子 → 映射到对应 L0 子分项 (通过 screen 的 source_weights 隔离)
                "wb_industry": "wb_industry_score",
                "wb_divergence": "wb_divergence_score",
                "wb_moneyflow": "wb_moneyflow_score",
                "wb_ind_residual": "wb_industry_score",
                "wb_turnover_price": "wb_divergence_score",
                "wb_vol_regime": "wb_industry_score",
                "wb_vp_divergence": "wb_divergence_score",
                "wb_capital_anomaly": "wb_moneyflow_score",
                "wb_smart_money": "wb_moneyflow_score",
                "wb_earnings_mo": "fundamental_score",
                "wb_peg_adj": "fundamental_score",
            }

            def norm(k):
                if not k:
                    return ""
                if k in FIELD:
                    return k
                # 白盒变体: wb_xxx_n5 → 映射到 wb_xxx 基类
                if k.startswith("wb_"):
                    parts = k.rsplit("_n", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        return norm(parts[0])
                return k

            # 命名冲突 → 直接淘汰 (候选 key/template_key 与池因子 key 完全同名, 已排除自指)
            conflicts = [key for key in active
                         if (template_key and template_key == key) or (variant_key and variant_key == key)]
            if conflicts:
                return False, 0.0, f"因子池中已存在同名/同模板: {conflicts[:3]}"

            cand_key = norm(variant_key) or norm(template_key)
            if cand_key not in FIELD:
                snapshots = ctx.get("snapshots")
                keys = sorted(str(key) for key in snapshots) if snapshots else []
                anchor = ctx.get("anchor_date")
                if anchor is not None:
                    anchor_key = anchor.isoformat() if hasattr(anchor, "isoformat") else str(anchor)[:10]
                    keys = [key for key in keys if key <= anchor_key]
                if not keys:
                    return False, 0.0, "候选表达式没有可用的相关性截面"
                date_key = keys[-1]
                import datetime as dt
                eval_date = dt.date.fromisoformat(date_key[:10])
                from core.domain.screening._screener import get_screener
                screener = get_screener()
                candidate_scores = self._variant_scores(
                    variant, eval_date, date_key, ctx.get("pool", "all"),
                    ctx.get("code_prefixes"), snapshots, screener, allow_composite=False,
                )
                if candidate_scores is None or len(candidate_scores[0]) < 20:
                    return False, 0.0, "候选表达式截面数据不足，无法完成相关性检查"
                candidate_map = {
                    code: float(score)
                    for code, score in zip(candidate_scores[0], candidate_scores[1])
                }
                references = {}
                result = screener.screen(
                    pool=ctx.get("pool", "all"), top_n=100000,
                    skip_l1=True, skip_l2=True, skip_debate=True,
                    skip_qlib=True, skip_sent=True, skip_fundamental=True,
                    skip_flow=True, eval_date=eval_date,
                    snapshot=snapshots.get(date_key),
                )
                rows = result.get("candidates", [])
                row_maps = {
                    field: {
                        row["code"]: float(row.get(field, 0) or 0)
                        for row in rows if row.get("code")
                    }
                    for field in set(FIELD.values())
                }
                for key, info in active.items():
                    pool_key = norm(info.get("template_key") or key)
                    if pool_key in FIELD:
                        references[key] = row_maps[FIELD[pool_key]]
                        continue
                    if not info.get("expr"):
                        continue
                    pool_variant = self._pool_entry_to_variant(key, info)
                    scores = self._variant_scores(
                        pool_variant, eval_date, date_key, ctx.get("pool", "all"),
                        ctx.get("code_prefixes"), snapshots, screener,
                        allow_composite=False,
                    )
                    if scores is not None and len(scores[0]) >= 20:
                        references[key] = {
                            code: float(score)
                            for code, score in zip(scores[0], scores[1])
                        }
                from core.domain.factors.candidate import max_abs_correlation
                max_corr, worst, compared = max_abs_correlation(candidate_map, references)
                self._last_deredundancy = {
                    "max_corr": max_corr,
                    "max_corr_with": worst,
                    "compared_count": compared,
                    "date": date_key,
                }
                if max_corr is None:
                    return False, 0.0, "没有可比较的在池因子截面，L5 证据不足"
                if max_corr > L5_CORR_THRESHOLD:
                    return False, 0.0, f"与因子池 {worst} 相关性 {max_corr:.2f} > {L5_CORR_THRESHOLD}, 冗余"
                return True, round(1.0 - max_corr, 3), f"与{compared}个可求值在池因子最大相关性 {max_corr:.2f} ({worst})"

            candidates = self._representative_candidates(ctx)
            if not candidates or len(candidates) < 20:
                return False, 0.0, "无足够截面数据，无法完成相关性检查"

            cand_vec = np.array([float(c.get(FIELD[cand_key], 0) or 0) for c in candidates])
            if np.std(cand_vec) < 1e-9:
                return False, 0.0, "候选因子子分项无区分度"

            max_corr, worst = 0.0, None
            for key in active:
                pk = norm(key)
                if pk == cand_key or pk not in FIELD:
                    continue
                vec = np.array([float(c.get(FIELD[pk], 0) or 0) for c in candidates])
                if np.std(vec) < 1e-9:
                    continue
                r = abs(float(np.corrcoef(cand_vec, vec)[0, 1]))
                if not np.isnan(r) and r > max_corr:
                    max_corr, worst = r, key

            if worst is not None and max_corr > L5_CORR_THRESHOLD:
                self._last_deredundancy = {
                    "max_corr": max_corr, "max_corr_with": worst,
                    "compared_count": len(active),
                }
                return False, 0.0, f"与因子池 {worst} 相关性 {max_corr:.2f} > {L5_CORR_THRESHOLD}, 冗余"
            self._last_deredundancy = {
                "max_corr": max_corr, "max_corr_with": worst,
                "compared_count": len(active),
            }
            return True, round(1.0 - max_corr, 3), f"与{len(active)}个活跃因子最大相关性 {max_corr:.2f}"
        except Exception as e:
            return False, 0.0, f"L5异常: {e}"

    def _representative_candidates(self, ctx):
        """取一个代表性 eval_date 的全截面 screen candidates (含各子分项), 复用 _screen_cache。"""
        try:
            from core.domain.screening._screener import get_screener
            pool = ctx.get("pool", "all")
            code_prefixes = ctx.get("code_prefixes")
            precomputed = ctx.get("snapshots")
            eval_dates = self._get_eval_dates(weeks=ctx.get("eval_weeks", 12), anchor_date=ctx.get("anchor_date"))
            if not eval_dates:
                return []
            eval_date = eval_dates[0]
            date_key = eval_date.isoformat() if hasattr(eval_date, 'isoformat') else str(eval_date)
            # 与 _factor_codes_scores 的综合分 screen 共用同一 cache key, 互相复用
            sc_key = (date_key, pool, "full", str(code_prefixes), "composite")
            if sc_key in self._screen_cache:
                return self._screen_cache[sc_key].get("candidates", [])
            screener = get_screener()
            snapshot = self._snapshot_for(precomputed, date_key, eval_date, pool, code_prefixes, screener)
            if snapshot is None:
                return []
            result = screener.screen(
                pool=pool, top_n=100000,
                skip_l1=True, skip_l2=True, skip_debate=True,
                skip_qlib=True, skip_sent=True, skip_fundamental=True, skip_flow=True,
                eval_date=eval_date, snapshot=snapshot,
            )
            self._screen_cache[sc_key] = result
            return result.get("candidates", [])
        except Exception:
            return []

    # ══════════════════════════════════════════════
    # L7: 置换检验 (Shuffle Test)
    # ══════════════════════════════════════════════

    def _check_shuffle(self, ctx):
        """随机打乱forward return标签N次，真实IC需>95%分位数。

        p<0.05 = 信号不是随机噪声。
        """
        variant = ctx["variant"]
        eval_weeks = ctx["eval_weeks"]
        n_shuffles = ctx.get("n_shuffles", 200)

        try:
            eval_dates = self._evidence_dates(ctx)
            if len(eval_dates) < 3:
                return False, 0.0, f"置换检验数据点不足({len(eval_dates)})"

            real_ic, real_series = self._calc_rank_ic(
                variant, eval_dates, ctx, return_series=True, return_dates=True)
            if len(real_series) < 3:
                return False, 0.0, f"置换检验有效每日IC不足({len(real_series)})"

            shuffle_ics = []
            for _ in range(n_shuffles):
                shuffled_ic = self._calc_rank_ic(variant, eval_dates, ctx, shuffle_labels=True)
                shuffle_ics.append(shuffled_ic)

            real_abs = abs(real_ic)
            shuffle_abs = np.abs(np.array(shuffle_ics))
            percentile_95 = np.percentile(shuffle_abs, 95)
            p_value = float(np.mean(shuffle_abs >= real_abs))

            passed = bool(real_abs > percentile_95 or p_value < 0.05)
            score = min(1.0, max(0.0, 1.0 - p_value))

            # 存储结构化统计数据 (供 check_all_layers 消费, 避免 regex 解析 detail 字符串)
            self._last_shuffle = {
                "real_abs": round(float(real_abs), 4),
                "p95": round(float(percentile_95), 4),
                "p_value": round(float(p_value), 4),
                "n_shuffles": n_shuffles,
            }

            return passed, score, (
                f"真实|IC|={real_abs:.4f}, 95%分位={percentile_95:.4f}, "
                f"p={p_value:.3f} ({n_shuffles}次置换) | {'信号真实' if passed else '可能是噪声'}"
            )
        except Exception as e:
            return False, 0.0, f"L7异常: {e}"

    # ══════════════════════════════════════════════
    # L8: 边际贡献 (正交化增量IC)
    # ══════════════════════════════════════════════

    def _check_marginal(self, ctx):
        """候选因子对因子池现有活跃因子做截面 OLS 取残差, 残差与 forward return 的 Spearman = 增量IC。
        要求增量IC与候选原始IC同号且 |增量IC| > 0.01 (加入因子池有真实增量价值)。
        因子池为空 → 直接通过 (首个因子无冗余)。"""
        try:
            from core.domain.factors.registry import get_pool
            from core.domain.screening._screener import get_screener
            from core.domain.factors.market_cap import get_market_cap_map
            from core.domain.factors.neutralize import cross_sectional_neutralize, orthogonalize_residual, time_weighted_mean
            from scipy.stats import spearmanr

            variant = ctx["variant"]
            pool = ctx.get("pool", "all")
            code_prefixes = ctx.get("code_prefixes")
            precomputed = ctx.get("snapshots")
            industry_map = self._get_industry_map()
            screener = get_screener()

            active = get_pool().get_active_factors()
            # 排除候选自身 (同名/同模板), 避免自我解释
            cand_keys = {variant.get("variant_key", ""), variant.get("template_key", "")}
            pool_variants = [self._pool_entry_to_variant(k, v) for k, v in active.items()
                             if k not in cand_keys and not any(ck and ck in k for ck in cand_keys)]
            if not pool_variants:
                return True, 1.0, "因子池为空/无其他因子, 边际贡献视为100%"

            eval_dates = self._evidence_dates(ctx)[:10]
            if not eval_dates:
                return False, 0.0, "无评估日期，无法计算边际贡献"

            raw_ics, inc_ics = [], []
            for eval_date in eval_dates:
                date_key = eval_date.isoformat() if hasattr(eval_date, "isoformat") else str(eval_date)
                cs = self._variant_scores(variant, eval_date, date_key, pool, code_prefixes, precomputed, screener)
                if cs is None:
                    continue
                codes, scores = cs
                if len(codes) < 20:
                    continue
                mktcap = get_market_cap_map(codes, eval_date)
                cand_resid = cross_sectional_neutralize(codes, scores, mktcap, industry_map)

                # 池因子对齐到同一 codes 顺序 (缺失填该因子截面均值)
                factor_cols = []
                for pv in pool_variants:
                    pcs = self._variant_scores(pv, eval_date, date_key, pool, code_prefixes, precomputed, screener)
                    if pcs is None:
                        continue
                    pcodes, pscores = pcs
                    pmap = {c: s for c, s in zip(pcodes, pscores)}
                    vals = [pmap.get(c) for c in codes]
                    present = [v for v in vals if v is not None]
                    if len(present) < 0.5 * len(codes):
                        continue
                    mean_v = float(np.mean(present)) if present else 0.0
                    factor_cols.append([float(v) if v is not None else mean_v for v in vals])

                fwd = self._batch_forward_returns(codes, 20, eval_date)
                cand_vec = np.array([cand_resid.get(c, 0.0) for c in codes], dtype=float)
                y = np.array(fwd, dtype=float)
                mask = np.array([f is not None for f in fwd]) & np.isfinite(y)
                if mask.sum() < 10:
                    continue

                raw_c, _ = spearmanr(cand_vec[mask], y[mask])
                if raw_c is not None and not np.isnan(raw_c):
                    raw_ics.append(float(raw_c))

                inc_resid = orthogonalize_residual(cand_vec, factor_cols)
                inc_c, _ = spearmanr(inc_resid[mask], y[mask])
                if inc_c is not None and not np.isnan(inc_c):
                    inc_ics.append(float(inc_c))

            if not inc_ics or not raw_ics:
                return True, 0.5, "数据不足, 跳过边际贡献"

            raw_ic = time_weighted_mean(raw_ics, decay=0.90)
            inc_ic = time_weighted_mean(inc_ics, decay=0.90)
            same_sign = (raw_ic >= 0) == (inc_ic >= 0)
            passed = same_sign and abs(inc_ic) > 0.015
            retention = abs(inc_ic) / max(abs(raw_ic), 1e-6)
            score = min(1.0, max(0.0, retention))

            return passed, score, (
                f"原始IC={raw_ic:.4f}, 正交后增量IC={inc_ic:.4f} (vs {len(pool_variants)}个池因子) | "
                f"保留率={retention:.0%}, {'有增量价值' if passed else '增量不足, 冗余'}"
            )
        except Exception as e:
            return True, 0.5, f"L8跳过: {e}"

    # ══════════════════════════════════════════════
    # 后置聚类去重 (补 L5 两两比对的链式冗余盲区)
    # ══════════════════════════════════════════════

    def cluster_deduplicate(self, passed_keys, variant_map, date_snapshots, pool,
                            code_prefixes, screener):
        """层次聚类去重: 多期截面平均相关 → 层次聚类 → 每类保留 ICIR 最高的因子。
        对 passed_keys 做两步去重: ①内部聚类 ②与活跃池联合聚类。
        返回精简后的 key 列表 (≤40 个因子跳过, 避免算力浪费)。
        """
        if len(passed_keys) <= 1:
            return list(passed_keys)

        all_keys = sorted(passed_keys)
        n = len(all_keys)
        if n <= 40:
            return list(all_keys)

        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        # 1. 取 5 个均匀间隔代表日期
        date_keys = sorted(date_snapshots.keys())
        nd = len(date_keys)
        if nd <= 5:
            rep_dates = date_keys
        else:
            step = max(1, nd // 5)
            rep_dates = [date_keys[i] for i in range(0, nd, step)][:5]

        # 2. 每日期计算因子得分向量; 同时累积每日 IC 用于 ICIR
        date_vecs = {}   # {dk: {key: (codes, scores)}}
        key_ics = {k: [] for k in all_keys}  # {key: [daily_IC]}
        for dk in rep_dates:
            snap = date_snapshots.get(dk)
            if snap is None:
                continue
            vecs = {}
            # 该日 forward return
            codes_all = sorted({c for s in snap.values() for c in [s[0].iloc[0].get("code", "")] if c} or
                               {c for c in snap})
            if len(codes_all) < 20:
                continue
            from core.domain.factors.market_cap import get_market_cap_map
            from core.domain.factors.neutralize import cross_sectional_neutralize
            import datetime as _dt
            ed = _dt.date.fromisoformat(dk[:10])
            fwd = self._batch_forward_returns(codes_all, 20, ed)
            fwd_map = {c: r for c, r in zip(codes_all, fwd)}
            industry_map = self._get_industry_map()
            mktcap_map = get_market_cap_map(codes_all, ed)

            for key in all_keys:
                v = variant_map.get(key)
                if v is None:
                    continue
                try:
                    cs = self._variant_scores(v, ed, dk, pool, code_prefixes,
                                               {dk: snap}, screener, allow_composite=False)
                except Exception:
                    continue
                if cs is None:
                    continue
                codes, scores = cs
                if len(codes) < 20:
                    continue
                vecs[key] = (codes, scores)
                # 该日中性化 IC
                try:
                    resid = cross_sectional_neutralize(codes, scores, mktcap_map, industry_map)
                    if resid and len(resid) >= 10:
                        xs = [resid.get(c, 0) for c in codes if c in fwd_map]
                        ys = [fwd_map[c] for c in codes if c in resid and c in fwd_map]
                        if len(xs) >= 10:
                            from scipy.stats import spearmanr
                            corr, _ = spearmanr(xs, ys)
                            if corr is not None and not np.isnan(corr):
                                key_ics[key].append(float(corr))
                except Exception:
                    pass
            date_vecs[dk] = vecs

        # 3. 多期平均相关矩阵
        corr_sum = np.zeros((n, n))
        corr_cnt = np.zeros((n, n))
        for dk, vecs in date_vecs.items():
            for i, ki in enumerate(all_keys):
                if ki not in vecs:
                    continue
                ci, si = vecs[ki]
                for j, kj in enumerate(all_keys):
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

        # 4. 距离矩阵 → 层次聚类
        dist = 1.0 - corr_avg
        dist = np.nan_to_num(dist, nan=1.0)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method='average')
        th = 1.0 - L5_CORR_THRESHOLD
        clusters = fcluster(Z, th, criterion='distance')

        # 5. ICIR 择优 (每类保留最高者)
        key_icir = {}
        for k in all_keys:
            ics = key_ics.get(k, [])
            if len(ics) >= 3:
                m = np.mean(ics)
                s = np.std(ics)
                key_icir[k] = abs(m) / max(s, 0.001)
            elif ics:
                key_icir[k] = abs(ics[0])
            else:
                key_icir[k] = 0.0

        cluster_best = {}
        cluster_members = {}
        for i, key in enumerate(all_keys):
            cid = int(clusters[i])
            cluster_members.setdefault(cid, []).append(key)
            cur_best = cluster_best.get(cid)
            if cur_best is None or key_icir.get(key, 0) > key_icir.get(cur_best, 0):
                cluster_best[cid] = key

        kept = sorted(cluster_best.values())
        dropped = [k for k in all_keys if k not in kept]
        max_corr_after = max(abs(corr_avg[i, j]) for i in range(n) for j in range(i + 1, n)) if n > 1 else 0.0

        logger.info(f"[ClusterDedup] {len(all_keys)} → {len(kept)} "
                     f"(聚类{len(cluster_best)}组, 淘汰{dropped[:5]}{'...' if len(dropped) > 5 else ''}) "
                     f"去重后最大相关 {max_corr_after:.2f}")
        for cid in sorted(cluster_best):
            members = cluster_members.get(cid, [])
            best = cluster_best[cid]
            if len(members) > 1:
                logger.debug(f"[ClusterDedup] 组{cid}: {members} → 保留 {best} (ICIR={key_icir[best]:.3f})")

        # 6. 第二步: kept ∪ 活跃池 → 再次聚类 (覆盖新旧因子间链式冗余)
        from core.domain.factors.registry import get_pool
        active = get_pool().get_active_factors(category="white_box")
        pool_keys = [k for k in active if k not in set(kept)]
        if pool_keys and len(kept) > 1:
            all_step2 = kept + pool_keys
            # 复用同样的聚类逻辑 (精简版: 只用一个代表日)
            if date_vecs:
                dk = rep_dates[0]
                vecs = date_vecs.get(dk, {})
                n2 = len(all_step2)
                corr2 = np.zeros((n2, n2))
                for i, ki in enumerate(all_step2):
                    if ki not in vecs:
                        continue
                    ci, si = vecs[ki]
                    for j, kj in enumerate(all_step2):
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
                            corr2[i, j] = r
                            corr2[j, i] = r
                np.fill_diagonal(corr2, 1.0)
                dist2 = 1.0 - np.abs(corr2)
                dist2 = np.nan_to_num(dist2, nan=1.0)
                Z2 = linkage(squareform(dist2, checks=False), method='average')
                c2 = fcluster(Z2, th, criterion='distance')
                c2_best = {}
                for i, key in enumerate(all_step2):
                    cid = int(c2[i])
                    is_new = key in set(kept)
                    cur = c2_best.get(cid)
                    # 该组有 passed 因子: 优先保留 passed 中 ICIR 高的
                    _, cur_is_new = cur if cur else (None, False)
                    if cur is None or (is_new and not cur_is_new) or \
                       (is_new == cur_is_new and key_icir.get(key, 0) > key_icir.get(cur[0], 0)):
                        c2_best[cid] = (key, is_new)
                kept2 = sorted(k for k, _ in c2_best.values())
                before = len(kept)
                kept = [k for k in kept2 if k in set(kept)]  # 只保留 passed 中幸存者
                removed_by_pool = before - len(kept)
                if removed_by_pool:
                    logger.info(f"[ClusterDedup] Step2 vs活跃池: {before} → {len(kept)} ({removed_by_pool}个与旧因子冗余)")

        return kept

    # ══════════════════════════════════════════════
    # 辅助方法
    # ══════════════════════════════════════════════

    def _db_trading_days(self, start, end):
        """从 daily_quotes 表取 [start, end] 的真实交易日 (可靠, 优于 akshare 日历)。
        失败返回 None (调用方回退 is_trading_day)。结果按实例缓存，同日期范围只查一次。"""
        cache_key = (start.isoformat(), end.isoformat())
        cached = self._td_cache.get(cache_key)
        if cached is not None:
            return cached or None
        try:
            from core.data import Database
            if not Database.is_available():
                self._td_cache[cache_key] = ()
                return None
            rows = Database.fetchall(
                "SELECT DISTINCT trade_date FROM daily_quotes WHERE trade_date>=%s AND trade_date<=%s ORDER BY trade_date",
                [start.isoformat(), end.isoformat()])
            import datetime as dt
            out = []
            for r in rows:
                v = r["trade_date"]
                out.append(v if isinstance(v, dt.date) else dt.date.fromisoformat(str(v)[:10]))
            self._td_cache[cache_key] = tuple(out) if out else ()
            return out or None
        except Exception:
            self._td_cache[cache_key] = ()
            return None

    def _get_eval_dates(self, weeks=18, step=5, anchor_date=None):
        """获取评估日期列表（截至 anchor_date 的最近N周，每隔step天取一个交易日）。
        anchor_date=None 时用今天; 传入历史日期则只用该日及之前的数据 (时序滚动/无前视)。
        优先查 daily_quotes 表 (可靠), 回退 is_trading_day。结果按实例缓存。"""
        import datetime as dt
        today = anchor_date or today_date()
        cache_key = (weeks, step, today.isoformat())
        cached = self._eval_dates_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        start = today - dt.timedelta(weeks=weeks * 7 // 5 * 5)
        db = self._db_trading_days(start, today)
        if db is not None:
            result = db[::step]
            self._eval_dates_cache[cache_key] = tuple(result)
            return result
        from core.shared.trading_time import is_trading_day
        dates = []
        d = start
        while d <= today:
            if is_trading_day(d):
                dates.append(d)
            d += dt.timedelta(days=1)
        result = dates[::step]
        self._eval_dates_cache[cache_key] = tuple(result)
        return result

    def _evidence_dates(self, ctx):
        explicit = list(ctx.get("eval_dates") or [])
        snapshots = ctx.get("snapshots")
        if explicit and snapshots:
            keys = {str(key)[:10] for key in snapshots}
            explicit = [value for value in explicit if str(value)[:10] in keys]
        if explicit:
            return explicit
        plan = ctx.get("l1_dates") or {}
        for name in ("test", "valid", "train"):
            values = list(plan.get(name) or [])
            if values:
                return values
        if snapshots:
            import datetime as dt
            return [dt.date.fromisoformat(str(key)[:10]) for key in sorted(snapshots)]
        return self._get_eval_dates(
            weeks=ctx["eval_weeks"], step=5, anchor_date=ctx.get("anchor_date"))

    def _dates_between(self, start, end, step=5):
        """[start, end] 区间内每隔 step 个交易日取一个日期。优先查库, 回退 is_trading_day。结果按实例缓存。"""
        import datetime as dt
        cache_key = (start.isoformat(), end.isoformat(), step)
        cached = self._dates_between_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        db = self._db_trading_days(start, end)
        if db is not None:
            result = db[::step]
            self._dates_between_cache[cache_key] = tuple(result)
            return result
        from core.shared.trading_time import is_trading_day
        dates = []
        d = start
        while d <= end:
            if is_trading_day(d):
                dates.append(d)
            d += dt.timedelta(days=1)
        result = dates[::step]
        self._dates_between_cache[cache_key] = tuple(result)
        return result

    def _classify_regimes(self, eval_dates):
        """对日期按市场状态分类。简化: 用沪深300指数日线的20日趋势和波动判断。

        Returns: {"trending_up": [...], "trending_down": [...], "ranging": [...], "volatile": [...]}
        """
        try:
            from core.data.fetcher import DataFetcher
            fetcher = DataFetcher()
            df = fetcher.get_index_daily("sh000300", days=500)
            if df is None or len(df) < 60:
                return {"trending_up": eval_dates, "trending_down": [], "ranging": [], "volatile": []}

            df["ma20"] = df["close"].rolling(20).mean()
            df["ret_20d"] = df["close"].pct_change(20, fill_method=None)
            df["vol_20d"] = df["close"].pct_change(fill_method=None).rolling(20).std()

            regimes = {"trending_up": [], "trending_down": [], "ranging": [], "volatile": []}

            for d in eval_dates:
                d_str = d.isoformat() if hasattr(d, 'isoformat') else str(d)
                row = df[df.index <= d_str]
                if len(row) < 20:
                    regimes["ranging"].append(d)
                    continue
                ret_20d = row["ret_20d"].iloc[-1]
                vol_20d = row["vol_20d"].iloc[-1]
                vol_median = row["vol_20d"].iloc[-60:].median() if len(row) >= 60 else vol_20d

                if vol_20d > vol_median * 1.5:
                    regimes["volatile"].append(d)
                elif ret_20d > 0.05:
                    regimes["trending_up"].append(d)
                elif ret_20d < -0.05:
                    regimes["trending_down"].append(d)
                else:
                    regimes["ranging"].append(d)

            return regimes
        except Exception:
            return {"trending_up": eval_dates, "trending_down": [], "ranging": [], "volatile": []}

    def _variant_isolation(self, variant):
        """把 variant 映射为「隔离单因子」的 screen kwargs; 不支持(占位白盒/公开Alpha)则返回 None。
        隔离后 candidate.l0_score == 该因子标准化分。仅覆盖 screen 原生支持的因子。"""
        ALL_SKIPS = {
            "skip_tech": True, "skip_qlib": True, "skip_sent": True,
            "skip_fundamental": True, "skip_flow": True, "skip_north_flow": True,
            "skip_size_turnover": True, "skip_momentum": True, "skip_volatility": True,
            "skip_ht": True, "skip_wb_industry": True, "skip_wb_divergence": True,
            "skip_wb_moneyflow": True,
        }
        SKIP_KEY = {
            "tech": "skip_tech", "qlib": "skip_qlib", "sentiment": "skip_sent",
            "fundamental": "skip_fundamental", "flow": "skip_flow", "north_flow": "skip_north_flow",
            "size_turnover": "skip_size_turnover", "momentum": "skip_momentum",
            "volatility": "skip_volatility", "ht_factors": "skip_ht",
            "wb_industry": "skip_wb_industry", "wb_divergence": "skip_wb_divergence",
            "wb_moneyflow": "skip_wb_moneyflow",
        }
        WINDOW_KEY = {
            "momentum": "momentum_n", "volatility": "volatility_n",
            "wb_industry": "wb_industry_n", "wb_divergence": "wb_divergence_n",
            "wb_moneyflow": "wb_moneyflow_n",
        }
        TECH_SUB = {
            "tech_macd": "macd_only", "tech_rsi": "rsi_only", "tech_ma": "ma_only",
            "tech_kdj": "kdj_only", "tech_bb": "bb_only", "tech_volume": "volume_only",
        }

        params = variant.get("params", {}) or {}
        n = params.get("n", variant.get("default_n"))

        def _resolve(k):
            if not k:
                return None
            if k in SKIP_KEY or k in TECH_SUB:
                return k
            for base in ("wb_industry", "wb_divergence", "wb_moneyflow"):
                if k.startswith(base):
                    return base
            return None

        key = _resolve(variant.get("variant_key", "")) or _resolve(variant.get("template_key", ""))
        if key is None:
            return None

        skips = dict(ALL_SKIPS)
        extra = {}
        if key in TECH_SUB:
            skips["skip_tech"] = False
            return {"skips": skips, "source_weights": {"tech": 1.0},
                    "extra": extra, "tech_preset": TECH_SUB[key], "sig": key}
        skips[SKIP_KEY[key]] = False
        if key in WINDOW_KEY and n:
            extra[WINDOW_KEY[key]] = int(n)
        sig = f"{key}:{n}" if key in WINDOW_KEY else key
        return {"skips": skips, "source_weights": {key: 1.0},
                "extra": extra, "tech_preset": "all", "sig": sig}

    def _resolve_eval(self, variant, quiet=False):
        """判定 variant 的求值方式, 返回 (iso, iso_sig, use_expr, inner_ast, outer_op)。
        iso 可隔离 → screen 隔离; 否则有 expr → 表达式求值; 都不行 → 综合分。"""
        iso = self._variant_isolation(variant)
        iso_sig = iso["sig"] if iso else "composite"
        use_expr = False
        inner_ast = outer_op = None
        expr = variant.get("expr")
        if iso is None and expr:
            try:
                from core.domain.factors.expression import parse_factor
                inner_ast, outer_op = parse_factor(expr)
                use_expr = True
                import hashlib
                expr_sig = hashlib.sha1(str(expr).encode("utf-8")).hexdigest()[:12]
                iso_sig = f"expr:{variant.get('variant_key', '')}:{expr_sig}"
            except Exception as e:
                if not quiet:
                    logger.debug(f"[IC] 表达式不支持 ({variant.get('variant_key')}): {e}, IC 基于综合分")
        if iso is None and not use_expr and not quiet:
            logger.debug(f"[IC] 因子无法隔离 (variant={variant.get('variant_key') or variant.get('template_key')}), IC 基于综合分")
        return iso, iso_sig, use_expr, inner_ast, outer_op

    def _variant_scores(self, variant, eval_date, date_key, pool, code_prefixes, precomputed, screener, allow_composite=True):
        """产出该 (variant, eval_date) 的全截面 (codes, scores); 数据不足或不可求值(且不允许综合分)返回 None。"""
        iso, iso_sig, use_expr, inner_ast, outer_op = self._resolve_eval(variant, quiet=True)
        if iso is None and not use_expr and not allow_composite:
            return None
        return self._factor_codes_scores(
            iso, iso_sig, use_expr, inner_ast, outer_op,
            eval_date, date_key, pool, code_prefixes, precomputed, screener)

    def _pool_entry_to_variant(self, key, info):
        """把因子池 entry 转成校验器 variant 格式 (供边际贡献解释变量求值)。"""
        return {
            "variant_key": key,
            "template_key": info.get("template_key") or key,
            "params": info.get("params") or {},
            "default_n": (info.get("params") or {}).get("n"),
            "expr": info.get("expr"),
        }

    def deduplicate_variants(self, variants, pool="all", code_prefixes=None, eval_date=None,
                             corr_thr=0.8, snapshots=None, selection_scores=None,
                             return_details=False):
        """对一组 variant 做候选集内部去同源: |Pearson|>corr_thr 每组留 |IC| 高者。
        返回保留的 variant_key 列表 (无法求值的 variant 保留, 交下游处理)。
        用于训练输入去重 (报告修正2 覆盖训练路径)。"""
        import datetime as dt
        from core.domain.screening._screener import get_screener
        from scipy.stats import spearmanr
        screener = get_screener()
        if snapshots:
            keys = sorted(str(key) for key in snapshots)
            requested = eval_date.isoformat() if hasattr(eval_date, "isoformat") else str(eval_date or "")[:10]
            eligible = [key for key in keys if not requested or key <= requested]
            date_key = (eligible or keys)[-1]
            eval_date = dt.date.fromisoformat(date_key[:10])
            snap_map = snapshots
        elif eval_date is None:
            eds = self._get_eval_dates(weeks=4, step=5)
            eval_date = eds[-1] if eds else today_date()
            date_key = eval_date.isoformat() if hasattr(eval_date, "isoformat") else str(eval_date)
            snap = self._snapshot_for(None, date_key, eval_date, pool, code_prefixes, screener)
            snap_map = {date_key: snap} if snap else None
        else:
            date_key = eval_date.isoformat() if hasattr(eval_date, "isoformat") else str(eval_date)
            snap = self._snapshot_for(None, date_key, eval_date, pool, code_prefixes, screener)
            snap_map = {date_key: snap} if snap else None

        vecs, unresolved = {}, []
        for v in variants:
            k = v.get("variant_key") or v.get("template_key")
            cs = self._variant_scores(v, eval_date, date_key, pool, code_prefixes, snap_map, screener,
                                      allow_composite=False)
            if cs is None or len(cs[0]) < 20:
                unresolved.append(k)
                continue
            vecs[k] = {c: float(s) for c, s in zip(cs[0], cs[1])}
        if len(vecs) < 2:
            kept = list(vecs)
            if return_details:
                return {"kept": kept, "unresolved": unresolved}
            return kept + unresolved

        selection_quality = {key: float(value or 0.0) for key, value in (selection_scores or {}).items()}
        if not selection_scores:
            all_codes = sorted({c for m in vecs.values() for c in m})
            fwd = self._batch_forward_returns(all_codes, 20, eval_date)
            fwd_map = {c: r for c, r in zip(all_codes, fwd)}
            for k, m in vecs.items():
                xs = [m[c] for c in m if fwd_map.get(c) is not None]
                ys = [fwd_map[c] for c in m if fwd_map.get(c) is not None]
                if len(xs) >= 10:
                    corr, _ = spearmanr(xs, ys)
                    selection_quality[k] = abs(float(corr)) if corr is not None and not np.isnan(corr) else 0.0
                else:
                    selection_quality[k] = 0.0

        kept = []
        for k in sorted(vecs, key=lambda x: selection_quality.get(x, 0.0), reverse=True):
            m = vecs[k]
            dup = False
            for kk in kept:
                common = [c for c in m if c in vecs[kk]]
                if len(common) < 20:
                    continue
                a = np.array([m[c] for c in common]); b = np.array([vecs[kk][c] for c in common])
                if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                    continue
                r = abs(float(np.corrcoef(a, b)[0, 1]))
                if not np.isnan(r) and r > corr_thr:
                    dup = True
                    break
            if not dup:
                kept.append(k)
        result = kept + unresolved
        logger.info(f"[DedupVariants] {len(variants)} → {len(result)} (可求值 {len(vecs)}→{len(kept)})")
        if return_details:
            return {"kept": kept, "unresolved": unresolved}
        return result

    def _calc_rank_ic(self, variant, eval_dates, ctx, forward_days=None, shuffle_labels=False, return_series=False, return_dates=False):
        try:
            from core.domain.screening._screener import get_screener
            from core.domain.factors.market_cap import get_market_cap_map
            from core.domain.factors.neutralize import cross_sectional_neutralize
            screener = get_screener()

            pool = ctx.get("pool", "all")
            code_prefixes = ctx.get("code_prefixes")
            precomputed = ctx.get("snapshots")
            industry_map = self._get_industry_map()

            iso, iso_sig, use_expr, inner_ast, outer_op = self._resolve_eval(variant, quiet=shuffle_labels)

            if forward_days is None:
                forward_days = int(ctx.get("label_horizon", 20))
            max_eval_dates = ctx.get("max_eval_dates", 8)
            dates = list(eval_dates)
            if max_eval_dates is not None:
                from core.domain.factors.walkforward import evenly_sample_dates
                dates = evenly_sample_dates(dates, max_eval_dates)
            daily_ics = []
            daily_pairs = []  # [(date_iso, ic)] 供持久化按日期过滤 (as_of 时序截断)

            for eval_date in dates:
                try:
                    date_key = eval_date.isoformat() if hasattr(eval_date, 'isoformat') else str(eval_date)
                    nb_key = (date_key, forward_days, pool, str(code_prefixes), iso_sig)
                    cached = self._neutral_cache.get(nb_key)
                    if cached is None:
                        # 因子值+残差与 forward_days 无关, 按 (date, iso_sig) 缓存, 各 horizon 复用
                        rd_key = (date_key, pool, str(code_prefixes), iso_sig)
                        rd = self._resid_cache.get(rd_key)
                        if rd is None:
                            cs = self._factor_codes_scores(
                                iso, iso_sig, use_expr, inner_ast, outer_op,
                                eval_date, date_key, pool, code_prefixes, precomputed, screener)
                            if cs is None:
                                continue
                            codes, scores = cs
                            mktcap_map = get_market_cap_map(codes, eval_date)
                            resid = cross_sectional_neutralize(codes, scores, mktcap_map, industry_map)
                            rd = (codes, resid)
                            self._resid_cache[rd_key] = rd
                        codes, resid = rd
                        forward_rets = self._batch_forward_returns(codes, forward_days, eval_date)
                        fwd_map = {code: r for code, r in zip(codes, forward_rets)}
                        rc, rr = [], []
                        for code in codes:
                            r = fwd_map.get(code)
                            if code in resid and r is not None:
                                rc.append(resid[code])
                                rr.append(r)
                        cached = (rc, rr)
                        self._neutral_cache[nb_key] = cached

                    rc, rr = cached
                    if len(rc) < 10:
                        continue
                    ys = rr
                    if shuffle_labels:
                        rng = np.random.default_rng()
                        ys = list(rng.permutation(rr))
                    from scipy.stats import spearmanr
                    corr, _ = spearmanr(rc, ys)
                    if corr is not None and not np.isnan(corr):
                        _ic = float(corr)
                        daily_ics.append(_ic)
                        daily_pairs.append((date_key, _ic))
                except Exception:
                    continue

            if not daily_ics:
                return (0.0, []) if return_series else 0.0
            from core.domain.factors.neutralize import time_weighted_mean
            ic_mean = time_weighted_mean(daily_ics, decay=0.90)
            if return_series:
                return (ic_mean, daily_pairs if return_dates else daily_ics)
            return ic_mean
        except Exception:
            return (0.0, []) if return_series else 0.0

    def _calc_icir(self, variant, eval_dates, ctx, forward_days=None):
        import numpy as np
        _, daily_ics = self._calc_rank_ic(variant, eval_dates, ctx,
                                           forward_days=forward_days, return_series=True)
        if not daily_ics:
            return 0.0, 0.0, 0, []
        arr = np.array([x for x in daily_ics if np.isfinite(x)])
        if len(arr) < 2:
            return 0.0, 0.0, 0, []
        ic_mean = float(np.mean(arr))
        ic_std = float(np.std(arr))
        icir = abs(ic_mean) / max(ic_std, 0.0001)
        return icir, ic_mean, len(arr), arr.tolist()

    def _snapshot_for(self, precomputed, date_key, eval_date, pool, code_prefixes, screener):
        if precomputed:
            return precomputed.get(date_key)  # 可能 None → 调用方跳过该日
        return screener.precompute_snapshot(
            pool=pool, code_prefixes=code_prefixes, eval_date=eval_date, skip_limit_up=True)

    def _factor_codes_scores(self, iso, iso_sig, use_expr, inner_ast, outer_op,
                             eval_date, date_key, pool, code_prefixes, precomputed, screener):
        """产出该 (eval_date, 因子) 的全截面 (codes, scores); 数据不足返回 None。"""
        FULL_N = 100000  # 取全 L0 截面 (skip_l1 时 candidates[:top_n])
        if use_expr:
            # 懒缓存: 按需求值单个因子, 不触发全量 batch_eval_all
            cached = self._date_expr_scores.get(date_key, {}).get(iso_sig)
            if cached is not None:
                return cached
            snapshot = self._snapshot_for(precomputed, date_key, eval_date, pool, code_prefixes, screener)
            if snapshot is None:
                return None
            codes, scores = self._expr_scores(inner_ast, outer_op, snapshot)
            if len(codes) >= 20:
                self._date_expr_scores.setdefault(date_key, {})[iso_sig] = (codes, scores)
                return (codes, scores)
            return None

        if iso is None:
            return None

        sc_key = (date_key, pool, "full", str(code_prefixes), iso_sig)
        result = self._screen_cache.get(sc_key)
        if result is None:
            snapshot = self._snapshot_for(precomputed, date_key, eval_date, pool, code_prefixes, screener)
            if snapshot is None:
                return None
            screen_kwargs = dict(
                pool=pool, top_n=FULL_N, skip_l1=True, skip_l2=True, skip_debate=True,
                eval_date=eval_date, snapshot=snapshot, skip_backfill=True,
            )
            if iso:
                screen_kwargs.update(iso["skips"])
                screen_kwargs["source_weights"] = iso["source_weights"]
                screen_kwargs["tech_preset"] = iso["tech_preset"]
                screen_kwargs.update(iso["extra"])
            else:
                screen_kwargs.update(skip_qlib=True, skip_sent=True,
                                     skip_fundamental=True, skip_flow=True)
            result = screener.screen(**screen_kwargs)
            self._screen_cache[sc_key] = result
        candidates = result.get("candidates", [])
        if len(candidates) < 20:
            return None
        codes = [c["code"] for c in candidates]
        scores = [c.get("l0_score", 0) for c in candidates]
        return codes, scores

    def _expr_scores(self, inner_ast, outer_op, snapshot):
        """对 snapshot 每只股票 df 求表达式值 → (codes, scores); 最外层截面算子跨截面转换。
        按交易日历(index)签名分组做面板批量向量化, 逐组失败回退逐股 (零回归)。"""
        from core.domain.factors.expression import eval_on_df, eval_on_panel
        import math, pandas as pd
        from collections import defaultdict

        # 按 index 签名分组: 只有日历完全相同的股票能拼进同一面板 (保证 shift/rolling 对齐)
        groups = defaultdict(dict)
        for code, tup in snapshot.items():
            df = tup[0] if isinstance(tup, (list, tuple)) else tup
            if df is None or len(df) < 2:
                continue
            sig = (len(df.index), df.index[0], df.index[-1])
            groups[sig][code] = df

        # 仅取最大的日历组 (ICIR 计算只需足够截面股票, 不需要全量)
        # 374 个不同上市日期的组 → 每组都 eval_on_panel → 极慢; 取最大组即可
        grp_items = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        if not grp_items:
            return [], []

        vals = {}
        for sig, grp in grp_items[:1]:  # 只处理最大的 1 个组
            # fast/skip_tech 快照的 df 索引是 RangeIndex, 需设 trade_date 为索引
            # (否则 FundVal reindex 日期对不上, 全返回 0)
            grp_dated = {}
            for code, df in grp.items():
                df_dated = df.copy()
                if "trade_date" in df_dated.columns:
                    df_dated["_td"] = pd.to_datetime(df_dated["trade_date"])
                    df_dated = df_dated.set_index("_td")
                else:
                    df_dated = df_dated.copy()
                grp_dated[code] = df_dated

            try:
                s = eval_on_panel(inner_ast, grp_dated)
                for code, v in s.items():
                    fv = float(v)
                    if math.isfinite(fv):
                        vals[code] = fv
            except Exception:
                # 面板失败 → 该组逐股回退
                for code, df in grp_dated.items():
                    try:
                        v = eval_on_df(inner_ast, df)
                    except Exception:
                        continue
                    if isinstance(v, float) and math.isfinite(v):
                        vals[code] = v

        if outer_op and len(vals) >= 2:
            codes = list(vals)
            arr = np.array([vals[c] for c in codes], float)
            if outer_op == "rank":
                ranks = arr.argsort().argsort() / max(len(arr) - 1, 1)
                vals = {c: float(r) for c, r in zip(codes, ranks)}
            else:  # scale (z-score)
                mu, sd = arr.mean(), arr.std()
                vals = {c: float((x - mu) / sd) if sd > 1e-12 else 0.0
                        for c, x in zip(codes, arr)}
        codes = list(vals)
        return codes, [vals[c] for c in codes]

    def _batch_forward_returns(self, codes, forward_days, eval_date):
        """获取批量 forward return (委托给 ForwardReturnCache 统一缓存层)。"""
        from core.domain.factors.forward_returns import get_forward_cache
        return get_forward_cache().get_list(codes, eval_date, forward_days)

    def _calc_hit_rate(self, variant, eval_dates, ctx, forward_days=20):
        """简化胜率计算"""
        try:
            ic = self._calc_rank_ic(variant, eval_dates, ctx, forward_days=forward_days)
            # IC>0 → 胜率约 0.5+IC/2 (经验公式)
            return min(0.9, max(0.1, 0.5 + ic / 2))
        except Exception:
            return 0.5
