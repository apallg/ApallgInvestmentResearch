"""
策略方向因子集精炼器 — 按"去冗余 + 滚动训练特征重要性 + 类内排名"思路

流程 (用户思路):
  1. 取方向候选因子 (该方向预设大类下的全部活跃因子, 未筛选, 含共线)
  2. 显式相关性聚类去冗余 (单截面 |r|>0.9 聚组, 每组留 ICIR 高者)
  3. 滚动训练 track_importance (适中容量, 不要搞大)
  4. 按 (n_rounds + win_rate_gain + mean_gain) 类内排名取 top-N
  5. 保存为方向精选集 + 输出报告

用法:
  # 完整跑一遍
  python scripts/refine_direction_factors.py trend_momentum \
      --pool csi300 --start 2023-01-01 --end 2025-12-31 --target-n 50

  # 只跑到去冗余 (先看看 1000 能压到多少)
  python scripts/refine_direction_factors.py trend_momentum --steps 1,2

  # 换个方向 / 目标数
  python scripts/refine_direction_factors.py oversold_rebound --target-n 21

依赖: 项目根目录运行, MySQL + qlib 数据可用 (与后端同环境)。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import datetime as dt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("refine_direction")


def step1_get_candidates(direction):
    """取方向候选因子: 该方向预设大类下的全部活跃因子 (未筛选, 含共线)。"""
    from core.domain.factors.selector import DIRECTION_PRESETS, group_by_category
    from core.domain.factors.registry import get_pool
    from core.domain.factors.adapter import PublicFactorAdapter

    preset = DIRECTION_PRESETS[direction]
    pool = get_pool()
    active = pool.get_active_factors()
    adapter = PublicFactorAdapter()

    category_groups = group_by_category(list(active.keys()), adapter, pool_active=active)
    candidates = []
    for cat in preset["categories"]:
        keys = category_groups.get(cat, [])
        candidates.extend(keys)
        logger.info(f"[候选] {cat}: {len(keys)} 个")
    candidates = list(dict.fromkeys(candidates))
    logger.info(f"[候选] 合计 {len(candidates)} 个因子 ({direction})")
    # 表达式去重: _n 扩展因子表达式重复, 剔除重复副本 (否则 importance 被重复因子污染)
    from core.domain.factors.selector import _dedupe_by_expression
    candidates = _dedupe_by_expression(candidates)
    logger.info(f"[候选] 表达式去重后 {len(candidates)} 个因子 ({direction})")
    return candidates, category_groups


def step2_dedupe(candidates, pool, anchor_date, corr_thr=0.95):
    """显式相关性聚类去冗余: 单截面 |r|>corr_thr 聚组, 每组留 ICIR 高者。"""
    from core.domain.screening._screener import get_screener
    from core.domain.factors.adapter import PublicFactorAdapter

    screener = get_screener()
    snapshot = screener.precompute_snapshot(pool=pool, eval_date=anchor_date, skip_limit_up=True)
    adapter = PublicFactorAdapter()
    deduped = adapter.deduplicate_factors(candidates, snapshot, anchor_date, pool=pool, corr_thr=corr_thr)
    logger.info(f"[去冗余] {len(candidates)} → {len(deduped)} (阈值 {corr_thr})")
    return deduped


def _resolve_pool(pool_str):
    """解析组合股票池 (csi300+csi500 → 合并股票列表)。单池原样返回。"""
    from pathlib import Path
    from core.shared import config as app_config
    if "+" not in str(pool_str):
        return pool_str
    combined = []
    for p in str(pool_str).split("+"):
        fpath = Path(app_config.QLIB_DATA_DIR) / "instruments" / f"{p.strip()}.txt"
        if fpath.exists():
            for line in fpath.read_text().splitlines():
                c = line.strip().split("\t")[0]
                if c and c not in combined:
                    combined.append(c)
    return combined if combined else "all"


def _latest_forward_date():
    """返回最近一个有 forward 收益数据的交易日 (daily_forward_ret.max)。

    去冗余用单日截面 IC 选代表, IC 依赖 forward 收益; 若 anchor 太新 (未来价格缺失),
    forward 收益全 0 → IC 全 0 → 选代表退化成随机删。故默认 anchor 取此日期。
    """
    try:
        from core.data import Database
        if Database.is_available():
            rows = Database.fetchall(
                "SELECT MAX(trade_date) AS mx FROM daily_forward_ret WHERE fwd_20d IS NOT NULL")
            if rows and rows[0].get("mx"):
                return rows[0]["mx"]
    except Exception as e:
        logger.warning(f"[anchor] 查询 forward 数据边界失败, 回退今天: {e}")
    return dt.date.today()


def step3_rolling_train(deduped_keys, args):
    """滚动训练 + 特征重要性追踪。返回 importance_summary。

    关键: 用适中容量 (param_profile=default/conservative), 不要 aggressive。
    预过滤: 用 resolve_factor_expression (与 WhiteBoxHandler 完全一致) 过滤无法解析的因子,
            否则 custom_factor_keys 与模型实际输入特征错位, importance 会映射到错误因子。
    """
    from core.adapters.qlib.rolling_train import rolling_train_dual
    from core.adapters.qlib.factor_resolver import resolve_factor_expression

    # 预过滤: 只保留能解析出 qlib 表达式的因子 (WhiteBoxHandler 内部也会跳过它们)
    valid_keys = []
    skipped = 0
    for key in deduped_keys:
        expr, skip_reason = resolve_factor_expression(key)
        if expr:
            valid_keys.append(key)
        else:
            skipped += 1
            logger.warning(f"[滚动训练] 跳过无法解析因子 {key}: {skip_reason}")
    if skipped:
        logger.info(f"[滚动训练] 预过滤: {len(deduped_keys)} → {len(valid_keys)} (跳过 {skipped} 个)")

    if not valid_keys:
        logger.error("[滚动训练] 无可用因子, 终止")
        return {}, {}

    logger.info(f"[滚动训练] {len(valid_keys)} 因子, {args.start}~{args.end or '今天'}, "
                f"step={args.step_months}月, param={args.param_profile}, "
                f"top_n={args.importance_top_n}")

    result = rolling_train_dual(
        pool=_resolve_pool(args.pool),
        pool_label=args.pool,
        start_anchor=args.start,
        end_anchor=args.end or None,
        step_months=args.step_months,
        train_years=args.train_years,
        valid_months=args.valid_months,
        model_name="lightgbm_rank",
        train_dual=args.train_dual,
        factor_set="WhiteBoxHandler",
        custom_factor_keys=valid_keys,
        label_horizon=args.label_horizon,
        label_norm="csrank",
        early_stop_metric="rank_ic",
        param_profile=args.param_profile,
        track_importance=True,
        feature_key_map=None,   # importance 靠 feature_names 顺序映射, 不依赖 key_map
        feature_names=valid_keys,
        importance_top_n=args.importance_top_n,
    )

    oos = result.get("oos", {})
    logger.info(f"[滚动训练] 全期 OOS: IC={oos.get('ic_mean')}, ICIR={oos.get('icir')}")
    summary = result.get("importance_summary", {})
    if not summary:
        logger.warning("[滚动训练] importance_summary 为空 — 可能轮次不足 8 或因子无 importance")
    return summary, result


def step4_rank_select(importance_summary, category_groups, target_n, direction, min_win_rate=0.3, rank_by="gain"):
    """类内排名取 top-N。

    rank_by="gain":  按 (n_rounds, win_rate_gain, mean_gain) 降序
    rank_by="split": 按 (n_rounds, win_rate, mean_split) 降序
    """
    from core.domain.factors.selector import DIRECTION_PRESETS

    cats = DIRECTION_PRESETS[direction]["categories"]
    cats_present = [c for c in cats
                    if any(k in importance_summary for k in category_groups.get(c, []))]
    if not cats_present:
        logger.warning("[排名] 无可用大类 (importance_summary 为空)")
        return [], {}
    per_cat = max(3, target_n // len(cats_present))

    win_key = "win_rate" if rank_by == "split" else "win_rate_gain"
    mean_key = "mean_split" if rank_by == "split" else "mean_gain"

    def score(k):
        info = importance_summary.get(k, {})
        return (info.get("n_rounds", 0), info.get(win_key, 0), info.get(mean_key, 0))

    selected = []
    report = {}
    for cat in cats_present:
        keys = [k for k in category_groups.get(cat, []) if k in importance_summary]
        keys = [k for k in keys if importance_summary.get(k, {}).get(win_key, 0) >= min_win_rate]
        if not keys:
            continue  # 该类无达标因子, 跳过
        ranked = sorted(keys, key=score, reverse=True)
        top = ranked[:per_cat]
        selected.extend(top)
        report[cat] = [(k, *score(k)) for k in top]
    return selected, report


def step5_save(direction, selected, importance_summary, report, args, rank_by="gain"):
    """保存为独立精选集 (trend_momentum_split / trend_momentum_gain), 不覆盖 curated[direction]。"""
    from core.domain.factors.registry import get_pool
    from core.domain.factors.selector import DIRECTION_PRESETS

    preset = DIRECTION_PRESETS[direction]
    win_key = "win_rate" if rank_by == "split" else "win_rate_gain"
    set_name = f"{direction}_{rank_by}"
    avg_win = sum(importance_summary.get(k, {}).get(win_key, 0) for k in selected) / max(len(selected), 1)
    get_pool().save_curated_set(
        name=set_name,
        factors_by_category={preset["label"]: [(k, 0.0, {}, {}) for k in selected]},
        config={
            "description": preset["desc"],
            "direction": direction,
            "categories": preset["categories"],
            "target_n": args.target_n,
            "method": f"rolling_{rank_by}",
        },
        stats={"total": len(selected), "avg_win_rate": round(avg_win, 4),
               "method": f"rolling_{rank_by}"},
    )
    logger.info(f"[保存] 精选集 '{set_name}': {len(selected)} 因子, 平均 {win_key}={avg_win:.3f}")
    return selected


def _cache_path(direction, anchor):
    return os.path.join("logs", f"refine_cache_{direction}_{anchor.isoformat()}.json")


def _load_cache(direction, anchor):
    try:
        from core.shared.persistence import load_json
        return load_json(_cache_path(direction, anchor)) or {}
    except Exception:
        return {}


def _save_cache(direction, anchor, data):
    try:
        from core.shared.persistence import save_json
        os.makedirs("logs", exist_ok=True)
        save_json(data, _cache_path(direction, anchor))
        logger.info(f"[缓存] 已保存 {_cache_path(direction, anchor)}")
    except Exception as e:
        logger.warning(f"[缓存] 保存失败: {e}")


def _run_one(direction, args):
    """跑单个方向的完整流程 (step1-5), 报告打印 + 落盘 JSON。

    step1/step2 结果落盘缓存 (logs/refine_cache_*.json),
    后续跑 --steps 3,4,5 时直接复用, 不重跑候选/去冗余。
    """
    anchor = dt.date.fromisoformat(args.anchor) if args.anchor else _latest_forward_date()
    steps = [int(s) for s in args.steps.split(",")]
    cache = _load_cache(direction, anchor)

    candidates = category_groups = None
    deduped = summary = selected = report = None

    # ── step1: 取候选 (结果缓存) ──
    if 1 in steps:
        candidates, category_groups = step1_get_candidates(direction)
        cache["candidates"] = candidates
        cache["category_groups"] = category_groups
        _save_cache(direction, anchor, cache)
    else:
        candidates = cache.get("candidates")
        category_groups = cache.get("category_groups")

    # ── step2: 去冗余 (结果缓存) ──
    if 2 in steps:
        if candidates is None:
            logger.error(f"[{direction}] step2 需 candidates, 先跑 --steps 1 (anchor={anchor.isoformat()})")
            return None
        deduped = step2_dedupe(candidates, args.pool, anchor, args.dedupe_threshold)
        cache["deduped"] = deduped
        _save_cache(direction, anchor, cache)
    else:
        deduped = cache.get("deduped")

    # ── step3: 滚动训练 ──
    if 3 in steps:
        if deduped is None:
            logger.error(f"[{direction}] step3 需 deduped, 先跑 --steps 2 (anchor={anchor.isoformat()})")
            return None
        summary, result = step3_rolling_train(deduped, args)

    # ── step4: 类内排名 (split + gain 两种) ──
    rank_bys = ["split", "gain"] if getattr(args, "rank_by", "both") == "both" else [args.rank_by]
    selected_by_rank = {}
    report_by_rank = {}
    if 4 in steps:
        if category_groups is None or summary is None:
            logger.error(f"[{direction}] step4 需 category_groups + summary (先跑 --steps 1,3)")
            return None
        for rb in rank_bys:
            sel, rep = step4_rank_select(summary, category_groups, args.target_n, direction, args.min_win_rate, rank_by=rb)
            selected_by_rank[rb] = sel
            report_by_rank[rb] = rep

    # ── step5: 保存 + 报告 ──
    if 5 in steps:
        for rb in rank_bys:
            sel = selected_by_rank.get(rb)
            rep = report_by_rank.get(rb)
            if sel is None or rep is None:
                logger.error(f"[{direction}] step5 需 {rb} 的 selected + report (先跑 --steps 4)")
                continue
            step5_save(direction, sel, summary, rep, args, rank_by=rb)

    if not report_by_rank:
        logger.info(f"[{direction}] 无报告 (步骤未跑完或排名为空)")
        return None

    for rb in rank_bys:
        selected = selected_by_rank.get(rb, [])
        report = report_by_rank.get(rb, {})
        if not report:
            continue
        logger.info("=" * 60)
        logger.info(f"[报告] {direction} [{rb}] 精选 {len(selected)} 因子")
        for cat, items in report.items():
            logger.info(f"  {cat}: {len(items)} 个")
            for k, nr, wr, ms in items:
                logger.info(f"    {k:<40} n_rounds={nr:<2} win_rate={wr:.2f} mean={ms:.1f}")
        logger.info("=" * 60)

        try:
            from core.shared.persistence import save_json, today_iso
            report_data = {
                "direction": direction,
                "rank_by": rb,
                "target_n": args.target_n,
                "selected": len(selected),
                "per_category": {
                    cat: [{"key": k, "n_rounds": nr, "win_rate": wr, "mean": ms}
                          for k, nr, wr, ms in items]
                    for cat, items in report.items()
                },
                "flat_list": selected,
            }
            os.makedirs("logs", exist_ok=True)
            path = f"logs/refine_direction_{direction}_{rb}_{today_iso()}.json"
            save_json(report_data, path)
            logger.info(f"[报告] 已落盘 {path}")
        except Exception as e:
            logger.warning(f"[报告] 落盘失败: {e}")

    logger.info("下一步: 训练时分别选 trend_momentum_split / trend_momentum_gain 对比")
    return selected_by_rank


def main():
    ap = argparse.ArgumentParser(description="策略方向因子集精炼器")
    ap.add_argument("direction", nargs="?",
                    choices=["trend_momentum", "value_defense", "oversold_rebound", "volume_breakout"],
                    help="策略方向, 或用 --all 跑全部")
    ap.add_argument("--all", action="store_true", help="跑全部 4 个策略方向")
    ap.add_argument("--pool", default="csi300")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="", help="空=今天")
    ap.add_argument("--anchor", default=None, help="去冗余快照日期, 默认=今天")
    ap.add_argument("--target-n", type=int, default=50)
    ap.add_argument("--step-months", type=int, default=3)
    ap.add_argument("--train-years", type=int, default=5)
    ap.add_argument("--valid-months", type=int, default=6)
    ap.add_argument("--label-horizon", type=int, default=20, help="label 周期, 周频用 5~10, 日频用 1")
    ap.add_argument("--train-dual", action="store_true", help="LGB+XGB 双模型")
    ap.add_argument("--param-profile", default="default", choices=["default", "conservative", "aggressive"],
                    help="超参容量: default/conservative=适中, 不要 aggressive")
    ap.add_argument("--importance-top-n", type=int, default=100,
                    help="每轮记录 top-N 因子的重要性 (设大覆盖更多候选)")
    ap.add_argument("--min-win-rate", type=float, default=0.3,
                    help="win_rate 下限, 低于此值的凑数因子剔除")
    ap.add_argument("--dedupe-threshold", type=float, default=0.95,
                    help="去冗余相关性阈值 (0.95=只去完全重复, 0.75=默认较严)")
    ap.add_argument("--steps", default="1,2,3,4,5", help="要执行的步骤, 如 1,2 只跑到去冗余")
    ap.add_argument("--rank-by", default="both", choices=["split", "gain", "both"],
                    help="排名键: split/gain/both (默认 both, 一次滚动训练产出两套)")
    args = ap.parse_args()

    if args.all:
        from core.domain.factors.selector import DIRECTION_PRESETS
        directions = list(DIRECTION_PRESETS.keys())
    else:
        if not args.direction:
            ap.error("必须指定 direction 或用 --all")
        directions = [args.direction]

    for direction in directions:
        logger.info("=" * 60)
        logger.info(f"=== 开始精炼方向: {direction} ===")
        _run_one(direction, args)


if __name__ == "__main__":
    main()
