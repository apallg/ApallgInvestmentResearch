import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('REPAIR_RESEARCH_OUTPUT', ROOT / 'data/experiments/repair_group_research_20261025'))
manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
(OUT / "status.json").write_text(json.dumps({"state": "running", "phase": "analysis", "research_only": True, "completed_model_fits": len(list(OUT.glob("*.checkpoint.pkl")))}, indent=2), encoding="utf-8")
audit = json.loads((OUT / "input_audit.json").read_text(encoding="utf-8"))
for name, expected_sha256 in manifest.get("input_files_sha256", {}).items():
    digest = hashlib.sha256()
    with (ROOT / name).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError(f"训练期间冻结输入发生变化：{name}")
completion = json.loads((OUT / "training_complete.json").read_text(encoding="utf-8"))
if completion["folds"] != list(manifest["folds"]) or completion["seeds"] != manifest["seeds"]:
    raise RuntimeError("训练矩阵未完成")
arms = ["A0", "A1", "B", "C"]
comparisons = [("A1", "A0"), ("B", "A1"), ("C", "B"), ("C", "A1")]
calendar = pd.DatetimeIndex(pd.to_datetime((ROOT / "data/qlib_data/calendars/day.txt").read_text(encoding="utf-8").splitlines()))
daily_rows, block_rows, group_rows, prediction_rows = [], [], [], []
context_coverage_rows = []
missing_columns = manifest.get("context_coverage", {}).get("missing_indicator_columns", [])
context_flags = pd.read_parquet(OUT / "context/context_features.parquet", columns=missing_columns) if missing_columns else None
for fold, window in manifest["folds"].items():
    members = {}
    labels = None
    group = None
    for seed in manifest["seeds"]:
        for arm in arms:
            frame = pd.read_parquet(OUT / f"{fold}_{seed}_{arm}.parquet").sort_index()
            if labels is None:
                labels = frame["label"]
                group = frame["group_id"]
            if not frame.index.equals(labels.index) or not frame["label"].equals(labels) or not frame["group_id"].equals(group):
                raise RuntimeError(f"股票日/标签/路由分组不一致：{fold} {seed} {arm}")
            if frame["score"].isna().any() or not np.isfinite(frame["score"]).all():
                raise RuntimeError("预测缺失，禁止缩小共同样本")
            members[(str(seed), arm)] = frame["score"]
            prediction_rows.append({"fold": fold, "seed": seed, "arm": arm, "rows": len(frame), "fallback_share": float(frame["fallback"].mean())})
    for arm in arms:
        members[("ensemble", arm)] = pd.concat([members[(str(seed), arm)] for seed in manifest["seeds"]], axis=1).mean(axis=1)
    if context_flags is not None:
        aligned_flags = context_flags.reindex(labels.index)
        if aligned_flags.isna().any().any():
            raise RuntimeError("测试样本上下文覆盖无法核对")
        context_coverage_rows.append({"fold": fold, "test_rows": len(labels), **{column: float(aligned_flags[column].mean()) for column in missing_columns}})
    dates = calendar[(calendar >= pd.Timestamp(window["test"][0])) & (calendar <= pd.Timestamp(window["test"][1]))]
    signals = {date for date in dates[::10] if calendar.get_loc(date) + 11 < len(calendar) and calendar[calendar.get_loc(date) + 11] <= dates[-1]}
    for (seed, arm), prediction in members.items():
        joined = pd.concat({"score": prediction, "label": labels, "group_id": group}, axis=1)
        for date, part in joined.groupby(level="datetime", sort=True):
            if len(part) < 50:
                raise RuntimeError(f"共同可评估截面不足50只：{date}")
            ranked = part.reset_index().sort_values(["score", "instrument"], ascending=[False, True], kind="stable")
            row = {"fold": fold, "seed": seed, "arm": arm, "date": date.strftime("%Y-%m-%d"), "stocks": len(part), "rank_ic": float(part["score"].corr(part["label"], method="spearman")), "universe_return": float(part["label"].mean()), "top5_return": float(ranked.head(5)["label"].mean()), "top50_return": float(ranked.head(50)["label"].mean())}
            row["top5_excess"] = row["top5_return"] - row["universe_return"]
            daily_rows.append(row)
            if date in signals:
                boundary = ranked.iloc[4]['score']
                above = ranked.loc[ranked['score'] > boundary]
                tied = ranked.loc[ranked['score'] == boundary]
                slots = 5 - len(above)
                row['tie_pool_size'] = len(tied)
                row['tie_slots'] = slots
                row['tie_expected_return'] = float((above['label'].sum() + slots * tied['label'].mean()) / 5)
                block_rows.append(row)
                if seed == "ensemble":
                    for group_id, selected in ranked.head(5).groupby("group_id"):
                        group_rows.append({"fold": fold, "arm": arm, "date": row["date"], "group_id": int(group_id), "selected": len(selected), "return_contribution": float(selected["label"].sum() / 5)})
daily = pd.DataFrame(daily_rows)
blocks = pd.DataFrame(block_rows)
daily.to_csv(OUT / "daily_metrics.csv", index=False, encoding="utf-8-sig")
blocks.to_csv(OUT / "label_baskets.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(group_rows).to_csv(OUT / "group_contribution.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(prediction_rows).to_csv(OUT / "prediction_coverage.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(context_coverage_rows).to_csv(OUT / "context_coverage_used.csv", index=False, encoding="utf-8-sig")
paired = blocks.loc[blocks["seed"] == "ensemble"].pivot(index=["fold", "date"], columns="arm", values="top5_return")
if paired.isna().any().any():
    raise RuntimeError("共同信号块缺失")
for candidate, baseline in comparisons:
    paired[f"{candidate}_minus_{baseline}"] = paired[candidate] - paired[baseline]
paired.to_csv(OUT / "paired_blocks.csv", encoding="utf-8-sig")
rng = np.random.default_rng(20261025)
effects = {}
for candidate, baseline in comparisons:
    column = f"{candidate}_minus_{baseline}"
    delta = paired[column]
    per_fold = {name: part.to_numpy() for name, part in delta.groupby(level="fold", sort=False)}
    sensitivity = {}
    for length in (2, 4, 6):
        short_folds = [name for name, values in per_fold.items() if len(values) < 2 * length]
        if short_folds:
            sensitivity[str(length)] = {'status': 'insufficient_blocks', 'short_folds': short_folds}
            continue
        sampled = []
        for fold_values in per_fold.values():
            n = len(fold_values)
            if n == 0:
                raise RuntimeError("测试折没有统计块")
            starts = rng.integers(0, n, size=(10000, int(np.ceil(n / length))))
            positions = ((starts[:, :, None] + np.arange(length)) % n).reshape(10000, -1)[:, :n]
            sampled.append(fold_values[positions])
        estimates = np.concatenate(sampled, axis=1).mean(axis=1)
        sensitivity[str(length)] = {"ci95": [float(value) for value in np.quantile(estimates, [0.025, 0.975])], "family_adjusted_ci9875": [float(value) for value in np.quantile(estimates, [0.00625, 0.99375])]}
    effects[column] = {"mean_return_difference": float(delta.mean()), "candidate_beats_baseline_share": float((delta > 1e-12).mean()), "tie_share": float((delta.abs() <= 1e-12).mean()), "by_fold": {name: float(values.mean()) for name, values in per_fold.items()}, "positive_folds": int(sum(values.mean() > 0 for values in per_fold.values())), "bootstrap": sensitivity}
summary = {}
for arm in arms:
    subset = blocks.loc[(blocks["seed"] == "ensemble") & (blocks["arm"] == arm)]
    day = daily.loc[(daily["seed"] == "ensemble") & (daily["arm"] == arm)]
    summary[arm] = {"dates": len(day), "signal_blocks": len(subset), "daily_rank_ic": float(day["rank_ic"].mean()), "top5_mean_gross_return": float(subset["top5_return"].mean()), "top5_mean_excess": float(subset["top5_excess"].mean()), "top50_mean_gross_return": float(subset["top50_return"].mean()), "profitable_basket_share": float((subset["top5_return"] > 0).mean())}
seed_effects = {}
for seed in manifest["seeds"]:
    seed_paired = blocks.loc[blocks["seed"] == str(seed)].pivot(index=["fold", "date"], columns="arm", values="top5_return")
    seed_effects[str(seed)] = {f"{candidate}_minus_{baseline}": float((seed_paired[candidate] - seed_paired[baseline]).mean()) for candidate, baseline in comparisons}
decision = {"research_only": True, "strict_pit_verified": False, "execution_verified": False, "deployment_eligible": False, "full_october_repair_effect_identified": False, "status": "historical_LightGBM_diagnostic_complete", "summary": summary, "effects": effects, "seed_effects": seed_effects, "limitations": manifest["limitations"]}
decision['tie_sensitivity'] = {arm: {'boundary_tie_blocks': int(((blocks['arm'] == arm) & (blocks['seed'] == 'ensemble') & (blocks['tie_pool_size'] > blocks['tie_slots'])).sum()),
    'tie_expected_mean_return': float(blocks.loc[(blocks['arm'] == arm) & (blocks['seed'] == 'ensemble'), 'tie_expected_return'].mean())} for arm in arms}
decision['leave_one_fold_out'] = {column: {fold: float(paired.loc[paired.index.get_level_values('fold') != fold, column].mean()) for fold in manifest['folds']}
    for column in [f'{candidate}_minus_{baseline}' for candidate, baseline in comparisons]}
splitting_ci = effects["C_minus_B"]["bootstrap"]["2"]["family_adjusted_ci9875"]
if splitting_ci[0] > 0:
    interpretation = "这套粗行业拆分在本轮历史标签篮子中显示正向增量，校正区间不含零；实际费用后收益、融合模型与上线效果仍未验证。"
elif splitting_ci[1] < 0:
    interpretation = "这套粗行业拆分在本轮历史标签篮子中落后于同信息统一模型；该结果不能否定其他股票分类方法。"
else:
    interpretation = "本轮尚不能证明这套粗行业拆分优于同信息统一模型，校正区间包含零。"
decision["research_interpretation"] = interpretation
lines = ["# 数据代际与分组训练：隔离研究结果", "", "仅为历史 LightGBM 再验证；未认证严格 PIT，没有真实成交/费用后账本，不具备上线资格。A0 只代表今日写入前，不能量化整个十月修复。", "", "| 方案 | 日均 RankIC | Top5 10日毛收益均值 | 同期全池超额均值 | 自身盈利块比例 |", "|---|---:|---:|---:|---:|"]
lines[3:3] = [interpretation, ""]
if 'FundVal v2' in audit.get('scope', ''):
    lines[2] = '本轮比较FundVal接入与缺失语义修复，行情、标签和上下文沿用冻结工件；仅为历史研究，未认证严格PIT，不具备上线资格。'
for arm, value in summary.items():
    lines.append(f"| {arm} | {value['daily_rank_ic']:.4f} | {value['top5_mean_gross_return']:.2%} | {value['top5_mean_excess']:.2%} | {value['profitable_basket_share']:.1%} |")
lines += ["", f"{len(manifest['folds'])} 折、{len(manifest['seeds'])} 个种子先平均校准分数后共同选股；共 {len(paired)} 个非重叠标签篮子。收益均值是每个10交易日篮子的平均毛收益，不能作为账户累计收益或年化收益。", "", "| 对比 | 配对均值差 | 战胜基线块比例 | 平局比例 | 95%区间 | 四比较校正98.75%区间 |", "|---|---:|---:|---:|---|---|"]
for name, effect in effects.items():
    ci = effect["bootstrap"]["2"]["ci95"]
    adjusted = effect["bootstrap"]["2"]["family_adjusted_ci9875"]
    lines.append(f"| {name} | {effect['mean_return_difference']:.2%} | {effect['candidate_beats_baseline_share']:.1%} | {effect['tie_share']:.1%} | [{ci[0]:.2%}, {ci[1]:.2%}] | [{adjusted[0]:.2%}, {adjusted[1]:.2%}] |")
scope_label = 'FundVal接入与缓存缺失处理的整体修复' if 'FundVal v2' in audit.get('scope', '') else '2026-10-25的输入代际'
lines += ["", f"A1−A0：{scope_label}；B−A1：新增信息；C−B：分别训练；C−A1：整体方案。置信区间以配对日期块重采样，股票和种子未被当作独立市场样本。完整逐折、逐种子和块长敏感性见 decision.json。", "", "局限："]
lines += ['- 所有方案均含行业分组收益校准与三种子集成，表中不是未经校准的原始模型排名。',
          '- 主区间采用折内2块重采样，条件于已实现的测试折；4/6块在短折不足两个完整块时不报告，防止循环重采样退化。',
          '- 同分边界按证券代码排序；同分等概率抽选的期望收益和删除单折敏感性保存在decision.json，不能只凭总均值判断稳定性。']
limitation_labels = {
    "historical publication vintages and independent certification remain unavailable": "历史发布版本证据及独立质量认证仍未完成。",
    "full pre-repair October dataset absent": "缺少整个十月修复前的完整输入快照，不能量化全部修复效果。",
    "fixed existing factor list selected using past research": "固定因子清单已在既往研究中选择过，历史区间不是研究者从未接触的最终留出集。",
    "no actual execution, limit/ST status or fee-adjusted NAV": "没有实际成交、历史涨跌停/ST状态与费用后账户净值，标签篮子收益不等于可实现收益。",
    "equal per-model parameters without tuning; expert system consumes more total compute": "各模型采用相同参数且未调参，专家系统使用更多总算力；尚未比较同总预算的更强统一模型。",
    "cannot infer effects for LGB+LSTM or all possible stock groupings": "仅验证纯LightGBM，不能推论LGB＋LSTM融合模型或其他分组方法。",
    "expanded 2019 context panel misses 98% overall coverage for market/industry relations": "扩展面板含早期预热样本，其市场/行业关系整体覆盖未达到98%；实际测试样本覆盖另行报告。",
    "shared research Corr computation aligns constant masks by date for unequal feature intervals; production adapter unchanged": "研究中的相关系数常数屏蔽按日期对齐，所有组共用；生产适配层未修改。",
}
lines += [f"- {limitation_labels.get(value, value)}" for value in manifest["limitations"]]
lines += ["", "C的六组为金融地产、资源材料、基建公用、工业制造、消费医疗、科技传媒；未知或专家不可用样本走B回退。它检验的是这套粗行业方案，不代表所有‘同逻辑/同类型’分组。"]
lines += ["", "因子清单参考工件为 `656749665990832865/5501ed88f12145f0a454d68d92dbad76`，40个因子键已再次逐项核对。该工件本身是LGB＋LSTM，本次只运行纯LightGBM；未据此认定该工件仍是当前生产Champion。详细来源见 reference_provenance.json。"]
if 'FundVal v2' in audit.get('scope', ''):
    lines += ['', 'A0为冻结的全零FundVal基线，A1为修复后的同40因子；B为修复后79因子全池，C为同79因子行业专家。A1−A0衡量接入与缓存缺失处理的整体修复效果，B−A1是上下文增量，C−B是拆分增量。',
              '输入验证见repair_verification.json：9列通过真实Qlib算子提取，并与直接缓存参考计算核对。']
else:
    lines += ["", "输入审计：", "", "- 今日旧新行情 bin 完全一致；当前40因子没有直接读取发生变动的财务/估值 bin，9列依赖独立基本面缓存。", "- FundVal按带交易所前缀的代码查找六位代码缓存，未命中时返回零；各折常数特征及旧新差异分别保存在 *_constant_features.json、*_feature_changes.json。若全部原始输入与标签相同，A0明确复用同种子A1，不重复训练。", "- 新代ROE、EPS、净利润同比、收入同比、资产负债率的2019年起历史有限值全部置空。这是历史可见时间证据约束造成的覆盖变化，不代表数据库原始财务行被删除；对依赖这些字段的其他模型需另测。", "- 严格PIT和独立质量签名继续未认证；本轮未修改这些状态。"]
lines += ["- 2019年起完整上下文面板包含早期预热样本，市场/行业关系缺失率4.72%/3.04%不能直接当作测试样本缺失率；各折实际共同测试样本的字段缺失率另存 context_coverage_used.csv。"]
coverage_path = OUT / "qlib_denominator_coverage.json"
if coverage_path.exists():
    qlib_coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    decision["qlib_denominator_coverage"] = qlib_coverage
    lines += [f"- 冻结Qlib池相对G0可交易分母的有效收盘价覆盖为{qlib_coverage['coverage']:.2%}（{qlib_coverage['total_positive_close']:,}/{qlib_coverage['total_expected']:,}股票日）；本轮是可用共同样本研究，不代表所有A股历史均已完整。"]
(OUT / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"summary": summary, "effects": effects}, ensure_ascii=False), flush=True)
