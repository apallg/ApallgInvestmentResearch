import gc
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.monthly_group_features import build_classifications
from scripts.run_monthly_group_research import OUTPUT, REFERENCE, ROOT, SOURCE, sha256
from scripts.train_monthly_group_research import industry_groups


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    membership = REFERENCE / "stock_index_membership_snapshot.json"
    target = OUTPUT / membership.name
    if target.exists() and sha256(target) != sha256(membership):
        raise RuntimeError("行业快照与冻结版本不一致")
    shutil.copy2(membership, target)
    preheat = OUTPUT / "preheat_2018.parquet"
    if not preheat.exists():
        raise RuntimeError("缺少2018年预热快照")
    reference = json.loads((REFERENCE / "manifest.json").read_text(encoding="utf-8"))
    end = pd.Timestamp(reference["folds"]["W8"]["test"][1])
    predicate = [("datetime", "<=", end)]
    fields = ["open", "high", "low", "close", "volume"]
    quotes = pd.concat([pd.read_parquet(path, columns=fields, filters=predicate)
                        for path in sorted(SOURCE.glob("new_2*.parquet"))]).sort_index()
    if quotes.index.has_duplicates:
        raise RuntimeError("冻结行情含重复股票日")
    context = pd.read_parquet(SOURCE / "context/context_features.parquet", filters=predicate).reindex(quotes.index)
    history = pd.concat([pd.read_parquet(preheat), quotes]).sort_index()
    industry, codes = industry_groups(history.index)
    calendar = pd.DatetimeIndex(pd.to_datetime((ROOT / "data/qlib_data/calendars/day.txt").read_text(encoding="utf-8").splitlines()))
    groups, audit = build_classifications(context, None, history, industry, calendar)
    if not groups.index.equals(quotes.index):
        raise RuntimeError("分类快照股票日未对齐")
    output = OUTPUT / "classification_snapshot.parquet"
    if output.exists():
        if not pd.read_parquet(output).equals(groups):
            raise RuntimeError("已有分类快照内容不同")
    else:
        groups.to_parquet(output, compression="zstd")
    (OUTPUT / "classification_audit.json").write_text(json.dumps({"groups": audit, "industry_codes": codes,
                                                                    "rows": len(groups), "snapshot_sha256": sha256(output)},
                                                                   ensure_ascii=False, indent=2, default=int), encoding="utf-8")
    print(json.dumps({"rows": len(groups), "snapshot_sha256": sha256(output)}, ensure_ascii=False))
    del context, quotes, history, industry, groups
    gc.collect()


if __name__ == "__main__":
    main()
