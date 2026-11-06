import sys
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scripts.h20_confirmation_v2 import OUTPUT, digest, save_json


def main():
    audit_path = OUTPUT / "repair/source_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "verified":
        raise RuntimeError("双源审计尚未全部通过")
    source = ROOT / "data/qlib_data/generations/a5d639b90686460eb04b25bedcbaf2c2"
    original = json.loads((OUTPUT / "inputs/full/2026-11-05/manifest.pre_audit_correction.json").read_text(encoding="utf-8"))
    if digest(source / "generation.json") != original["source_sha256"]["qlib_manifest"]:
        raise RuntimeError("原冻结 Qlib 代签名冲突")
    destination = OUTPUT / "repair/provider"
    final = destination / "repair_manifest.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8"))
        if result["source_audit_sha256"] != digest(audit_path):
            raise RuntimeError("隔离修复输入签名冲突")
        print("existing_verified_provider")
        return
    if shutil.disk_usage("G:/").free < 12 * 1024 ** 3:
        raise RuntimeError("G盘空间不足以保留10GiB与修复余量")
    destination.mkdir(parents=True, exist_ok=True)
    state = destination / "build_state.json"
    signature = {"source": str(source), "source_audit_sha256": digest(audit_path), "code_sha256": digest(__file__)}
    if state.exists() and json.loads(state.read_text(encoding="utf-8")) != signature:
        raise RuntimeError("修复断点签名冲突")
    save_json(state, signature)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    calendar = pd.DatetimeIndex(pd.to_datetime((destination / "calendars/day.txt").read_text().splitlines()))
    intervals = {line.split('\t')[0]: line for line in (source / "instruments/all.txt").read_text().splitlines()}
    hashes = {}
    for row in audit["stocks"]:
        code = row["code"]
        instrument = ("sh" if code.startswith("6") else "sz") + code
        captured = OUTPUT / "repair/captured_sources" / code
        data = pd.read_parquet(captured / "tushare_normalized.parquet").set_index("trade_date").sort_index()
        data = data.loc[data.vol.gt(0) & data.index.isin(calendar)]
        if data.empty or not np.isfinite(data.adj_factor).all() or not data.adj_factor.gt(0).all():
            raise RuntimeError(f"非法原始行情或复权因子: {code}")
        ratio = data.adj_factor / data.adj_factor.iloc[-1]
        frame = data[["open", "high", "low", "close"]].mul(ratio, axis=0)
        frame["volume"] = data.vol * 100
        frame["amount"] = data.amount * 1000
        frame["vwap"] = frame.amount / frame.volume * ratio
        frame["factor"] = 1.0
        if not np.isfinite(frame.to_numpy()).all() or not frame[["open", "high", "low", "close"]].gt(0).all().all():
            raise RuntimeError(f"复权重建非法: {code}")
        positions = calendar.get_indexer(frame.index)
        folder = destination / "features" / instrument
        folder.mkdir(parents=True, exist_ok=True)
        for field in frame:
            values = np.full(positions[-1] - positions[0] + 1, np.nan, dtype=np.float32)
            values[positions - positions[0]] = frame[field].to_numpy(dtype=np.float32)
            path = folder / f"{field}.day.bin"
            temporary = path.with_suffix(".tmp")
            np.r_[np.float32(positions[0]), values].astype('<f4').tofile(temporary)
            temporary.replace(path)
            restored = np.fromfile(path, dtype='<f4')
            np.testing.assert_array_equal(restored[1:][positions - positions[0]], frame[field].to_numpy(dtype=np.float32))
            hashes[str(path.relative_to(destination))] = digest(path)
        intervals[instrument] = f"{instrument}\t{frame.index[0].date()}\t{frame.index[-1].date()}"
    instruments = destination / "instruments/all.txt"
    instruments.write_text('\n'.join(intervals[key] for key in sorted(intervals)) + '\n', encoding="utf-8")
    save_json(destination / "generation.json", {"type": "isolated_research_repair", "parent_generation_sha256": digest(source / "generation.json"), "source_audit_sha256": digest(audit_path)})
    save_json(final, {"status": "verified", **signature, "repaired_stocks": len(audit["stocks"]),
                     "as_of": "2026-11-05", "sha256": hashes, "instruments_sha256": digest(instruments),
                     "research_only": True, "production_qualified": False})
    print(json.dumps({"status": "verified", "stocks": len(audit["stocks"]), "provider": str(destination)}))


if __name__ == "__main__":
    main()
