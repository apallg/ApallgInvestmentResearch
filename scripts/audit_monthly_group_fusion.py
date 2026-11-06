import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.adapters.qlib.lstm_fusion import _LSTMReg, _DEVICE
from scripts.confirm_monthly_group_fusion import predict
from scripts.run_monthly_group_research import OUTPUT, SEEDS
from scripts.train_monthly_group_research import load_fold


def main():
    source = OUTPUT / "fusion_confirmation"
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    report = {"status": "passed", "prediction_files": 0, "training_audits": 0,
              "common_stock_days_and_labels": True, "finite_scores": True, "restore_chunk_max_error": {}}
    for fold in manifest["folds"]:
        reference = None
        for seed in SEEDS:
            detail_path = source / f"{fold}_{seed}_audit.json"
            if not detail_path.exists():
                raise RuntimeError(f"融合训练审计缺失: {detail_path.name}")
            json.loads(detail_path.read_text(encoding="utf-8"))
            report["training_audits"] += 1
            for arm in ("D", "DM", "UM", "SL4_M"):
                path = source / f"{fold}_{seed}_{arm}.parquet"
                frame = pd.read_parquet(path).sort_index()
                if reference is None:
                    reference = frame["label"]
                if not frame.index.equals(reference.index) or not np.allclose(frame["label"], reference, rtol=0, atol=1e-7):
                    raise RuntimeError(f"融合共同股票日或标签不一致: {path.name}")
                if not np.isfinite(frame["score"]).all():
                    raise RuntimeError(f"融合预测分数非有限值: {path.name}")
                report["prediction_files"] += 1
    arms, _, _, _, _, _, _ = load_fold(manifest, "W1")
    for arm, group_id in (("D", None), ("DM", None), ("UM", None), ("SL4_M", 0)):
        name = arm if group_id is None else f"{arm}_{group_id}"
        path = source / f"W1_42_{name}.pkl"
        with path.open("rb") as handle:
            stored = pickle.load(handle)
        bundle = stored["bundle"]
        x = arms["UM" if arm == "SL4_M" else arm].sort_index(level=[1, 0]).fillna(0)
        if list(x.columns) != bundle["features"]:
            raise RuntimeError(f"保存模型特征列不一致: {name}")
        target = stored["scores"]["test"].iloc[:128]
        positions = x.index.get_indexer(target.index)
        if (positions < 19).any():
            raise RuntimeError(f"模型恢复样本预热不足: {name}")
        values = x.to_numpy(dtype=np.float32)[positions[:, None] - np.arange(19, -1, -1)]
        scaled = (values - np.asarray(bundle["mean"], dtype=np.float32)) / np.asarray(bundle["scale"], dtype=np.float32)
        model = _LSTMReg(x.shape[1]).to(_DEVICE)
        model.load_state_dict(bundle["state"])
        model.eval()
        whole = predict(model, scaled)
        chunked = np.concatenate([predict(model, scaled[start:start + 31]) for start in range(0, len(scaled), 31)])
        restore_error = float(np.max(np.abs(whole - target.to_numpy())))
        chunk_error = float(np.max(np.abs(whole - chunked)))
        if restore_error > 5e-4 or chunk_error > 5e-4:
            raise RuntimeError(f"模型恢复或分片超出容差: {name}, {restore_error}, {chunk_error}")
        report["restore_chunk_max_error"][name] = {"saved_score": restore_error, "chunked": chunk_error}
    (source / "output_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
