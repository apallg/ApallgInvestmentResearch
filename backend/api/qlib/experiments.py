"""DEPRECATED: 实验概览页签已移除，本模块不再挂载到 qlib router。"""
from fastapi import APIRouter, Body, HTTPException
from backend.api.qlib.common import *  # noqa: F403

router = APIRouter()

@router.get("/experiments")
async def list_experiments():
    """列出 MLflow 实验"""
    from core.adapters.qlib.config import init_qlib
    init_qlib()
    from qlib.workflow import R
    from qlib.workflow.expm import MLflowExpManager
    try:
        exp_manager = R.exp_manager
        experiments = exp_manager.client.search_experiments()
        return {"data": [{"id": e.experiment_id, "name": e.name} for e in experiments]}
    except Exception as e:
        return {"data": [], "error": str(e)}
@router.get("/experiments/{exp_id}")
async def get_experiment(exp_id: str):
    """获取实验详情及所有 run"""
    import yaml
    from pathlib import Path
    from core.shared import config as app_config

    mlruns_dir = Path(app_config.QLIB_MLFLOW_URI.replace("file:", "").replace("file:", ""))
    exp_dir = mlruns_dir / exp_id
    if not exp_dir.exists():
        return {"data": {"runs": [], "name": exp_id}}

    exp_meta_file = exp_dir / "meta.yaml"
    exp_name = exp_id
    if exp_meta_file.exists():
        try:
            with open(exp_meta_file) as f:
                exp_name = yaml.safe_load(f).get("name", exp_id)
        except Exception:
            pass

    runs = []
    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name == "meta.yaml":
            continue
        run_data = {"run_id": run_dir.name, "metrics": {}, "params": {}}

        metrics_dir = run_dir / "metrics"
        if metrics_dir.exists():
            for mf in metrics_dir.iterdir():
                try:
                    parts = mf.read_text().strip().split()
                    if len(parts) >= 2:
                        run_data["metrics"][mf.name] = float(parts[1])
                except Exception:
                    pass

        params_dir = run_dir / "params"
        if params_dir.exists():
            for pf in params_dir.iterdir():
                try:
                    run_data["params"][pf.name] = pf.read_text().strip()
                except Exception:
                    pass

        if (run_dir / "artifacts" / "params.pkl").exists():
            run_data["has_model"] = True
        if (run_dir / "artifacts" / "pred.pkl").exists():
            run_data["has_predictions"] = True

        runs.append(run_data)

    runs.sort(key=lambda r: r["metrics"].get("IC", 0), reverse=True)
    return {"data": {"name": exp_name, "runs": runs}}


# ── 每日自动采集调度器 API ──
