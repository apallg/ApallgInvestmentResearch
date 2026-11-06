"""DEPRECATED: 已由 backend/api/qlib/predict_v2.py 和 backend/service/prediction.py 取代。"""
from fastapi import APIRouter, BackgroundTasks, Body, HTTPException
from backend.api.qlib.common import logger, task_manager, _get_qlib_data_dir
from backend.service.qlib_path_guard import assert_run_path as _assert_run_path
from backend.service.qlib_access import run_with_qlib_read_access
from backend.api.qlib.models import PredictRequest, BacktestRequest

router = APIRouter()

def _get_top_features(model, dataset, top_n=10):
    """从模型和数据集提取 Top-N 特征重要性及名称"""
    import numpy as np

    top_features = []
    try:
        if hasattr(model, "booster_"):
            booster = model.booster_
            imp = booster.feature_importance()
            names = booster.feature_name()
            idx = np.argsort(imp)[::-1][:top_n]
            for rank, i in enumerate(idx, 1):
                top_features.append({
                    "name": names[i],
                    "importance": round(float(imp[i]), 2),
                    "rank": rank,
                })
    except Exception:
        logger.warning("特征重要性提取失败", exc_info=True)
    return top_features
def _get_feature_values(dataset, pred_index, top_features):
    """从 dataset 提取预测日期的因子值"""
    import pandas as pd

    try:
        X_test, _ = dataset.prepare("test")
        if not isinstance(X_test, pd.DataFrame):
            return {}
        feature_names = [f["name"] for f in top_features if f["name"] in X_test.columns]
        if not feature_names:
            return {}

        X_sub = X_test[feature_names]
        result = {}
        for dt, inst in pred_index:
            key = (pd.Timestamp(dt), inst) if not isinstance(dt, pd.Timestamp) else (dt, inst)
            try:
                row = X_sub.loc[key]
                result[f"{dt}_{inst}"] = {
                    fn: round(float(row[fn]), 4)
                    for fn in feature_names
                }
            except (KeyError, TypeError):
                continue
        return result
    except Exception:
        logger.warning("因子值提取失败", exc_info=True)
        return {}
def _do_predict(experiment_id: str, run_id: str, instruments: str,
                start_time: str, end_time: str, topk: int,
                skip_features: bool, code_prefixes: list):
    """预测执行体 — 在后台线程中运行"""
    import pickle
    from pathlib import Path as _Path
    from core.shared import config as app_config

    mlruns_dir = _Path(app_config.QLIB_MLFLOW_URI.replace("file:", "").replace("file:", ""))
    model_path = _assert_run_path(mlruns_dir, experiment_id, run_id) / "artifacts" / "params.pkl"
    from core.adapters.qlib.fundval import require_run_contract
    trained_keys = require_run_contract(model_path.parent)

    factor_set = "Alpha158"
    params_dir = mlruns_dir / experiment_id / run_id / "params"
    if params_dir.exists():
        for pf in params_dir.iterdir():
            try:
                if pf.name == "dataset.kwargs.handler.class":
                    factor_set = pf.read_text().strip()
            except Exception:
                pass

    from core.adapters.qlib.config import init_qlib
    init_qlib()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": factor_set,
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "instruments": instruments,
                    "start_time": start_time,
                    "end_time": end_time,
                    "fit_start_time": start_time,
                    "fit_end_time": end_time,
                },
            },
            "segments": {"test": (start_time, end_time)},
        },
    }

    handler_module = "core.adapters.qlib.custom_handler" if factor_set in ("WhiteBoxHandler",) or factor_set.startswith("WhiteBox") else "qlib.contrib.data.handler"
    dataset_config["kwargs"]["handler"]["module_path"] = handler_module
    if factor_set.startswith('WhiteBox'):
        if not trained_keys:
            raise RuntimeError('无法还原模型的实际因子清单')
        dataset_config['kwargs']['handler']['kwargs']['factor_keys'] = trained_keys

    from core.adapters.qlib.config import build_dataset_cached
    dataset = build_dataset_cached(dataset_config)
    top_features = [] if skip_features else _get_top_features(model, dataset)
    pred = model.predict(dataset)

    if code_prefixes:
        prefixes = tuple(str(p) for p in code_prefixes)

    dates_summary = {}
    records = []
    if hasattr(pred, "reset_index"):
        pred_df = pred.reset_index()
        if len(pred_df.columns) >= 3:
            pred_df.columns = ["datetime", "instrument", "score"]
        pred_df["datetime"] = pred_df["datetime"].astype(str).str[:10]
        pred_df["instrument"] = pred_df["instrument"].astype(str)

        if code_prefixes:
            raw_code = pred_df["instrument"].str[2:]
            pred_df = pred_df[raw_code.str.startswith(prefixes)]

        pred_df = pred_df.sort_values("score", ascending=False)
        top_rows = pred_df.head(topk * 5)
        pred_index = []
        for _, r in top_rows.iterrows():
            inst = str(r["instrument"])
            dt = r["datetime"]
            pred_index.append((dt, inst))
            records.append({
                "datetime": dt, "instrument": inst,
                "score": float(r["score"]),
            })

        feature_values = _get_feature_values(dataset, pred_index, top_features) if top_features else {}
        for rec in records:
            key = f"{rec['datetime']}_{rec['instrument']}"
            rec["factors"] = feature_values.get(key, {})

        for dt, group in pred_df.groupby("datetime"):
            top = group.head(topk)
            day_stocks = []
            for _, r in top.iterrows():
                inst_d = str(r["instrument"])
                key = f"{dt}_{inst_d}"
                day_stocks.append({
                    "instrument": inst_d,
                    "score": float(r["score"]),
                    "factors": feature_values.get(key, {}),
                })
            dates_summary[dt] = day_stocks

        logger.info(f"预测完成: records={len(records)}, dates={len(dates_summary)}")

        try:
            model_id = f"{experiment_id}/{run_id}"
            db_rows = []
            all_pred = pred_df.copy() if hasattr(pred, "reset_index") else None
            if all_pred is not None and not all_pred.empty:
                for _, r in all_pred.iterrows():
                    inst = str(r["instrument"])
                    code = inst[2:] if inst.startswith(("sh", "sz")) else inst
                    dt = str(r["datetime"])[:10]
                    db_rows.append((model_id, code, dt, float(r["score"])))
            if db_rows:
                from core.data import Database
                batch_size = 500
                for bs in range(0, len(db_rows), batch_size):
                    batch = db_rows[bs:bs + batch_size]
                    ph = ",".join(["(%s,%s,%s,%s)"] * len(batch))
                    flat = []
                    for b in batch:
                        flat.extend(b)
                    Database.execute(
                        f"INSERT INTO qlib_predictions (model_id, code, trade_date, score) "
                        f"VALUES {ph} ON DUPLICATE KEY UPDATE score=VALUES(score)",
                        tuple(flat),
                    )
                logger.info(f"预测入库: {len(db_rows)} 条")
        except Exception as e:
            logger.warning(f"预测入库失败: {e}")
    elif hasattr(pred, "to_dict"):
        records = [{"score": float(v)} for v in list(pred)]

    return {
        "predictions": records,
        "count": len(records),
        "topk": topk,
        "by_date": dict(list(dates_summary.items())[:30]),
        "feature_importance": top_features,
    }
@router.post("/models/predict")
async def predict(req: PredictRequest):
    """加载已训练模型并生成预测 (异步后台执行)"""
    from core.shared import config as app_config
    from pathlib import Path as _Path

    if not req.experiment_id or not req.run_id:
        raise HTTPException(400, "必须指定 experiment_id 和 run_id")

    from core.shared.disabled_models import is_disabled
    if is_disabled(req.experiment_id, req.run_id):
        raise HTTPException(400, "模型已停用，请先恢复后再使用")

    mlruns_dir = _Path(app_config.QLIB_MLFLOW_URI.replace("file:", "").replace("file:", ""))
    model_path = _assert_run_path(mlruns_dir, req.experiment_id, req.run_id) / "artifacts" / "params.pkl"
    if not model_path.exists():
        raise HTTPException(400, f"模型文件不存在: {model_path}")

    job_id = task_manager.submit(
        lambda: run_with_qlib_read_access(
            _do_predict,
            experiment_id=req.experiment_id,
            run_id=req.run_id,
            instruments=req.instruments,
            start_time=req.start_time,
            end_time=req.end_time,
            topk=req.topk,
            skip_features=req.skip_features,
            code_prefixes=req.code_prefixes,
        ),
        description=f"预测 {req.experiment_id}/{req.run_id}",
        pool="default",
    )
    return {"data": {"job_id": job_id, "status": "started"}}
@router.get("/models/predict/{job_id}")
async def get_predict_result(job_id: str):
    """查询预测任务结果"""
    task = task_manager.get_dict(job_id)
    if task is None:
        return {"data": None, "error": f"任务 {job_id} 不存在"}
    return {"data": task}
def _do_predict_dual(experiment_id: str, run_id: str, instruments: str,
                     start_time: str, end_time: str, topk: int,
                     skip_features: bool, code_prefixes: list):
    """双模型预测执行体 — 在后台线程中运行"""
    import numpy as np, pandas as pd

    from core.adapters.qlib.models import QlibModelManager
    from core.adapters.qlib.config import init_qlib
    init_qlib()

    manager = QlibModelManager()
    fused, dataset = manager.predict_dual(
        experiment_id, run_id,
        predict_start=start_time, predict_end=end_time,
        instruments=instruments,
    )

    if hasattr(fused, 'reset_index'):
        pred_df = fused.reset_index()
        if len(pred_df.columns) >= 3:
            pred_df.columns = ["datetime", "instrument", "score"]
        pred_df["datetime"] = pred_df["datetime"].astype(str).str[:10]
        pred_df["instrument"] = pred_df["instrument"].astype(str)
    else:
        arr = np.array(fused).ravel()
        pred_df = pd.DataFrame({"score": arr, "instrument": "", "datetime": ""})

    if code_prefixes:
        prefixes = tuple(str(p) for p in code_prefixes)
        raw_code = pred_df["instrument"].str[2:]
        pred_df = pred_df[raw_code.str.startswith(prefixes)]

    pred_df = pred_df.sort_values("score", ascending=False)
    top_rows = pred_df.head(topk * 5)

    records = []
    dates_summary = {}
    for _, r in top_rows.iterrows():
        records.append({"datetime": str(r["datetime"])[:10], "instrument": str(r["instrument"]),
                        "score": float(r["score"])})
    for dt, group in pred_df.groupby("datetime"):
        dates_summary[str(dt)[:10]] = [
            {"instrument": str(r["instrument"]), "score": float(r["score"])}
            for _, r in group.head(topk).iterrows()
        ]

    try:
        model_id = f"{experiment_id}/{run_id}/dual"
        db_rows = []
        for _, r in pred_df.iterrows():
            inst = str(r["instrument"])
            code = inst[2:] if inst.startswith(("sh", "sz")) else inst
            db_rows.append((model_id, code, str(r["datetime"])[:10], float(r["score"])))
        if db_rows:
            from core.data import Database
            for bs in range(0, len(db_rows), 500):
                batch = db_rows[bs:bs + 500]
                ph = ",".join(["(%s,%s,%s,%s)"] * len(batch))
                flat = [x for b in batch for x in b]
                Database.execute(
                    f"INSERT INTO qlib_predictions (model_id, code, trade_date, score) "
                    f"VALUES {ph} ON DUPLICATE KEY UPDATE score=VALUES(score)", tuple(flat))
            logger.info(f"[PredictDual] 入库: {len(db_rows)} 条")
    except Exception as e:
        logger.warning(f"[PredictDual] 入库失败: {e}")

    return {
        "predictions": records, "count": len(records), "topk": topk,
        "by_date": dict(list(dates_summary.items())[:30]),
    }
@router.post("/models/predict-dual")
async def predict_dual(req: PredictRequest):
    """双模型等权融合预测 (异步后台执行)"""
    if not req.experiment_id or not req.run_id:
        raise HTTPException(400, "必须指定 experiment_id 和 run_id")

    from core.shared.disabled_models import is_disabled
    if is_disabled(req.experiment_id, req.run_id):
        raise HTTPException(400, "模型已停用，请先恢复后再使用")

    job_id = task_manager.submit(
        lambda: run_with_qlib_read_access(
            _do_predict_dual,
            experiment_id=req.experiment_id,
            run_id=req.run_id,
            instruments=req.instruments,
            start_time=req.start_time,
            end_time=req.end_time,
            topk=req.topk,
            skip_features=req.skip_features,
            code_prefixes=req.code_prefixes,
        ),
        description=f"双模型预测 {req.experiment_id}/{req.run_id}",
        pool="default",
    )
    return {"data": {"job_id": job_id, "status": "started"}}
@router.get("/predictions/{code}")
async def get_stock_predictions(code: str, days: int = 60, model_id: str = None):
    """查询某只股票的历史预测分数"""
    from core.data import Database

    if model_id:
        rows = Database.fetchall(
            "SELECT trade_date, score FROM qlib_predictions "
            "WHERE code=%s AND model_id=%s AND trade_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY) "
            "ORDER BY trade_date DESC LIMIT 200",
            (code, model_id, days),
        )
    else:
        rows = Database.fetchall(
            "SELECT trade_date, score, model_id FROM qlib_predictions "
            "WHERE code=%s AND trade_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY) "
            "ORDER BY trade_date DESC LIMIT 200",
            (code, days),
        )
    return {"data": [{"date": str(r["trade_date"]), "score": float(r["score"]),
                     "model_id": r.get("model_id", "")} for r in rows]}


@router.post("/backtest/run")
async def run_backtest(req: BacktestRequest):
    """运行 qlib 回测"""
    import pickle
    from pathlib import Path
    from core.shared import config as app_config

    pred_path = Path(req.predictions_file).resolve()
    mlruns_dir = Path(app_config.QLIB_MLFLOW_URI.replace("file:", "").replace("file:", "")).resolve()
    data_root = Path(_get_qlib_data_dir()).resolve()
    try:
        pred_path.relative_to(mlruns_dir)
    except ValueError:
        try:
            pred_path.relative_to(data_root)
        except ValueError:
            raise HTTPException(400, f"预测文件必须在服务端数据目录内: {req.predictions_file}")
    if not pred_path.exists():
        raise HTTPException(400, f"预测文件不存在: {req.predictions_file}")

    data = pickle.loads(pred_path.read_bytes())

    from core.adapters.qlib.backtest import QlibBacktestRunner
    runner = QlibBacktestRunner()

    if isinstance(data, tuple) and len(data) == 2:
        predictions, label = data
    else:
        predictions = data
        label = None

    if label is None:
        raise HTTPException(400, "预测文件缺少标签数据，请保存为 (predictions, label) 元组")

    result = runner.run_from_predictions(
        predictions=predictions,
        label=label,
        topk=req.topk,
        n_drop=req.n_drop,
        start_time=req.start_time,
        end_time=req.end_time,
    )
    return {"data": result}
