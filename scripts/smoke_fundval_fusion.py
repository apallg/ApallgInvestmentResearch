import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('POLARS_SKIP_CPU_CHECK', '1')
os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE', 'true')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from core.shared import config as app_config
from core.adapters.qlib import config as qlib_config

OUT = ROOT / 'data/experiments/fundval_repair_research_20261026/fusion_smoke'
OUT.mkdir(parents=True, exist_ok=True)
app_config.QLIB_PROVIDER_URI = str(ROOT / 'data/qlib_data_prewrite_20261025')
app_config.QLIB_MLFLOW_URI = str(OUT / 'mlruns')
qlib.init(provider_uri=app_config.QLIB_PROVIDER_URI, region=REG_CN, kernels=2,
          joblib_backend='threading', expression_cache=None, dataset_cache=None,
          exp_manager={'class': 'MLflowExpManager', 'module_path': 'qlib.workflow.expm',
                       'kwargs': {'uri': (OUT / 'mlruns').as_uri(), 'default_exp_name': 'fundval_integration_smoke'}})
sys.excepthook = sys.__excepthook__
qlib_config._initialized = True
qlib_config._patch_qlib_git_encoding()
from core.adapters.qlib.models import QlibModelManager, PARAM_PROFILES
from core.adapters.qlib.lstm_fusion import train_lgb_lstm_fusion, persist_fusion_run, predict_lgb_lstm_fusion
from core.adapters.qlib.fundval import require_run_contract
import torch

torch.set_num_threads(2)
PARAM_PROFILES['conservative']['lightgbm']['num_threads'] = 2
keys = ['fund_turnover', 'SP_n5', 'dp_historical_n12_delta_window5', 'a158_ma10_60']
instruments = ['sh600000', 'sz000001', 'sh600036', 'sz000333', 'sh601318', 'sh600519',
               'sz000651', 'sh600009', 'sh601166', 'sh600016', 'sz000002', 'sh600030']
train, valid, test = ('2023-01-01', '2023-06-30'), ('2023-07-01', '2023-12-31'), ('2024-01-01', '2024-03-29')
manager = QlibModelManager()
task = manager._build_task_config('lightgbm_rank', 'WhiteBoxHandler', instruments, train, valid, test,
    label_horizon=10, custom_factor_keys=keys, label_norm='none', enable_cross_sectional=False)
dataset = init_instance_by_config(task['dataset'])
result = train_lgb_lstm_fusion(keys, instruments, train, valid, test, label_horizon=10,
    label_norm='none', lstm_epochs=2, dataset=dataset, seed=42)
run_id = persist_fusion_run(result, keys, instruments, train, valid, test, label_horizon=10,
    label_norm='none', experiment_name='fundval_integration_smoke',
    training_metadata={'research_only': True, 'deployment_eligible': False, 'training_mode': 'integration_smoke'})
from qlib.workflow import R
experiment = R.get_exp(experiment_name='fundval_integration_smoke')
experiment_id = str(experiment.id)
artifacts = OUT / 'mlruns' / experiment_id / run_id / 'artifacts'
require_run_contract(artifacts, keys)
predicted, _ = predict_lgb_lstm_fusion(experiment_id, run_id, test[0], test[1],
                                     instruments=instruments, dataset=dataset, use_cache=False)
expected = result['fused_pred'].reindex(predicted.index)
np.testing.assert_allclose(expected, predicted, atol=1e-7, equal_nan=False)
report = {'passed': True, 'research_only': True, 'deployment_eligible': False,
          'epochs': 2, 'stocks': len(instruments), 'factor_keys': keys,
          'restored_predictions': len(predicted), 'model_id': f'{experiment_id}/{run_id}',
          'maximum_roundtrip_difference': float(np.max(np.abs(predicted - expected))),
          'feature_contract': result['feature_contract'],
          'scope': 'real Handler + LightGBM/LSTM training, isolated MLflow persistence and restored prediction; no performance claim'}
(OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False), flush=True)
