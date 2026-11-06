import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/experiments/repair_group_research_20261025'
OUT = ROOT / 'data/experiments/fundval_repair_research_20261026'
verification = json.loads((OUT / 'repair_verification.json').read_text(encoding='utf-8'))
if not verification['passed']:
    raise RuntimeError('FundVal逐值验证尚未通过')
if not json.loads((OUT / 'full_reference_verification.json').read_text(encoding='utf-8'))['passed']:
    raise RuntimeError('完整逐值参考核对尚未通过')
for name in ['reference_provenance.json', 'qlib_denominator_coverage.json']:
    if not (OUT / name).exists():
        shutil.copy2(BASE / name, OUT / name)
source = (ROOT / 'scripts/run_repair_group_research.py').read_text(encoding='utf-8')
source = source.replace('ROOT = Path(__file__).resolve().parents[1]', f'ROOT = Path({str(ROOT)!r})')
source = source.replace('data/experiments/repair_group_research_20261025', 'data/experiments/fundval_repair_research_20261026')
source = source.replace('data/qlib_data/calendars/day.txt', 'data/qlib_data_prewrite_20261025/calendars/day.txt')
start = source.index('native = pd.read_parquet')
end = source.index('input_files = {}', start)
source = source[:start] + source[end:]
source = source.replace('[OUT / "native_reference_2019.parquet", context_path]', '[context_path, OUT / "repair_verification.json", OUT / "full_reference_verification.json", OUT / "input_identity.json", ROOT / "data/qlib_data_prewrite_20261025/calendars/day.txt"]')
source = source.replace('2026-10-25 prewrite inputs, pooled 40 features', 'frozen broken FundVal baseline, pooled 40 features')
source = source.replace('current inputs, pooled same 40 features', 'corrected FundVal v2, pooled same 40 features')
source = source.replace("today's input generation effect only", 'FundVal integration and cache missing-value repair effect')
source = source.replace('"scripts/research_feature_ops.py"]', '"scripts/research_feature_ops.py", "core/data/fundamental.py", "core/adapters/qlib/fundval.py", "core/adapters/qlib/factor_resolver.py", "core/shared/fundamental_contract.py", "core/adapters/qlib/models.py", "core/adapters/qlib/factor_set_evaluator.py", "scripts/verify_fundval_repair.py", "scripts/build_stock_context_p1_panel.py", "scripts/analyze_repair_group_research.py"]')
runner = OUT / 'run_training.py'
if runner.exists() and runner.read_text(encoding='utf-8') != source:
    raise RuntimeError('研究脚本已冻结，不能改变后混用模型检查点')
runner.write_text(source, encoding='utf-8')
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', PYTHONIOENCODING='utf-8',
           REPAIR_RESEARCH_OUTPUT=str(OUT))
try:
    for phase, script in [('training', runner), ('analysis', ROOT / 'scripts/analyze_repair_group_research.py')]:
        print(phase, flush=True)
        with (OUT / f'{phase}.log').open('a', encoding='utf-8') as handle:
            subprocess.run([sys.executable, '-u', str(script)], cwd=ROOT, env=env,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
    (OUT / 'status.json').write_text(json.dumps({'state': 'complete', 'phase': 'analysis_complete',
        'research_only': True, 'deployment_eligible': False, 'report': str(OUT / 'report.md')}, indent=2), encoding='utf-8')
except Exception:
    (OUT / 'status.json').write_text(json.dumps({'state': 'failed', 'phase': phase,
        'research_only': True}, indent=2), encoding='utf-8')
    raise
