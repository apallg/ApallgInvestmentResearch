import sys
import os
import atexit
import shutil
import tempfile
import pytest
import numpy as np

_runtime_dir = tempfile.mkdtemp(prefix=f"apallg_pytest_{os.getpid()}_")
os.environ["APALLG_FACTOR_POOL_STATE_PATH"] = os.path.join(
    _runtime_dir, "factor_pool_state.json",
)
atexit.register(shutil.rmtree, _runtime_dir, ignore_errors=True)

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(42)
