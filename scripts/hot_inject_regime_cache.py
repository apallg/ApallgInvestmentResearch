"""独立进程热注入 regime 缓存到 Redis，阻止 get_current_regime() 触发 DB 查询"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv()

from core.domain.regime.classifier import hot_inject_regime_cache

hot_inject_regime_cache()
print("OK")
