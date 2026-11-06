"""
配置校验 — 单元测试
覆盖: 类型/范围检查, 布尔值校验, 边界值, 空字符串, 极值
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestValidateConfig:
    """validate_config() 检测配置错误和警告"""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """重置 env 和 config module 状态"""
        # 清除可能存在的环境变量
        for key in list(os.environ.keys()):
            if key.startswith(("MYSQL_", "QLIB_", "REDIS_", "BUY_SIDE_",
                               "FUSION_", "COLLECT_", "REGIME_", "BARRA_",
                               "MACRO_", "ERROR_LOG_", "FORCE_CLOSE_",
                               "SIGNAL_", "SMOKE_")) and key != "MYSQL_PASSWORD":
                monkeypatch.delenv(key, raising=False)
        # 需要保留以通过 MYSQL_ENABLED 检查
        monkeypatch.setenv("MYSQL_PASSWORD", "test123")

    def _reload_config(self):
        """重新加载 config 包 (env 已设置) — 清除所有 config 子模块强制重导入"""
        import importlib
        # 按依赖逆序清除: __init__ → _factors/_regime/_trading/_notify → _data
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("core.shared.config.") or mod_name == "core.shared.config":
                del sys.modules[mod_name]
        importlib.invalidate_caches()
        return importlib.import_module("core.shared.config")

    def test_all_defaults_pass(self):
        """默认值全部通过校验 (0 error)"""
        config = self._reload_config()
        result = config.validate_config()
        assert result["errors"] == 0, f"默认配置有错误: {result['details']}"

    def test_invalid_boolean_string(self, monkeypatch):
        """非法布尔值字符串报错"""
        monkeypatch.setenv("QLIB_ENABLED", "yes")  # 应该是 true/false
        monkeypatch.setenv("MYSQL_PASSWORD", "test123")
        config = self._reload_config()
        result = config.validate_config()
        qlib_issues = [m for lv, m in result["details"] if "QLIB_ENABLED" in m]
        assert qlib_issues, "QLIB_ENABLED=yes 应被检测"

    def test_valid_boolean_variants(self, monkeypatch):
        """合法布尔值大小写变体"""
        for val in ("true", "True", "TRUE", "false", "False", "FALSE"):
            monkeypatch.setenv("QLIB_ENABLED", val)
            monkeypatch.setenv("MYSQL_PASSWORD", "test123")
            config = self._reload_config()
            result = config.validate_config()
            qlib_issues = [m for lv, m in result["details"] if "QLIB_ENABLED" in m]
            assert not qlib_issues, f"QLIB_ENABLED={val} 应合法"

    def test_negative_port_number(self, monkeypatch):
        """负数端口号报错"""
        monkeypatch.setenv("MYSQL_PORT", "-5")
        monkeypatch.setenv("MYSQL_PASSWORD", "test123")
        config = self._reload_config()
        result = config.validate_config()
        port_issues = [m for lv, m in result["details"] if "MYSQL_PORT" in m]
        assert port_issues, "负端口号应被检测"

    def test_zero_pool_size(self, monkeypatch):
        """连接池大小为 0 报错"""
        monkeypatch.setenv("MYSQL_POOL_SIZE", "0")
        monkeypatch.setenv("MYSQL_PASSWORD", "test123")
        config = self._reload_config()
        result = config.validate_config()
        pool_issues = [m for lv, m in result["details"] if "MYSQL_POOL_SIZE" in m]
        assert pool_issues, "POOL_SIZE=0 应被检测"

    def test_icir_out_of_range(self, monkeypatch):
        """QLIB_MODEL_MIN_ICIR 超出合理范围"""
        monkeypatch.setenv("QLIB_MODEL_MIN_ICIR", "999")
        monkeypatch.setenv("MYSQL_PASSWORD", "test123")
        config = self._reload_config()
        result = config.validate_config()
        icir_issues = [m for lv, m in result["details"] if "QLIB_MODEL_MIN_ICIR" in m]
        assert icir_issues, "ICIR=999 应告警"

    def test_negative_thread_count(self, monkeypatch):
        """采集线程数为负数"""
        monkeypatch.setenv("COLLECT_THREAD_COUNT", "-10")
        monkeypatch.setenv("MYSQL_PASSWORD", "test123")
        config = self._reload_config()
        result = config.validate_config()
        thread_issues = [m for lv, m in result["details"] if "COLLECT_THREAD_COUNT" in m]
        assert thread_issues, "负线程数应被检测"

    def test_fusion_weights_not_sum_to_one(self, monkeypatch):
        """选股权重之和偏离 1.0 告警"""
        monkeypatch.setenv("FUSION_BUY_WEIGHT_TECH", "0.8")
        monkeypatch.setenv("FUSION_BUY_WEIGHT_QLIB", "0.8")
        monkeypatch.setenv("FUSION_BUY_WEIGHT_STRATEGY", "0.8")
        monkeypatch.setenv("MYSQL_PASSWORD", "test123")
        config = self._reload_config()
        result = config.validate_config()
        weight_issues = [m for lv, m in result["details"] if "权重之和" in m]
        assert weight_issues, "权重之和=2.4 应产生告警"

    def test_threshold_inversion(self, monkeypatch):
        """BARRA danger < warning 报错"""
        monkeypatch.setenv("BARRA_RISK_WARNING_THRESHOLD", "0.8")
        monkeypatch.setenv("BARRA_RISK_DANGER_THRESHOLD", "0.3")
        monkeypatch.setenv("MYSQL_PASSWORD", "test123")
        config = self._reload_config()
        result = config.validate_config()
        barra_issues = [m for lv, m in result["details"] if "BARRA" in m and "danger" in m]
        assert barra_issues, "danger < warning 应报错"

    def test_placeholder_password_warns(self, monkeypatch):
        """占位密码告警"""
        monkeypatch.setenv("MYSQL_PASSWORD", "your_mysql_password_here")
        config = self._reload_config()
        result = config.validate_config()
        pw_issues = [m for lv, m in result["details"] if "MYSQL_PASSWORD" in m]
        assert pw_issues

    def test_empty_config_no_crash(self, monkeypatch):
        """所有 env 未设置时 validate_config 不崩溃"""
        monkeypatch.setenv("MYSQL_PASSWORD", "test123")
        config = self._reload_config()
        try:
            result = config.validate_config()
        except Exception as e:
            pytest.fail(f"空配置时 validate_config 崩溃: {e}")
        assert "errors" in result
        assert "warnings" in result

    def test_get_config_issues_persists(self, monkeypatch):
        """get_config_issues 返回最近一次校验结果"""
        monkeypatch.setenv("MYSQL_PASSWORD", "test123")
        config = self._reload_config()
        config.validate_config()
        issues = config.get_config_issues()
        assert isinstance(issues, list)

    def test_edge_max_values(self, monkeypatch):
        """边界最大值不报 error"""
        monkeypatch.setenv("MYSQL_PORT", "65535")
        monkeypatch.setenv("MYSQL_POOL_SIZE", "100")
        monkeypatch.setenv("COLLECT_THREAD_COUNT", "64")
        monkeypatch.setenv("QLIB_NUM_THREADS", "32")
        monkeypatch.setenv("QLIB_TRAIN_TIMEOUT", "86400")
        monkeypatch.setenv("MYSQL_PASSWORD", "test123")
        config = self._reload_config()
        result = config.validate_config()
        assert result["errors"] == 0, f"边界最大值应无 error: {result['details']}"

    def test_non_numeric_string_in_numeric_field(self, monkeypatch):
        """非数字字符串在数字字段 — 由 Python int()/float() 自然拦截"""
        # Python int("abc") 会抛出 ValueError, 在 import config 时就炸
        # 这是期望的行为 — 配错类型立即中止启动
        monkeypatch.setenv("MYSQL_PORT", "abc")
        monkeypatch.setenv("MYSQL_PASSWORD", "test123")
        with pytest.raises(ValueError):
            self._reload_config()
