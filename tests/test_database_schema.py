"""
数据库 schema 单源化 — 单元测试
覆盖: create_tables() 读取 setup.sql, 文件缺失, 空SQL, USE/CREATE DATABASE 剥离
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data import Database


class TestCreateTables:
    """create_tables() 从 setup.sql 读取并执行"""

    def test_file_not_found_raises(self):
        """setup.sql 不存在时抛出 FileNotFoundError"""
        with patch.object(Path, 'exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="setup.sql"):
                Database.create_tables()

    def test_reads_setup_sql(self):
        """正常路径: 读取 setup.sql 并逐条执行"""
        mock_sql = (
            "CREATE TABLE IF NOT EXISTS holdings (\n"
            "  id INT AUTO_INCREMENT PRIMARY KEY,\n"
            "  code VARCHAR(10) NOT NULL\n"
            ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;\n"
            "CREATE TABLE IF NOT EXISTS risk_alerts (\n"
            "  id INT AUTO_INCREMENT PRIMARY KEY,\n"
            "  stock_code VARCHAR(10) NOT NULL\n"
            ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
        )
        with patch.object(Path, 'exists', return_value=True):
            with patch.object(Path, 'read_text', return_value=mock_sql):
                with patch.object(Database, 'execute') as mock_exec:
                    Database.create_tables()
        assert mock_exec.call_count == 2
        # 第一条 SQL 包含 holdings
        assert 'holdings' in mock_exec.call_args_list[0][0][0]
        # 第二条 SQL 包含 risk_alerts
        assert 'risk_alerts' in mock_exec.call_args_list[1][0][0]

    def test_strips_use_statement(self):
        """USE `database` 语句被剥离"""
        mock_sql = (
            "USE `qilin_stock`;\n"
            "CREATE TABLE IF NOT EXISTS test_table (id INT);"
        )
        with patch.object(Path, 'exists', return_value=True):
            with patch.object(Path, 'read_text', return_value=mock_sql):
                with patch.object(Database, 'execute') as mock_exec:
                    Database.create_tables()
        # 只执行了 CREATE TABLE, USE 被剥离
        assert mock_exec.call_count == 1
        executed_sql = mock_exec.call_args[0][0]
        assert 'USE' not in executed_sql.upper()
        assert 'test_table' in executed_sql

    def test_strips_create_database(self):
        """CREATE DATABASE 语句被剥离"""
        mock_sql = (
            "CREATE DATABASE IF NOT EXISTS `qilin_stock`\n"
            "  DEFAULT CHARACTER SET utf8mb4;\n"
            "CREATE TABLE IF NOT EXISTS test_table (id INT);"
        )
        with patch.object(Path, 'exists', return_value=True):
            with patch.object(Path, 'read_text', return_value=mock_sql):
                with patch.object(Database, 'execute') as mock_exec:
                    Database.create_tables()
        assert mock_exec.call_count == 1
        executed_sql = mock_exec.call_args[0][0]
        assert 'CREATE DATABASE' not in executed_sql.upper()
        assert 'test_table' in executed_sql

    def test_skips_comments_and_empty_statements(self):
        """注释行和空语句被跳过"""
        mock_sql = (
            "-- 这是注释\n"
            "CREATE TABLE IF NOT EXISTS holdings (id INT);\n"
            "\n"
            "-- 另一个注释\n"
            ";"
        )
        with patch.object(Path, 'exists', return_value=True):
            with patch.object(Path, 'read_text', return_value=mock_sql):
                with patch.object(Database, 'execute') as mock_exec:
                    Database.create_tables()
        assert mock_exec.call_count == 1

    def test_skips_delimiter_blocks(self):
        """DELIMITER 块不导致崩溃 (核心场景: setup.sql 已不含存储过程)"""
        mock_sql = (
            "CREATE TABLE IF NOT EXISTS test_table (id INT);\n"
            "DELIMITER $$\n"
            "CREATE PROCEDURE test_proc() BEGIN SELECT 1; END$$\n"
            "DELIMITER ;"
        )
        with patch.object(Path, 'exists', return_value=True):
            with patch.object(Path, 'read_text', return_value=mock_sql):
                with patch.object(Database, 'execute') as mock_exec:
                    Database.create_tables()
        # 至少 CREATE TABLE 被执行; DELIMITER 行被跳过不会崩溃
        executed_sqls = [c[0][0] for c in mock_exec.call_args_list]
        assert any('test_table' in s for s in executed_sqls), \
            f"CREATE TABLE 未被执行: {executed_sqls}"
        # DELIMITER 开头的语句不执行
        assert not any(s.strip().startswith('DELIMITER') for s in executed_sqls), \
            f"DELIMITER 语句被执行: {executed_sqls}"

    def test_execute_failure_is_silent(self):
        """单条SQL执行失败不中断后续SQL (存量兼容: 重复索引/列)"""
        call_count = [0]

        def side_effect(sql):
            call_count[0] += 1
            if 'risk_alerts' in sql:
                raise Exception("模拟: 表已存在")

        mock_sql = (
            "CREATE TABLE IF NOT EXISTS holdings (id INT);\n"
            "CREATE TABLE IF NOT EXISTS risk_alerts (id INT);\n"
            "CREATE TABLE IF NOT EXISTS settings (id INT);"
        )
        with patch.object(Path, 'exists', return_value=True):
            with patch.object(Path, 'read_text', return_value=mock_sql):
                with patch.object(Database, 'execute', side_effect=side_effect):
                    Database.create_tables()
        assert call_count[0] == 3

    def test_real_setup_sql_parses_without_error(self):
        """真实 setup.sql 能被正确解析 (不实际执行SQL)"""
        sql_path = Path(__file__).parent.parent / "db" / "setup.sql"
        if not sql_path.exists():
            pytest.skip("setup.sql 不存在")
        sql_text = sql_path.read_text(encoding="utf-8")
        import re
        sql_text = re.sub(r'(?i)CREATE\s+DATABASE\s+IF\s+NOT\s+EXISTS\s+`[^`]+`[^;]*;', '', sql_text)
        sql_text = re.sub(r'(?i)USE\s+`[^`]+`\s*;', '', sql_text)
        statements = [s.strip() for s in sql_text.split(';')]
        valid = 0
        for stmt in statements:
            if not stmt or stmt.startswith('DELIMITER'):
                continue
            lines = [l for l in stmt.split('\n') if not l.strip().startswith('--')]
            clean = '\n'.join(lines).strip()
            if clean:
                valid += 1
        assert valid > 30, f"setup.sql 应包含30+条有效SQL, 实际 {valid}"

    def test_essential_tables_present(self):
        """关键业务表都在 setup.sql 中定义"""
        sql_path = Path(__file__).parent.parent / "db" / "setup.sql"
        if not sql_path.exists():
            pytest.skip("setup.sql 不存在")
        sql_text = sql_path.read_text(encoding="utf-8")
        essential_tables = [
            "holdings", "daily_quotes", "risk_alerts", "settings",
            "factor_pool_state", "factor_archive", "factor_validation_scores",
            "rolling_screen_results", "training_results", "fundamental_data",
            "north_flow", "capital_flow_daily", "stock_sectors", "raw_news",
            "data_cache", "sentiment_daily", "stock_notices",
            "backtest_results", "scan_results", "orders", "trades", "accounts",
            "daily_collection_log", "smoke_test_log", "error_logs",
            "async_tasks", "market_regime_history",
            "etf_fund_flow", "qlib_predictions",
            "north_flow_summary", "forecast_data", "income_data",
            "analyst_forecast", "daily_basic",
            "macro_shibor", "macro_lpr", "macro_money_supply",
            "macro_cpi", "macro_ppi", "macro_pmi",
            "daily_forward_ret", "source_health",
            "rolling_factor_importance", "factor_elimination_log",
            "curation_logs", "debate_backtest_logs",
        ]
        for table in essential_tables:
            assert f"`{table}`" in sql_text or f"IF NOT EXISTS {table}" in sql_text, \
                f"表 {table} 缺失于 setup.sql"


class TestSchemaColumnParity:
    """setup.sql 与 database.py Repository 用到的列保持一致"""

    def test_north_flow_has_buy_sell_amounts(self):
        """north_flow 表包含 buy_amount/sell_amount (create_tables 原有列)"""
        sql_path = Path(__file__).parent.parent / "db" / "setup.sql"
        sql_text = sql_path.read_text(encoding="utf-8")
        assert 'buy_amount' in sql_text
        assert 'sell_amount' in sql_text

    def test_sentiment_daily_has_avg_confidence(self):
        """sentiment_daily 表包含 avg_confidence"""
        sql_path = Path(__file__).parent.parent / "db" / "setup.sql"
        sql_text = sql_path.read_text(encoding="utf-8")
        assert 'avg_confidence' in sql_text

    def test_regime_history_has_auditable_volatility_fields(self):
        sql_path = Path(__file__).parent.parent / "db" / "setup.sql"
        sql_text = sql_path.read_text(encoding="utf-8")
        for field in ("vol_20d", "vol_percentile", "vol_state", "method"):
            assert field in sql_text

    def test_factor_evidence_persists_family(self):
        sql_path = Path(__file__).parent.parent / "db" / "setup.sql"
        sql_text = sql_path.read_text(encoding="utf-8")
        table = sql_text.split("CREATE TABLE IF NOT EXISTS `factor_fold_evidence`", 1)[1]
        table = table.split(") ENGINE=InnoDB", 1)[0]
        assert "family VARCHAR(64)" in table

    def test_raw_news_has_hash_constraint(self):
        """raw_news 表有 hash 列和唯一约束"""
        sql_path = Path(__file__).parent.parent / "db" / "setup.sql"
        sql_text = sql_path.read_text(encoding="utf-8")
        assert 'hash VARCHAR(12)' in sql_text
        assert 'uk_hash' in sql_text

    def test_daily_quotes_has_trade_date_index(self):
        """daily_quotes 有 idx_trade_date 索引"""
        sql_path = Path(__file__).parent.parent / "db" / "setup.sql"
        sql_text = sql_path.read_text(encoding="utf-8")
        assert 'idx_trade_date' in sql_text


class TestDbSetupModule:
    """db/setup.py 仍能正常工作"""

    def test_setup_imports_database(self):
        """db/setup.py 能正常导入 Database"""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from core.data import Database
        assert hasattr(Database, 'create_tables')
