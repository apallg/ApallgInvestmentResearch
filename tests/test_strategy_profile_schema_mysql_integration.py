import re
import uuid
from pathlib import Path

import pymysql
import pytest

import core.data.connection as connection
from core.data.connection import Database
from core.data.repos.strategy_profile import StrategyProfileRepo
from core.shared import config


@pytest.mark.integration
def test_security_reference_provenance_migration_preserves_legacy_rows_as_unverified():
    database_name = "codex_pit_schema_" + uuid.uuid4().hex[:12]
    try:
        admin = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            autocommit=True, connect_timeout=3,
        )
    except Exception as error:
        pytest.skip(f"隔离 MySQL 不可用: {type(error).__name__}")

    original_database = config.MYSQL_DATABASE
    original_ready = StrategyProfileRepo._ready
    created = False
    try:
        with admin.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE `{database_name}` CHARACTER SET utf8mb4")
        created = True
        if connection._pool is not None:
            assert connection._pool.stats()["borrowed"] == 0
            connection._pool.shutdown()
            connection._pool = None
        config.MYSQL_DATABASE = database_name
        StrategyProfileRepo._ready = False
        assert Database.fetchone("SELECT DATABASE() AS name")["name"] == database_name
        Database.execute(
            "CREATE TABLE security_reference_history ("
            "as_of DATE NOT NULL,code VARCHAR(10) NOT NULL,name VARCHAR(80) DEFAULT '',"
            "industry VARCHAR(100) DEFAULT '',board VARCHAR(32) DEFAULT '',"
            "is_st TINYINT(1) DEFAULT 0,is_suspended TINYINT(1) DEFAULT 0,"
            "listed_date DATE DEFAULT NULL,delisted_date DATE DEFAULT NULL,"
            "source VARCHAR(32) DEFAULT 'snapshot',PRIMARY KEY(as_of,code)) "
            "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
        )
        Database.execute(
            "INSERT INTO security_reference_history (as_of,code,name,industry,source) "
            "VALUES ('2026-10-30','600000','浦发银行','银行','pit_import')"
        )
        schema = (Path(__file__).resolve().parents[1] / "db" / "setup.sql").read_text(encoding="utf-8")
        for table in ("screening_saves", "decision_runs", "rebalance_signals", "rebalance_sessions"):
            match = re.search(
                rf"CREATE TABLE IF NOT EXISTS `{table}` \([\s\S]*?\) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
                schema,
            )
            assert match is not None
            Database.execute(match.group())

        StrategyProfileRepo.ensure_schema()

        columns = {row["Field"] for row in Database.fetchall("SHOW COLUMNS FROM security_reference_history")}
        assert {
            "source_name", "source_version", "source_available_at",
            "source_evidence_sha256", "provenance_status", "imported_at",
        } <= columns
        legacy = Database.fetchone(
            "SELECT source_name,source_version,source_available_at,source_evidence_sha256,"
            "provenance_status,imported_at FROM security_reference_history "
            "WHERE as_of='2026-10-30' AND code='600000'"
        )
        assert legacy["source_name"] is None
        assert legacy["source_version"] is None
        assert legacy["source_available_at"] is None
        assert legacy["source_evidence_sha256"] is None
        assert legacy["provenance_status"] == "unverified"
        assert legacy["imported_at"] is not None
    finally:
        StrategyProfileRepo._ready = original_ready
        config.MYSQL_DATABASE = original_database
        if connection._pool is not None:
            connection._pool.shutdown()
            connection._pool = None
        if created:
            with admin.cursor() as cursor:
                cursor.execute(f"DROP DATABASE `{database_name}`")
        admin.close()
