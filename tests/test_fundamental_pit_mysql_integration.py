import json
import os
import uuid

import pymysql
import pytest

import core.data.connection as connection
from core.data import Database
from core.shared import config


@pytest.fixture
def isolated_fundamental_schema():
    if os.environ.get("FUNDAMENTAL_PIT_TEST_MYSQL") != "1":
        pytest.skip("set FUNDAMENTAL_PIT_TEST_MYSQL=1 to run isolated MySQL fundamental PIT integration")

    original_database = config.MYSQL_DATABASE
    database_name = "codex_pi02_" + uuid.uuid4().hex[:12]
    if database_name == original_database:
        raise RuntimeError("Refusing to use the configured application database")
    admin = pymysql.connect(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        charset="utf8mb4",
        autocommit=True,
        connect_timeout=3,
    )
    original_pool = connection._pool
    if original_pool is not None:
        if original_pool.stats()["borrowed"] != 0:
            admin.close()
            raise RuntimeError("Refusing to detach a MySQL pool with borrowed connections")
        original_pool.shutdown()
        connection._pool = None
    created = False
    try:
        with admin.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE `{database_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        created = True
        config.MYSQL_DATABASE = database_name
        Database.execute("""CREATE TABLE fundamental_data_versions (
            source_capture_id VARCHAR(64) NOT NULL,
            source_record_hash CHAR(64) NOT NULL,
            code VARCHAR(10) NOT NULL,
            report_date DATE NOT NULL,
            ann_date DATE DEFAULT NULL,
            source_name VARCHAR(32) NOT NULL,
            version_kind VARCHAR(32) NOT NULL,
            source_observed_at DATETIME(6) NOT NULL,
            payload_json LONGTEXT NOT NULL,
            PRIMARY KEY (source_capture_id,source_record_hash),
            KEY idx_fundamental_version_asof (code,report_date,ann_date,source_observed_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
        assert Database.fetchone("SELECT DATABASE() AS name")["name"] == database_name
        yield database_name
    finally:
        config.MYSQL_DATABASE = original_database
        if connection._pool is not None:
            connection._pool.shutdown()
            connection._pool = None
        if created:
            with admin.cursor() as cursor:
                cursor.execute(f"DROP DATABASE IF EXISTS `{database_name}`")
        admin.close()


@pytest.mark.integration
def test_fundamental_asof_reads_version_and_official_revision_from_isolated_mysql(isolated_fundamental_schema, monkeypatch):
    from backend.service.portfolio_decision import load_fundamental_quality
    from core.data.fundamental import _load_fundamental_data

    monkeypatch.setattr("core.data.fetcher.DataFetcher.get_industry_map", lambda self: {})

    records = [
        ("provider-1", "a" * 64, "tushare_fina_indicator", "provider_observation", "2024-01-05 18:00:00",
         {"roe": "5", "netprofit_yoy": "8", "or_yoy": "10", "debt_to_assets": "40", "profit_dedt": "10"}),
        ("official-1", "b" * 64, "official_report", "official_field_reconciliation", "2024-01-10 18:00:00",
         {"official_values": {"debt_to_assets": "35", "profit_dedt": "12"}}),
        ("provider-2", "c" * 64, "tushare_fina_indicator", "provider_observation", "2024-01-20 18:00:00",
         {"roe": "6", "netprofit_yoy": "9", "or_yoy": "11", "debt_to_assets": "42", "profit_dedt": "9"}),
    ]
    Database.executemany(
        "INSERT INTO fundamental_data_versions "
        "(source_capture_id,source_record_hash,code,report_date,ann_date,source_name,version_kind,source_observed_at,payload_json) "
        "VALUES (%s,%s,'000001','2023-12-31','2024-01-03',%s,%s,%s,%s)",
        [(capture, digest, source, kind, observed, json.dumps(payload))
         for capture, digest, source, kind, observed, payload in records],
    )

    assert load_fundamental_quality(["000001"], "2024-01-05") == {}
    before_revision = load_fundamental_quality(["000001"], "2024-01-06")["000001"]
    assert {key: before_revision[key] for key in ("profit_dedt", "debt_to_assets")} == {
        "profit_dedt": 10.0, "debt_to_assets": 40.0,
    }
    at_revision = load_fundamental_quality(["000001"], "2024-01-11")["000001"]
    assert {key: at_revision[key] for key in ("profit_dedt", "debt_to_assets")} == {
        "profit_dedt": 12.0, "debt_to_assets": 35.0,
    }
    after_later_provider = load_fundamental_quality(["000001"], "2024-01-21")["000001"]
    assert {key: after_later_provider[key] for key in ("profit_dedt", "debt_to_assets")} == {
        "profit_dedt": 12.0, "debt_to_assets": 35.0,
    }

    frame = _load_fundamental_data()
    assert frame["avail_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-06", "2024-01-11", "2024-01-21",
    ]
    assert frame["profit_dedt"].tolist() == [10.0, 12.0, 12.0]
    assert frame["debt_to_assets"].tolist() == [40.0, 35.0, 35.0]
