"""
数据库初始化脚本 - 创建数据库和表
"""
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.shared import config

logger = logging.getLogger(__name__)

def setup():
    logger.info("Apallg投研 - MySQL 数据库初始化")
    logger.info("=" * 40)
    if not config.MYSQL_PASSWORD or config.MYSQL_PASSWORD == "your_mysql_password_here":
        logger.error("请先编辑 .env 文件，填入 MYSQL_PASSWORD")
        logger.error("    示例: MYSQL_PASSWORD=your_root_password")
        return False
    import pymysql
    try:
        conn = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            charset="utf8mb4",
        )
        with conn.cursor() as cur:
            cur.execute(
                "CREATE DATABASE IF NOT EXISTS `" + config.MYSQL_DATABASE + "` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            logger.info("数据库 %s 已就绪", config.MYSQL_DATABASE)
        conn.close()
    except Exception as e:
        logger.error("连接 MySQL 失败: %s", e)
        return False
    from core.data import Database
    try:
        Database.create_tables()
        if not Database.fetchone("SHOW TABLES LIKE 'holdings_adjustments'"):
            raise RuntimeError("持仓调整审计表未创建")
        for table in ("local_ledger_accounts", "local_ledger_events", "local_order_intents",
                      "broker_order_states", "broker_order_reports", "broker_trade_reports",
                      "broker_order_submissions", "broker_trade_report_conflicts"):
            if not Database.fetchone(
                "SELECT 1 FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name=%s",
                (table,),
            ):
                raise RuntimeError(f"W1 账本或券商回报表未创建: {table}")
        from core.data.quote_audit_migration import ensure_quote_audit_integrity
        ensure_quote_audit_integrity()
        from core.data.daily_quotes_turnover_migration import ensure_daily_quotes_turnover_precision
        ensure_daily_quotes_turnover_precision()
        from core.data.fundamental_data_migration import ensure_fundamental_data_columns
        ensure_fundamental_data_columns()
        from core.data.income_data_migration import ensure_income_data_oper_cost
        ensure_income_data_oper_cost()
        from core.data.income_data_migration import ensure_income_data_version_fields
        ensure_income_data_version_fields()
        from core.data.repos.fundamental_versions import ensure_fundamental_versions_table
        ensure_fundamental_versions_table()
        from core.data.repos.dividend_source_versions import ensure_dividend_source_versions
        ensure_dividend_source_versions()
        from core.data.macro_lpr_migration import ensure_macro_lpr_source_columns
        ensure_macro_lpr_source_columns()
        from core.data.north_flow_migration import ensure_north_flow_quarantine
        ensure_north_flow_quarantine()
        from core.data.stock_sectors_migration import ensure_stock_sector_type_key
        ensure_stock_sector_type_key()
        from core.data.local_ledger_migration import ensure_local_ledger_schema
        ensure_local_ledger_schema()
        from core.data.risk_notification_migration import ensure_risk_notification_outbox
        ensure_risk_notification_outbox()
        for table in ("risk_notification_outbox", "risk_scan_runs", "risk_notification_outbox_resolutions"):
            if not Database.fetchone(
                "SELECT 1 FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name=%s",
                (table,),
            ):
                raise RuntimeError(f"风控通知/扫描表未创建: {table}")
    except FileNotFoundError as e:
        logger.error("数据库初始化脚本缺失: %s", e)
        logger.error("请确认 db/setup.sql 文件存在")
        return False
    logger.info("所有数据表已创建")
    logger.info("MySQL 持久化就绪！")
    return True


if __name__ == "__main__":
    setup()
