SET @has_task_id = (
  SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema = DATABASE()
    AND table_name = 'daily_quote_ingest_audit'
    AND column_name = 'task_id'
);
SET @add_task_id = IF(
  @has_task_id = 0,
  'ALTER TABLE daily_quote_ingest_audit ADD COLUMN task_id VARCHAR(64) DEFAULT NULL AFTER trade_date',
  'SELECT 1'
);
PREPARE add_task_id_stmt FROM @add_task_id;
EXECUTE add_task_id_stmt;
DEALLOCATE PREPARE add_task_id_stmt;

SET @has_task_index = (
  SELECT COUNT(*) FROM information_schema.statistics
  WHERE table_schema = DATABASE()
    AND table_name = 'daily_quote_ingest_audit'
    AND index_name = 'idx_quote_audit_task'
);
SET @add_task_index = IF(
  @has_task_index = 0,
  'CREATE INDEX idx_quote_audit_task ON daily_quote_ingest_audit (task_id)',
  'SELECT 1'
);
PREPARE add_task_index_stmt FROM @add_task_index;
EXECUTE add_task_index_stmt;
DEALLOCATE PREPARE add_task_index_stmt;

CREATE TRIGGER trg_quote_audit_no_update BEFORE UPDATE ON daily_quote_ingest_audit
FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='daily_quote_ingest_audit is append-only';

CREATE TRIGGER trg_quote_audit_no_delete BEFORE DELETE ON daily_quote_ingest_audit
FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='daily_quote_ingest_audit is append-only';

INSERT IGNORE INTO schema_migrations (version)
VALUES ('20261020_quote_audit_task_link_immutable');
