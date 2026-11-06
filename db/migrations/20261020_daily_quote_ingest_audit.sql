CREATE TABLE IF NOT EXISTS `daily_quote_ingest_audit` (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(10) NOT NULL,
  trade_date DATE NOT NULL,
  task_id VARCHAR(64) DEFAULT NULL,
  batch_id VARCHAR(64) NOT NULL,
  source VARCHAR(32) NOT NULL,
  provider_version VARCHAR(64) NOT NULL,
  adjustment_mode VARCHAR(16) NOT NULL,
  anchor_date DATE NOT NULL,
  captured_at DATETIME(6) NOT NULL,
  input_payload_sha256 CHAR(64) DEFAULT NULL,
  action VARCHAR(16) NOT NULL,
  old_payload_sha256 CHAR(64) DEFAULT NULL,
  new_payload_sha256 CHAR(64) DEFAULT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  INDEX idx_quote_audit_code_date (code, trade_date),
  INDEX idx_quote_audit_task (task_id),
  INDEX idx_quote_audit_batch (batch_id),
  INDEX idx_quote_audit_captured (captured_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

SET @has_input_payload_sha256 = (
  SELECT COUNT(*) FROM information_schema.columns
  WHERE table_schema = DATABASE()
    AND table_name = 'daily_quote_ingest_audit'
    AND column_name = 'input_payload_sha256'
);
SET @add_input_payload_sha256 = IF(
  @has_input_payload_sha256 = 0,
  'ALTER TABLE daily_quote_ingest_audit ADD COLUMN input_payload_sha256 CHAR(64) DEFAULT NULL AFTER captured_at',
  'SELECT 1'
);
PREPARE add_input_payload_sha256_stmt FROM @add_input_payload_sha256;
EXECUTE add_input_payload_sha256_stmt;
DEALLOCATE PREPARE add_input_payload_sha256_stmt;

ALTER TABLE `daily_quote_ingest_audit`
  MODIFY COLUMN `new_payload_sha256` CHAR(64) DEFAULT NULL;

INSERT IGNORE INTO `schema_migrations` (`version`)
VALUES ('20261020_daily_quote_ingest_audit');
