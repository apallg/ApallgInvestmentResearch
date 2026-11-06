SET @turnover_scale = (
  SELECT numeric_scale FROM information_schema.columns
  WHERE table_schema = DATABASE()
    AND table_name = 'daily_quotes'
    AND column_name = 'turnover'
);
SET @widen_turnover = IF(
  @turnover_scale IS NOT NULL AND @turnover_scale < 6,
  'ALTER TABLE daily_quotes MODIFY COLUMN turnover DECIMAL(12,6) NULL DEFAULT NULL, ALGORITHM=COPY, LOCK=SHARED',
  'SELECT 1'
);
PREPARE widen_turnover_stmt FROM @widen_turnover;
EXECUTE widen_turnover_stmt;
DEALLOCATE PREPARE widen_turnover_stmt;

INSERT IGNORE INTO `schema_migrations` (`version`)
VALUES ('20261024_daily_quotes_turnover_precision');
