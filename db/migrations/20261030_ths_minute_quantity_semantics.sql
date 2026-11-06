SET @has_quantity_status = (
    SELECT COUNT(*)
    FROM information_schema.columns
    WHERE table_schema = DATABASE()
      AND table_name = 'ths_minute_bars_v2'
      AND column_name = 'quantity_status'
);
SET @quantity_status_sql = IF(
    @has_quantity_status = 0,
    'ALTER TABLE ths_minute_bars_v2 ADD COLUMN quantity_status VARCHAR(24) NOT NULL DEFAULT ''unknown'' AFTER amount',
    'SELECT 1'
);
PREPARE quantity_status_stmt FROM @quantity_status_sql;
EXECUTE quantity_status_stmt;
DEALLOCATE PREPARE quantity_status_stmt;
