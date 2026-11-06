ALTER TABLE fundamental_data
  MODIFY COLUMN net_profit_yoy DECIMAL(20,4) DEFAULT NULL,
  MODIFY COLUMN revenue_yoy DECIMAL(20,4) DEFAULT NULL,
  MODIFY COLUMN roe_yoy DECIMAL(20,4) DEFAULT NULL;

INSERT IGNORE INTO schema_migrations (version)
VALUES ('20261024_fundamental_metric_precision');
