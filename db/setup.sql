-- Apallg投研 A股量化平台 - 数据库初始化脚本
-- 用法: mysql -u root -p < setup.sql
-- 或: mysql -u root -p -e "source setup.sql"

CREATE DATABASE IF NOT EXISTS `qilin_stock`
  DEFAULT CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE `qilin_stock`;

-- 持仓表
CREATE TABLE IF NOT EXISTS `holdings` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  account_id VARCHAR(64) DEFAULT NULL,
  code VARCHAR(10) NOT NULL,
  name VARCHAR(50) NOT NULL,
  shares INT NOT NULL DEFAULT 0,
  cost_price DECIMAL(12,2) NOT NULL DEFAULT 0,
  buy_date DATE,
  alerts_enabled TINYINT(1) DEFAULT 1,
  risk_threshold INT DEFAULT 7,
  version BIGINT NOT NULL DEFAULT 1,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uk_code (code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `holdings_adjustments` (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  account_id VARCHAR(64) NOT NULL DEFAULT 'LOCAL-LIVE',
  code VARCHAR(10) NOT NULL,
  action VARCHAR(12) NOT NULL,
  before_json JSON DEFAULT NULL,
  after_json JSON DEFAULT NULL,
  reason VARCHAR(500) NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_holdings_adjustments_code_time (code, created_at),
  INDEX idx_holdings_adjustments_account_code_time (account_id, code, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 风控告警表
CREATE TABLE IF NOT EXISTS `risk_alerts` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  stock_code VARCHAR(10) NOT NULL,
  stock_name VARCHAR(50) NOT NULL,
  risk_level VARCHAR(10) NOT NULL,
  risk_score INT NOT NULL,
  risk_detail TEXT,
  suggestion TEXT,
  notified TINYINT(1) DEFAULT 0,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_created (created_at),
  INDEX idx_stock (stock_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `risk_notification_outbox` (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  alert_id INT DEFAULT NULL,
  scan_id CHAR(36) DEFAULT NULL,
  idempotency_key VARCHAR(160) NOT NULL,
  payload_json LONGTEXT NOT NULL,
  status VARCHAR(24) NOT NULL DEFAULT 'pending',
  delivery_channel VARCHAR(32) DEFAULT NULL,
  provider_response_json LONGTEXT NULL,
  resolution_status VARCHAR(40) DEFAULT NULL,
  resolved_by VARCHAR(128) DEFAULT NULL,
  resolution_reason VARCHAR(1000) DEFAULT NULL,
  resolution_evidence VARCHAR(255) DEFAULT NULL,
  resolved_at DATETIME DEFAULT NULL,
  attempt_count INT NOT NULL DEFAULT 0,
  available_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  lease_until DATETIME DEFAULT NULL,
  last_error VARCHAR(1000) DEFAULT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  sent_at DATETIME DEFAULT NULL,
  UNIQUE KEY uq_risk_notification_idempotency (idempotency_key),
  KEY idx_risk_notification_ready (status, available_at, id),
  KEY idx_risk_notification_scan (scan_id, id),
  KEY idx_risk_notification_alert (alert_id, id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `risk_notification_outbox_resolutions` (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  outbox_id BIGINT NOT NULL,
  previous_status VARCHAR(24) NOT NULL,
  resolution_status VARCHAR(40) NOT NULL,
  actor VARCHAR(128) NOT NULL,
  reason VARCHAR(1000) NOT NULL,
  evidence_ref VARCHAR(255) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY idx_risk_outbox_resolution (outbox_id, id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `risk_scan_runs` (
  scan_id CHAR(36) PRIMARY KEY,
  source VARCHAR(32) NOT NULL,
  session_name VARCHAR(64) NOT NULL,
  regime_name VARCHAR(64) DEFAULT NULL,
  status VARCHAR(24) NOT NULL,
  expected_count INT NOT NULL DEFAULT 0,
  scanned_count INT NOT NULL DEFAULT 0,
  unknown_count INT NOT NULL DEFAULT 0,
  dynamic_unknown_count INT NOT NULL DEFAULT 0,
  triggered_count INT NOT NULL DEFAULT 0,
  assessment_json LONGTEXT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY idx_risk_scan_created (created_at, scan_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 设置表
CREATE TABLE IF NOT EXISTS `settings` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  setting_key VARCHAR(100) NOT NULL UNIQUE,
  setting_value TEXT,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 分析历史表
CREATE TABLE IF NOT EXISTS `analysis_history` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  stock_code VARCHAR(10),
  stock_name VARCHAR(50),
  analysis_type VARCHAR(50),
  analysis_result LONGTEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_type_created (analysis_type, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 日K线行情表
CREATE TABLE IF NOT EXISTS `daily_quotes` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(10) NOT NULL,
  trade_date DATE NOT NULL,
  open DECIMAL(10,2),
  high DECIMAL(10,2),
  low DECIMAL(10,2),
  close DECIMAL(10,2),
  volume BIGINT,
  amount DECIMAL(16,2),
  amplitude DECIMAL(10,2),
  pct_chg DECIMAL(10,2),
  turnover DECIMAL(12,6),
  adj_factor DECIMAL(12,6) DEFAULT NULL COMMENT '后复权因子 (Tushare adj_factor, 原始价×adj_factor=后复权价)',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_code_date (code, trade_date),
  INDEX idx_trade_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

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

-- 预计算 forward return 物化表 (LEAD 窗口函数, MySQL 8.0+)
-- 供因子校验和调优的 IC 计算直接查询, 避免每次都扫 daily_quotes 做窗口计算
CREATE TABLE IF NOT EXISTS `daily_forward_ret` (
  code VARCHAR(10) NOT NULL,
  trade_date DATE NOT NULL,
  fwd_5d DECIMAL(10,6) DEFAULT NULL,
  fwd_10d DECIMAL(10,6) DEFAULT NULL,
  fwd_20d DECIMAL(10,6) DEFAULT NULL,
  fwd_40d DECIMAL(10,6) DEFAULT NULL,
  PRIMARY KEY (code, trade_date),
  KEY idx_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 通用数据缓存表
CREATE TABLE IF NOT EXISTS `data_cache` (
  cache_key VARCHAR(255) NOT NULL PRIMARY KEY,
  data_blob LONGBLOB NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  expires_at DATETIME,
  INDEX idx_expires (expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 情绪日度汇总表
CREATE TABLE IF NOT EXISTS `sentiment_daily` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(10) NOT NULL,
  trade_date DATE NOT NULL,
  avg_score DECIMAL(10,4),
  pos_ratio DECIMAL(10,4),
  neg_ratio DECIMAL(10,4),
  news_count INT DEFAULT 0,
  avg_confidence DECIMAL(10,4) DEFAULT 0.5000,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_code_date (code, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- 原始新闻表
CREATE TABLE IF NOT EXISTS `raw_news` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(10) NOT NULL,
  title VARCHAR(500) NOT NULL,
  url VARCHAR(500),
  date VARCHAR(20),
  source VARCHAR(50),
  hash VARCHAR(12),
  sentiment VARCHAR(10) DEFAULT 'neutral',
  sent_score DOUBLE DEFAULT 0,
  collected_at DATETIME,
  UNIQUE KEY uk_hash (hash),
  INDEX idx_code_date (code, date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 股票板块关联表
CREATE TABLE IF NOT EXISTS `stock_sectors` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(10) NOT NULL,
  sector_name VARCHAR(100) NOT NULL,
  sector_type VARCHAR(20) NOT NULL DEFAULT 'industry',
  updated_at DATETIME DEFAULT NULL COMMENT '采集时间 (概念板块成分周频采集)',
  UNIQUE KEY uk_code_sector_type (code, sector_name, sector_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `stock_sector_membership_observations` (
  observed_at DATETIME(6) NOT NULL,
  source VARCHAR(32) NOT NULL,
  sector_code VARCHAR(16) NOT NULL,
  sector_name VARCHAR(100) NOT NULL,
  code VARCHAR(10) NOT NULL,
  PRIMARY KEY (observed_at, sector_code, code),
  KEY idx_sector_observation_code (code, observed_at),
  KEY idx_sector_observation_name (sector_name, observed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 回测结果表
CREATE TABLE IF NOT EXISTS `backtest_results` (
  id VARCHAR(64) PRIMARY KEY,
  strategy_name VARCHAR(100),
  stock_code VARCHAR(10),
  params JSON,
  metrics JSON,
  trades LONGTEXT,
  equity_curve LONGTEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 选股扫描结果表
CREATE TABLE IF NOT EXISTS `scan_results` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  stock_code VARCHAR(10),
  stock_name VARCHAR(50),
  price DECIMAL(12,2),
  pct_chg DECIMAL(10,2),
  ai_risk_score INT,
  risk_level VARCHAR(10),
  reason TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 委托记录表
CREATE TABLE IF NOT EXISTS `orders` (
  id VARCHAR(32) PRIMARY KEY,
  account_id VARCHAR(64) DEFAULT NULL,
  broker VARCHAR(20) DEFAULT NULL,
  environment VARCHAR(20) DEFAULT NULL,
  source VARCHAR(20) DEFAULT NULL,
  symbol VARCHAR(10) NOT NULL,
  name VARCHAR(50) NOT NULL,
  side VARCHAR(5) NOT NULL,
  quantity INT NOT NULL DEFAULT 0,
  price_type VARCHAR(10) NOT NULL DEFAULT 'market',
  price DECIMAL(12,3) NOT NULL DEFAULT 0,
  filled_qty INT DEFAULT 0,
  filled_price DECIMAL(12,3) DEFAULT 0,
  status VARCHAR(20) NOT NULL DEFAULT 'pending',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_symbol (symbol),
  INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 成交记录表
CREATE TABLE IF NOT EXISTS `trades` (
  id VARCHAR(32) PRIMARY KEY,
  account_id VARCHAR(64) DEFAULT NULL,
  broker VARCHAR(20) DEFAULT NULL,
  environment VARCHAR(20) DEFAULT NULL,
  source VARCHAR(20) DEFAULT NULL,
  order_id VARCHAR(32) NOT NULL,
  symbol VARCHAR(10) NOT NULL,
  name VARCHAR(50) NOT NULL,
  side VARCHAR(5) NOT NULL,
  price DECIMAL(12,3) NOT NULL,
  quantity INT NOT NULL,
  amount DECIMAL(16,2) NOT NULL,
  commission DECIMAL(10,2) DEFAULT 0,
  stamp_duty DECIMAL(10,2) DEFAULT 0,
  trade_time DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_symbol (symbol),
  INDEX idx_order (order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 账户资金表
CREATE TABLE IF NOT EXISTS `accounts` (
  id VARCHAR(32) PRIMARY KEY,
  cash DECIMAL(16,2) NOT NULL DEFAULT 0,
  frozen DECIMAL(16,2) NOT NULL DEFAULT 0,
  market_value DECIMAL(16,2) NOT NULL DEFAULT 0,
  total_assets DECIMAL(16,2) NOT NULL DEFAULT 0,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `local_ledger_accounts` (
  account_id VARCHAR(64) PRIMARY KEY,
  environment VARCHAR(20) NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'uninitialized',
  cash DECIMAL(18,2) DEFAULT NULL,
  frozen DECIMAL(18,2) NOT NULL DEFAULT 0,
  version BIGINT NOT NULL DEFAULT 0,
  opening_fingerprint CHAR(64) DEFAULT NULL,
  broker_account_id VARCHAR(128) DEFAULT NULL,
  opened_at DATETIME DEFAULT NULL,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `local_ledger_events` (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  event_key VARCHAR(128) NOT NULL,
  account_id VARCHAR(64) NOT NULL,
  event_type VARCHAR(24) NOT NULL,
  signal_id INT DEFAULT NULL,
  code VARCHAR(10) DEFAULT NULL,
  side VARCHAR(10) DEFAULT NULL,
  shares INT DEFAULT NULL,
  price DECIMAL(12,3) DEFAULT NULL,
  commission DECIMAL(12,2) NOT NULL DEFAULT 0,
  stamp_duty DECIMAL(12,2) NOT NULL DEFAULT 0,
  cash_delta DECIMAL(18,2) NOT NULL DEFAULT 0,
  cash_after DECIMAL(18,2) NOT NULL,
  account_version BIGINT NOT NULL,
  actor VARCHAR(64) NOT NULL,
  reason VARCHAR(500) DEFAULT NULL,
  payload_json JSON DEFAULT NULL,
  occurred_at DATETIME NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_local_ledger_event_key (event_key),
  INDEX idx_local_ledger_account_time (account_id, created_at),
  INDEX idx_local_ledger_signal (signal_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `local_order_intents` (
  intent_id CHAR(36) PRIMARY KEY,
  account_id VARCHAR(64) NOT NULL,
  broker_account_id VARCHAR(128) NOT NULL,
  signal_id INT NOT NULL,
  code VARCHAR(10) NOT NULL,
  side VARCHAR(10) NOT NULL,
  quantity INT NOT NULL,
  limit_price DECIMAL(12,3) NOT NULL,
  fee_cap DECIMAL(12,2) NOT NULL,
  reserved_cash DECIMAL(18,2) NOT NULL DEFAULT 0,
  remaining_reserved_cash DECIMAL(18,2) NOT NULL DEFAULT 0,
  broker_order_id VARCHAR(64) DEFAULT NULL,
  filled_quantity INT NOT NULL DEFAULT 0,
  state VARCHAR(24) NOT NULL,
  account_version BIGINT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_local_intent_account_state (account_id,state),
  INDEX idx_local_intent_signal (signal_id),
  UNIQUE KEY uk_local_intent_broker_order (broker_account_id,broker_order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `broker_order_states` (
  broker VARCHAR(20) NOT NULL,
  broker_account_id VARCHAR(128) NOT NULL,
  order_id VARCHAR(64) NOT NULL,
  symbol VARCHAR(16) NOT NULL,
  side VARCHAR(10) NOT NULL,
  quantity INT NOT NULL,
  price_type VARCHAR(10) DEFAULT NULL,
  limit_price DECIMAL(12,3) DEFAULT NULL,
  filled_qty INT NOT NULL DEFAULT 0,
  status VARCHAR(20) NOT NULL DEFAULT 'unknown',
  last_report_key CHAR(64) DEFAULT NULL,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (broker,broker_account_id,order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `broker_order_submissions` (
  broker VARCHAR(20) NOT NULL,
  broker_account_id VARCHAR(128) NOT NULL,
  order_id VARCHAR(64) NOT NULL,
  submission_json JSON NOT NULL,
  observed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (broker,broker_account_id,order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `broker_order_reports` (
  report_key CHAR(64) PRIMARY KEY,
  broker VARCHAR(20) NOT NULL,
  broker_account_id VARCHAR(128) NOT NULL,
  order_id VARCHAR(64) NOT NULL,
  symbol VARCHAR(16) NOT NULL,
  side VARCHAR(10) NOT NULL,
  quantity INT NOT NULL,
  filled_qty INT NOT NULL,
  status VARCHAR(20) NOT NULL,
  source VARCHAR(20) NOT NULL,
  disposition VARCHAR(20) NOT NULL,
  raw_json JSON NOT NULL,
  resolved_at DATETIME DEFAULT NULL,
  resolved_by VARCHAR(64) DEFAULT NULL,
  resolution_reason VARCHAR(500) DEFAULT NULL,
  observed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_broker_report_order (broker,broker_account_id,order_id,observed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `broker_trade_reports` (
  broker VARCHAR(20) NOT NULL,
  broker_account_id VARCHAR(128) NOT NULL,
  trade_id VARCHAR(64) NOT NULL,
  order_id VARCHAR(64) NOT NULL,
  symbol VARCHAR(16) NOT NULL,
  side VARCHAR(10) NOT NULL,
  quantity INT NOT NULL,
  price DECIMAL(12,3) NOT NULL,
  amount DECIMAL(18,2) NOT NULL,
  commission DECIMAL(12,2) DEFAULT NULL,
  stamp_duty DECIMAL(12,2) DEFAULT NULL,
  payload_hash CHAR(64) NOT NULL,
  raw_json JSON NOT NULL,
  first_source VARCHAR(20) NOT NULL,
  observed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (broker,broker_account_id,trade_id),
  INDEX idx_broker_trade_order (broker,broker_account_id,order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `broker_trade_report_conflicts` (
  conflict_key CHAR(64) PRIMARY KEY,
  broker VARCHAR(20) NOT NULL,
  broker_account_id VARCHAR(128) NOT NULL,
  trade_id VARCHAR(64) NOT NULL,
  payload_hash CHAR(64) NOT NULL,
  raw_json JSON NOT NULL,
  resolved_at DATETIME DEFAULT NULL,
  resolved_by VARCHAR(64) DEFAULT NULL,
  resolution_reason VARCHAR(500) DEFAULT NULL,
  resolution_evidence_json JSON DEFAULT NULL,
  observed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_broker_trade_conflict (broker,broker_account_id,trade_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `broker_report_resolution_events` (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    broker VARCHAR(20) NOT NULL,
    broker_account_id VARCHAR(128) NOT NULL,
    report_type VARCHAR(10) NOT NULL,
    report_key CHAR(64) NOT NULL,
    action VARCHAR(12) NOT NULL,
    actor VARCHAR(64) NOT NULL,
    reason VARCHAR(500) NOT NULL,
    evidence_json JSON NOT NULL,
    occurred_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_broker_resolution_history (broker,broker_account_id,report_type,report_key,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT IGNORE INTO local_ledger_accounts (account_id, environment) VALUES ('LOCAL-LIVE', 'live');

-- 北向资金每日个股净流入表
CREATE TABLE IF NOT EXISTS `north_flow` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(10) NOT NULL,
  trade_date DATE NOT NULL,
  net_flow DECIMAL(16,2),
  buy_amount DECIMAL(16,2) DEFAULT 0,
  sell_amount DECIMAL(16,2) DEFAULT 0,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_code_date (code, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `north_flow_quarantine` (
  code VARCHAR(10) NOT NULL,
  trade_date DATE NOT NULL,
  net_flow DECIMAL(16,2),
  quarantine_reason VARCHAR(255) NOT NULL,
  quarantined_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (code, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 资金流向每日个股表 (主力净流入)
CREATE TABLE IF NOT EXISTS `capital_flow_daily` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(10) NOT NULL,
  trade_date DATE NOT NULL,
  main_net_flow DECIMAL(16,2) DEFAULT 0,
  main_net_pct DECIMAL(10,4) DEFAULT 0,
  turnover DECIMAL(16,2) DEFAULT 0,
  sm_net_flow DECIMAL(16,2) DEFAULT NULL COMMENT '散户净流入(小单<5万, 万元)',
  md_net_flow DECIMAL(16,2) DEFAULT NULL COMMENT '中户净流入(中单5-20万, 万元)',
  lg_net_flow DECIMAL(16,2) DEFAULT NULL COMMENT '大户净流入(大单20-100万, 万元)',
  elg_net_flow DECIMAL(16,2) DEFAULT NULL COMMENT '机构净流入(超大单>100万, 万元)',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_code_date (code, trade_date),
  INDEX idx_trade_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 新闻源健康状态表
CREATE TABLE IF NOT EXISTS `source_health` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  source_tag VARCHAR(30) NOT NULL,
  source_name VARCHAR(50) NOT NULL,
  status VARCHAR(10) NOT NULL DEFAULT 'ok',
  error_msg VARCHAR(500) DEFAULT NULL,
  stock_count INT DEFAULT 0,
  news_count INT DEFAULT 0,
  elapsed_ms INT DEFAULT 0,
  recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_source_time (source_tag, recorded_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 滚动筛选结果表 — 持久化每个因子的每期校验结果
CREATE TABLE IF NOT EXISTS `rolling_screen_results` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  job_id VARCHAR(32) NOT NULL,
  anchor_date DATE NOT NULL,
  factor_key VARCHAR(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  label VARCHAR(200) DEFAULT NULL,
  passed TINYINT(1) NOT NULL DEFAULT 0,
  direction VARCHAR(10) DEFAULT 'long',
  passed_count INT DEFAULT 0,
  layer_detail JSON DEFAULT NULL,
  error TEXT DEFAULT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_job_id (job_id),
  INDEX idx_anchor (anchor_date),
  INDEX idx_factor (factor_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 滚动训练因子重要性 (嵌入式二次筛选: 每轮滚动训练的特征重要性记录)
CREATE TABLE IF NOT EXISTS `rolling_factor_importance` (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  job_id VARCHAR(32) NOT NULL,
  round_idx INT NOT NULL,
  anchor_date DATE NOT NULL,
  feature_name VARCHAR(300) NOT NULL COMMENT 'qlib表达式原始名',
  factor_key VARCHAR(128) DEFAULT NULL COMMENT '映射后的因子池key',
  importance_split DOUBLE NOT NULL,
  importance_gain DOUBLE NOT NULL,
  rank_split INT NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_job_round (job_id, round_idx),
  INDEX idx_factor (factor_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 因子校验数据集市 (滚动筛选结果完整持久化, 供精选集/监控/训练复用)
CREATE TABLE IF NOT EXISTS `factor_validation_scores` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  factor_key VARCHAR(200) NOT NULL,
  factor_label VARCHAR(200) DEFAULT NULL,
  anchor_date DATE NOT NULL,
  source VARCHAR(50) DEFAULT NULL COMMENT 'Alpha158|Alpha101|Alpha191',
  category VARCHAR(50) DEFAULT NULL COMMENT '原生分类',
  unified_category VARCHAR(50) DEFAULT NULL COMMENT '统一8大类',

  -- IC 汇总
  icir FLOAT DEFAULT NULL COMMENT 'abs(mean/std) of daily neutralized IC',
  ic_mean FLOAT DEFAULT NULL,
  ic_std FLOAT DEFAULT NULL,
  n_ic_dates INT DEFAULT 0,

  -- 逐层得分 (0~1, NULL=未执行)
  l1_score FLOAT DEFAULT NULL COMMENT 'Walk-Forward',
  l2_score FLOAT DEFAULT NULL COMMENT '多行情分段',
  l3_score FLOAT DEFAULT NULL COMMENT 'IC衰减',
  l4_score FLOAT DEFAULT NULL COMMENT '参数鲁棒性',
  l5_score FLOAT DEFAULT NULL COMMENT '去冗余',
  l6_score FLOAT DEFAULT NULL COMMENT '换手率约束',
  l7_score FLOAT DEFAULT NULL COMMENT '置换检验',
  l8_score FLOAT DEFAULT NULL COMMENT '边际贡献',

  -- 综合
  all_pass TINYINT(1) NOT NULL DEFAULT 0,
  passed_count INT DEFAULT 0,
  direction VARCHAR(10) DEFAULT 'long',
  mode VARCHAR(10) DEFAULT 'full' COMMENT 'fast|full',
  early_terminated TINYINT(1) DEFAULT 0,

  -- 明细
  layer_detail JSON DEFAULT NULL COMMENT 'check_all_layers() 完整返回值',
  daily_ics JSON DEFAULT NULL COMMENT '[["2024-01-02",0.032],...]',

  -- 附加指标 (2026-07 新增, 从 check_all_layers 返回的结构化数据)
  ic_decay JSON DEFAULT NULL COMMENT '{"1":0.052,"5":0.048,"10":0.031,"20":0.021}',
  shuffle_p_value FLOAT DEFAULT NULL COMMENT 'L7置换检验 p-value',
  shuffle_p95 FLOAT DEFAULT NULL COMMENT 'L7置换检验 95%分位数 |IC|',
  shuffle_n INT DEFAULT NULL COMMENT 'L7置换次数',
  max_corr FLOAT DEFAULT NULL COMMENT 'L5与因子池最大Pearson相关性',

  -- 来源
  job_id VARCHAR(32) DEFAULT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

  UNIQUE KEY uk_factor_anchor (factor_key, anchor_date),
  INDEX idx_anchor (anchor_date),
  INDEX idx_factor (factor_key),
  INDEX idx_unified_cat (unified_category),
  INDEX idx_icir (icir),
  INDEX idx_all_pass (all_pass)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `factor_evidence_runs` (
  run_id VARCHAR(32) PRIMARY KEY,
  status VARCHAR(32) NOT NULL,
  pool_snapshot_hash VARCHAR(64) DEFAULT NULL,
  selection_cutoff DATE DEFAULT NULL,
  config_json JSON DEFAULT NULL,
  result_json JSON DEFAULT NULL,
  error TEXT DEFAULT NULL,
  created_at DATETIME(3) NOT NULL,
  updated_at DATETIME(3) NOT NULL,
  INDEX idx_factor_evidence_runs_status (status),
  INDEX idx_factor_evidence_runs_updated (updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `factor_evidence_candidates` (
  run_id VARCHAR(32) NOT NULL,
  factor_key VARCHAR(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  expression_text TEXT DEFAULT NULL,
  expression_fingerprint VARCHAR(64) DEFAULT NULL,
  family VARCHAR(64) DEFAULT NULL,
  source VARCHAR(64) DEFAULT NULL,
  status VARCHAR(32) NOT NULL,
  reject_reason VARCHAR(255) DEFAULT NULL,
  metadata_json JSON DEFAULT NULL,
  PRIMARY KEY (run_id, factor_key),
  INDEX idx_factor_evidence_candidates_status (run_id, status),
  INDEX idx_factor_evidence_candidates_expr (expression_fingerprint)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `factor_fold_evidence` (
  run_id VARCHAR(32) NOT NULL,
  factor_key VARCHAR(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  family VARCHAR(64) DEFAULT NULL,
  fold_id VARCHAR(32) NOT NULL,
  anchor_date DATE NOT NULL,
  state VARCHAR(24) NOT NULL,
  train_end DATE DEFAULT NULL,
  valid_start DATE DEFAULT NULL,
  valid_end DATE DEFAULT NULL,
  test_start DATE DEFAULT NULL,
  test_end DATE DEFAULT NULL,
  label_matured_at DATE DEFAULT NULL,
  direction_sign SMALLINT NOT NULL DEFAULT 1,
  raw_ic DOUBLE DEFAULT NULL,
  oriented_ic DOUBLE DEFAULT NULL,
  icir DOUBLE DEFAULT NULL,
  positive_ratio DOUBLE DEFAULT NULL,
  sample_count INT NOT NULL DEFAULT 0,
  coverage DOUBLE DEFAULT NULL,
  metrics_json JSON DEFAULT NULL,
  created_at DATETIME(3) NOT NULL,
  PRIMARY KEY (run_id, factor_key, fold_id, state),
  INDEX idx_factor_fold_evidence_factor (run_id, factor_key),
  INDEX idx_factor_fold_evidence_matured (run_id, label_matured_at),
  INDEX idx_factor_fold_evidence_state (run_id, state)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `factor_set_versions` (
  set_id VARCHAR(64) PRIMARY KEY,
  run_id VARCHAR(32) NOT NULL,
  role VARCHAR(32) NOT NULL,
  version_no INT NOT NULL DEFAULT 1,
  status VARCHAR(24) NOT NULL,
  factor_keys_json JSON DEFAULT NULL,
  selection_config_json JSON DEFAULT NULL,
  metrics_json JSON DEFAULT NULL,
  fingerprint VARCHAR(64) NOT NULL,
  created_at DATETIME(3) NOT NULL,
  updated_at DATETIME(3) NOT NULL,
  UNIQUE KEY uq_factor_set_version (run_id, role, version_no),
  INDEX idx_factor_set_versions_role (run_id, role, status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `factor_set_evaluations` (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  run_id VARCHAR(32) NOT NULL,
  set_id VARCHAR(64) NOT NULL,
  arm VARCHAR(255) NOT NULL,
  fold_id VARCHAR(32) DEFAULT NULL,
  metrics_json JSON DEFAULT NULL,
  created_at DATETIME(3) NOT NULL,
  INDEX idx_factor_set_evaluations_set (run_id, set_id),
  INDEX idx_factor_set_evaluations_arm (run_id, arm)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 每日市场状态 (一次性批量回填 + 每日自动追加)
CREATE TABLE IF NOT EXISTS `market_regime_history` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  trade_date DATE NOT NULL UNIQUE,
  regime VARCHAR(20) NOT NULL COMMENT 'trending_up|trending_down|ranging|volatile|bull_steady|bear_steady',
  confidence FLOAT DEFAULT 0 COMMENT '置信度 0~1',
  trending_up FLOAT DEFAULT 0,
  trending_down FLOAT DEFAULT 0,
  ranging FLOAT DEFAULT 0,
  volatile FLOAT DEFAULT 0,
  vol_20d DOUBLE DEFAULT NULL COMMENT '截至trade_date的20日年化波动率',
  vol_percentile DOUBLE DEFAULT NULL COMMENT '截至trade_date的252日波动率滚动分位数',
  vol_state VARCHAR(16) DEFAULT NULL COMMENT 'high_vol|low_vol',
  method VARCHAR(32) DEFAULT NULL COMMENT '状态构建方法版本',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_date (trade_date),
  INDEX idx_regime (regime)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 训练结果记录 (每次 Qlib 训练的结果持久化, 供模型对比和历史追踪)
CREATE TABLE IF NOT EXISTS `training_results` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  run_id VARCHAR(64) NOT NULL COMMENT 'MLflow run name / 唯一标识',
  model_type VARCHAR(20) DEFAULT 'dual' COMMENT 'dual|single',
  factor_set VARCHAR(50) NOT NULL COMMENT 'Alpha158|WhiteBoxHandler|Alpha360',
  instruments VARCHAR(20) NOT NULL COMMENT 'csi300|csi500|all',
  train_start DATE NOT NULL,
  train_end DATE NOT NULL,
  valid_start DATE NOT NULL,
  valid_end DATE NOT NULL,
  test_start DATE NOT NULL,
  test_end DATE NOT NULL,
  label_horizon INT DEFAULT 20,

  -- 融合后指标
  ic_mean FLOAT DEFAULT NULL,
  ric_mean FLOAT DEFAULT NULL,
  icir FLOAT DEFAULT NULL,

  -- 子模型指标
  lgb_ic_mean FLOAT DEFAULT NULL,
  lgb_weight FLOAT DEFAULT NULL,
  xgb_ic_mean FLOAT DEFAULT NULL,
  xgb_weight FLOAT DEFAULT NULL,
  linear_ic_mean FLOAT DEFAULT NULL,
  linear_weight FLOAT DEFAULT NULL,
  cat_ic_mean FLOAT DEFAULT NULL,
  cat_weight FLOAT DEFAULT NULL,

  -- 扩展评估 (来自 _extended_eval_metrics)
  monthly_win_rate FLOAT DEFAULT NULL COMMENT '月度IC>0占比',
  annual_icir FLOAT DEFAULT NULL COMMENT '年化ICIR',
  long_short_spread FLOAT DEFAULT NULL COMMENT 'Q5-Q1多空收益差',
  max_neg_ic_months INT DEFAULT NULL COMMENT '最长连续负IC月数',

  -- 因子配置
  custom_factor_keys JSON DEFAULT NULL COMMENT '训练时使用的白盒因子key列表',
  extra_metrics JSON DEFAULT NULL COMMENT '其他扩展指标',

  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_run_id (run_id),
  INDEX idx_factor_set (factor_set),
  INDEX idx_train_end (train_end),
  INDEX idx_icir (icir),
  INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 因子淘汰日志 (滚动训练 filter_mode 淘汰记录, 供因子生命周期追踪)
CREATE TABLE IF NOT EXISTS `factor_elimination_log` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  job_id VARCHAR(64) NOT NULL,
  factor_key VARCHAR(200) NOT NULL,
  factor_label VARCHAR(200) DEFAULT NULL,
  reason VARCHAR(100) DEFAULT 'low_importance' COMMENT 'elimination_candidate|low_importance|manual',
  importance_rank INT DEFAULT NULL COMMENT '重要性排名 (越小越差)',
  mean_split FLOAT DEFAULT NULL COMMENT '平均 split importance',
  win_rate FLOAT DEFAULT NULL COMMENT '各轮进入top-N的占比',
  eliminated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_job (job_id),
  INDEX idx_factor (factor_key),
  INDEX idx_eliminated (eliminated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 每日市值表 (TuShare daily_basic, 单位: 万元) — 用于因子截面市值中性化
CREATE TABLE IF NOT EXISTS `daily_basic` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(10) NOT NULL,
  trade_date DATE NOT NULL,
  total_mv DECIMAL(20,4),
  circ_mv DECIMAL(20,4),
  float_share DECIMAL(20,4),
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_code_date (code, trade_date),
  INDEX idx_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 异步任务表 — 持久化后台任务状态/进度/结果
CREATE TABLE IF NOT EXISTS `async_tasks` (
  `job_id` VARCHAR(32) PRIMARY KEY,
  `status` VARCHAR(20) NOT NULL DEFAULT 'pending',
  `description` VARCHAR(200) DEFAULT '',
  `progress` JSON,
  `result` JSON,
  `error` TEXT,
  `created_at` DATETIME(3) NOT NULL,
  `updated_at` DATETIME(3) NOT NULL,
  INDEX idx_status (`status`),
  INDEX idx_created (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- 宏观经济数据 (2026-07 新增, Tushare 采集)
-- ============================================================

CREATE TABLE IF NOT EXISTS `macro_shibor` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  trade_date DATE NOT NULL,
  on_rate DECIMAL(10,4) DEFAULT NULL COMMENT '隔夜利率%',
  w1_rate DECIMAL(10,4) DEFAULT NULL COMMENT '1周利率%',
  w2_rate DECIMAL(10,4) DEFAULT NULL COMMENT '2周利率%',
  m1_rate DECIMAL(10,4) DEFAULT NULL COMMENT '1个月利率%',
  m3_rate DECIMAL(10,4) DEFAULT NULL COMMENT '3个月利率%',
  m6_rate DECIMAL(10,4) DEFAULT NULL COMMENT '6个月利率%',
  m9_rate DECIMAL(10,4) DEFAULT NULL COMMENT '9个月利率%',
  y1_rate DECIMAL(10,4) DEFAULT NULL COMMENT '1年利率%',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `macro_lpr` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  trade_date DATE NOT NULL,
  lpr_1y DECIMAL(10,4) DEFAULT NULL COMMENT '1年期LPR%',
  lpr_5y DECIMAL(10,4) DEFAULT NULL COMMENT '5年期LPR%',
  source_name VARCHAR(32) NOT NULL DEFAULT 'unknown',
  source_url VARCHAR(512) DEFAULT NULL,
  source_observed_at DATETIME DEFAULT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `macro_money_supply` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  trade_date DATE NOT NULL COMMENT '报告月末日',
  m1 DECIMAL(18,4) DEFAULT NULL COMMENT 'M1余额(亿元)',
  m1_yoy DECIMAL(10,4) DEFAULT NULL COMMENT 'M1同比%',
  m2 DECIMAL(18,4) DEFAULT NULL COMMENT 'M2余额(亿元)',
  m2_yoy DECIMAL(10,4) DEFAULT NULL COMMENT 'M2同比%',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `macro_cpi` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  trade_date DATE NOT NULL COMMENT '报告月',
  cpi_yoy DECIMAL(10,4) DEFAULT NULL COMMENT 'CPI同比%',
  cpi_mom DECIMAL(10,4) DEFAULT NULL COMMENT 'CPI环比%',
  core_cpi_yoy DECIMAL(10,4) DEFAULT NULL COMMENT '核心CPI同比%',
  food_yoy DECIMAL(10,4) DEFAULT NULL COMMENT '食品同比%',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `macro_ppi` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  trade_date DATE NOT NULL COMMENT '报告月',
  ppi_yoy DECIMAL(10,4) DEFAULT NULL COMMENT 'PPI全部工业品同比%',
  ppi_mom DECIMAL(10,4) DEFAULT NULL COMMENT 'PPI环比%',
  ppi_mp_yoy DECIMAL(10,4) DEFAULT NULL COMMENT '生产资料同比%',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `macro_pmi` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  trade_date DATE NOT NULL COMMENT '报告月',
  pmi DECIMAL(10,4) DEFAULT NULL COMMENT '制造业PMI',
  pmi_new_order DECIMAL(10,4) DEFAULT NULL COMMENT '新订单指数',
  pmi_employment DECIMAL(10,4) DEFAULT NULL COMMENT '从业人员指数',
  pmi_input_price DECIMAL(10,4) DEFAULT NULL COMMENT '原材料购进价格',
  pmi_non_mfg DECIMAL(10,4) DEFAULT NULL COMMENT '非制造业PMI',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- 文件数据迁移至MySQL (2026-07 新增)
-- ============================================================

-- 因子池状态表 (替代 core/factor_pool_state.json)
CREATE TABLE IF NOT EXISTS `factor_pool_state` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  state_json JSON NOT NULL COMMENT '因子池完整状态JSON',
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 因子归档表 (替代 core/factor_archive.json)
CREATE TABLE IF NOT EXISTS `factor_archive` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  factor_key VARCHAR(200) NOT NULL COMMENT '因子key',
  category VARCHAR(50) DEFAULT NULL COMMENT '因子类别(white_box/traditional/hybrid)',
  label VARCHAR(100) DEFAULT NULL COMMENT '因子标签',
  archived_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '归档时间',
  reason VARCHAR(500) DEFAULT NULL COMMENT '归档原因',
  snapshot_json JSON DEFAULT NULL COMMENT '归档时的因子快照'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 精选报表日志表 (替代 logs/curation_*.json)
CREATE TABLE IF NOT EXISTS `curation_logs` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  curation_name VARCHAR(100) NOT NULL COMMENT '精选集名称',
  report_json JSON NOT NULL COMMENT '报表内容JSON',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_name (curation_name),
  INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 辩论回测日志表 (替代 logs/debate_backtest_*.json)
CREATE TABLE IF NOT EXISTS `debate_backtest_logs` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  eval_date DATE NOT NULL COMMENT '评估日期',
  result_json JSON NOT NULL COMMENT '回测结果JSON',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_eval_date (eval_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ETF资金流数据
CREATE TABLE IF NOT EXISTS `etf_fund_flow` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  etf_code VARCHAR(16) NOT NULL COMMENT 'ETF代码',
  etf_name VARCHAR(64) COMMENT 'ETF名称',
  trade_date DATE NOT NULL COMMENT '交易日',
  nav DECIMAL(12,4) COMMENT '单位净值',
  net_inflow DECIMAL(20,2) COMMENT '资金净流入(元)',
  tracking_index VARCHAR(32) COMMENT '跟踪指数/行业标签',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_etf_date (etf_code, trade_date),
  INDEX idx_date (trade_date),
  INDEX idx_tracking (tracking_index, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- 全局错误日志 (2026-08 新增)
-- 按日期+指纹去重，同一天同类异常只增计数不增行
-- ============================================================
CREATE TABLE IF NOT EXISTS `error_logs` (
  `id`          BIGINT AUTO_INCREMENT PRIMARY KEY,
  `error_date`  DATE NOT NULL,
  `created_at`  DATETIME(3) NOT NULL,
  `level`       ENUM('critical','error','warning') NOT NULL DEFAULT 'error',
  `source`      VARCHAR(64) NOT NULL,
  `method`      VARCHAR(10) DEFAULT '',
  `path`        VARCHAR(512) DEFAULT '',
  `exc_type`    VARCHAR(128) NOT NULL,
  `exc_msg`     TEXT,
  `traceback`   MEDIUMTEXT,
  `count`       INT NOT NULL DEFAULT 1,
  `fingerprint` CHAR(32) DEFAULT NULL,
  `last_at`     DATETIME(3) NOT NULL,
  INDEX idx_date (`error_date`),
  INDEX idx_level (`level`),
  INDEX idx_source (`source`),
  UNIQUE KEY uk_date_fp (`error_date`, `fingerprint`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- stock_sectors 存量补列已移至文件末尾迁移区

-- ============================================================
-- 以下表原仅在 database.py create_tables() 中定义，现统一至此
-- ============================================================

-- 个股公告表
CREATE TABLE IF NOT EXISTS `stock_notices` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(10) NOT NULL,
  title VARCHAR(500) NOT NULL,
  url VARCHAR(500),
  notice_date DATE NOT NULL,
  notice_type VARCHAR(30) DEFAULT '其他',
  source VARCHAR(30),
  source_confidence DECIMAL(4,2) DEFAULT 0.95,
  hash VARCHAR(32) NOT NULL,
  collected_at DATETIME,
  INDEX idx_code_notice_date (code, notice_date),
  INDEX idx_notice_date (notice_date),
  UNIQUE KEY uk_hash (hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 每日采集日志表
CREATE TABLE IF NOT EXISTS `daily_collection_log` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  source VARCHAR(20) NOT NULL DEFAULT 'scheduled',
  data_source VARCHAR(20) NOT NULL,
  stock_pool VARCHAR(20) NOT NULL DEFAULT 'all',
  total_codes INT DEFAULT 0,
  saved INT DEFAULT 0,
  new_dates INT DEFAULT 0,
  new_stocks INT DEFAULT 0,
  elapsed_seconds DECIMAL(10,1) DEFAULT 0,
  status VARCHAR(20) NOT NULL DEFAULT 'pending',
  error_message TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 冒烟测试日志表
CREATE TABLE IF NOT EXISTS `smoke_test_log` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  run_date DATE NOT NULL,
  source VARCHAR(20) NOT NULL DEFAULT 'scheduled',
  total INT DEFAULT 0,
  passed INT DEFAULT 0,
  failed INT DEFAULT 0,
  skipped INT DEFAULT 0,
  elapsed_seconds DECIMAL(10,1) DEFAULT 0,
  status VARCHAR(20) NOT NULL DEFAULT 'pending',
  detail TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_smoke_created (created_at),
  INDEX idx_smoke_run_date (run_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 基本面数据表 (季频)
CREATE TABLE IF NOT EXISTS `fundamental_data` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(10) NOT NULL,
  report_date DATE NOT NULL,
  ann_date DATE DEFAULT NULL,
  roe DECIMAL(10,4) DEFAULT NULL,
  net_profit_yoy DECIMAL(20,4) DEFAULT NULL,
  revenue_yoy DECIMAL(20,4) DEFAULT NULL,
  debt_to_assets DECIMAL(10,4) DEFAULT NULL,
  eps DECIMAL(10,4) DEFAULT NULL,
  grossprofit_margin DECIMAL(10,4) DEFAULT NULL,
  roa DECIMAL(20,4) DEFAULT NULL,
  ebit DECIMAL(20,4) DEFAULT NULL,
  netprofit_margin DECIMAL(20,4) DEFAULT NULL,
  ocf_to_debt DECIMAL(10,4) DEFAULT NULL,
  roe_yoy DECIMAL(20,4) DEFAULT NULL,
  assets_turn DECIMAL(10,4) DEFAULT NULL,
  quick_ratio DECIMAL(10,4) DEFAULT NULL,
  profit_dedt DECIMAL(20,4) DEFAULT NULL,
  diluted2_eps DECIMAL(10,4) DEFAULT NULL,
  bps DECIMAL(10,4) DEFAULT NULL,
  cfps DECIMAL(10,4) DEFAULT NULL,
  pb DECIMAL(10,4) DEFAULT NULL,
  pe DECIMAL(10,4) DEFAULT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_code_report (code, report_date),
  INDEX idx_code (code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `fundamental_data_versions` (
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
  KEY idx_fundamental_version_asof (code,report_date,ann_date,source_observed_at),
  KEY idx_fundamental_version_capture (source_capture_id,source_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `stock_dividend_events` (
  id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  ts_code VARCHAR(12) NOT NULL,
  end_date DATE NOT NULL,
  ann_date DATE DEFAULT NULL,
  div_proc VARCHAR(32) DEFAULT NULL,
  stk_div DECIMAL(18,6) DEFAULT NULL,
  cash_div DECIMAL(18,6) DEFAULT NULL,
  cash_div_tax DECIMAL(18,6) DEFAULT NULL,
  record_date DATE DEFAULT NULL,
  ex_date DATE DEFAULT NULL,
  pay_date DATE DEFAULT NULL,
  imp_ann_date DATE DEFAULT NULL,
  data_quality_status VARCHAR(32) NOT NULL DEFAULT 'unreviewed',
  ann_date_key DATE GENERATED ALWAYS AS (COALESCE(ann_date,'1000-01-01')) STORED,
  div_proc_key VARCHAR(32) GENERATED ALWAYS AS (COALESCE(div_proc,'')) STORED,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uk_dividend_version_nullsafe (ts_code,end_date,ann_date_key,div_proc_key),
  KEY idx_dividend_ann (ann_date),
  KEY idx_dividend_ex (ex_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `stock_dividend_source_versions` (
  capture_id CHAR(64) NOT NULL,
  source_row_no BIGINT UNSIGNED NOT NULL,
  observed_at DATETIME(6) NOT NULL,
  ts_code VARCHAR(12) NOT NULL,
  end_date DATE NOT NULL,
  ann_date DATE DEFAULT NULL,
  div_proc VARCHAR(32) DEFAULT NULL,
  imp_ann_date DATE DEFAULT NULL,
  stk_div DECIMAL(18,6) DEFAULT NULL,
  cash_div DECIMAL(18,6) DEFAULT NULL,
  cash_div_tax DECIMAL(18,6) DEFAULT NULL,
  record_date DATE DEFAULT NULL,
  ex_date DATE DEFAULT NULL,
  pay_date DATE DEFAULT NULL,
  payload_sha256 CHAR(64) NOT NULL,
  payload_json JSON NOT NULL,
  PRIMARY KEY (capture_id,source_row_no),
  KEY idx_dividend_source_pit (ts_code,ann_date,imp_ann_date),
  KEY idx_dividend_source_business (ts_code,end_date,ann_date,div_proc),
  KEY idx_dividend_source_observed (observed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 北向资金汇总表
CREATE TABLE IF NOT EXISTS `north_flow_summary` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  trade_date DATE NOT NULL UNIQUE,
  north_net DECIMAL(16,2) DEFAULT NULL,
  north_buy DECIMAL(16,2) DEFAULT NULL,
  south_net DECIMAL(16,2) DEFAULT NULL,
  hgt_net DECIMAL(16,2) DEFAULT NULL,
  sgt_net DECIMAL(16,2) DEFAULT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 盈利预测数据表
CREATE TABLE IF NOT EXISTS `forecast_data` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(10) NOT NULL,
  end_date DATE NOT NULL,
  ann_date DATE DEFAULT NULL,
  net_profit_min DECIMAL(20,4) DEFAULT NULL,
  net_profit_max DECIMAL(20,4) DEFAULT NULL,
  p_change_min DECIMAL(10,4) DEFAULT NULL,
  p_change_max DECIMAL(10,4) DEFAULT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_code_enddate (code, end_date),
  INDEX idx_code (code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 利润表数据
CREATE TABLE IF NOT EXISTS `income_data` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(10) NOT NULL,
  end_date DATE NOT NULL,
  ann_date DATE DEFAULT NULL,
  f_ann_date DATE DEFAULT NULL,
  update_flag VARCHAR(8) DEFAULT NULL,
  operate_profit DECIMAL(20,4) DEFAULT NULL,
  total_revenue DECIMAL(20,4) DEFAULT NULL,
  oper_cost DECIMAL(20,4) DEFAULT NULL,
  sell_exp DECIMAL(20,4) DEFAULT NULL,
  admin_exp DECIMAL(20,4) DEFAULT NULL,
  fin_exp DECIMAL(20,4) DEFAULT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_code_enddate (code, end_date),
  INDEX idx_code (code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 利润表供应商原始版本，保留不同 report_type/comp_type 供冲突审计
CREATE TABLE IF NOT EXISTS `income_data_source_variants` (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  variant_key CHAR(64) NOT NULL,
  code VARCHAR(10) NOT NULL,
  end_date DATE NOT NULL,
  ann_date DATE DEFAULT NULL,
  report_type VARCHAR(8) DEFAULT NULL,
  comp_type VARCHAR(8) DEFAULT NULL,
  payload JSON NOT NULL,
  collected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_variant_key (variant_key),
  INDEX idx_code_end_date (code, end_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 分析师一致预期表
CREATE TABLE IF NOT EXISTS `analyst_forecast` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(10) NOT NULL,
  report_date DATE NOT NULL,
  net_profit_forecast DECIMAL(20,4) DEFAULT NULL,
  net_profit_yoy DECIMAL(12,4) DEFAULT NULL,
  revenue_forecast DECIMAL(20,4) DEFAULT NULL,
  revenue_yoy DECIMAL(12,4) DEFAULT NULL,
  eps_forecast DECIMAL(12,4) DEFAULT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_code_date (code, report_date),
  INDEX idx_code (code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Qlib预测结果缓存表
CREATE TABLE IF NOT EXISTS `qlib_predictions` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  model_id VARCHAR(64) NOT NULL,
  code VARCHAR(10) NOT NULL,
  trade_date DATE NOT NULL,
  score DECIMAL(10,6) NOT NULL DEFAULT 0,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_model_code_date (model_id, code, trade_date),
  INDEX idx_code_date (code, trade_date),
  INDEX idx_model_date (model_id, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 选股结果保存表
CREATE TABLE IF NOT EXISTS `screening_saves` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  save_name VARCHAR(100) NOT NULL,
  job_id VARCHAR(32) NOT NULL,
  pool VARCHAR(30) NOT NULL DEFAULT 'all',
  eval_date DATE NOT NULL,
  direction VARCHAR(30) DEFAULT NULL,
  candidates JSON NOT NULL,
  recommended JSON DEFAULT NULL,
  params_json JSON DEFAULT NULL,
  note VARCHAR(500) DEFAULT '',
  profile_id VARCHAR(64) DEFAULT NULL,
  profile_version INT DEFAULT NULL,
  profile_fingerprint CHAR(64) DEFAULT NULL,
  data_fingerprint CHAR(64) DEFAULT NULL,
  model_fingerprint CHAR(64) DEFAULT NULL,
  factor_fingerprint CHAR(64) DEFAULT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_eval_date (eval_date),
  INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `decision_policy_versions` (
  policy_id VARCHAR(64) NOT NULL,
  version INT NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'shadow',
  policy_json JSON NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (policy_id, version),
  INDEX idx_decision_policy_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `decision_runs` (
  decision_id VARCHAR(36) PRIMARY KEY,
  screening_save_id INT NOT NULL,
  eval_date DATE NOT NULL,
  data_cutoff DATE NOT NULL,
  policy_id VARCHAR(64) NOT NULL,
  policy_version INT NOT NULL,
  mode VARCHAR(20) NOT NULL,
  top_n INT NOT NULL,
  source_fingerprint CHAR(64) NOT NULL,
  result_fingerprint CHAR(64) NOT NULL,
  request_json JSON DEFAULT NULL,
  result_json JSON NOT NULL,
  profile_id VARCHAR(64) DEFAULT NULL,
  profile_version INT DEFAULT NULL,
  profile_fingerprint CHAR(64) DEFAULT NULL,
  data_fingerprint CHAR(64) DEFAULT NULL,
  model_fingerprint CHAR(64) DEFAULT NULL,
  factor_fingerprint CHAR(64) DEFAULT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_decision_screening (screening_save_id),
  INDEX idx_decision_eval_date (eval_date),
  INDEX idx_decision_policy (policy_id, policy_version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `decision_candidates` (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  decision_id VARCHAR(36) NOT NULL,
  code VARCHAR(10) NOT NULL,
  original_rank INT NOT NULL,
  final_rank INT DEFAULT NULL,
  decision_status VARCHAR(20) NOT NULL,
  base_score DECIMAL(20,10) DEFAULT 0,
  overlay_delta DECIMAL(20,10) DEFAULT 0,
  risk_penalty DECIMAL(20,10) DEFAULT 0,
  cost_penalty DECIMAL(20,10) DEFAULT 0,
  utility_score DECIMAL(20,10) DEFAULT 0,
  hard_excluded TINYINT(1) DEFAULT 0,
  exclusion_reason VARCHAR(255) DEFAULT '',
  candidate_json JSON NOT NULL,
  UNIQUE KEY uk_decision_candidate (decision_id, code),
  INDEX idx_decision_candidate_rank (decision_id, final_rank)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `decision_policy_evaluations` (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  policy_id VARCHAR(64) NOT NULL,
  policy_version INT NOT NULL,
  eval_start DATE NOT NULL,
  eval_end DATE NOT NULL,
  baseline_policy_id VARCHAR(64) DEFAULT 'legacy',
  metrics_json JSON NOT NULL,
  passed TINYINT(1) NOT NULL DEFAULT 0,
  note VARCHAR(500) DEFAULT '',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_policy_evaluation (policy_id, policy_version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 调仓信号表 (含执行追踪)
CREATE TABLE IF NOT EXISTS `rebalance_signals` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  account_id VARCHAR(64) DEFAULT NULL,
  account_version BIGINT DEFAULT NULL,
  planned_shares INT DEFAULT NULL,
  filled_shares INT NOT NULL DEFAULT 0,
  batch_id VARCHAR(32) NOT NULL COMMENT '批次ID, 同一次检查的信号共享',
  screening_save_id INT DEFAULT NULL COMMENT '关联的选股保存ID',
  check_type VARCHAR(20) NOT NULL DEFAULT 'rebalance' COMMENT 'daily_risk/rebalance',
  code VARCHAR(10) NOT NULL,
  name VARCHAR(50) NOT NULL DEFAULT '',
  side VARCHAR(10) NOT NULL COMMENT 'buy/sell/reduce/add',
  action VARCHAR(50) NOT NULL COMMENT '建议买入/止损卖出/止盈卖出/减仓调整/加仓',
  current_weight DECIMAL(8,6) DEFAULT 0,
  target_weight DECIMAL(8,6) DEFAULT 0,
  delta_weight DECIMAL(8,6) DEFAULT 0,
  amount DECIMAL(16,2) DEFAULT 0,
  urgency VARCHAR(20) NOT NULL DEFAULT 'normal' COMMENT 'stop_loss/take_profit/normal',
  reason VARCHAR(500) DEFAULT '',
  execution_status VARCHAR(20) NOT NULL DEFAULT 'pending' COMMENT 'pending/executed/skipped/cancelled',
  executed_at DATETIME DEFAULT NULL,
  executed_detail JSON DEFAULT NULL COMMENT '执行详情 {shares,price,date,notes}',
  profile_id VARCHAR(64) DEFAULT NULL,
  profile_version INT DEFAULT NULL,
  profile_fingerprint CHAR(64) DEFAULT NULL,
  data_fingerprint CHAR(64) DEFAULT NULL,
  model_fingerprint CHAR(64) DEFAULT NULL,
  factor_fingerprint CHAR(64) DEFAULT NULL,
  planned_price DECIMAL(12,3) DEFAULT NULL,
  actual_price DECIMAL(12,3) DEFAULT NULL,
  slippage DECIMAL(12,6) DEFAULT NULL,
  unfilled_reason VARCHAR(255) DEFAULT NULL,
    source_mode VARCHAR(20) DEFAULT 'live',
    input_fingerprint CHAR(64) DEFAULT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_batch (batch_id),
  INDEX idx_code (code),
  INDEX idx_exec_status (execution_status),
  INDEX idx_check_type (check_type),
  INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 调仓历史快照表
CREATE TABLE IF NOT EXISTS `rebalance_sessions` (
  id INT AUTO_INCREMENT PRIMARY KEY,
  account_id VARCHAR(64) DEFAULT NULL,
  account_version BIGINT DEFAULT NULL,
  commit_request_id CHAR(36) DEFAULT NULL,
  plan_fingerprint CHAR(64) DEFAULT NULL,
  batch_id VARCHAR(32) NOT NULL UNIQUE,
  check_type VARCHAR(20) NOT NULL DEFAULT 'rebalance',
  screening_save_id INT DEFAULT NULL,
  eval_date DATE NOT NULL,
  account_json JSON NOT NULL,
  holdings_json JSON NOT NULL,
  signals_json JSON NOT NULL,
  constraints_json JSON DEFAULT NULL,
  regime_json JSON DEFAULT NULL,
  timing_ratio DECIMAL(5,4) DEFAULT 1.0,
  turnover DECIMAL(6,4) DEFAULT 0,
  stop_loss_pct DECIMAL(5,4) DEFAULT -0.08,
  take_profit_half DECIMAL(5,4) DEFAULT 0.15,
  take_profit_full DECIMAL(5,4) DEFAULT 0.30,
  profile_id VARCHAR(64) DEFAULT NULL,
  profile_version INT DEFAULT NULL,
  profile_fingerprint CHAR(64) DEFAULT NULL,
  data_fingerprint CHAR(64) DEFAULT NULL,
  model_fingerprint CHAR(64) DEFAULT NULL,
  factor_fingerprint CHAR(64) DEFAULT NULL,
  source_mode VARCHAR(20) DEFAULT 'live',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_eval_date (eval_date),
  INDEX idx_check_type (check_type),
  INDEX idx_screening_save (screening_save_id),
  UNIQUE KEY uq_rebalance_sessions_commit_request (commit_request_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `schema_migrations` (
  version VARCHAR(80) PRIMARY KEY,
  applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT IGNORE INTO `schema_migrations` (version) VALUES ('20260915_strategy_evidence_integrity');
INSERT IGNORE INTO `schema_migrations` (version) VALUES ('20260915_strategy_runtime_guard');
INSERT IGNORE INTO `schema_migrations` (version) VALUES ('20261020_daily_quote_ingest_audit');
INSERT IGNORE INTO `schema_migrations` (version) VALUES ('20261030_risk_notification_outbox');
INSERT IGNORE INTO `schema_migrations` (version) VALUES ('20261030_risk_scan_runs');

CREATE TABLE IF NOT EXISTS `strategy_profile_versions` (
  profile_id VARCHAR(64) NOT NULL,
  version INT NOT NULL,
  name VARCHAR(120) NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'draft',
  config_json JSON NOT NULL,
  fingerprint CHAR(64) NOT NULL,
  parent_version INT DEFAULT NULL,
  blocked_reasons_json JSON DEFAULT NULL,
  created_by VARCHAR(64) DEFAULT 'system',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (profile_id, version),
  INDEX idx_strategy_status (status),
  INDEX idx_strategy_fingerprint (fingerprint)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `strategy_profile_active` (
  environment VARCHAR(20) PRIMARY KEY,
  profile_id VARCHAR(64) NOT NULL,
  version INT NOT NULL,
  activated_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `strategy_profile_evaluations` (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  profile_id VARCHAR(64) NOT NULL,
  version INT NOT NULL,
  evaluation_key CHAR(64) NOT NULL,
  metrics_json JSON NOT NULL,
  folds_json JSON NOT NULL,
  passed TINYINT(1) NOT NULL DEFAULT 0,
  failure_reasons_json JSON DEFAULT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_strategy_eval (profile_id, version, evaluation_key),
  INDEX idx_strategy_eval_profile (profile_id, version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `strategy_profile_transitions` (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  profile_id VARCHAR(64) NOT NULL,
  version INT NOT NULL,
  from_status VARCHAR(20) DEFAULT NULL,
  to_status VARCHAR(20) NOT NULL,
  action VARCHAR(32) NOT NULL,
  details_json JSON DEFAULT NULL,
  created_by VARCHAR(64) DEFAULT 'system',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_strategy_transition (profile_id, version, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `data_snapshots` (
  snapshot_id CHAR(64) PRIMARY KEY,
  as_of DATE NOT NULL,
  latest_trade_date DATE DEFAULT NULL,
  universe_name VARCHAR(64) NOT NULL,
  member_count INT NOT NULL DEFAULT 0,
  freshness_status VARCHAR(20) NOT NULL,
  pit_status VARCHAR(20) NOT NULL,
  snapshot_json JSON NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_data_snapshot_asof (as_of)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `universe_membership_history` (
  snapshot_id CHAR(64) NOT NULL,
  as_of DATE NOT NULL,
  universe_name VARCHAR(64) NOT NULL,
  code VARCHAR(10) NOT NULL,
  observation_count INT NOT NULL,
  first_trade_date DATE DEFAULT NULL,
  last_trade_date DATE DEFAULT NULL,
  PRIMARY KEY (snapshot_id, code),
  INDEX idx_universe_asof (universe_name, as_of, code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `security_reference_history` (
  as_of DATE NOT NULL,
  code VARCHAR(10) NOT NULL,
  name VARCHAR(80) DEFAULT '',
  industry VARCHAR(100) DEFAULT '',
  board VARCHAR(32) DEFAULT '',
  is_st TINYINT(1) DEFAULT 0,
  is_suspended TINYINT(1) DEFAULT 0,
  listed_date DATE DEFAULT NULL,
  delisted_date DATE DEFAULT NULL,
  source VARCHAR(32) DEFAULT 'snapshot',
  PRIMARY KEY (as_of, code),
  INDEX idx_security_reference_code (code, as_of)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `paper_runs` (
  paper_run_id VARCHAR(36) PRIMARY KEY,
  profile_id VARCHAR(64) NOT NULL,
  version INT NOT NULL,
  status VARCHAR(20) NOT NULL DEFAULT 'running',
  started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  ended_at DATETIME DEFAULT NULL,
  UNIQUE KEY uk_active_paper (profile_id, version),
  INDEX idx_paper_profile (profile_id, version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `paper_periods` (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  paper_run_id VARCHAR(36) NOT NULL,
  decision_date DATE NOT NULL,
  matured TINYINT(1) NOT NULL DEFAULT 0,
  valid TINYINT(1) NOT NULL DEFAULT 0,
  period_json JSON NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_paper_period (paper_run_id, decision_date),
  INDEX idx_paper_maturity (paper_run_id, matured, valid)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `strategy_profile_runtime_health` (
  profile_id VARCHAR(64) NOT NULL,
  version INT NOT NULL,
  equity_peak DECIMAL(20,4) NOT NULL DEFAULT 0,
  last_equity DECIMAL(20,4) NOT NULL DEFAULT 0,
  current_drawdown DECIMAL(12,8) NOT NULL DEFAULT 0,
  max_drawdown DECIMAL(12,8) NOT NULL DEFAULT 0,
  drawdown_breached TINYINT(1) NOT NULL DEFAULT 0,
  observed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (profile_id, version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `stock_index_membership_observations` (
  observed_at DATETIME(6) NOT NULL,
  ts_code VARCHAR(12) NOT NULL,
  index_code VARCHAR(16) NOT NULL,
  index_name VARCHAR(64),
  in_date DATE NOT NULL,
  out_date DATE,
  is_new VARCHAR(2),
  PRIMARY KEY (observed_at,ts_code,index_code,in_date),
  KEY idx_observation_asof (ts_code,observed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- 列级增量迁移已移至 Python 层 (core/database.py migrate_schema)
-- MariaDB 的 ADD COLUMN IF NOT EXISTS 不兼容 MySQL 8.x
-- ============================================================
