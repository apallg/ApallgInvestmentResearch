# 新增数据源表：融资融券、龙虎榜、十大股东
# 2020-01-01 ~ 今

-- ============================================================
-- 1. 融资融券日频数据（沪深统一）
-- ============================================================
CREATE TABLE IF NOT EXISTS margin_daily (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    trade_date  DATE NOT NULL,
    stock_code  VARCHAR(10) NOT NULL,
    stock_name  VARCHAR(20),
    -- 融资
    margin_balance   BIGINT   COMMENT '融资余额(元)',
    margin_buy       BIGINT   COMMENT '融资买入额(元)',
    margin_repay     BIGINT   COMMENT '融资偿还额(元)',
    -- 融券
    short_volume     BIGINT   COMMENT '融券余量(股)',
    short_balance    BIGINT   COMMENT '融券余额(元)',
    short_sell       BIGINT   COMMENT '融券卖出量(股)',
    short_repay      BIGINT   COMMENT '融券偿还量(股)',
    -- 汇总
    total_balance    BIGINT   COMMENT '融资融券余额(元)',
    -- 元信息
    source          VARCHAR(10) DEFAULT '' COMMENT 'SSE/SZSE',
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_date_code (trade_date, stock_code),
    INDEX idx_date (trade_date),
    INDEX idx_code (stock_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ============================================================
-- 2. 龙虎榜数据
-- ============================================================
CREATE TABLE IF NOT EXISTS lhb_daily (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    trade_date      DATE NOT NULL,
    stock_code      VARCHAR(10) NOT NULL,
    stock_name      VARCHAR(20),
    close           DECIMAL(10,2)   COMMENT '收盘价',
    pct_chg         DECIMAL(8,2)    COMMENT '涨跌幅(%)',
    -- 龙虎榜资金
    lhb_net_buy     BIGINT          COMMENT '龙虎榜净买额(元)',
    lhb_buy         BIGINT          COMMENT '龙虎榜买入额(元)',
    lhb_sell        BIGINT          COMMENT '龙虎榜卖出额(元)',
    lhb_turnover    BIGINT          COMMENT '龙虎榜成交额(元)',
    market_turnover BIGINT          COMMENT '市场总成交额(元)',
    net_pct         DECIMAL(8,2)    COMMENT '净买额占总成交比(%)',
    turnover_pct    DECIMAL(8,2)    COMMENT '成交额占总成交比(%)',
    -- 股票属性
    turnover_rate   DECIMAL(8,2)    COMMENT '换手率(%)',
    float_mv        DECIMAL(18,2)   COMMENT '流通市值(元)',
    -- 上榜后表现
    post_1d         DECIMAL(8,2)    COMMENT '上榜后1日涨跌幅(%)',
    post_2d         DECIMAL(8,2)    COMMENT '上榜后2日涨跌幅(%)',
    post_5d         DECIMAL(8,2)    COMMENT '上榜后5日涨跌幅(%)',
    post_10d        DECIMAL(8,2)    COMMENT '上榜后10日涨跌幅(%)',
    -- 文本
    reason          VARCHAR(500)    COMMENT '上榜原因',
    interpretation  VARCHAR(500)    COMMENT '解读',
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_date_code (trade_date, stock_code),
    INDEX idx_date (trade_date),
    INDEX idx_code (stock_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ============================================================
-- 3. 十大股东（季度数据）
-- ============================================================
CREATE TABLE IF NOT EXISTS shareholder_quarterly (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    stock_code      VARCHAR(10) NOT NULL,
    stock_name      VARCHAR(20),
    report_date     DATE NOT NULL          COMMENT '报告期(季度末)',
    announce_date   DATE                   COMMENT '公告日期',
    -- 聚合指标
    total_holders   INT                    COMMENT '股东总数',
    avg_hold        DECIMAL(12,0)          COMMENT '户均持股数',
    top10_hold_pct  DECIMAL(8,2)           COMMENT '前十大股东持股比例合计(%)',
    top10_float_pct DECIMAL(8,2)           COMMENT '前十大流通股东持股比例合计(%)',
    -- 机构持仓
    fund_holders    INT                    COMMENT '基金持股家数',
    fund_hold_pct   DECIMAL(8,2)           COMMENT '基金持股比例合计(%)',
    -- 变更
    holders_change  DECIMAL(8,2)           COMMENT '股东人数环比变化(%)',
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_stock_quarter (stock_code, report_date),
    INDEX idx_date (report_date),
    INDEX idx_code (stock_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
