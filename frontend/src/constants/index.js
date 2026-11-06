// 股票代码前缀选项 — 统一管理，不再在多个页面中重复定义
// 数组格式: [[prefixes], label] — 用于 checkbox 遍历
export const CODE_PREFIX_GROUPS = [
  [['600','601','603','605'], '沪主板'],
  [['000','001','002','003'], '深主板'],
  [['300','301'], '创业板'],
  [['688','689'], '科创板'],
]

// 对象格式: label → prefix_csv — 用于 API 参数
export const CODE_PREFIX_OPTIONS = [
  { label: '沪主板', value: '600,601,603,605' },
  { label: '深主板', value: '000,001,002,003' },
  { label: '创业板', value: '300,301' },
  { label: '科创板', value: '688,689' },
]

// 信号源键值和中文名映射
export const SIGNAL_LABELS = {
  tech: '技术指标',
  qlib: 'AI预测',
  sentiment: '情绪分析',
  fundamental: '基本面',
  flow: '资金流向',
  north_flow: '北向资金',
  size_turnover: '小市值换手',
  momentum: '动量因子',
  volatility: '波动率',
  ht_factors: '华泰多因子',
  wb_industry: '白盒-行业残差',
  wb_divergence: '白盒-量价背离',
  wb_moneyflow: '白盒-资金异动',
  illiq: '非流动性(Amihud)',
  overnight_gap: '隔夜跳空',
}

// 信号源缩写（用于紧凑展示）
export const SIGNAL_ABBR = {
  tech: '技', qlib: 'Q', sentiment: '情', fundamental: '基',
  flow: '资', north_flow: '北', size_turnover: '小',
  momentum: '动', volatility: '波',
  illiq: '流', overnight_gap: '跳',
}

// 市场状态标签 (2×3 网格: 波动率×方向)
export const REGIME_LABELS = {
  volatile_up: '高波快涨',
  volatile_down: '高波快跌',
  volatile_flat: '高波震荡',
  steady_up: '稳步上涨',
  steady_down: '稳步下跌',
  steady_flat: '低波盘整',
}

// 技术指标预设
export const TECH_PRESETS = [
  { value: 'all', label: '全部指标' },
  { value: 'trend', label: '趋势类（指数平滑、均线、布林带）' },
  { value: 'reversal', label: '反转类 (RSI/KDJ/BB)' },
  { value: 'momentum', label: '动量类 (MACD/RSI/VOL)' },
  { value: 'macd_only', label: '仅 MACD' },
  { value: 'rsi_only', label: '仅 RSI' },
  { value: 'ma_only', label: '仅 MA均线' },
  { value: 'kdj_only', label: '仅 KDJ' },
  { value: 'bb_only', label: '仅 布林带' },
  { value: 'volume_only', label: '仅 成交量' },
]

// 股票池选项
export const POOL_OPTIONS = [
  { value: 'all', label: '全A股' },
  { value: 'csi300', label: '沪深300' },
  { value: 'csi500', label: '中证500' },
  { value: 'csi300+csi500', label: '沪深300+中证500' },
  { value: 'stable_history_500d', label: '稳定历史池500D' },
  { value: 'non_csi300', label: '非沪深300' },
  { value: 'small_cap', label: '小市值 (中证1000+国证2000)' },
  { value: 'sector_rotation', label: '板块轮动（前5个行业）' },
  { value: 'concept_rotation', label: '概念轮动（前5个概念）' },
]

// 宏观指标标签
export const MACRO_LABELS = {
  shibor_on: 'Shibor隔夜', shibor_1y: 'Shibor1年', shibor_3m: 'Shibor3月',
  lpr_1y: 'LPR1年', lpr_5y: 'LPR5年', spread: '期限利差',
  m1_yoy: 'M1', m2_yoy: 'M2', m1_m2: 'M1-M2剪刀差',
  cpi_yoy: 'CPI', core_cpi: '核心CPI', ppi_yoy: 'PPI',
  pmi: 'PMI', pmi_new_order: '新订单',
}

// 宏观风险等级
export const MACRO_RISK_COLORS = {
  normal: 'var(--success)',
  warning: '#f59e0b',
  danger: 'var(--danger)',
}

export const MACRO_RISK_LABELS = {
  normal: '正常', warning: '关注', danger: '危险',
}
