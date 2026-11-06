export const VOLATILITY_MODEL_STATES = [
  { key: 'high_vol', label: '高波动', hint: '建议超卖反弹', direction: 'oversold_rebound' },
  { key: 'low_vol', label: '低波动', hint: '建议趋势动量', direction: 'trend_momentum' },
]

export const REGIME_MODEL_STATES = [
  { key: 'volatile_up', label: '高波快涨', hint: '建议趋势动量', direction: 'trend_momentum' },
  { key: 'volatile_down', label: '高波快跌', hint: '建议超卖反弹', direction: 'oversold_rebound' },
  { key: 'volatile_flat', label: '高波震荡', hint: '建议量价突破', direction: 'volume_breakout' },
  { key: 'steady_up', label: '稳步上涨', hint: '建议趋势动量', direction: 'trend_momentum' },
  { key: 'steady_down', label: '稳步下跌', hint: '建议量价突破', direction: 'volume_breakout' },
  { key: 'steady_flat', label: '低波盘整', hint: '建议超卖反弹', direction: 'oversold_rebound' },
]

export function activeModelStates(useVolSwitch) {
  return useVolSwitch ? VOLATILITY_MODEL_STATES : REGIME_MODEL_STATES
}

export function modelRefPayload(modelId) {
  const parts = String(modelId || '').split('/')
  if (parts.length !== 2 || !parts.every(Boolean)) {
    throw new Error('请选择一个有效的已训练模型')
  }
  return parts
}

export function stateModelsPayload(mapping, useVolSwitch) {
  const states = activeModelStates(useVolSwitch)
  const selected = states.filter(state => mapping?.[state.key])
  if (!selected.length) return {}
  if (selected.length !== states.length) {
    throw new Error(`请完成全部 ${states.length} 个状态的模型选择，或全部留空使用自动匹配`)
  }
  return Object.fromEntries(states.map(state => [state.key, modelRefPayload(mapping[state.key])]))
}
