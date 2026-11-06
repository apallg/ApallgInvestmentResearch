import { describe, expect, it } from 'vitest'
import { localizeStrategyText, strategyStatusText } from '../src/utils/strategyTerms.js'

describe('策略术语中文化', () => {
  it('转换策略状态', () => {
    expect(strategyStatusText('champion')).toBe('正式策略')
    expect(strategyStatusText('challenger')).toBe('候选策略')
    expect(strategyStatusText('paper')).toBe('模拟运行')
    expect(strategyStatusText('blocked')).toBe('已阻断')
  })

  it('转换后端返回的流程术语', () => {
    expect(localizeStrategyText('Champion Profile OOS walk-forward run ID PIT')).toBe(
      '正式策略 策略版本 样本外 滚动样本外 运行编号 时点一致性',
    )
  })

  it('转换核心卫星因子集名称', () => {
    expect(localizeStrategyText('core_trend_satellite')).toBe('核心加趋势卫星因子集')
  expect(localizeStrategyText('core_rebound_satellite')).toBe('核心加反弹卫星因子集')
  expect(localizeStrategyText('running')).toBe('正在运行')
  })
})
