import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const checkIntraday = vi.hoisted(() => vi.fn())

vi.mock('../src/api/rebalance.js', () => ({ checkIntraday }))

import IntradayCheckPanel from '../src/components/IntradayCheckPanel.vue'

function verifiedResponse(code) {
  return {
    data: {
      results: [{
        code,
        analysis_status: 'verified',
        data_provenance: { quantity_status: 'verified', semantic_status: 'verified' },
        entry_quality_label: '优秀',
        entry_quality_score: 80,
        current_price: 10,
        supports: [],
        signals: [],
        recent_bars: [],
        execution: { can_execute: true, action: 'limit', status: 'ready', reason: 'verified' },
      }],
      sell_results: [],
      source: 'ths',
      timestamp: '10:00:00',
      unavailable: [],
    },
  }
}

function deferred() {
  let resolve
  let reject
  const promise = new Promise((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

describe('IntradayCheckPanel request validity', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('clears prior verified advice when a refresh fails', async () => {
    checkIntraday.mockResolvedValueOnce(verifiedResponse('000001'))
    const wrapper = mount(IntradayCheckPanel, {
      props: { buyCodes: ['000001'], autoRefresh: false },
    })
    await flushPromises()
    expect(wrapper.text()).toContain('条件通过 1')
    expect(wrapper.text()).toContain('优秀 80')

    checkIntraday.mockRejectedValueOnce(new Error('minute source unavailable'))
    await wrapper.get('.icp-btn--sm').trigger('click')
    await flushPromises()

    expect(wrapper.find('.icp-card').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('条件通过 1')
    expect(wrapper.text()).toContain('minute source unavailable')
    wrapper.unmount()
  })

  it('ignores a late response for the previous target after inputs change', async () => {
    const oldRequest = deferred()
    const newRequest = deferred()
    checkIntraday.mockReturnValueOnce(oldRequest.promise).mockReturnValueOnce(newRequest.promise)
    const wrapper = mount(IntradayCheckPanel, {
      props: { buyCodes: ['000001'], autoRefresh: false },
    })
    await flushPromises()
    await wrapper.setProps({ buyCodes: ['000002'] })
    await flushPromises()
    expect(checkIntraday).toHaveBeenCalledTimes(2)

    newRequest.resolve(verifiedResponse('000002'))
    await flushPromises()
    oldRequest.resolve(verifiedResponse('000001'))
    await flushPromises()

    expect(wrapper.findAll('.icp-card__code').map(node => node.text())).toEqual(['000002'])
    expect(wrapper.text()).toContain('000002')
    expect(wrapper.text()).not.toContain('000001')
    wrapper.unmount()
  })

  it('hides minute-derived metrics and blocks execution when provenance is unverified', async () => {
    checkIntraday.mockResolvedValueOnce({
      data: {
        results: [{
          code: '000001',
          analysis_status: 'verified',
          data_provenance: { quantity_status: 'verified', semantic_status: 'unverified' },
          entry_quality_label: '优秀',
          entry_quality_score: 80,
          current_price: 10,
          vwap: 9.9,
          supports: [{ name: 'MA20', price: 9.8, distance_pct: -0.02 }],
          nearest_support: { name: 'MA20', price: 9.8, distance_pct: -0.02 },
          signals: [{ type: 'near_support', description: '接近支撑', strength: 3 }],
          recent_bars: [{ time: '10:00', open: 10, close: 10, low: 9.9, high: 10.1 }],
          execution: { can_execute: true, action: 'limit', intent_action: 'limit', limit_price: 9.9, status: 'ready', reason: 'ready' },
        }],
        sell_results: [{
          code: '000002',
          name: '测试股票',
          side: 'sell',
          urgency: 'normal',
          suggested_shares: 200,
          analysis_status: 'verified',
          data_provenance: { quantity_status: 'verified', semantic_status: 'unverified' },
          current_price: 10,
          execution: { can_execute: true, action: 'limit', intent_action: 'limit', limit_price: 10.1, status: 'ready', reason: 'ready' },
        }],
        source: 'ths',
        timestamp: '10:00:00',
        unavailable: [],
      },
    })
    const wrapper = mount(IntradayCheckPanel, {
      props: { buyCodes: ['000001'], autoRefresh: false },
    })
    await flushPromises()

    expect(wrapper.text()).toContain('分钟数据 未认证')
    expect(wrapper.text()).toContain('分钟数据来源语义未核验，执行建议不可用')
    expect(wrapper.text()).toContain('调仓计划量 200股')
    expect(wrapper.text()).not.toContain('条件通过')
    expect(wrapper.text()).not.toContain('优秀 80')
    expect(wrapper.text()).not.toContain('9.80')
    expect(wrapper.text()).not.toContain('@9.90')
    expect(wrapper.find('.icp-chart').exists()).toBe(false)
    wrapper.unmount()
  })
})
