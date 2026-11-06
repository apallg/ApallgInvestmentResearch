import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const checkT0 = vi.hoisted(() => vi.fn())

vi.mock('../src/api/rebalance.js', () => ({ checkT0 }))

import T0Panel from '../src/components/T0Panel.vue'

function response(code) {
  return {
    data: {
      results: [{
        code,
        current_price: 10,
        vwap: 9.9,
        day_low: 9.8,
        day_high: 10.1,
        atr_abs: 0.2,
        rsi: 45,
        swing_factor: 1,
        aggressive: {
          name: 'I', label: '激进', allowed: false, suggested_shares: 0,
          suggested_sell_shares: 0, flags: [],
        },
        conservative: null,
      }],
      unavailable: [],
      unavailable_details: {},
      data_status: 'verified',
      mode: 'live',
      semantic_status: 'unverified',
      coverage: { available: 1, requested: 1 },
      data_source: 'ths_live_minute',
      as_of: '2026-10-30',
      available_at: '2026-10-30T10:00:00+08:00',
      source: 'ths',
      timestamp: '10:00:00',
      sellable_source: 'local_ledger_t1',
      account_id: 'local-account',
      account_version: 3,
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

describe('T0Panel request validity', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('removes old T+0 results as soon as a refresh begins and on failure', async () => {
    checkT0.mockResolvedValueOnce(response('000001'))
    const wrapper = mount(T0Panel, {
      props: { codes: ['000001'], refreshInterval: 0 },
    })
    await flushPromises()
    expect(wrapper.text()).toContain('000001')

    const pending = deferred()
    checkT0.mockReturnValueOnce(pending.promise)
    await wrapper.get('.t0-btn').trigger('click')
    expect(wrapper.find('.t0-card').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('账本版本 3')

    pending.reject(new Error('T+0 source unavailable'))
    await flushPromises()
    expect(wrapper.find('.t0-card').exists()).toBe(false)
    expect(wrapper.text()).toContain('T+0 source unavailable')
    wrapper.unmount()
  })

  it('ignores a late T+0 response for the previous symbol', async () => {
    const oldRequest = deferred()
    const newRequest = deferred()
    checkT0.mockReturnValueOnce(oldRequest.promise).mockReturnValueOnce(newRequest.promise)
    const wrapper = mount(T0Panel, {
      props: { codes: ['000001'], refreshInterval: 0 },
    })
    await flushPromises()
    await wrapper.setProps({ codes: ['000002'] })
    await flushPromises()
    expect(checkT0).toHaveBeenCalledTimes(2)

    newRequest.resolve(response('000002'))
    await flushPromises()
    oldRequest.resolve(response('000001'))
    await flushPromises()

    expect(wrapper.find('.t0-code').text()).toBe('000002')
    expect(wrapper.text()).not.toContain('000001')
    wrapper.unmount()
  })

  it('hides minute-derived fields and variants when minute provenance is absent', async () => {
    checkT0.mockResolvedValueOnce(response('000001'))
    const wrapper = mount(T0Panel, {
      props: { codes: ['000001'], refreshInterval: 0 },
    })
    await flushPromises()

    expect(wrapper.text()).toContain('分钟来源语义未核验，不展示分钟派生指标或做T建议')
    expect(wrapper.get('.t0-price').text()).toBe('--')
    expect(wrapper.find('.t0-variant').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('VWAP 9.90')
    wrapper.unmount()
  })
})
