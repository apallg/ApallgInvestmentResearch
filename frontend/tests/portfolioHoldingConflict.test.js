import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() }))
const loadBroker = vi.hoisted(() => vi.fn())

vi.mock('../src/api/index.js', () => ({ default: api }))
vi.mock('../src/composables/useBroker.js', () => ({
  useBroker: () => ({ cashEnabled: false, brokerLabel: 'mock', loadBroker }),
}))

import PortfolioView from '../src/views/PortfolioView.vue'
import { useHoldingsStore } from '../src/stores/holdings.js'

const original = {
  id: 5, version: 2, code: '000001', name: '平安银行', shares: 100,
  cost_price: 10, buy_date: '2026-01-05', risk_threshold: 8,
}
const current = { ...original, version: 3, shares: 200, cost_price: 11 }

describe('PortfolioView holding version conflict', () => {
  beforeEach(() => {
    api.get.mockReset()
    api.post.mockReset()
    api.put.mockReset()
    api.delete.mockReset()
    loadBroker.mockReset()
    let holdingReads = 0
    api.get.mockImplementation(async path => {
      if (path === '/trading/account') return { data: {} }
      if (path === '/rebalance/account/local') return { data: { status: 'uninitialized' } }
      if (path === '/portfolio/holdings') {
        holdingReads += 1
        return { data: [holdingReads === 1 ? original : current] }
      }
      return { data: {} }
    })
    api.post.mockResolvedValue({ data: {} })
    const store = useHoldingsStore()
    store.holdings.value = []
    store.invalidate()
  })

  it('closes the stale edit form and displays freshly loaded shares after 409', async () => {
    const conflict = new Error('持仓已变化，请刷新后重试')
    conflict.status = 409
    api.put.mockRejectedValue(conflict)
    const wrapper = mount(PortfolioView, {
      global: { stubs: { MetricCard: true, RouterLink: true } },
    })
    await flushPromises()

    await wrapper.findAll('button').find(button => button.text() === '编辑').trigger('click')
    await wrapper.findAll('button').find(button => button.text() === '保存').trigger('click')
    await flushPromises()

    expect(api.put).toHaveBeenCalledWith('/portfolio/holdings/000001', expect.objectContaining({
      expected_id: original.id,
      expected_version: original.version,
    }))
    expect(wrapper.text()).toContain('已重新读取当前持仓，请重新打开编辑后再提交')
    expect(wrapper.text()).toContain('200')
    expect(wrapper.find('input').exists()).toBe(false)
    wrapper.unmount()
  })
})
