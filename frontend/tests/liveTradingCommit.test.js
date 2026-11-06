import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() }))
const holdingStore = vi.hoisted(() => ({ load: vi.fn(), holdings: null }))

vi.mock('../src/api/index.js', () => ({ default: api }))
vi.mock('../src/stores/holdings.js', async () => {
  const { ref } = await import('vue')
  holdingStore.holdings = ref([])
  return { useHoldingsStore: () => holdingStore }
})

import LiveTradingView from '../src/views/LiveTradingView.vue'
import { todayStr } from '../src/utils/date.js'

const storageKey = 'rebalance-plan-commit:LOCAL-LIVE'
const fillStorageKey = 'rebalance-fill-7'
const pending = {
  version: 1,
  screening_save_id: 17,
  payload: {
    decision_id: 'decision-17',
    commit_request_id: '123e4567-e89b-42d3-a456-426614174000',
    expected_plan_fingerprint: 'a'.repeat(64),
  },
}
const pendingFill = {
  shares: 100, price: '10.000', commission: '0.30', stamp_duty: '0',
  request_id: '323e4567-e89b-42d3-a456-426614174000', expected_account_version: 1,
  date: '2026-10-31',
}

function mountView() {
  return mount(LiveTradingView, {
    global: {
      stubs: {
        IntradayCheckPanel: true,
        T0Panel: true,
        StrategyProfileBanner: true,
      },
    },
  })
}

function retryButton(wrapper) {
  return wrapper.findAll('button').find(button => button.text().includes('用原请求ID核对并重试'))
}

describe('LiveTradingView pending rebalance commit', () => {
  beforeEach(() => {
    sessionStorage.clear()
    localStorage.clear()
    api.get.mockReset()
    api.post.mockReset()
    api.put.mockReset()
    api.delete.mockReset()
    api.get.mockReset()
    api.get.mockImplementation(async path => {
      if (path === '/screening/saves') return { data: [] }
      if (path === '/rebalance/account/local') {
        return { data: { account_id: 'LOCAL-LIVE', status: 'inactive', integrity: { ok: false }, legacy_unknown: {} } }
      }
      if (path === '/rebalance/account/local/reconciliation') return { data: { can_execute: false, reason: '待核对' } }
      if (path === '/rebalance/cash') return { data: { cash: 10000 } }
      if (path === '/rebalance/account/local/broker-trades/unresolved' ||
          path === '/rebalance/account/local/broker-reports/unresolved' ||
          path === '/rebalance/account/local/intents' || path === '/rebalance/signals/pending' ||
          path === '/rebalance/cooldown') return { data: [] }
      if (path === '/rebalance/account/local/quarantine') {
        return { data: { counts: { orders: 0, trades: 0, signals: 0, sessions: 0 }, orders: [], trades: [], signals: [], sessions: [] } }
      }
      return { data: [] }
    })
    holdingStore.load.mockReset()
    holdingStore.holdings.value = []
    holdingStore.load.mockImplementation(async () => holdingStore.holdings.value)
    api.post.mockResolvedValue({ data: {} })
  })

  it('restores the same pending request after remount', async () => {
    localStorage.setItem(storageKey, JSON.stringify(pending))
    const first = mountView()
    await flushPromises()
    expect(first.text()).toContain('上次调仓提交结果尚未确认')
    expect(first.text()).toContain(pending.payload.commit_request_id)
    first.unmount()

    const refreshed = mountView()
    await flushPromises()
    expect(refreshed.text()).toContain('上次调仓提交结果尚未确认')
    expect(refreshed.text()).toContain(pending.payload.commit_request_id)
    expect(api.post).not.toHaveBeenCalled()
    refreshed.unmount()
  })

  it('keeps the original request ID and blocks new plans when outcome is unknown', async () => {
    localStorage.setItem(storageKey, JSON.stringify(pending))
    api.post.mockRejectedValueOnce(new Error('network lost'))
    const wrapper = mountView()
    await flushPromises()

    await retryButton(wrapper).trigger('click')
    await flushPromises()

    expect(api.post).toHaveBeenCalledWith('/rebalance/run', expect.objectContaining({
      screening_save_id: pending.screening_save_id,
      commit_request_id: pending.payload.commit_request_id,
      expected_plan_fingerprint: pending.payload.expected_plan_fingerprint,
      preview_only: false,
    }), expect.any(Object))
    expect(JSON.parse(localStorage.getItem(storageKey))).toEqual(pending)
    expect(wrapper.text()).toContain('上次调仓提交结果尚未确认')
    expect(wrapper.text()).toContain('调仓方案提交结果尚未确认')
    wrapper.unmount()
  })

  it('clears the saved request only after a definitive noncommit response', async () => {
    localStorage.setItem(storageKey, JSON.stringify(pending))
    const failure = new Error('本地账户版本已变化')
    failure.status = 400
    failure.commitOutcome = 'not_committed'
    api.post.mockRejectedValueOnce(failure)
    const wrapper = mountView()
    await flushPromises()

    await retryButton(wrapper).trigger('click')
    await flushPromises()

    expect(localStorage.getItem(storageKey)).toBeNull()
    expect(wrapper.text()).not.toContain('上次调仓提交结果尚未确认')
    expect(wrapper.text()).toContain('服务确认本次调仓未写入')
    wrapper.unmount()
  })

  it('migrates legacy session-scoped plan requests to durable browser storage', async () => {
    sessionStorage.setItem(storageKey, JSON.stringify(pending))
    const wrapper = mountView()
    await flushPromises()

    expect(localStorage.getItem(storageKey)).toBe(JSON.stringify(pending))
    expect(sessionStorage.getItem(storageKey)).toBeNull()
    wrapper.unmount()
  })

  it.each([
    ['blocked', { status: 'inactive' }],
    ['bound to a broker', { status: 'active', broker_account_id: 'QMT-1' }],
  ])('restores and retries a committed fill after refresh when the account is %s', async (_label, accountState) => {
    localStorage.setItem(fillStorageKey, JSON.stringify(pendingFill))
    const originalGet = api.get.getMockImplementation()
    api.get.mockImplementation(async (path, ...args) => {
      if (path === '/rebalance/account/local') {
        return { data: { account_id: 'LOCAL-LIVE', version: 2, integrity: { ok: false }, legacy_unknown: {}, ...accountState } }
      }
      if (path === '/rebalance/signals/7') {
        return { data: {
          id: 7, account_id: 'LOCAL-LIVE', code: '000001', name: '平安银行', side: 'buy',
          planned_shares: 100, filled_shares: 100, execution_status: 'executed', account_version: 1,
        } }
      }
      return originalGet(path, ...args)
    })
    api.put.mockResolvedValueOnce({ data: {
      execution_status: 'executed', already_recorded: true, remaining_plan_shares: 0, account_version: 2,
    } })
    const wrapper = mountView()
    await flushPromises()

    expect(wrapper.text()).toContain('存在尚未确认的成交请求，已暂停新调仓方案')
    expect(wrapper.text()).toContain(pendingFill.request_id)
    await wrapper.findAll('button').find(button => button.text().includes('读取信号并用原请求ID核对')).trigger('click')
    await flushPromises()
    await wrapper.findAll('button').find(button => button.text() === '确认记录').trigger('click')
    await flushPromises()

    expect(api.put).toHaveBeenCalledWith('/rebalance/signals/7/execute', pendingFill)
    expect(localStorage.getItem(fillStorageKey)).toBeNull()
    expect(wrapper.text()).not.toContain('存在尚未确认的成交请求，已暂停新调仓方案')
    wrapper.unmount()
  })

  it('migrates an uncertain legacy fill request and does not submit it with a new ID', async () => {
    sessionStorage.setItem(fillStorageKey, JSON.stringify(pendingFill))
    const wrapper = mountView()
    await flushPromises()

    expect(localStorage.getItem(fillStorageKey)).toBe(JSON.stringify(pendingFill))
    expect(sessionStorage.getItem(fillStorageKey)).toBeNull()
    expect(wrapper.text()).toContain(pendingFill.request_id)
    expect(api.put).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('migrates a legacy session-scoped fill request to durable browser storage', async () => {
    sessionStorage.setItem(fillStorageKey, JSON.stringify(pendingFill))
    const wrapper = mountView()
    await flushPromises()

    expect(localStorage.getItem(fillStorageKey)).toBe(JSON.stringify(pendingFill))
    expect(sessionStorage.getItem(fillStorageKey)).toBeNull()
    expect(wrapper.text()).toContain(pendingFill.request_id)
    wrapper.unmount()
  })

  it('synchronizes an unconfirmed fill written or resolved by another tab', async () => {
    const wrapper = mountView()
    await flushPromises()

    localStorage.setItem(fillStorageKey, JSON.stringify(pendingFill))
    window.dispatchEvent(new StorageEvent('storage', {
      key: fillStorageKey, newValue: JSON.stringify(pendingFill), storageArea: localStorage,
    }))
    await flushPromises()
    expect(wrapper.text()).toContain(pendingFill.request_id)
    expect(wrapper.text()).toContain('暂停新调仓方案')

    localStorage.removeItem(fillStorageKey)
    window.dispatchEvent(new StorageEvent('storage', { key: fillStorageKey, newValue: null, storageArea: localStorage }))
    await flushPromises()
    expect(wrapper.text()).not.toContain(pendingFill.request_id)
    wrapper.unmount()
  })

  it('synchronizes pending plan commits written or resolved by another tab', async () => {
    const wrapper = mountView()
    await flushPromises()

    localStorage.setItem(storageKey, JSON.stringify(pending))
    window.dispatchEvent(new StorageEvent('storage', {
      key: storageKey, newValue: JSON.stringify(pending), storageArea: localStorage,
    }))
    await flushPromises()
    expect(wrapper.text()).toContain(pending.payload.commit_request_id)

    localStorage.removeItem(storageKey)
    window.dispatchEvent(new StorageEvent('storage', { key: storageKey, newValue: null, storageArea: localStorage }))
    await flushPromises()
    expect(wrapper.text()).not.toContain(pending.payload.commit_request_id)
    wrapper.unmount()
  })

  it('persists the full fill request before submission and blocks a new plan after an unknown outcome', async () => {
    const signal = {
      id: 7, account_id: 'LOCAL-LIVE', code: '000001', name: '平安银行', side: 'buy',
      planned_shares: 100, filled_shares: 0, execution_status: 'pending', account_version: 1,
      created_at: `${todayStr()} 10:00:00`, planned_price: 10, amount: 1000,
    }
    const originalGet = api.get.getMockImplementation()
    api.get.mockImplementation(async (path, ...args) => {
      if (path === '/rebalance/account/local') {
        return { data: { account_id: 'LOCAL-LIVE', status: 'active', version: 1, integrity: { ok: true, account_version: 1 }, legacy_unknown: {} } }
      }
      if (path === '/rebalance/signals/pending') return { data: [signal] }
      return originalGet(path, ...args)
    })
    api.put.mockRejectedValueOnce(new Error('network lost'))
    const wrapper = mountView()
    await flushPromises()

    await wrapper.findAll('button').find(button => button.text() === '记录成交').trigger('click')
    await wrapper.get('#execution-shares').setValue('100')
    await wrapper.get('#execution-price').setValue('10.000')
    await wrapper.get('#execution-commission').setValue('0.30')
    await wrapper.findAll('button').find(button => button.text() === '确认记录').trigger('click')
    await flushPromises()

    const saved = JSON.parse(localStorage.getItem(fillStorageKey))
    expect(saved).toMatchObject({
      shares: 100, price: '10', commission: '0.3', stamp_duty: '0',
      expected_account_version: 1,
    })
    expect(saved.request_id).toMatch(/^[0-9a-f-]{36}$/i)
    expect(wrapper.text()).toContain('存在尚未确认的成交请求，已暂停新调仓方案')
    expect(wrapper.text()).toContain('成交结果未确认')
    wrapper.unmount()
  })

  it('closes stale holding edits and invalidates the plan after a version conflict', async () => {
    const original = {
      id: 5, version: 2, code: '000001', name: '平安银行', shares: 100,
      cost_price: 10, buy_date: '2026-01-05', risk_threshold: 8,
    }
    holdingStore.holdings.value = [original]
    api.get.mockImplementation(async path => {
      if (path === '/rebalance/account/local') {
        return { data: { account_id: 'LOCAL-LIVE', status: 'uninitialized', version: 2, integrity: { ok: false }, legacy_unknown: {} } }
      }
      if (path === '/screening/saves') return { data: [] }
      if (path === '/rebalance/account/local/reconciliation') return { data: { can_execute: false, reason: '待核对' } }
      if (path === '/rebalance/cash') return { data: { cash: 10000 } }
      if (path === '/rebalance/account/local/broker-trades/unresolved' ||
          path === '/rebalance/account/local/broker-reports/unresolved' ||
          path === '/rebalance/account/local/intents' || path === '/rebalance/signals/pending' ||
          path === '/rebalance/cooldown') return { data: [] }
      if (path === '/rebalance/account/local/quarantine') {
        return { data: { counts: { orders: 0, trades: 0, signals: 0, sessions: 0 }, orders: [], trades: [], signals: [], sessions: [] } }
      }
      return { data: [] }
    })
    const conflict = new Error('持仓已变化，请刷新后重试')
    conflict.status = 409
    api.put.mockRejectedValueOnce(conflict)
    const wrapper = mountView()
    await flushPromises()

    await wrapper.get('[aria-label="编辑000001持仓"]').trigger('click')
    await wrapper.findAll('button').find(button => button.text() === '保存').trigger('click')
    await flushPromises()

    expect(api.put).toHaveBeenCalledWith('/portfolio/holdings/000001', expect.objectContaining({
      expected_id: original.id,
      expected_version: original.version,
    }))
    expect(wrapper.text()).toContain('已刷新持仓与账本，请重新打开编辑后再提交')
    expect(wrapper.text()).not.toContain('保存持仓')
    wrapper.unmount()
  })

  it('shows cross-batch stale plans invalidated by a broker fill', async () => {
    const intentId = 'intent-1'
    let evidenceReads = 0
    api.get.mockImplementation(async path => {
      if (path === '/rebalance/account/local') {
        return { data: { account_id: 'LOCAL-LIVE', broker_account_id: 'QMT-1', status: 'active', version: 3, available: 1000, integrity: { ok: true, account_version: 3 }, legacy_unknown: {} } }
      }
      if (path === '/rebalance/account/local/intents') {
        return { data: [{ intent_id: intentId, code: '000001', side: 'buy', quantity: 100, limit_price: 10, state: 'partial', broker_order_id: 'O-1', created_at: 'now', filled_quantity: 40 }] }
      }
      if (path === `/rebalance/account/local/intents/${intentId}/broker-evidence`) {
        evidenceReads += 1
        return { data: { intent: { intent_id: intentId }, order: { status: 'cancelled', filled_qty: 40, quantity: 100 }, trades: [{ trade_id: 'T-1', symbol: '000001.SZ', side: 'buy', quantity: 40, price: 10, amount: 400, commission: '0.12', stamp_duty: '0', observed_at: 'now', booked: evidenceReads > 1 }], conflicts: 0 } }
      }
      if (path === '/rebalance/account/local/reconciliation') return { data: { can_execute: false, reason: '待核对' } }
      if (path === '/rebalance/cash') return { data: { cash: 1000 } }
      if (path === '/rebalance/account/local/broker-trades/unresolved' ||
          path === '/rebalance/account/local/broker-reports/unresolved' ||
          path === '/rebalance/signals/pending' || path === '/rebalance/cooldown') return { data: [] }
      if (path === '/rebalance/account/local/quarantine') {
        return { data: { counts: { orders: 0, trades: 0, signals: 0, sessions: 0 }, orders: [], trades: [], signals: [], sessions: [] } }
      }
      return { data: [] }
    })
    api.post.mockImplementation(async path => {
      if (path.endsWith('/broker-fill')) return { data: {
        broker_trade_id: 'T-1', already_recorded: false,
        invalidated_signal_ids: [91],
        invalidated_signal_details: [{ signal_id: 91, code: '000002', planned_shares: 300, filled_shares: 100, invalidated_remaining_shares: 200 }],
      } }
      if (path.endsWith('/finalize')) return { data: {
        intent_id: intentId, state: 'cancelled', already_finalized: false,
        invalidated_signal_ids: [92],
        invalidated_signal_details: [{ signal_id: 92, code: '000003', planned_shares: 100, filled_shares: 0, invalidated_remaining_shares: 100 }],
      } }
      return { data: {} }
    })

    const wrapper = mountView()
    await flushPromises()
    expect(api.get.mock.calls.map(([path]) => path)).toContain('/rebalance/account/local/intents')
    expect(wrapper.text()).toContain(intentId)
    await wrapper.findAll('button').find(button => button.text() === '读取券商回报').trigger('click')
    await flushPromises()
    await wrapper.findAll('button').find(button => button.text() === '核对回报并入账').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('000002 #91')
    expect(wrapper.text()).toContain('失效余量 200 股')
    await wrapper.findAll('button').find(button => button.text() === '核验终态并释放剩余预留').trigger('click')
    await flushPromises()
    expect(wrapper.text()).toContain('000003 #92')
    expect(wrapper.text()).toContain('失效余量 100 股')
    wrapper.unmount()
  })

  it('blocks trade booking and explains an unresolved cumulative order report conflict', async () => {
    const intentId = 'intent-terminal-conflict'
    api.get.mockImplementation(async path => {
      if (path === '/rebalance/account/local') {
        return { data: { account_id: 'LOCAL-LIVE', broker_account_id: 'QMT-1', status: 'active', version: 5, available: 1000, integrity: { ok: true, account_version: 5 }, legacy_unknown: {} } }
      }
      if (path === '/rebalance/account/local/intents') {
        return { data: [{ intent_id: intentId, code: '000001', side: 'buy', quantity: 100, limit_price: 10, state: 'cancelled', broker_order_id: 'O-1', created_at: 'now', filled_quantity: 40 }] }
      }
      if (path === `/rebalance/account/local/intents/${intentId}/broker-evidence`) {
        return { data: {
          intent: { intent_id: intentId, state: 'cancelled' },
          order: { status: 'cancelled', filled_qty: 40, quantity: 100 },
          trades: [{ trade_id: 'T-LATE', symbol: '000001.SZ', side: 'buy', quantity: 20, price: 10, amount: 200, commission: '0.06', stamp_duty: '0', observed_at: 'now', booked: false }],
          conflicts: 1,
          trade_conflicts: 0,
          order_conflicts: 1,
        } }
      }
      if (path === '/rebalance/account/local/reconciliation') return { data: { can_execute: false, reason: '待核对' } }
      if (path === '/rebalance/cash') return { data: { cash: 1000 } }
      if (path === '/rebalance/account/local/broker-trades/unresolved' ||
          path === '/rebalance/account/local/broker-reports/unresolved' ||
          path === '/rebalance/signals/pending' || path === '/rebalance/cooldown') return { data: [] }
      if (path === '/rebalance/account/local/quarantine') {
        return { data: { counts: { orders: 0, trades: 0, signals: 0, sessions: 0 }, orders: [], trades: [], signals: [], sessions: [] } }
      }
      return { data: [] }
    })

    const wrapper = mountView()
    await flushPromises()
    await wrapper.findAll('button').find(button => button.text() === '读取券商回报').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('委托状态或累计成交数有 1 条未解决冲突')
    const bookButton = wrapper.findAll('button').find(button => button.text() === '核对回报并入账')
    expect(bookButton).toBeDefined()
    expect(bookButton.element.disabled).toBe(true)
    wrapper.unmount()
  })
})
