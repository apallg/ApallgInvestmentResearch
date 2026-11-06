import { beforeEach, describe, expect, it, vi } from 'vitest'

const api = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() }))

vi.mock('../src/api/index.js', () => ({ default: api }))

import { useHoldingsStore } from '../src/stores/holdings.js'

const original = {
  id: 5, version: 2, code: '000001', name: '平安银行', shares: 100,
  cost_price: 10, buy_date: '2026-01-05', risk_threshold: 8,
}
const current = { ...original, version: 3, shares: 200, cost_price: 11 }

describe('holdings optimistic conflict refresh', () => {
  beforeEach(() => {
    api.get.mockReset()
    api.post.mockReset()
    api.put.mockReset()
    api.delete.mockReset()
    api.get.mockResolvedValue({ data: [current] })
    const store = useHoldingsStore()
    store.holdings.value = [original]
    store.loadError.value = ''
    store.invalidate()
  })

  it.each([
    ['update threshold', store => store.updateThreshold(original.code, 9)],
    ['edit', store => store.update(original.code, {
      name: original.name, expected_id: original.id, expected_version: original.version,
    })],
    ['delete', store => store.remove(original.code, {
      reason: '管理员核对后的快照修正', expected_id: original.id,
      expected_version: original.version, expected_shares: original.shares,
      expected_cost_price: original.cost_price, expected_buy_date: original.buy_date,
    })],
    ['duplicate create', store => store.add({ ...original })],
  ])('%s reloads current holdings and preserves the conflict error', async (label, mutate) => {
    const conflict = new Error('持仓已变化，请刷新后重试')
    conflict.status = 409
    api.put.mockRejectedValue(conflict)
    api.delete.mockRejectedValue(conflict)
    api.post.mockRejectedValue(conflict)
    const store = useHoldingsStore()

    await expect(mutate(store)).rejects.toBe(conflict)

    expect(api.get).toHaveBeenCalledWith('/portfolio/holdings')
    expect(store.holdings.value).toEqual([current])
  })

  it('reports both the write conflict and failed refresh', async () => {
    const conflict = new Error('持仓版本已过期')
    conflict.status = 409
    api.put.mockRejectedValue(conflict)
    api.get.mockRejectedValueOnce(new Error('服务不可用'))
    const store = useHoldingsStore()

    await expect(store.update(original.code, {
      name: original.name, expected_id: original.id, expected_version: original.version,
    })).rejects.toThrow('持仓版本已过期；刷新最新持仓失败：服务不可用')
  })

  it('clears the previous snapshot and exposes an error when refresh fails, then recovers', async () => {
    const store = useHoldingsStore()
    api.get.mockRejectedValueOnce(new Error('服务不可用')).mockResolvedValueOnce({ data: [current] })

    await expect(store.load(true)).rejects.toThrow('服务不可用')

    expect(store.holdings.value).toEqual([])
    expect(store.loadError.value).toBe('服务不可用')
    expect(store.loading.value).toBe(false)

    await expect(store.load(true)).resolves.toEqual([current])
    expect(store.holdings.value).toEqual([current])
    expect(store.loadError.value).toBe('')
  })

  it('treats malformed holding responses as unavailable instead of empty', async () => {
    const store = useHoldingsStore()
    api.get.mockResolvedValueOnce({ data: null })

    await expect(store.load(true)).rejects.toThrow('持仓响应格式无效')

    expect(store.holdings.value).toEqual([])
    expect(store.loadError.value).toBe('持仓响应格式无效')
  })

  it('does not let an older forced refresh overwrite a newer snapshot', async () => {
    const store = useHoldingsStore()
    let resolveOlder
    api.get
      .mockImplementationOnce(() => new Promise(resolve => { resolveOlder = resolve }))
      .mockResolvedValueOnce({ data: [current] })

    const olderRequest = store.load(true)
    const newerRequest = store.load(true)
    await expect(newerRequest).resolves.toEqual([current])
    resolveOlder({ data: [original] })
    await expect(olderRequest).resolves.toEqual([current])

    expect(store.holdings.value).toEqual([current])
    expect(store.loading.value).toBe(false)
  })

  it('ignores an older failed refresh after a newer refresh succeeds', async () => {
    const store = useHoldingsStore()
    let rejectOlder
    api.get
      .mockImplementationOnce(() => new Promise((resolve, reject) => { rejectOlder = reject }))
      .mockResolvedValueOnce({ data: [current] })

    const olderRequest = store.load(true)
    const newerRequest = store.load(true)
    await expect(newerRequest).resolves.toEqual([current])
    rejectOlder(new Error('迟到的旧请求失败'))
    await expect(olderRequest).resolves.toEqual([current])

    expect(store.holdings.value).toEqual([current])
    expect(store.loadError.value).toBe('')
    expect(store.loading.value).toBe(false)
  })
})
