/**
 * 持仓数据共享 store — 模块级 ref，跨视图复用
 * 替代 PortfolioView 和 AlertView 中各自独立的 /portfolio/holdings 请求
 */
import { ref } from 'vue'
import api from '../api/index.js'

const holdings = ref([])
const loading = ref(false)
const loadError = ref('')
const lastFetch = ref(0)

const TTL = 30_000
let requestGeneration = 0
let pendingLoad = null

export function useHoldingsStore() {
  async function refreshAfterConflict(error) {
    if (Number(error?.status ?? error?.response?.status) !== 409) return
    try {
      await load(true)
    } catch (refreshError) {
      const combined = new Error(`${error.message || '持仓版本冲突'}；刷新最新持仓失败：${refreshError.message || refreshError}`)
      combined.status = 409
      throw combined
    }
  }

  async function load(force = false) {
    if (pendingLoad && !force) return pendingLoad
    if (!force && lastFetch.value > 0 && Date.now() - lastFetch.value < TTL) {
      return holdings.value
    }
    const generation = ++requestGeneration
    loading.value = true
    const request = (async () => {
      try {
        const r = await api.get('/portfolio/holdings')
        const rows = r?.data ?? r
        if (!Array.isArray(rows)) throw new Error('持仓响应格式无效')
        if (generation !== requestGeneration) return holdings.value
        holdings.value = rows
        lastFetch.value = Date.now()
        loadError.value = ''
        return rows
      } catch (error) {
        if (generation !== requestGeneration) return pendingLoad || holdings.value
        holdings.value = []
        lastFetch.value = 0
        loadError.value = error.message || '读取失败'
        throw error
      } finally {
        if (generation === requestGeneration) {
          loading.value = false
          if (pendingLoad === request) pendingLoad = null
        }
      }
    })()
    pendingLoad = request
    return request
  }

  async function updateThreshold(code, threshold) {
    const holding = holdings.value.find(row => row.code === code)
    if (!holding || !Number.isInteger(Number(holding.id)) || Number(holding.id) < 1 ||
        !Number.isInteger(Number(holding.version)) || Number(holding.version) < 1) {
      throw new Error('持仓快照不可用，请刷新后重试')
    }
    try {
      await api.put(`/portfolio/holdings/${code}`, {
        risk_threshold: threshold,
        expected_id: Number(holding.id),
        expected_version: Number(holding.version),
      })
    } catch (error) {
      await refreshAfterConflict(error)
      throw error
    }
    const idx = holdings.value.findIndex(h => h.code === code)
    if (idx >= 0) holdings.value[idx] = { ...holdings.value[idx], risk_threshold: threshold, version: Number(holding.version) + 1 }
  }

  async function update(code, data) {
    const holding = holdings.value.find(row => row.code === code)
    const expectedId = data.expected_id ?? holding?.id
    const expectedVersion = data.expected_version ?? holding?.version
    if (!Number.isInteger(Number(expectedId)) || Number(expectedId) < 1 ||
        !Number.isInteger(Number(expectedVersion)) || Number(expectedVersion) < 1) {
      throw new Error('持仓快照不可用，请刷新后重试')
    }
    try {
      await api.put('/portfolio/holdings/' + code, {
        ...data,
        expected_id: Number(expectedId),
        expected_version: Number(expectedVersion),
      })
    } catch (error) {
      await refreshAfterConflict(error)
      throw error
    }
    await load(true)
  }

  async function remove(code, adjustment) {
    const holding = holdings.value.find(row => row.code === code)
    const expectedId = adjustment.expected_id ?? holding?.id
    const expectedVersion = adjustment.expected_version ?? holding?.version
    if (!Number.isInteger(Number(expectedId)) || Number(expectedId) < 1 ||
        !Number.isInteger(Number(expectedVersion)) || Number(expectedVersion) < 1) {
      throw new Error('持仓快照不可用，请刷新后重试')
    }
    try {
      await api.delete('/portfolio/holdings/' + code, {
        data: { ...adjustment, expected_id: Number(expectedId), expected_version: Number(expectedVersion) },
      })
    } catch (error) {
      await refreshAfterConflict(error)
      throw error
    }
    await load(true)
  }

  async function add(form) {
    try {
      await api.post('/portfolio/holdings', form)
    } catch (error) {
      await refreshAfterConflict(error)
      throw error
    }
    await load(true)
  }

  function invalidate() {
    lastFetch.value = 0
    requestGeneration += 1
    pendingLoad = null
    loading.value = false
  }

  return { holdings, loading, loadError, load, updateThreshold, update, remove, add, invalidate }
}
