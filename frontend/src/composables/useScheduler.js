/**
 * 统一调度器控制 composable
 * 替代 AlertView/SentimentView/SchedulerView 中各自重复的 start/stop/trigger 逻辑
 */
import { ref, onUnmounted } from 'vue'
import api from '../api'

export function useScheduler(type, { pollInterval = 15000 } = {}) {
  const running = ref(false)
  const lastScan = ref(null)
  const scanCount = ref(0)
  const lastError = ref(null)
  const loading = ref(false)
  let timer = null
  let polling = false

  // 调度器类型 → API 子路径映射
  const PATH_MAP = { 'risk-scan': '', 'data-collection': 'collection', 'sentiment': 'sentiment', 'live': 'live', 'smoke-test': 'smoke', 'notice': 'notice' }
  const subPath = PATH_MAP[type] ?? type
  const endpoint = subPath ? `/scheduler/${subPath}` : '/scheduler'

  async function loadStatus() {
    try {
      const res = await api.get(`${endpoint}/status`)
      const d = res?.data ?? res
      running.value = d?.running ?? false
      lastScan.value = d?.last_scan ?? d?.last_collection ?? d?.last_run ?? null
      scanCount.value = d?.scan_count ?? d?.collection_count ?? d?.count ?? 0
      lastError.value = d?.last_error ?? d?.last_result ?? null
    } catch (e) {
      lastError.value = e.status === 404 ? 'API未就绪(需重启后端)' : e.message
    }
  }

  async function start() {
    loading.value = true
    try {
      await api.post(`${endpoint}/start`)
      await loadStatus()
    } finally {
      loading.value = false
    }
  }

  async function stop() {
    loading.value = true
    try {
      await api.post(`${endpoint}/stop`)
      await loadStatus()
    } finally {
      loading.value = false
    }
  }

  async function trigger() {
    loading.value = true
    try {
      await api.post(`${endpoint}/trigger`)
      await loadStatus()
    } finally {
      loading.value = false
    }
  }

  function scheduleNext() {
    if (!polling) return
    timer = setTimeout(async () => {
      await loadStatus()
      if (polling) scheduleNext()
    }, pollInterval)
  }

  function startPolling() {
    polling = true
    loadStatus()
    scheduleNext()
  }

  function stopPolling() {
    polling = false
    if (timer) {
      clearTimeout(timer)
      timer = null
    }
  }

  onUnmounted(() => stopPolling())

  return { running, lastScan, scanCount, lastError, loading, start, stop, trigger, loadStatus, startPolling, stopPolling }
}
