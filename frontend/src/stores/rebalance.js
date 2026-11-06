import { ref, computed } from 'vue'
import * as rebalanceApi from '../api/rebalance.js'

const pendingSignals = ref([])
const lastResult = ref(null)
const loading = ref(false)
const error = ref(null)

export function useRebalanceStore() {
  async function loadPendingSignals(checkType = null) {
    loading.value = true
    error.value = null
    try {
      const res = await rebalanceApi.getPendingSignals(checkType)
      pendingSignals.value = res?.data || res || []
    } catch (e) {
      error.value = e.message || '加载失败'
    } finally {
      loading.value = false
    }
  }

  function isExecuted(signalId) {
    const s = pendingSignals.value.find(x => x.id === signalId)
    return s?.execution_status === 'executed'
  }

  function isSkipped(signalId) {
    const s = pendingSignals.value.find(x => x.id === signalId)
    return s?.execution_status === 'skipped'
  }

  async function markExecuted(signalId, detail = {}) {
    try {
      await rebalanceApi.executeSignal(signalId, detail)
      const s = pendingSignals.value.find(x => x.id === signalId)
      if (s) {
        s.execution_status = 'executed'
        s.executed_detail = detail
      }
      return true
    } catch (e) {
      error.value = e.message || '执行失败'
      return false
    }
  }

  async function markSkipped(signalId, reason = '') {
    try {
      await rebalanceApi.skipSignal(signalId, reason)
      const s = pendingSignals.value.find(x => x.id === signalId)
      if (s) s.execution_status = 'skipped'
      return true
    } catch (e) {
      error.value = e.message || '操作失败'
      return false
    }
  }

  async function batchExecute(signalIds, detail = {}) {
    try {
      await rebalanceApi.batchExecuteSignals(signalIds, detail)
      for (const id of signalIds) {
        const s = pendingSignals.value.find(x => x.id === id)
        if (s) {
          s.execution_status = 'executed'
          s.executed_detail = detail
        }
      }
      return true
    } catch (e) {
      error.value = e.message || '批量执行失败'
      return false
    }
  }

  const urgentSignals = computed(() =>
    pendingSignals.value.filter(s =>
      s.execution_status === 'pending' &&
      (s.urgency === 'stop_loss' || s.urgency === 'take_profit')
    )
  )

  const normalSignals = computed(() =>
    pendingSignals.value.filter(s =>
      s.execution_status === 'pending' && s.urgency === 'normal'
    )
  )

  const pendingCount = computed(() =>
    pendingSignals.value.filter(s => s.execution_status === 'pending').length
  )

  function clear() {
    pendingSignals.value = []
    lastResult.value = null
    error.value = null
  }

  return {
    pendingSignals,
    lastResult,
    loading,
    error,
    loadPendingSignals,
    isExecuted,
    isSkipped,
    markExecuted,
    markSkipped,
    batchExecute,
    urgentSignals,
    normalSignals,
    pendingCount,
    clear,
  }
}
