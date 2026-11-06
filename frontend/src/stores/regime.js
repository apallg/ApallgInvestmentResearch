/**
 * 市场状态 + 宏观数据共享 store — 模块级 ref，跨视图复用
 * 替代 MarketView 和 StockScreening 中各自独立的 /regime/current + /macro/overview 请求
 */
import { ref } from 'vue'
import api from '../api/index.js'

const regime = ref(null)
const macro = ref(null)
const loading = ref(false)
const lastFetch = ref(0)

const TTL = 60_000

export function useRegimeStore() {
  async function load(force = false) {
    if (!force && regime.value && Date.now() - lastFetch.value < TTL) {
      return { regime: regime.value, macro: macro.value }
    }
    loading.value = true
    try {
      const [rr, mr] = await Promise.all([
        api.get('/regime/current').catch(() => null),
        api.get('/macro/overview').catch(() => null),
      ])
      if (rr !== null && mr !== null) {
        regime.value = rr?.data ?? rr
        macro.value = mr?.data ?? mr
        lastFetch.value = Date.now()
      } else {
        regime.value = rr?.data ?? rr ?? regime.value
        macro.value = mr?.data ?? mr ?? macro.value
      }
    } finally {
      loading.value = false
    }
    return { regime: regime.value, macro: macro.value }
  }

  function invalidate() {
    lastFetch.value = 0
  }

  return { regime, macro, loading, load, invalidate }
}
