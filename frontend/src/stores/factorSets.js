/**
 * 精选因子集共享 store — 模块级 ref，跨视图复用
 * 替代 FactorManager 和 QlibLab 中各自独立的 /factor/curated-sets 请求
 */
import { ref } from 'vue'
import api from '../api/index.js'

const sets = ref([])
const loading = ref(false)
const lastFetch = ref(0)

const TTL = 60_000

export function useFactorSetsStore() {
  async function load(force = false) {
    if (!force && sets.value.length > 0 && Date.now() - lastFetch.value < TTL) {
      return sets.value
    }
    loading.value = true
    try {
      const r = await api.get('/factor/curated-sets')
      sets.value = r.data?.data || r.data || []
      lastFetch.value = Date.now()
    } finally {
      loading.value = false
    }
    return sets.value
  }

  const _detailCache = {}

  async function getByName(name) {
    if (_detailCache[name] && Date.now() - _detailCache[name].ts < TTL) {
      return _detailCache[name].data
    }
    const r = await api.get(`/factor/curated-sets/${name}`)
    const data = r.data?.data || r.data || {}
    _detailCache[name] = { data, ts: Date.now() }
    return data
  }

  async function remove(name) {
    await api.delete(`/factor/curated-sets/${name}`)
    sets.value = sets.value.filter(s => s.name !== name && s.key !== name)
  }

  function invalidate() {
    lastFetch.value = 0
  }

  return { sets, loading, load, getByName, remove, invalidate }
}
