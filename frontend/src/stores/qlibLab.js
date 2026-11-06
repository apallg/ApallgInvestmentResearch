/**
 * QlibLab 共享状态 store。
 * 目标：把 QlibLab.vue data() 中跨 Tab 共享的状态逐步迁到这里，
 * 让壳组件和各个 Tab 子组件只负责自己的 UI 状态。
 */
import { ref, computed } from 'vue'
import api from '../api'
import qlibApi from '../api/qlib'

const dataStatus = ref({ synced: false })
const availableModels = ref([])
const trainedModels = ref([])
const loadingModels = ref(false)
const schedulerStatuses = ref({})
const poolFactors = ref([])
const curatedSets = ref([])
const optunaParamsAvailable = ref('')

const bestIC = computed(() => {
  const models = trainedModels.value.filter(m => m.ic !== null && !Number.isNaN(m.ic))
  if (!models.length) return null
  return Math.max(...models.map(m => m.ic))
})

const bestICIR = computed(() => {
  const models = trainedModels.value.filter(m => m.icir !== null && !Number.isNaN(m.icir))
  if (!models.length) return null
  return Math.max(...models.map(m => m.icir))
})

const dualCount = computed(() => trainedModels.value.filter(m => m.is_dual).length)
const singleCount = computed(() => trainedModels.value.filter(m => !m.is_dual).length)
const predictableCount = computed(() => trainedModels.value.filter(m => m.has_predictions).length)

export function useQlibLabStore() {
  async function loadDataStatus() {
    try {
      const r = await qlibApi.getDataStatus()
      if (r?.data) dataStatus.value = r.data
    } catch (e) {
      dataStatus.value.synced = false
    }
  }

  async function loadModels() {
    try {
      const r = await qlibApi.getModels()
      if (r?.data) availableModels.value = r.data
    } catch (e) {
      availableModels.value = []
    }
  }

  async function loadTrainedModels() {
    loadingModels.value = true
    try {
      const r = await qlibApi.getTrainedModels()
      if (r?.data) trainedModels.value = r.data
    } catch (e) {
      trainedModels.value = []
    }
    loadingModels.value = false
  }

  async function loadSchedulerStatuses() {
    try {
      const r = await qlibApi.getSchedulerStatus('collection/')
      schedulerStatuses.value = { collection: r?.data || {} }
    } catch (e) {
      schedulerStatuses.value = {}
    }
  }

  async function loadOptunaParams() {
    try {
      const r = await api.get('/qlib/optuna-params')
      const sets = r?.data || []
      if (sets.length) {
        optunaParamsAvailable.value = sets.map(s => s.factor_set).join(', ')
      }
    } catch (e) {
      // 保留原 QlibLab 的静默失败行为
    }
  }

  return {
    dataStatus,
    availableModels,
    trainedModels,
    loadingModels,
    schedulerStatuses,
    poolFactors,
    curatedSets,
    optunaParamsAvailable,
    bestIC,
    bestICIR,
    dualCount,
    singleCount,
    predictableCount,
    loadDataStatus,
    loadModels,
    loadTrainedModels,
    loadSchedulerStatuses,
    loadOptunaParams,
  }
}
