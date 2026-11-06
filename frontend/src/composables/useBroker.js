import { ref, computed } from 'vue'
import api from '../api/index.js'

const LABELS = { qmt: 'QMT 实盘', easytrader: '同花顺实盘', fake: '模拟账户' }

export function useBroker() {
  const cashEnabled = ref(false)
  const brokerType = ref('')
  const brokerError = ref('')
  const brokerLabel = computed(() => LABELS[brokerType.value] || brokerType.value)

  async function loadBroker() {
    brokerError.value = ''
    try {
      const r = await api.get('/trading/broker')
      cashEnabled.value = r.data?.cash_enabled === true && r.data?.connected === true
      brokerType.value = r.data?.type || ''
    } catch (e) {
      cashEnabled.value = false
      brokerError.value = '券商状态加载失败'
    }
  }

  return { cashEnabled, brokerType, brokerLabel, brokerError, loadBroker }
}
