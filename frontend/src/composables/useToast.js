import { ref } from 'vue'
const _instance = ref(null)
export function setToastRef(ref) { _instance.value = ref }
export function useToast() {
  return {
    success(msg) { _instance.value?.show(msg, 'success') },
    warning(msg) { _instance.value?.show(msg, 'warning', 5000) },
    error(msg)   { _instance.value?.show(msg, 'error', 8000) },
    info(msg)    { _instance.value?.show(msg, 'info') },
  }
}
