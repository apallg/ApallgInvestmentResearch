/**
 * 统一异步任务轮询 composable。
 * 替代 QlibLab 中训练/Optuna/滚动/择时/回测各自重复的
 * `for + sleep + getStatus` 轮询循环。
 */
import { ref, onUnmounted } from 'vue'
import { pollSleep } from '../utils/poll'

/**
 * @param {Object} options
 * @param {number} options.intervalMs    轮询间隔
 * @param {number} options.maxAttempts   最大轮询次数
 * @param {Function} options.isDestroyed 组件卸载信号，返回 true 时停止
 */
export function useAsyncJob({ intervalMs = 10000, maxAttempts = 720, isDestroyed } = {}) {
  const status = ref('idle')
  const message = ref('')
  const result = ref(null)
  const error = ref('')
  const attempts = ref(0)

  let disposed = false

  function stop() {
    disposed = true
  }

  /**
   * @param {Function} submitFn   提交任务，返回 { data: { job_id } }
   * @param {Function} statusFn   (jobId) => 返回 { data: { status, result, error, ... } }
   * @param {Object} options
   * @param {string} options.startMessage
   * @param {string} options.completedStatus  默认 'completed'
   * @param {string} options.errorStatus      默认 'error'
   * @param {Function} options.onCompleted    完成后回调，可用于刷新模型列表等
   * @param {Function} options.onError
   */
  async function run(submitFn, statusFn, {
    startMessage = '任务已启动，轮询中...',
    completedStatus = 'completed',
    errorStatus = 'error',
    onCompleted,
    onError,
  } = {}) {
    disposed = false
    status.value = 'running'
    message.value = startMessage
    result.value = null
    error.value = ''
    attempts.value = 0

    try {
      const res = await submitFn()
      const jobId = res?.data?.job_id
      if (!jobId) {
        throw new Error('未返回 job_id')
      }

      for (let i = 0; i < maxAttempts; i++) {
        if (disposed || (isDestroyed && isDestroyed())) {
          status.value = 'disposed'
          return null
        }

        await pollSleep(intervalMs)
        attempts.value = i + 1

        try {
          const job = await statusFn(jobId)
          const data = job?.data ?? job
          if (data?.status === completedStatus) {
            result.value = data.result
            status.value = 'completed'
            message.value = ''
            if (onCompleted) await onCompleted(data.result)
            return data.result
          }
          if (data?.status === errorStatus) {
            status.value = 'error'
            error.value = data.error || data.message || '任务失败'
            message.value = data.error || data.message || '任务失败'
            if (onError) await onError(error.value)
            return null
          }
          message.value = data?.message || data?.error || `轮询中... (第${i + 1}次)`
        } catch (e) {
          // 404 等网络抖动不立即失败，继续轮询，与 QlibLab 现有行为一致
          message.value = '轮询中... (网络抖动，继续等待)'
        }
      }

      status.value = 'timeout'
      error.value = '任务超时'
      message.value = '任务超时'
      return null
    } catch (e) {
      status.value = 'error'
      error.value = e.response?.data?.detail || e.message || String(e)
      message.value = error.value
      return null
    }
  }

  onUnmounted(stop)

  return { status, message, result, error, attempts, run, stop }
}
