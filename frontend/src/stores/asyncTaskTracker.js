/**
 * 全局异步任务追踪器
 * 模块级 reactive，不绑定任何组件，KeepAlive 被清除后仍可恢复任务状态
 */
import { reactive, computed } from 'vue'

const tasks = reactive({})
const STORAGE_KEY = 'apallg.asyncTaskTracker'

try {
  const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
  Object.assign(tasks, saved)
} catch (e) { /* 浏览器存储不可用时保持内存追踪 */ }

function persist() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(tasks)) } catch (e) { /* ignore */ }
}

export function useAsyncTaskTracker() {
  function register(page, jobId, meta = {}) {
    tasks[page] = { jobId, status: 'running', startedAt: Date.now(), ...meta }
    persist()
  }

  function update(page, updates) {
    if (tasks[page]) { Object.assign(tasks[page], updates); persist() }
  }

  function unregister(page) {
    delete tasks[page]
    persist()
  }

  function getTask(page) {
    return tasks[page] || null
  }

  function hasRunningTask(page) {
    return tasks[page]?.status === 'running'
  }

  const runningCount = computed(() =>
    Object.values(tasks).filter(t => t.status === 'running').length
  )

  return { tasks, register, update, unregister, getTask, hasRunningTask, runningCount }
}
