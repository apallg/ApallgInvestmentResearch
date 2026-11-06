/**
 * 轮询睡眠 — 页面在后台 (document.hidden) 时拉长间隔。
 * App.vue 用 keep-alive 包裹路由视图, 组件离页不触发 onUnmounted, 无界训练轮询
 * (FactorManager/QlibLab) 会持续打后端。此函数把后台轮询降到 60s 一次,
 * 既减少无效请求又不丢失任务完成状态。
 */
export async function pollSleep(ms) {
  const actual = document.hidden ? Math.min(ms * 6, 60000) : ms
  await new Promise((resolve) => setTimeout(resolve, actual))
}
