import { ref, onUnmounted, nextTick } from 'vue'

export function useChart(initOptions) {
  const chartRef = ref(null)
  let instance = null
  let initPromise = null
  let resizeBound = false
  let resizeRaf = null

  function onResize() {
    if (resizeRaf) return
    resizeRaf = requestAnimationFrame(() => {
      resizeRaf = null
      if (instance && !instance.isDisposed()) instance.resize()
    })
  }

  async function render(getOptions) {
    await nextTick()
    const el = chartRef.value
    if (!el) return

    if (!instance) {
      if (!initPromise) {
        initPromise = import('echarts').then(echarts => {
          const target = chartRef.value
          if (!target) return null
          const existing = 'getInstanceByDom' in echarts ? echarts.getInstanceByDom(target) : null
          instance = existing || echarts.init(target)
          if (!resizeBound) {
            window.addEventListener('resize', onResize)
            resizeBound = true
          }
          return instance
        })
      }
      await initPromise
    }

    if (!instance || instance.isDisposed()) return
    const opts = getOptions()
    if (opts) instance.setOption(opts, { notMerge: true })
  }

  function dispose() {
    if (resizeRaf) {
      cancelAnimationFrame(resizeRaf)
      resizeRaf = null
    }
    if (instance) {
      if (!instance.isDisposed()) instance.dispose()
      instance = null
    }
    initPromise = null
    if (resizeBound) {
      window.removeEventListener('resize', onResize)
      resizeBound = false
    }
  }

  onUnmounted(dispose)

  return { chartRef, render, dispose }
}
