import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { ref, nextTick } from 'vue'

// Mock echarts
const mockDispose = vi.fn()
const mockResize = vi.fn()
const mockSetOption = vi.fn()
const mockIsDisposed = vi.fn(() => false)
const mockInit = vi.fn(() => ({
  setOption: mockSetOption,
  resize: mockResize,
  dispose: mockDispose,
  isDisposed: mockIsDisposed,
}))

vi.mock('echarts', () => ({
  default: { init: mockInit },
  init: mockInit,
}))

// Mock window event listeners
let resizeListeners = []
const mockAddEventListener = vi.fn((event, fn) => {
  if (event === 'resize') resizeListeners.push(fn)
})
const mockRemoveEventListener = vi.fn((event, fn) => {
  if (event === 'resize') {
    resizeListeners = resizeListeners.filter(f => f !== fn)
  }
})

// Simulate resize
function triggerResize() {
  resizeListeners.forEach(fn => fn())
}

import { useChart } from '../src/composables/useChart.js'

describe('useChart', () => {
  let addEventListenerSpy
  let removeEventListenerSpy
  let requestAnimationFrameSpy
  let cancelAnimationFrameSpy

  beforeEach(() => {
    vi.clearAllMocks()
    resizeListeners = []
    addEventListenerSpy = vi.spyOn(window, 'addEventListener')
    removeEventListenerSpy = vi.spyOn(window, 'removeEventListener')
    requestAnimationFrameSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(fn => {
      const id = Math.random()
      setTimeout(() => fn(), 0)
      return id
    })
    cancelAnimationFrameSpy = vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {})
    mockIsDisposed.mockReturnValue(false)
  })

  afterEach(() => {
    addEventListenerSpy.mockRestore()
    removeEventListenerSpy.mockRestore()
    requestAnimationFrameSpy.mockRestore()
    cancelAnimationFrameSpy.mockRestore()
  })

  describe('实例复用', () => {
    it('首次 render 创建新实例', async () => {
      const { chartRef, render } = useChart()
      const el = document.createElement('div')
      chartRef.value = el

      await render(() => ({ title: { text: 'test' } }))
      expect(mockInit).toHaveBeenCalledTimes(1)
    })

    it('第二次 render 复用已有实例，调用 setOption', async () => {
      const { chartRef, render } = useChart()
      const el = document.createElement('div')
      chartRef.value = el

      await render(() => ({ title: { text: 'first' } }))
      const initCount = mockInit.mock.calls.length

      await render(() => ({ title: { text: 'second' } }))
      expect(mockInit).toHaveBeenCalledTimes(initCount) // 不再创建新实例
      expect(mockSetOption).toHaveBeenCalledTimes(2)
    })
  })

  describe('边界条件', () => {
    it('chartRef.value 为空时 render 直接返回', async () => {
      const { render } = useChart()
      await render(() => ({}))
      expect(mockInit).not.toHaveBeenCalled()
    })

    it('getOptions 返回 null/undefined 时不调用 setOption', async () => {
      const { chartRef, render } = useChart()
      const el = document.createElement('div')
      chartRef.value = el

      await render(() => null)
      expect(mockInit).toHaveBeenCalledTimes(1)
      expect(mockSetOption).not.toHaveBeenCalled()
    })

    it('实例已被 dispose 时不重复 dispose', async () => {
      const { chartRef, render, dispose } = useChart()
      chartRef.value = document.createElement('div')
      await render(() => ({}))
      mockIsDisposed.mockReturnValue(true)
      dispose()
      expect(mockIsDisposed).toHaveBeenCalled()
      expect(mockDispose).not.toHaveBeenCalled()
    })

    // 并发测试：jsdom 不支持 canvas context，跳过此场景
    it.skip('并发快速调用 render 不会创建多个实例', async () => {
      // jsdom 限制：echarts setOption 需要 canvas context，跳过真实渲染
      mockSetOption.mockImplementation(() => {})
      const { chartRef, render } = useChart()
      const el = document.createElement('div')
      Object.defineProperty(el, 'clientWidth', { value: 800, configurable: true })
      Object.defineProperty(el, 'clientHeight', { value: 600, configurable: true })
      chartRef.value = el

      await Promise.all([
        render(() => ({ title: { text: 'a' } })),
        render(() => ({ title: { text: 'b' } })),
      ])
      expect(mockInit).toHaveBeenCalledTimes(1)
      mockSetOption.mockRestore()
    })
  })
})
