<template>
  <div class="mi-root">
    <div class="mi-hero">
      <div class="mi-hero-left">
        <div class="mi-hero-label">上证指数</div>
        <div class="mi-hero-price">{{ fmtPrice(sse.price) }}</div>
        <div class="mi-hero-change">
          <span :class="['mi-hero-delta', sse.pct_chg == null ? '' : sse.pct_chg >= 0 ? 'up' : 'down']">{{ fmtChange(sse.change) }}</span>
          <span :class="['mi-hero-pct', sse.pct_chg == null ? '' : sse.pct_chg >= 0 ? 'up' : 'down']">{{ fmtDelta(sse.pct_chg) }}</span>
        </div>
      </div>
      <div class="mi-hero-ohlc">
        <div class="mi-ohlc-item"><span class="mi-ohlc-label">开盘</span><span class="mi-ohlc-val">{{ fmtPrice(sse.open) }}</span></div>
        <div class="mi-ohlc-item"><span class="mi-ohlc-label">最高</span><span class="mi-ohlc-val up">{{ fmtPrice(sse.high) }}</span></div>
        <div class="mi-ohlc-item"><span class="mi-ohlc-label">最低</span><span class="mi-ohlc-val down">{{ fmtPrice(sse.low) }}</span></div>
        <div class="mi-ohlc-item"><span class="mi-ohlc-label">昨收</span><span class="mi-ohlc-val">{{ fmtPrice(sse.prev_close) }}</span></div>
        <div class="mi-ohlc-item"><span class="mi-ohlc-label">成交</span><span class="mi-ohlc-val">{{ fmtAmt(sse.amount) }}</span></div>
      </div>
    </div>

    <div class="mi-others">
      <div v-for="(label, code) in otherIndices" :key="code" class="mi-other">
        <span class="mi-other-name">{{ label }}</span>
        <span class="mi-other-price">{{ fmtPrice((indices[code] || {}).price) }}</span>
        <span :class="['mi-other-chg', (indices[code] || {}).pct_chg == null ? '' : (indices[code] || {}).pct_chg >= 0 ? 'up' : 'down']">
          {{ fmtDelta((indices[code] || {}).pct_chg) }}
        </span>
      </div>
    </div>

    <div class="card">
      <div class="card-title">上证指数 K线 (近120日)</div>
      <div ref="chartRef" class="mi-chart"></div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import { useChart } from '../../composables/useChart.js'
import { cssVar, fmtPrice, fmtChange, fmtDelta, fmtAmt } from '../../utils/marketFormat.js'

const props = defineProps({
  indices: { type: Object, default: () => ({}) },
  sseChartData: { type: Array, default: () => [] },
})

const otherIndices = { '399001': '深证成指', '399006': '创业板指', '000688': '科创50' }

const sse = computed(() => {
  const idx = props.indices['000001'] || {}
  const lastBar = props.sseChartData.length ? props.sseChartData[props.sseChartData.length - 1] : {}
  const previousBar = props.sseChartData.length > 1 ? props.sseChartData[props.sseChartData.length - 2] : {}
  return {
    price: idx.price,
    change: idx.change,
    pct_chg: idx.pct_chg,
    open: idx.open ?? lastBar.open ?? null,
    high: idx.high ?? lastBar.high ?? null,
    low: idx.low ?? lastBar.low ?? null,
    prev_close: idx.prev_close ?? previousBar.close ?? lastBar.close ?? null,
    amount: idx.amount ?? lastBar.amount ?? null,
  }
})

const { chartRef, render } = useChart()

watch(() => props.sseChartData, async (data) => {
  if (!data?.length) return
  await nextTick()
  const upC = cssVar('--up')
  const downC = cssVar('--down')
  const primaryC = cssVar('--primary')
  const warnC = cssVar('--warning')
  const dates = data.map(d => String(d.date || '').slice(5, 10))
  await render(() => ({
    tooltip: { trigger: 'axis' },
    grid: { left: '3%', right: '3%', bottom: '3%', containLabel: true },
    xAxis: { type: 'category', data: dates, axisLabel: { fontSize: 10 } },
    yAxis: { type: 'value', scale: true, axisLabel: { fontSize: 10 } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 20 }],
    series: [
      { type: 'candlestick', name: '上证', data: data.map(d => [d.open, d.close, d.low, d.high]),
        itemStyle: { color: upC, color0: downC, borderColor: upC, borderColor0: downC } },
      { type: 'line', name: 'MA5', data: ma(data.map(d => d.close), 5), smooth: true, lineStyle: { color: warnC, width: 1 }, symbol: 'none' },
      { type: 'line', name: 'MA20', data: ma(data.map(d => d.close), 20), smooth: true, lineStyle: { color: primaryC, width: 1 }, symbol: 'none' },
    ],
  }))
}, { immediate: true })

function ma(values, n) {
  const result = []
  for (let i = 0; i < values.length; i++) {
    if (i < n - 1) { result.push(null); continue }
    let sum = 0
    for (let j = i - n + 1; j <= i; j++) sum += values[j]
    result.push(sum / n)
  }
  return result
}
</script>

<style scoped>
.mi-root { display: flex; flex-direction: column; gap: 14px; }

.mi-hero { display: flex; justify-content: space-between; align-items: flex-start; gap: 24px; background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px 24px; }
.mi-hero-label { font-size: var(--fs-sm); color: var(--text-muted); letter-spacing: .6px; margin-bottom: 4px; }
.mi-hero-price { font-family: var(--font-data); font-size: var(--fs-3xl); font-weight: 700; color: var(--text); margin-bottom: 4px; }
.mi-hero-change { display: flex; align-items: baseline; gap: 10px; }
.mi-hero-delta { font-family: var(--font-data); font-size: var(--fs-xl); font-weight: 600; }
.mi-hero-pct { font-family: var(--font-data); font-size: var(--fs-lg); opacity: .85; }
.mi-hero-delta.up, .mi-hero-pct.up { color: var(--up); }
.mi-hero-delta.down, .mi-hero-pct.down { color: var(--down); }

.mi-hero-ohlc { display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px 16px; text-align: right; }
.mi-ohlc-item { display: flex; flex-direction: column; align-items: flex-end; gap: 1px; }
.mi-ohlc-label { font-size: var(--fs-caption); color: var(--text-muted); }
.mi-ohlc-val { font-family: var(--font-data); font-size: var(--fs-sm); color: var(--text-secondary); font-weight: 500; }
.mi-ohlc-val.up { color: var(--up); }
.mi-ohlc-val.down { color: var(--down); }

.mi-others { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.mi-other { display: flex; align-items: baseline; gap: 10px; padding: 12px 16px; background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius); }
.mi-other-name { font-size: var(--fs-sm); color: var(--text-muted); }
.mi-other-price { font-family: var(--font-data); font-size: var(--fs-base); font-weight: 600; color: var(--text); }
.mi-other-chg { font-size: var(--fs-sm); font-weight: 600; }
.mi-other-chg.up { color: var(--up); }
.mi-other-chg.down { color: var(--down); }

.mi-chart { height: 420px; }
</style>
