<template>
  <div class="mo-root">
    <!-- ROW 1: 上证 Hero + 宏观摘要 -->
    <div class="mo-hero-row">
      <div class="mo-hero">
        <div class="mo-hero-top">
          <div>
            <div class="mo-hero-label">上证指数</div>
            <div class="mo-hero-price">{{ fmtPrice(sse.price) }}</div>
            <div class="mo-hero-change">
              <span :class="['mo-hero-delta', sse.pct_chg == null ? '' : sse.pct_chg >= 0 ? 'up' : 'down']">{{ fmtChange(sse.change) }}</span>
              <span :class="['mo-hero-pct', sse.pct_chg == null ? '' : sse.pct_chg >= 0 ? 'up' : 'down']">{{ fmtDelta(sse.pct_chg) }}</span>
            </div>
          </div>
          <div class="mo-hero-ohlc">
            <div class="mo-ohlc-item"><span class="mo-ohlc-label">开盘</span><span class="mo-ohlc-val">{{ fmtPrice(sse.open) }}</span></div>
            <div class="mo-ohlc-item"><span class="mo-ohlc-label">最高</span><span class="mo-ohlc-val up">{{ fmtPrice(sse.high) }}</span></div>
            <div class="mo-ohlc-item"><span class="mo-ohlc-label">最低</span><span class="mo-ohlc-val down">{{ fmtPrice(sse.low) }}</span></div>
            <div class="mo-ohlc-item"><span class="mo-ohlc-label">昨收</span><span class="mo-ohlc-val">{{ fmtPrice(sse.prev_close) }}</span></div>
            <div class="mo-ohlc-item"><span class="mo-ohlc-label">成交</span><span class="mo-ohlc-val">{{ fmtAmt(sse.amount) }}</span></div>
          </div>
        </div>
        <div ref="sparkRef" class="mo-spark"></div>
        <button class="mo-more" @click="$emit('switch-tab', 'indices')">指数详情 →</button>
      </div>

      <div class="card mo-macro">
        <div class="card-title flex-between">
          <span>宏观环境</span>
          <button class="btn-outline btn-sm" @click="$emit('switch-tab', 'macro')">详情</button>
        </div>
        <div v-if="macro" class="mo-macro-grid">
          <div class="mo-macro-item" v-for="item in macroItems" :key="item.label">
            <span class="mo-macro-label">{{ item.label }}</span>
            <span :class="['mo-macro-val', item.trend]">{{ item.value }}</span>
          </div>
          <div class="mo-risk-bar" v-if="macro.risk_score != null" :class="'risk-' + macro.risk_level">
            <div class="mo-risk-track"><div class="mo-risk-fill" :style="{ width: (macro.risk_score * 100) + '%' }"></div></div>
            <div class="mo-risk-text">{{ macro.risk_level === 'danger' ? '⚠ 危险' : macro.risk_level === 'warning' ? '⚡ 关注' : '✓ 正常' }} ({{ (macro.risk_score * 100).toFixed(0) }})</div>
          </div>
        </div>
        <div v-else class="empty">暂无宏观数据</div>
      </div>
    </div>

    <!-- ROW 2: 指数条 + 情绪摘要 -->
    <div class="mo-index-row">
      <div v-for="(label, code) in otherIndices" :key="code" class="mo-index-chip">
        <span class="mo-chip-name">{{ label }}</span>
        <span class="mo-chip-price">{{ fmtPrice((indices[code] || {}).price) }}</span>
        <span :class="['mo-chip-chg', (indices[code] || {}).pct_chg == null ? '' : (indices[code] || {}).pct_chg >= 0 ? 'up' : 'down']">{{ fmtDelta((indices[code] || {}).pct_chg) }}</span>
      </div>
      <span class="mo-chip-sep"></span>
      <div class="mo-index-chip mo-chip-sent">
        <span class="mo-chip-name">情绪指数</span>
        <span class="mo-chip-price" :style="{ color: scoreInfo.color }">{{ scoreInfo.text }}</span>
        <span class="mo-chip-sub">{{ posPct }}% / {{ negPct }}% / {{ sentOverview.today?.total_news || 0 }}条</span>
        <button class="mo-more-inline" @click="$emit('switch-tab', 'sentiment')">详情 →</button>
      </div>
    </div>

    <!-- ROW 3: 板块热力图 + 散户资金 -->
    <div class="mo-row-2">
      <div class="card">
        <div class="card-title flex-between">
          <span>板块热力图</span>
          <button class="btn-outline btn-sm" @click="$emit('switch-tab', 'sectors')">详情</button>
        </div>
        <div ref="heatmapRef" class="mo-chart"></div>
      </div>
      <RetailHeatPanel />
    </div>

    <!-- ROW 4: 涨跌榜摘要 -->
    <div class="mo-row-3">
      <DataTable title="涨幅榜 TOP5" :rows="gainers.slice(0, 5)" :columns="columns" empty-text="暂无数据" />
      <DataTable title="跌幅榜 TOP5" :rows="losers.slice(0, 5)" :columns="columns" empty-text="暂无数据" />
      <div class="card mo-breadth-more">
        <div class="card-title">涨跌榜</div>
        <p class="mo-more-hint">查看完整 TOP20 涨幅榜与跌幅榜，以及市场涨跌家数分布。</p>
        <button class="btn btn-sm" @click="$emit('switch-tab', 'breadth')">查看涨跌榜 →</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import DataTable from '../DataTable.vue'
import RetailHeatPanel from '../RetailHeatPanel.vue'
import { useChart } from '../../composables/useChart.js'
import { cssVar, fmtPrice, fmtChange, fmtDelta, fmtAmt, fmtM } from '../../utils/marketFormat.js'

const props = defineProps({
  indices: { type: Object, default: () => ({}) },
  sseChartData: { type: Array, default: () => [] },
  macro: { type: Object, default: null },
  sectors: { type: Array, default: () => [] },
  sentOverview: { type: Object, default: () => ({}) },
  gainers: { type: Array, default: () => [] },
  losers: { type: Array, default: () => [] },
})
defineEmits(['switch-tab'])

const otherIndices = { '399001': '深证成指', '399006': '创业板指', '000688': '科创50' }

const sse = computed(() => {
  const idx = props.indices['000001'] || {}
  const lastBar = props.sseChartData.length ? props.sseChartData[props.sseChartData.length - 1] : {}
  const previousBar = props.sseChartData.length > 1 ? props.sseChartData[props.sseChartData.length - 2] : {}
  return {
    price: idx.price, change: idx.change, pct_chg: idx.pct_chg,
    open: idx.open ?? lastBar.open ?? null,
    high: idx.high ?? lastBar.high ?? null,
    low: idx.low ?? lastBar.low ?? null,
    prev_close: idx.prev_close ?? previousBar.close ?? lastBar.close ?? null,
    amount: idx.amount ?? lastBar.amount ?? null,
  }
})

const macroItems = computed(() => {
  if (!props.macro?.indicators) return []
  const keys = ['制造业PMI', 'CPI同比', 'PPI同比', 'Shibor隔夜', 'Shibor3月', 'LPR5年', 'M1-M2剪刀差']
  return keys.filter(k => props.macro.indicators[k]).map(k => ({
    label: k,
    value: fmtM(props.macro.indicators[k].value) + (k.includes('PMI') || k.includes('M1') ? '' : '%'),
    trend: props.macro.indicators[k].trend || '',
  }))
})

const scoreInfo = computed(() => {
  const v = props.sentOverview?.today?.avg_score
  const num = Number(v || 0)
  return { text: v != null ? num.toFixed(4) : '--', color: num > 0 ? 'var(--up)' : num < -0.05 ? 'var(--down)' : '' }
})
const posPct = computed(() => ((props.sentOverview?.today?.pos_ratio || 0) * 100).toFixed(1))
const negPct = computed(() => ((props.sentOverview?.today?.neg_ratio || 0) * 100).toFixed(1))

const columns = [
  { key: 'code', label: '代码' },
  { key: 'name', label: '名称' },
  { key: 'price', label: '最新价', format: v => Number(v || 0).toFixed(2) },
  { key: 'pct_chg', label: '涨跌幅', format: v => v == null || !Number.isFinite(Number(v)) ? '--' : `${Number(v) > 0 ? '+' : ''}${Number(v).toFixed(2)}%`,
    tdClass: row => row.pct_chg == null ? '' : row.pct_chg > 0 ? 'profit-up' : row.pct_chg < 0 ? 'profit-down' : '' },
]

const { chartRef: sparkRef, render: renderSpark } = useChart()
const { chartRef: heatmapRef, render: renderHeatmap } = useChart()

watch(() => props.sseChartData, async (data) => {
  if (!data?.length) return
  await nextTick()
  const upC = cssVar('--up')
  const downC = cssVar('--down')
  const closes = data.map(d => d.close)
  const isUp = (closes[closes.length - 1] || 0) >= (closes[0] || 0)
  const lineColor = isUp ? upC : downC
  const dates = data.map(d => String(d.date || '').slice(5, 10))
  await renderSpark(() => ({
    grid: { top: 4, right: 0, bottom: 0, left: 0 },
    xAxis: { type: 'category', show: false, data: dates },
    yAxis: { type: 'value', show: false, scale: true },
    series: [{ type: 'line', data: closes, smooth: true, symbol: 'none', lineStyle: { color: lineColor, width: 1.5 }, areaStyle: { color: lineColor + '18' } }],
  }))
}, { immediate: true })

watch(() => props.sectors, async (sectors) => {
  if (!sectors?.length) return
  await nextTick()
  const maxPct = Math.max(...sectors.slice(0, 30).map(s => Math.abs(sectorPct(s) ?? 0)), 0.01)
  const treeData = sectors.slice(0, 30).map(s => {
    const pct = sectorPct(s)
    if (pct == null) return { name: s['板块名称'] || s.name || '', value: 0.001, pct: null, itemStyle: { color: cssVar('--bg-secondary') } }
    const absPct = Math.abs(pct)
    const intensity = Math.min(1, absPct / (maxPct || 1))
    const alpha = 0.2 + intensity * 0.6
    const color = pct >= 0 ? `rgba(239, 68, 68, ${alpha.toFixed(2)})` : `rgba(34, 197, 94, ${alpha.toFixed(2)})`
    return { name: s['板块名称'] || s.name || '', value: Math.max(absPct, 0.001), pct, itemStyle: { color } }
  })
  await renderHeatmap(() => ({
    tooltip: { formatter: p => `${p.name}<br/>涨跌幅: ${p.data?.pct == null ? '--' : `${p.data.pct >= 0 ? '+' : ''}${p.data.pct.toFixed(2)}%`}` },
    series: [{
      type: 'treemap', data: treeData, width: '100%', height: '100%', roam: false, nodeClick: false,
      breadcrumb: { show: false },
      label: { show: true, formatter: p => `${p.name}\n${p.data?.pct == null ? '--' : `${p.data.pct >= 0 ? '+' : ''}${p.data.pct.toFixed(1)}%`}`, fontSize: 10, color: cssVar('--text') },
      itemStyle: { borderColor: cssVar('--bg'), borderWidth: 2, gapWidth: 2 },
      levels: [{ itemStyle: { borderWidth: 2, gapWidth: 2 } }],
    }],
  }))
}, { immediate: true })

function sectorPct(sector) {
  const value = sector['涨跌幅'] ?? sector.pct_chg
  return value == null || !Number.isFinite(Number(value)) ? null : Number(value)
}
</script>

<style scoped>
.mo-root { display: flex; flex-direction: column; gap: 14px; }

.mo-hero-row { display: grid; grid-template-columns: 2fr 1fr; gap: 14px; }
.mo-hero { background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px 24px 36px; display: flex; flex-direction: column; position: relative; }
.mo-hero:hover { border-color: var(--border-light); }
.mo-hero-top { display: flex; justify-content: space-between; align-items: flex-start; }
.mo-hero-label { font-size: var(--fs-sm); color: var(--text-muted); letter-spacing: .6px; margin-bottom: 4px; }
.mo-hero-price { font-family: var(--font-data); font-size: var(--fs-2xl); font-weight: 700; color: var(--text); margin-bottom: 4px; }
.mo-hero-change { display: flex; align-items: baseline; gap: 10px; }
.mo-hero-delta { font-family: var(--font-data); font-size: var(--fs-lg); font-weight: 600; }
.mo-hero-pct { font-family: var(--font-data); font-size: var(--fs-base); opacity: .85; }
.mo-hero-delta.up, .mo-hero-pct.up { color: var(--up); }
.mo-hero-delta.down, .mo-hero-pct.down { color: var(--down); }

.mo-hero-ohlc { display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px 16px; text-align: right; }
.mo-ohlc-item { display: flex; flex-direction: column; align-items: flex-end; gap: 1px; }
.mo-ohlc-label { font-size: var(--fs-caption); color: var(--text-muted); }
.mo-ohlc-val { font-family: var(--font-data); font-size: var(--fs-sm); color: var(--text-secondary); font-weight: 500; }
.mo-ohlc-val.up { color: var(--up); }
.mo-ohlc-val.down { color: var(--down); }

.mo-spark { height: 180px; margin-top: 12px; }
.mo-more { position: absolute; bottom: 10px; right: 14px; background: none; border: none; color: var(--text-muted); font-size: var(--fs-caption); cursor: pointer; }
.mo-more:hover { color: var(--primary); }

.mo-macro { display: flex; flex-direction: column; overflow: hidden; }
.mo-macro-grid { display: flex; flex-direction: column; gap: 6px; font-size: var(--fs-sm); }
.mo-macro-item { display: flex; justify-content: space-between; align-items: center; padding: 3px 0; border-bottom: 1px solid rgba(42, 42, 80, .25); }
.mo-macro-item:last-of-type { border-bottom: none; }
.mo-macro-label { font-size: var(--fs-caption); color: var(--text-muted); }
.mo-macro-val { font-family: var(--font-mono); font-size: var(--fs-sm); font-weight: 500; color: var(--text-secondary); }
.mo-macro-val.up { color: var(--up); }
.mo-macro-val.down { color: var(--down); }
.mo-risk-bar { margin-top: 8px; }
.mo-risk-track { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; margin-bottom: 4px; }
.mo-risk-fill { height: 100%; border-radius: 3px; transition: width .5s; }
.risk-warning .mo-risk-fill { background: var(--warning); }
.risk-danger .mo-risk-fill { background: var(--danger); }
.risk-normal .mo-risk-fill { background: var(--success); }
.mo-risk-text { font-size: var(--fs-sm); font-weight: 600; color: var(--text-secondary); }
.risk-danger .mo-risk-text { color: var(--danger); }

.mo-index-row { display: flex; align-items: center; gap: 0; padding: 12px 0; overflow-x: auto; }
.mo-index-chip { display: flex; align-items: baseline; gap: 8px; padding: 0 16px; flex-shrink: 0; }
.mo-chip-name { font-size: var(--fs-sm); color: var(--text-muted); }
.mo-chip-price { font-family: var(--font-data); font-size: var(--fs-base); font-weight: 600; color: var(--text); font-variant-numeric: tabular-nums; }
.mo-chip-chg { font-size: var(--fs-sm); font-weight: 600; font-variant-numeric: tabular-nums; }
.mo-chip-chg.up { color: var(--up); }
.mo-chip-chg.down { color: var(--down); }
.mo-chip-sep { width: 1px; height: 28px; background: var(--border); flex-shrink: 0; }
.mo-chip-sent { flex: 1; min-width: 200px; }
.mo-chip-sub { font-size: var(--fs-caption); color: var(--text-secondary); }
.mo-more-inline { background: none; border: none; color: var(--text-muted); font-size: var(--fs-caption); cursor: pointer; }
.mo-more-inline:hover { color: var(--primary); }

.mo-row-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.mo-row-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
.mo-chart { height: 340px; }
@media (max-width: 900px) { .mo-hero-row, .mo-row-2, .mo-row-3 { grid-template-columns: 1fr; } }
@media (max-width: 600px) {
  .mo-hero { padding: 16px 16px 42px; }
  .mo-hero-top { flex-direction: column; gap: 16px; }
  .mo-hero-ohlc { width: 100%; grid-template-columns: repeat(2, minmax(0, 1fr)); text-align: left; }
  .mo-ohlc-item { align-items: flex-start; }
  .mo-spark { height: 150px; }
}

.mo-breadth-more { display: flex; flex-direction: column; justify-content: center; align-items: flex-start; gap: 10px; }
.mo-more-hint { color: var(--text-muted); font-size: var(--fs-sm); margin: 0; }

.empty { text-align: center; padding: 20px; color: var(--text-muted); font-size: var(--fs-sm); }
</style>
