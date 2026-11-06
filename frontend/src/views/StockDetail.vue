<template>
  <div class="stock-detail">
    <div class="loading" v-if="loading">加载中...</div>
    <div class="error" v-if="error">{{ error }}</div>
    <div class="no-data" v-if="!rt && !loading && !error">
      <p>暂无实时数据</p>
       <p class="hint">请确认实时行情源正常，当前没有可用的最新报价</p>
    </div>

    <template v-if="rt">
      <header class="price-bar" :class="chgClass">
        <div class="pb-left">
          <div class="pb-name-row">
            <h1 class="pb-name">{{ rt.名称 || rt.name }}</h1>
            <span class="pb-code">{{ code }}</span>
            <span class="pb-status" :class="{ trading: rt.交易状态 && rt.交易状态.includes('竞价') }">{{ rt.交易状态 || '--' }}</span>
          </div>
          <div class="pb-ohlc">
            <div class="ohlc-item">开<span class="num">{{ rt.开盘价 ?? '--' }}</span></div>
            <div class="ohlc-item">高<span class="num">{{ rt.最高价 ?? '--' }}</span></div>
            <div class="ohlc-item">低<span class="num">{{ rt.最低价 ?? '--' }}</span></div>
            <div class="ohlc-item">昨收<span class="num">{{ rt.昨收 ?? '--' }}</span></div>
          </div>
        </div>
        <div class="pb-center">
          <div class="pb-price">{{ rt.最新价 ?? '--' }}</div>
          <div class="pb-change">
            <span class="pb-chg-amt">{{ formatChg(rt.涨跌额) }}</span>
            <span class="pb-chg-pct">{{ formatPct(rt.涨跌幅) }}</span>
          </div>
        </div>
        <div class="pb-right">
          <div class="trade-sig" v-if="signals && signals.composite" :class="signals.composite">
            <span class="ts-dot"></span>
            <span class="ts-label">{{ signalLabel(signals.composite) }}</span>
          </div>
        </div>
      </header>
      <div class="data-freshness" :class="rtMeta?.freshness || 'unavailable'">
        {{ rtMeta?.freshness === 'fresh' ? '实时' : '实时行情不可用' }}
        <span v-if="rtMeta?.source">· {{ rtMeta.source }}</span>
        <span v-if="rtMeta?.age_seconds != null">· {{ Number(rtMeta.age_seconds).toFixed(1) }}秒前</span>
      </div>

      <!-- 紧凑信息条 -->
      <div class="info-strip">
        <div class="is-block ob-block">
          <div class="is-label">买一</div>
          <div class="is-val up" v-if="rt.买盘 && rt.买盘[0]">{{ rt.买盘[0].price }}<span class="is-sub"> {{ fmtVol(rt.买盘[0].volume) }}</span></div>
          <div class="is-val dim" v-else>--</div>
          <div class="is-label" style="margin-top:2px">卖一</div>
          <div class="is-val down" v-if="rt.卖盘 && rt.卖盘[0]">{{ rt.卖盘[0].price }}<span class="is-sub"> {{ fmtVol(rt.卖盘[0].volume) }}</span></div>
          <div class="is-val dim" v-else>--</div>
        </div>
        <div class="is-block flow-block">
          <div class="is-label">主力净流入</div>
          <div class="is-val" :class="rt.主力净流入 > 0 ? 'up' : 'down'">{{ fmtAmt(rt.主力净流入) }}</div>
          <div class="flow-mini">
            <div class="flow-mini-bar"><div class="fm-in" :style="{ width: inRatio + '%' }"></div></div>
            <div class="flow-mini-lbl"><span>入{{ fmtAmt(rt.大单净流入) }}</span><span>出{{ fmtAmt(rt.大单净流出) }}</span></div>
          </div>
        </div>
        <div class="is-block metrics-block">
          <div class="is-row"><span class="is-lbl">成交量</span><span class="is-num">{{ fmtVol(rt.成交量) }}</span></div>
          <div class="is-row"><span class="is-lbl">成交额</span><span class="is-num">{{ fmtAmt(rt.成交额) }}</span></div>
          <div class="is-row"><span class="is-lbl">换手率</span><span class="is-num">{{ rt.换手率 != null ? rt.换手率 + '%' : '--' }}</span></div>
          <div class="is-row"><span class="is-lbl">量比</span><span class="is-num">{{ rt.量比 != null ? rt.量比 : '--' }}</span></div>
        </div>
        <div class="is-block metrics-block">
          <div class="is-row"><span class="is-lbl">市盈率</span><span class="is-num">{{ rt.市盈率 != null ? rt.市盈率 : '--' }}</span></div>
          <div class="is-row"><span class="is-lbl">市净率</span><span class="is-num">{{ rt.市净率 != null ? rt.市净率 : '--' }}</span></div>
          <div class="is-row"><span class="is-lbl">总市值</span><span class="is-num">{{ fmtMCap(rt.总市值) }}</span></div>
          <div class="is-row"><span class="is-lbl">流通市值</span><span class="is-num">{{ fmtMCap(rt.流通市值) }}</span></div>
        </div>
        <div class="is-block sig-block" v-if="signals">
          <div class="sig-row" :class="{ on: signals.ofi_alert }">
            <span class="sig-dot"></span>
            <span class="sig-lbl">OFI</span>
            <span class="sig-n">{{ signals.ofi != null ? signals.ofi : '--' }}</span>
            <span class="sig-tag" v-if="signals.ofi_alert">背离</span>
          </div>
          <div class="sig-row" :class="{ on: signals.big_order_alert }">
            <span class="sig-dot"></span>
            <span class="sig-lbl">大单Z</span>
            <span class="sig-n">{{ signals.big_order_z != null ? signals.big_order_z.toFixed(2) : '--' }}</span>
            <span class="sig-tag" v-if="signals.big_order_alert">异动</span>
          </div>
          <div class="sig-row" :class="{ on: signals.vwap_alert }">
            <span class="sig-dot"></span>
            <span class="sig-lbl">VWAP</span>
            <span class="sig-n">{{ signals.vwap_deviation != null ? signals.vwap_deviation.toFixed(2) + 'σ' : '--' }}</span>
            <span class="sig-tag" v-if="signals.vwap_alert">极端</span>
          </div>
          <div class="sig-row" :class="{ on: signals.vol_acc_alert }">
            <span class="sig-dot"></span>
            <span class="sig-lbl">量加速</span>
            <span class="sig-n">{{ signals.vol_ratio_acc != null ? signals.vol_ratio_acc.toFixed(2) : '--' }}</span>
            <span class="sig-tag" v-if="signals.vol_acc_alert">加速</span>
          </div>
        </div>
        <div class="is-block metrics-block" v-if="!signals"></div>
      </div>
    </template>

    <!-- 全宽图表 Tab -->
    <section class="card chart-card">
      <div class="card-head">
        <div class="chart-tabs">
          <button class="ctab" :class="{ active: activeChart === 'intra' }" @click="switchChart('intra')">分时走势<span v-if="intraday"> {{ intraday.length }}分</span></button>
          <button class="ctab" :class="{ active: activeChart === 'daily' }" @click="switchChart('daily')">日K线<span v-if="daily"> {{ daily.length }}条</span></button>
        </div>
      </div>
      <div v-if="activeChart === 'intra' && intraday?.length && intradayMeta?.quantity_status !== 'verified'" class="intra-data-note" role="status">
        分钟量额与均价口径未核验；当前仅展示价格轨迹，均价线和成交量不作为有效数据。
      </div>
      <div v-if="activeChart === 'daily' && benchmarkStatus" class="intra-data-note" role="status">
        {{ benchmarkStatus }}
      </div>
      <div class="chart-wrap tab-chart" ref="intraChartRef" v-show="activeChart === 'intra'"></div>
      <div class="chart-wrap tab-chart" ref="chartRef" v-show="activeChart === 'daily'"></div>
    </section>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import * as echarts from 'echarts'
import api from '../api/index.js'
import { alignBenchmarkToStockDates } from '../utils/benchmark.js'
import { cssVar } from '../utils/marketFormat.js'

const route = useRoute()
const code = computed(() => route.params.code || route.query.code || '600519')
const rt = ref(null)
const daily = ref(null)
const intraday = ref(null)
const intradayMeta = ref(null)
const signals = ref(null)
const rtMeta = ref(null)
const loading = ref(false)
const error = ref(null)
const benchmarkStatus = ref('')
let timer = null
let requestSeq = 0

const activeChart = ref('intra')
const chartRef = ref(null)
const intraChartRef = ref(null)
let chartInstance = null
let intraChartInstance = null

function switchChart(tab) {
  activeChart.value = tab
  nextTick(() => {
    if (tab === 'daily' && chartInstance) chartInstance.resize()
    if (tab === 'intra' && intraChartInstance) intraChartInstance.resize()
  })
}

const chgClass = computed(() => {
  const v = rt.value?.涨跌幅
  if (v == null) return ''
  return v > 0 ? 'up' : v < 0 ? 'down' : ''
})

const inRatio = computed(() => {
  const inn = rt.value?.大单净流入 || 0
  const out = rt.value?.大单净流出 || 0
  const total = inn + out
  if (!total) return '1'
  return String(Math.round(inn / total * 100))
})

const outRatio = computed(() => {
  const inn = rt.value?.大单净流入 || 0
  const out = rt.value?.大单净流出 || 0
  const total = inn + out
  if (!total) return '1'
  return String(Math.round(out / total * 100))
})

function askBarPct(vol) {
  if (!vol) return '0%'
  const max = maxObVol()
  if (!max) return '0%'
  return Math.round(vol / max * 100) + '%'
}

function bidBarPct(vol) {
  if (!vol) return '0%'
  const max = maxObVol()
  if (!max) return '0%'
  return Math.round(vol / max * 100) + '%'
}

function maxObVol() {
  let m = 0
  const bid = rt.value?.买盘 || []
  const ask = rt.value?.卖盘 || []
  for (const x of [...bid, ...ask]) {
    if (x.volume > m) m = x.volume
  }
  return m
}

function fmtVol(v) {
  if (!v) return '--'
  if (v > 1e8) return (v / 1e8).toFixed(2) + '亿'
  if (v > 1e4) return (v / 1e4).toFixed(0) + '万'
  return v.toLocaleString()
}

function fmtAmt(v) {
  if (!v) return '--'
  const abs = Math.abs(v)
  if (abs > 1e8) return (v / 1e8).toFixed(2) + '亿'
  if (abs > 1e4) return (v / 1e4).toFixed(0) + '万'
  return v.toFixed(0)
}

function fmtMCap(v) {
  if (!v) return '--'
  if (v > 1e12) return (v / 1e12).toFixed(2) + '万亿'
  if (v > 1e8) return (v / 1e8).toFixed(0) + '亿'
  return v.toLocaleString()
}

function formatChg(v) { return v != null ? (v > 0 ? '+' : '') + v.toFixed(2) : '--' }
function formatPct(v) { return v != null ? (v > 0 ? '+' : '') + v.toFixed(2) + '%' : '--' }
function signalLabel(s) {
  const map = { strong_buy: '强烈买入', buy: '买入', hold: '观望', sell: '卖出', strong_sell: '强烈卖出' }
  return map[s] || s
}

function calcMA(data, period) {
  const result = new Array(data.length).fill(null)
  for (let i = period - 1; i < data.length; i++) {
    let sum = 0
    for (let j = 0; j < period; j++) sum += data[i - j]
    result[i] = sum / period
  }
  return result
}

function findCrossPoints(dates, closes) {
  const ma5 = calcMA(closes, 5)
  const ma20 = calcMA(closes, 20)
  const buyPoints = []
  const sellPoints = []
  for (let i = 21; i < closes.length; i++) {
    if (ma5[i] == null || ma20[i] == null || ma5[i-1] == null || ma20[i-1] == null) continue
    if (ma5[i-1] <= ma20[i-1] && ma5[i] > ma20[i]) {
      buyPoints.push({ name: '金叉(MA5↑MA20)', coord: [dates[i], closes[i]], value: '金叉', symbol: 'triangle', symbolSize: 12, itemStyle: { color: cssVar('--up') }, label: { show: true, color: cssVar('--up'), fontSize: 9, position: 'bottom', distance: 4, formatter: '金叉' } })
    }
    if (ma5[i-1] >= ma20[i-1] && ma5[i] < ma20[i]) {
      sellPoints.push({ name: '死叉(MA5↓MA20)', coord: [dates[i], closes[i]], value: '死叉', symbol: 'triangle', symbolSize: 12, symbolRotate: 180, itemStyle: { color: cssVar('--down') }, label: { show: true, color: cssVar('--down'), fontSize: 9, position: 'top', distance: 4, formatter: '死叉' } })
    }
  }
  return { buyPoints, sellPoints, ma5, ma20 }
}

function isCurrentRequest(seq, requestedCode) {
  return seq === requestSeq && requestedCode === String(code.value)
}

async function buildChart(rows, seq, requestedCode) {
  if (!isCurrentRequest(seq, requestedCode)) return
  if (!chartRef.value) return
  chartInstance = echarts.getInstanceByDom(chartRef.value)
  if (!chartInstance) {
    await new Promise(r => requestAnimationFrame(r))
    if (!isCurrentRequest(seq, requestedCode) || !chartRef.value) return
    chartInstance = echarts.init(chartRef.value)
  }

  const dates = rows.map(r => r.trade_date).reverse()
  const ohlc = rows.map(r => [r.open, r.close, r.low, r.high]).reverse()
  const closes = rows.map(r => r.close).reverse()
  const vols = rows.map(r => r.volume).reverse()

  let indexLine = null
  benchmarkStatus.value = ''
  try {
    const indexCode = requestedCode.startsWith('6') || requestedCode.startsWith('68') ? '000001' : '399001'
    const idxResp = await api.get(`/ths/index/${indexCode}/daily`)
    if (!isCurrentRequest(seq, requestedCode)) return
    if (idxResp.data && idxResp.data.length > 0) {
      const aligned = alignBenchmarkToStockDates(rows.slice().reverse(), idxResp.data)
      indexLine = aligned.line
      benchmarkStatus.value = aligned.baseDate
        ? `大盘线按首个共同有效交易日 ${aligned.baseDate} 归一化；缺失日期保留断点。`
        : '指数与个股区间没有共同有效交易日，已隐藏大盘线。'
    } else {
      benchmarkStatus.value = '基准指数数据不可用，已隐藏大盘线。'
    }
  } catch (e) {
    benchmarkStatus.value = '基准指数数据读取失败，已隐藏大盘线。'
  }
  if (!isCurrentRequest(seq, requestedCode)) return

  const { buyPoints, sellPoints, ma5, ma20 } = findCrossPoints(dates, closes)
  const allMarkPoints = [...buyPoints, ...sellPoints]

  const composite = signals.value?.composite
  if (composite && (composite === 'strong_buy' || composite === 'buy')) {
    const lastIdx = dates.length - 1
    allMarkPoints.push({
      name: '当前信号: ' + signalLabel(composite), coord: [dates[lastIdx], closes[lastIdx]],
      symbol: 'pin', symbolSize: 40, symbolRotate: 180,
      itemStyle: { color: cssVar('--up') },
      label: { show: true, color: '#fff', fontSize: 12, position: 'bottom', distance: 16, formatter: '当前: ' + signalLabel(composite), backgroundColor: 'rgba(239,68,68,0.85)', padding: [3, 8], borderRadius: 3, fontWeight: 'bold' },
    })
  } else if (composite && (composite === 'strong_sell' || composite === 'sell')) {
    const lastIdx = dates.length - 1
    allMarkPoints.push({
      name: '当前信号: ' + signalLabel(composite), coord: [dates[lastIdx], closes[lastIdx]],
      symbol: 'pin', symbolSize: 40,
      itemStyle: { color: cssVar('--down') },
      label: { show: true, color: '#fff', fontSize: 12, position: 'top', distance: 16, formatter: '当前: ' + signalLabel(composite), backgroundColor: 'rgba(34,197,94,0.85)', padding: [3, 8], borderRadius: 3, fontWeight: 'bold' },
    })
  }

  chartInstance.setOption({
    backgroundColor: '#0f1319',
    grid: [
      { left: 8, right: 8, top: 10, height: '56%' },
      { left: 8, right: 8, top: '69%', height: '12%' },
      { left: 8, right: 8, top: '83%', height: '14%' },
    ],
    xAxis: [
      { type: 'category', data: dates, gridIndex: 0, axisLabel: { show: false }, axisTick: { show: false } },
      { type: 'category', data: dates, gridIndex: 1, axisLabel: { show: false }, axisTick: { show: false } },
      { type: 'category', data: dates, gridIndex: 2, axisLabel: { color: '#555', fontSize: 9, interval: Math.floor(dates.length / 6) }, axisTick: { show: false }, splitLine: { show: false } },
    ],
    yAxis: [
      { type: 'value', gridIndex: 0, scale: true, axisLabel: { color: '#555', fontSize: 10 }, splitLine: { lineStyle: { color: '#1a1f2b' } }, position: 'right' },
      { type: 'value', gridIndex: 1, axisLabel: { show: false }, splitLine: { show: false } },
      { type: 'value', gridIndex: 2, scale: true, axisLabel: { color: '#555', fontSize: 9 }, splitLine: { lineStyle: { color: '#1a1f2b' } }, position: 'right' },
    ],
    series: [
      {
        type: 'candlestick', name: 'K线',
        data: ohlc,
        itemStyle: { color: cssVar('--up'), color0: cssVar('--down'), borderColor: cssVar('--up'), borderColor0: cssVar('--down') },
        markPoint: { data: allMarkPoints, symbolOffset: [0, 0] },
      },
      {
        type: 'line', name: 'MA5',
        data: ma5, symbol: 'none', xAxisIndex: 0, yAxisIndex: 0,
        lineStyle: { color: 'rgba(239,68,68,0.5)', width: 1 },
      },
      {
        type: 'line', name: 'MA20',
        data: ma20, symbol: 'none', xAxisIndex: 0, yAxisIndex: 0,
        lineStyle: { color: 'rgba(34,197,94,0.5)', width: 1 },
      },
      {
        type: 'bar', name: '成交量',
        data: vols,
        xAxisIndex: 1, yAxisIndex: 1,
        itemStyle: {
          color: (p) => {
            const d = ohlc[p.dataIndex]
            return d ? (d[1] >= d[0] ? 'rgba(239,68,68,0.5)' : 'rgba(34,197,94,0.5)') : 'rgba(107,114,128,0.3)'
          }
        },
      },
      ...(indexLine ? [{
        type: 'line', name: '大盘',
        data: indexLine, symbol: 'none', xAxisIndex: 2, yAxisIndex: 2,
        lineStyle: { color: '#d4a853', width: 1.2 },
        areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: 'rgba(212,168,83,0.15)' }, { offset: 1, color: 'rgba(212,168,83,0)' }] } },
      }] : []),
    ],
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' },
      backgroundColor: '#131820',
      borderColor: '#1e2632',
      textStyle: { color: '#e8eaed', fontSize: 12 },
    },
  }, { notMerge: true })
  chartInstance.resize()
}

async function buildIntraChart(rows, seq, requestedCode) {
  if (!isCurrentRequest(seq, requestedCode)) return
  if (!intraChartRef.value) return
  intraChartInstance = echarts.getInstanceByDom(intraChartRef.value)
  if (!intraChartInstance) {
    await new Promise(r => requestAnimationFrame(r))
    if (!isCurrentRequest(seq, requestedCode) || !intraChartRef.value) return
    intraChartInstance = echarts.init(intraChartRef.value)
  }

  const times = rows.map(r => r.bar_time.slice(11, 16))
  const prices = rows.map(r => r.price)
  const avgs = rows.map(r => r.avg_price)
  const vols = rows.map(r => r.volume)
  const prevClose = rt.value?.昨收 || prices[0]

  const composite = signals.value?.composite
  const sigColor = (composite === 'strong_buy' || composite === 'buy') ? cssVar('--up')
    : (composite === 'strong_sell' || composite === 'sell') ? cssVar('--down') : '#6b7280'
  const sigBg = composite && composite !== 'hold'
    ? `rgba(${composite.includes('buy') ? '239,68,68' : '34,197,94'},0.08)` : 'transparent'

  const graphic = composite ? [{
    type: 'text',
    right: 12, top: 8,
    style: {
      text: signalLabel(composite),
      font: 'bold 14px "JetBrains Mono", Consolas, monospace',
      fill: sigColor,
    },
  }] : []

  if (!isCurrentRequest(seq, requestedCode)) return
  intraChartInstance.setOption({
    backgroundColor: '#0f1319',
    graphic,
    grid: [
      { left: 8, right: 40, top: 32, height: '62%' },
      { left: 8, right: 40, top: '78%', height: '18%' },
    ],
    xAxis: [
      { type: 'category', data: times, gridIndex: 0, axisLabel: { show: false }, axisTick: { show: false }, boundaryGap: false },
      { type: 'category', data: times, gridIndex: 1, axisLabel: { color: '#555', fontSize: 9, interval: Math.floor(times.length / 8) }, axisTick: { show: false }, splitLine: { show: false }, boundaryGap: false },
    ],
    yAxis: [
      { type: 'value', gridIndex: 0, scale: true, axisLabel: { color: '#555', fontSize: 10 }, splitLine: { lineStyle: { color: '#1a1f2b' } }, position: 'right' },
      { type: 'value', gridIndex: 1, axisLabel: { color: '#555', fontSize: 9 }, splitLine: { show: false } },
    ],
    series: [
      {
        type: 'line', name: '价格',
        data: prices,
        smooth: true, symbol: 'none',
        lineStyle: { color: '#e8eaed', width: 1.5 },
        markArea: composite && composite !== 'hold' ? {
          silent: true,
          data: [[{ xAxis: '90%', yAxis: 'min' }, { xAxis: '100%', yAxis: 'max' }]],
          itemStyle: { color: sigBg },
        } : undefined,
        markLine: {
          silent: true, symbol: 'none',
          lineStyle: { color: '#555', type: 'dashed' },
          data: [{ yAxis: prevClose, label: { show: false } }],
        },
      },
      {
        type: 'line', name: '均价',
        data: avgs,
        smooth: true, symbol: 'none',
        lineStyle: { color: '#d4a853', width: 1 },
      },
      {
        type: 'bar', name: '成交量',
        data: vols,
        xAxisIndex: 1, yAxisIndex: 1,
        itemStyle: { color: 'rgba(107,114,128,0.25)' },
      },
    ],
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#131820',
      borderColor: '#1e2632',
      textStyle: { color: '#e8eaed', fontSize: 12 },
    },
  }, { notMerge: true })
  intraChartInstance.resize()
}

async function fetchData() {
  const requestedCode = String(code.value)
  const seq = ++requestSeq
  try {
    const r = await api.get(`/ths/stock/${requestedCode}/realtime`)
    if (!isCurrentRequest(seq, requestedCode)) return
    rt.value = r.data
    rtMeta.value = r.meta || null
    error.value = null
  } catch (e) {
    if (!isCurrentRequest(seq, requestedCode)) return
    rt.value = null
    rtMeta.value = { freshness: 'unavailable' }
    if (e.response?.status === 404) error.value = null
    else error.value = e.message || '加载失败'
  }
  try {
    const s = await api.get(`/ths/signals/${requestedCode}`)
    if (!isCurrentRequest(seq, requestedCode)) return
    signals.value = s.data
  } catch (e) {
    if (!isCurrentRequest(seq, requestedCode)) return
    signals.value = null
  }
  try {
    const d = await api.get(`/ths/stock/${requestedCode}/daily`)
    if (!isCurrentRequest(seq, requestedCode)) return
    daily.value = Array.isArray(d.data) ? d.data : null
    await nextTick()
    if (!isCurrentRequest(seq, requestedCode)) return
    if (daily.value?.length) await buildChart(daily.value, seq, requestedCode)
    else if (chartInstance && !chartInstance.isDisposed()) chartInstance.clear()
  } catch (e) {
    if (!isCurrentRequest(seq, requestedCode)) return
    daily.value = null
    if (chartInstance && !chartInstance.isDisposed()) chartInstance.clear()
  }
  try {
    const i = await api.get(`/ths/stock/${requestedCode}/intraday`)
    if (!isCurrentRequest(seq, requestedCode)) return
    intraday.value = Array.isArray(i.data) ? i.data : null
    intradayMeta.value = i.meta || null
    await nextTick()
    if (!isCurrentRequest(seq, requestedCode)) return
    if (intraday.value?.length) await buildIntraChart(intraday.value, seq, requestedCode)
    else if (intraChartInstance && !intraChartInstance.isDisposed()) intraChartInstance.clear()
  } catch (e) {
    if (!isCurrentRequest(seq, requestedCode)) return
    intraday.value = null
    intradayMeta.value = { freshness: 'unavailable', quantity_status: 'unknown' }
    if (intraChartInstance && !intraChartInstance.isDisposed()) intraChartInstance.clear()
  }
  if (isCurrentRequest(seq, requestedCode)) loading.value = false
}

onMounted(() => {
  fetchData()
  timer = setInterval(fetchData, 10000)
})

onUnmounted(() => {
  requestSeq += 1
  clearInterval(timer)
  if (chartInstance && !chartInstance.isDisposed()) chartInstance.dispose()
  if (intraChartInstance && !intraChartInstance.isDisposed()) intraChartInstance.dispose()
})

watch(code, () => {
  requestSeq += 1
  loading.value = true
  rt.value = null
  rtMeta.value = null
  signals.value = null
  error.value = null
  daily.value = null
  intraday.value = null
  intradayMeta.value = null
  benchmarkStatus.value = ''
  if (chartInstance && !chartInstance.isDisposed()) chartInstance.clear()
  if (intraChartInstance && !intraChartInstance.isDisposed()) intraChartInstance.clear()
  fetchData()
})
</script>

<style scoped>
.stock-detail {
  --bg: #0a0e14;
  --card: #131820;
  --border: #1e2632;
  --gold: #d4a853;
  --up: #ef4444;
  --down: #22c55e;
  --text: #e8eaed;
  --muted: #6b7280;
  --dim: #3a3f4a;
  --font-mono: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
  padding: 16px 20px;
  color: var(--text);
  background: var(--bg);
  min-height: 100vh;
}

.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 6px;
}
.card-head {
  display: flex; justify-content: space-between; align-items: center;
  padding: 10px 16px; border-bottom: 1px solid var(--border);
}
.card-head h3 { font-size: 12px; font-weight: 500; color: var(--muted); margin: 0; letter-spacing: 0.04em; }
.card-hint { font-size: 10px; color: var(--dim); }

/* ── 价格头部 ──────────────────────────────── */
.price-bar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 20px 24px; margin-bottom: 10px;
  background: var(--card); border: 1px solid var(--border); border-radius: 6px;
  transition: box-shadow 0.6s;
}
.price-bar.up   { box-shadow: inset 0 1px 0 rgba(239,68,68,0.12); }
.price-bar.down { box-shadow: inset 0 1px 0 rgba(34,197,94,0.12); }

.pb-left { display: flex; flex-direction: column; gap: 8px; }
.pb-name-row { display: flex; align-items: center; gap: 8px; }
.pb-name { font-size: 18px; font-weight: 600; margin: 0; color: #fff; }
.pb-code { font-size: 11px; color: var(--muted); font-family: var(--font-mono); background: rgba(255,255,255,0.04); padding: 1px 6px; border-radius: 3px; }
.pb-status { font-size: 10px; padding: 1px 6px; border-radius: 3px; background: rgba(255,255,255,0.05); color: var(--muted); }
.pb-status.trading { background: rgba(34,197,94,0.12); color: var(--down); }
.pb-ohlc { display: flex; gap: 16px; }
.ohlc-item { font-size: 11px; color: var(--muted); display: flex; flex-direction: column; gap: 1px; }
.ohlc-item .num { font-family: var(--font-mono); font-size: 13px; color: var(--text); }

.pb-center { text-align: center; }
.pb-price { font-family: var(--font-mono); font-size: 42px; font-weight: 600; line-height: 1.1; letter-spacing: 0; }
.up   .pb-price { color: var(--up); }
.down .pb-price { color: var(--down); }
.pb-price { color: var(--text); }
.pb-change { display: flex; gap: 8px; justify-content: center; margin-top: 2px; }
.pb-chg-amt, .pb-chg-pct { font-family: var(--font-mono); font-size: 16px; font-weight: 500; }
.up   .pb-chg-amt, .up   .pb-chg-pct { color: var(--up); }
.down .pb-chg-amt, .down .pb-chg-pct { color: var(--down); }

.pb-right { display: flex; align-items: center; }
.trade-sig {
  display: flex; align-items: center; gap: 6px;
  padding: 8px 18px; border-radius: 4px;
  font-weight: 600; font-size: 13px; white-space: nowrap;
}
.ts-dot { width: 7px; height: 7px; border-radius: 50%; animation: pulse-dot 1.4s ease-in-out infinite; }
@keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(1.3)} }
.trade-sig.strong_buy { background: rgba(239,68,68,0.12); color: #ef4444; }
.trade-sig.strong_buy .ts-dot { background: #ef4444; box-shadow: 0 0 8px rgba(239,68,68,0.5); }
.trade-sig.buy        { background: rgba(239,68,68,0.08); color: #f97070; }
.trade-sig.buy .ts-dot        { background: #f97070; }
.trade-sig.hold       { background: rgba(107,114,128,0.08); color: #8b949e; }
.trade-sig.hold .ts-dot       { background: #8b949e; }
.trade-sig.sell       { background: rgba(34,197,94,0.08); color: #4ade80; }
.trade-sig.sell .ts-dot       { background: #4ade80; }
.trade-sig.strong_sell { background: rgba(34,197,94,0.12); color: #22c55e; }
.trade-sig.strong_sell .ts-dot { background: #22c55e; box-shadow: 0 0 8px rgba(34,197,94,0.5); }

/* ── 信息条 ────────────────────────────────── */
.info-strip {
  display: grid;
  grid-template-columns: 110px 170px 1fr 1fr 200px;
  gap: 1px;
  background: var(--border);
  border-radius: 6px;
  overflow: hidden;
  margin-bottom: 10px;
}
.is-block {
  background: var(--card);
  padding: 10px 14px;
  display: flex; flex-direction: column; justify-content: center;
}
.is-label { font-size: 10px; color: var(--muted); margin-bottom: 2px; letter-spacing: 0.04em; }
.is-val { font-family: var(--font-mono); font-size: 15px; font-weight: 500; }
.is-sub { font-size: 11px; color: var(--muted); font-weight: 400; }
.is-val.dim { color: var(--dim); }

.metrics-block { gap: 1px; padding: 6px 14px; }
.is-row { display: flex; justify-content: space-between; align-items: center; padding: 2px 0; }
.is-lbl { font-size: 10px; color: var(--muted); }
.is-num { font-family: var(--font-mono); font-size: 12px; color: var(--text); }

.flow-mini { margin-top: 4px; }
.flow-mini-bar { height: 3px; background: rgba(34,197,94,0.2); border-radius: 1px; overflow: hidden; }
.fm-in { height: 100%; background: var(--up); border-radius: 1px; }
.flow-mini-lbl { display: flex; justify-content: space-between; font-size: 9px; color: var(--dim); margin-top: 2px; }

.sig-block { gap: 0; padding: 4px 10px; }
.sig-row { display: flex; align-items: center; gap: 6px; padding: 2px 0; }
.sig-row .sig-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--dim); flex-shrink: 0; transition: background .3s, box-shadow .3s; }
.sig-row.on .sig-dot { background: #d4a853; box-shadow: 0 0 5px rgba(212,168,83,0.4); }
.sig-lbl { font-size: 10px; color: var(--muted); width: 36px; }
.sig-n { font-family: var(--font-mono); font-size: 12px; color: var(--text); flex: 1; text-align: right; }
.sig-tag { font-size: 9px; padding: 1px 5px; border-radius: 2px; background: rgba(212,168,83,0.15); color: var(--gold); font-weight: 600; }

/* ── 图表 ──────────────────────────────────── */
.chart-card { margin-bottom: 10px; }
.chart-card .card-head { border-bottom: none; padding: 6px 16px; }
.chart-tabs { display: flex; gap: 0; }
.ctab {
  background: none; border: none; color: var(--dim); font-size: 12px;
  padding: 7px 16px; cursor: pointer; border-bottom: 2px solid transparent;
  transition: color .2s, border-color .2s; font-family: inherit;
}
.ctab:hover { color: var(--muted); }
.ctab.active { color: var(--text); border-bottom-color: var(--text); }
.ctab span { font-size: 10px; color: var(--dim); margin-left: 2px; }
.intra-data-note { padding: 8px 16px; color: var(--gold); font-size: 11px; border-top: 1px solid var(--border); }
.tab-chart { height: 480px; }

/* ── 通用 ──────────────────────────────────── */
.up   { color: var(--up); }
.down { color: var(--down); }
.loading, .error, .no-data { text-align: center; padding: 80px 40px; color: var(--muted); font-size: 14px; }
.no-data .hint { font-size: 12px; color: var(--dim); margin-top: 8px; }

@media (max-width: 1100px) {
  .info-strip { grid-template-columns: 100px 150px 1fr 1fr 170px; }
}
@media (max-width: 860px) {
  .info-strip { grid-template-columns: 1fr 1fr; }
  .sig-block { display: none; }
  .price-bar { flex-wrap: wrap; gap: 10px; }
  .pb-price { font-size: 32px; }
}
@media (max-width: 540px) {
  .info-strip { grid-template-columns: 1fr; }
  .stock-detail { padding: 8px; }
  .pb-price { font-size: 28px; }
}
</style>
