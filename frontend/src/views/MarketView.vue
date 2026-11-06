<template>
  <div class="mv-root">
    <h1 class="sr-only">行情监控</h1>
    <MarketPulse
      :regime="regime" :timing="timingSignal" :breadth="breadthData"
      :riskLabel="riskLabel" :pendingCount="pendingSignalsCount" />

    <div v-if="error" class="error">{{ error }}</div>
    <div v-if="indexMeta && indexMeta.freshness !== 'fresh'" class="error">
      实时指数暂不可用：{{ indexMeta.freshness || 'unknown' }}
    </div>
    <div v-if="sectorMeta && sectorMeta.data_envelope?.status !== 'available'" class="error">
      板块行情{{ sectorMeta.data_envelope?.status === 'stale' ? '已过期' : sectorMeta.data_envelope?.status === 'partial' ? '覆盖不完整' : '不可用' }}：
      {{ sectorMeta.data_envelope?.coverage?.accepted ?? 0 }}/{{ sectorMeta.data_envelope?.coverage?.requested ?? 0 }} 有报价；
      {{ sectorMeta.data_envelope?.missing_reason || '原因未知' }}
      <span v-if="sectorMeta.source">（来源：{{ sectorMeta.source }}）</span>
      <span v-if="sectorMeta.as_of">（统计日：{{ sectorMeta.as_of }}）</span>
    </div>
    <LoadingSpinner v-if="loading" />

    <template v-if="!loading">
      <nav class="mv-tabs" aria-label="行情视图" role="tablist">
        <button v-for="t in tabs" :key="t.key"
          :class="['mv-tab', { active: activeTab === t.key }]"
          role="tab" :aria-selected="activeTab === t.key"
          @click="activeTab = t.key">{{ t.label }}</button>
      </nav>

      <MarketOverview v-if="activeTab === 'overview'"
        :indices="indices" :sse-chart-data="sseChartData" :macro="macro"
        :sectors="sectors" :sent-overview="sentOverview" :gainers="gainers" :losers="losers"
        @switch-tab="activeTab = $event" />
      <MarketIndices v-else-if="activeTab === 'indices'" :indices="indices" :sse-chart-data="sseChartData" />
      <MarketSectors v-else-if="activeTab === 'sectors'" :sectors="sectors" :meta="sectorMeta" @open-sector="openSectorDetail" />
      <RetailHeatPanel v-else-if="activeTab === 'retail'" />
      <MarketSentiment v-else-if="activeTab === 'sentiment'" :sent-overview="sentOverview" />
      <MarketBreadth v-else-if="activeTab === 'breadth'" :gainers="gainers" :losers="losers" :breadth="breadthData" />
      <MarketMacro v-else-if="activeTab === 'macro'" :macro="macro" />
    </template>

    <!-- ═══ 个股详情侧边面板 ═══ -->
    <div v-if="stockPanel.visible" class="stock-overlay" @click.self="closeStock"></div>
    <div v-if="stockPanel.visible" class="stock-panel">
      <div class="stock-panel-header">
        <span class="stock-panel-title">{{ stockPanel.name || stockPanel.code }}</span>
        <button class="stock-panel-close" @click="closeStock">✕</button>
      </div>
      <LoadingSpinner v-if="stockPanel.loading" />
      <template v-if="!stockPanel.loading && stockPanel.info">
        <div class="stock-tabs">
          <button v-for="t in stockTabs" :key="t.key"
            :class="['stock-tab', { active: stockTab === t.key }]"
            @click="switchStockTab(t.key)">{{ t.label }}</button>
        </div>
        <div v-show="stockTab === 'kline'" class="stock-tab-content">
          <div ref="stockChartRef" class="mv-chart"></div>
        </div>
        <div v-show="stockTab === 'indicators'" class="stock-tab-content">
          <LoadingSpinner v-if="stockPanel.indicatorsLoading" />
          <template v-else-if="stockPanel.indicators">
            <div class="ind-grid" v-if="Object.keys(stockPanel.indicators.indicators || {}).length">
              <div class="ind-cell" v-for="(v, k) in stockPanel.indicators.indicators" :key="k">
                <span class="ind-name">{{ k }}</span><span class="ind-val">{{ fmtM(v) }}</span>
              </div>
            </div>
            <div class="ind-section" v-if="stockPanel.indicators.signals?.length">
              <div class="ind-subtitle">交叉信号</div>
              <div class="ind-signal" v-for="(s, i) in stockPanel.indicators.signals" :key="i"
                :class="s.includes('金叉') || s.includes('买入') ? 'signal-buy' : 'signal-sell'">{{ s }}</div>
            </div>
            <div class="ind-section" v-if="stockPanel.indicators.patterns?.length">
              <div class="ind-subtitle">K线形态</div>
              <div class="ind-signal" v-for="(p, i) in stockPanel.indicators.patterns" :key="i">{{ p }}</div>
            </div>
            <div v-if="!Object.keys(stockPanel.indicators.indicators || {}).length && !stockPanel.indicators.signals?.length && !stockPanel.indicators.patterns?.length" class="empty">暂无指标数据</div>
          </template>
          <div v-else class="empty">暂无指标数据</div>
        </div>
        <div v-show="stockTab === 'ai'" class="stock-tab-content">
          <div class="ai-actions" v-if="!stockPanel.aiStreaming">
            <button class="ai-btn" @click="runAiAnalysis('single')">首席分析师</button>
            <button class="ai-btn ai-btn-panel" @click="runAiAnalysis('panel')">辩论模式 (4专家+风控+主持)</button>
          </div>
          <div class="ai-output" v-if="stockPanel.aiContent">
            <div v-for="(block, i) in stockPanel.aiBlocks" :key="i">
              <div v-if="block.type === 'status'" class="ai-status">{{ block.text }}</div>
              <div v-else-if="block.type === 'error'" class="ai-error">{{ block.text }}</div>
              <div v-else class="ai-agent-block">
                <div class="ai-agent-name">{{ block.agent }}</div>
                <pre class="ai-agent-text">{{ block.text }}</pre>
              </div>
            </div>
            <LoadingSpinner v-if="stockPanel.aiStreaming" />
          </div>
        </div>
        <div v-show="stockTab === 'financial'" class="stock-tab-content">
          <LoadingSpinner v-if="stockPanel.finLoading" />
          <div v-else-if="stockPanel.financial && Object.keys(stockPanel.financial).length">
            <div class="fin-grid">
              <div class="fin-cell" v-for="(v, k) in stockPanel.financial" :key="k">
                <span class="fin-name">{{ k }}</span><span class="fin-val">{{ fmtM(v) }}</span>
              </div>
            </div>
          </div>
          <div v-else class="empty">暂无财务数据</div>
        </div>
        <div v-show="stockTab === 'news'" class="stock-tab-content">
          <LoadingSpinner v-if="stockPanel.newsLoading" />
          <div v-else-if="stockPanel.news?.length">
            <div v-for="(item, i) in stockPanel.news" :key="i" class="mv-news-row">
              <span :class="['tag', sentTag(item.sent_score)]">{{ sentLabel(item.sent_score) }}</span>
              <span class="mv-news-date">{{ item.source || '' }}</span>
              <span class="mv-news-title truncate" :title="item.title">{{ item.title?.slice(0, 50) }}</span>
              <span class="mv-news-date">{{ item.date || item.pub_date || '' }}</span>
            </div>
          </div>
          <div v-else class="empty">暂无新闻</div>
        </div>
        <div v-show="stockTab === 'qlib'" class="stock-tab-content">
          <LoadingSpinner v-if="stockPanel.qlibLoading" />
          <div v-else-if="stockPanel.qlibPred">
            <div class="qlib-score">
              <span class="qlib-label">预测分数</span>
              <span :class="['qlib-val', (stockPanel.qlibPred.score || 0) > 0 ? 'profit-up' : 'profit-down']">
                {{ Number(stockPanel.qlibPred.score || 0).toFixed(4) }}
              </span>
            </div>
            <div class="qlib-meta" v-if="stockPanel.qlibPred.model">模型: {{ stockPanel.qlibPred.model }}</div>
          </div>
          <div v-else class="empty">暂无量化预测数据</div>
        </div>
      </template>
    </div>

    <!-- ═══ 板块分析侧边面板 ═══ -->
    <div v-if="sectorPanel.visible" class="stock-overlay" @click.self="closeSectorPanel"></div>
    <div v-if="sectorPanel.visible" class="stock-panel">
      <div class="stock-panel-header">
        <span class="stock-panel-title">{{ sectorPanel.name }}</span>
        <button class="stock-panel-close" @click="closeSectorPanel">✕</button>
      </div>
      <LoadingSpinner v-if="sectorPanel.loading" />
      <template v-if="!sectorPanel.loading">
        <div class="stock-tabs">
          <button v-for="t in sectorTabs" :key="t.key"
            :class="['stock-tab', { active: sectorTab === t.key }]"
            @click="sectorTab = t.key; loadSectorTab(t.key)">{{ t.label }}</button>
        </div>
        <div v-show="sectorTab === 'stocks'" class="stock-tab-content">
          <div v-if="sectorPanel.membershipMeta?.status !== 'available'" class="error">
            板块分类快照{{ sectorPanel.membershipMeta?.status === 'stale' ? '已过期' : '未通过可用性核验' }}：
            {{ sectorPanel.membershipMeta?.missing_reason || sectorPanel.membershipMeta?.source || 'unknown' }}
          </div>
          <div v-if="sectorPanel.stocksMeta?.data_envelope?.status && sectorPanel.stocksMeta.data_envelope.status !== 'available'" class="error">
            成分股行情{{ sectorPanel.stocksMeta.data_envelope.status === 'stale' ? '已过期' : '部分或全部不可用' }}：
            {{ sectorPanel.stocksMeta?.coverage?.accepted ?? 0 }}/{{ sectorPanel.stocksMeta?.coverage?.requested ?? 0 }} 有报价
            <span v-if="sectorPanel.stocksMeta?.quote_source">（来源：{{ sectorPanel.stocksMeta.quote_source }}）</span>
          </div>
          <div v-if="sectorPanel.stocks?.length">
            <div class="empty">成员快照 {{ sectorPanel.snapshotId }}；当前展示 {{ sectorPanel.membershipMeta?.coverage?.returned ?? sectorPanel.stocks.length }}/{{ sectorPanel.membershipMeta?.coverage?.members ?? '--' }} 个成员</div>
            <table class="mini-table">
              <thead><tr><th>代码</th><th>名称</th><th>最新价</th><th>涨跌幅</th></tr></thead>
              <tbody>
                <tr v-for="s in sectorPanel.stocks.slice(0, 50)" :key="s.code"
                  @click="openStockByCode(s.code)" style="cursor:pointer">
                  <td class="td-mono">{{ s.code }}</td><td>{{ s.name || '--' }}</td>
                  <td>{{ s.price == null ? '--' : Number(s.price).toFixed(2) }}</td>
                  <td :class="s.pct_chg == null ? '' : s.pct_chg > 0 ? 'profit-up' : s.pct_chg < 0 ? 'profit-down' : ''">{{ s.pct_chg == null || !Number.isFinite(Number(s.pct_chg)) ? '--' : `${Number(s.pct_chg) > 0 ? '+' : ''}${Number(s.pct_chg).toFixed(2)}%` }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div v-else class="empty">{{ sectorPanel.membershipMeta?.status === 'unavailable' ? '行业成员快照不可用，当前无法载入该板块成分' : '该快照中没有成员数据' }}</div>
        </div>
        <div v-show="sectorTab === 'sr'" class="stock-tab-content">
          <LoadingSpinner v-if="sectorPanel.srLoading" />
          <div v-else-if="sectorPanel.srMeta && sectorPanel.srMeta.membership_status !== 'available'" class="error">
            板块成员快照{{ sectorPanel.srMeta.membership_status === 'stale' ? '已过期' : '不可用' }}：{{ sectorPanel.srMeta?.missing_reason || 'unknown' }}
          </div>
          <div v-else-if="sectorPanel.srScan?.length">
            <div class="empty">成员快照 {{ sectorPanel.srMeta?.snapshot_id || sectorPanel.snapshotId }}；扫描候选 {{ sectorPanel.srMeta?.candidates ?? '--' }} 只，返回 {{ sectorPanel.srMeta?.returned ?? sectorPanel.srScan.length }} 条</div>
            <div v-for="(item, i) in sectorPanel.srScan" :key="i" class="sr-card">
              <div class="sr-header">
                <span class="sr-code">{{ item.code }}</span>
                <span class="sr-name">{{ item.name }}</span>
                <span :class="['sr-type', item.support_or_resistance === 'support' ? 'signal-buy' : 'signal-sell']">{{ item.support_or_resistance === 'support' ? '支撑' : '压力' }}</span>
              </div>
              <div class="sr-detail">
                <span>信号价 {{ Number(item.signal_price || 0).toFixed(2) }}</span>
                <span>现价 {{ Number(item.price || 0).toFixed(2) }}</span>
                <span>距离 {{ Number(item.distance_pct || 0).toFixed(2) }}%</span>
              </div>
            </div>
          </div>
          <div v-else class="empty">暂无支撑/压力信号</div>
        </div>
        <div v-show="sectorTab === 'dense'" class="stock-tab-content">
          <LoadingSpinner v-if="sectorPanel.dzLoading" />
          <div v-else-if="sectorPanel.denseMeta && sectorPanel.denseMeta.membership_status !== 'available'" class="error">
            板块成员快照{{ sectorPanel.denseMeta.membership_status === 'stale' ? '已过期' : '不可用' }}：{{ sectorPanel.denseMeta?.missing_reason || 'unknown' }}
          </div>
          <div v-else-if="sectorPanel.denseZones?.stocks?.length">
            <div class="empty">成员快照 {{ sectorPanel.denseMeta?.snapshot_id || sectorPanel.snapshotId }}；成功分析 {{ sectorPanel.denseMeta?.scanned ?? sectorPanel.denseZones.scanned }} / {{ sectorPanel.denseMeta?.candidates ?? sectorPanel.denseZones.total }} 只候选（成员共 {{ sectorPanel.denseMeta?.membership_count ?? '--' }} 只）</div>
            <div class="dz-summary" v-if="sectorPanel.denseZones.summary">
              <span class="ind-cell">支撑 {{ ((sectorPanel.denseZones.summary.support_ratio || 0) * 100).toFixed(0) }}%</span>
              <span class="ind-cell">压力 {{ ((sectorPanel.denseZones.summary.resist_ratio || 0) * 100).toFixed(0) }}%</span>
              <span :class="['ind-cell', sectorPanel.denseZones.summary.bias === 'bullish' ? 'profit-up' : 'profit-down']">偏向 {{ sectorPanel.denseZones.summary.bias === 'bullish' ? '多头' : '空头' }}</span>
            </div>
            <div v-for="(item, i) in sectorPanel.denseZones.stocks.slice(0, 30)" :key="i" class="sr-card">
              <div class="sr-header">
                <span class="sr-code">{{ item.code }}</span>
                <span class="sr-name">{{ item.name }}</span>
                <span :class="item.bias === 'bullish' ? 'profit-up' : 'profit-down'">{{ item.bias === 'bullish' ? '多头' : '空头' }}</span>
              </div>
            </div>
          </div>
          <div v-else class="empty">暂无数据</div>
        </div>
        <div v-show="sectorTab === 'sent'" class="stock-tab-content">
          <LoadingSpinner v-if="sectorPanel.sentLoading" />
          <div v-else-if="sectorPanel.sectorSent?.length">
            <div v-if="sectorPanel.sentimentMeta?.status !== 'available'" class="error">
              行业情绪样本{{ sectorPanel.sentimentMeta?.status === 'stale' ? '已过期' : '覆盖不完整或不可用' }}：
              {{ sectorPanel.sentimentMeta?.missing_reason || 'unknown' }}
            </div>
            <div class="empty">成分股情绪样本{{ sectorPanel.sectorSent[0]?.latest_trade_date ? `（截至 ${sectorPanel.sectorSent[0].latest_trade_date}）` : '' }}</div>
            <div v-for="(item, i) in sectorPanel.sectorSent.slice(0, 30)" :key="i" class="ind-cell" style="justify-content:space-between">
              <span>{{ item.code || item.sector_name }}</span>
              <span :class="item.score == null ? '' : item.score > 0.05 ? 'profit-up' : item.score < -0.05 ? 'profit-down' : ''">{{ item.score == null ? '--' : Number(item.score).toFixed(4) }}</span>
            </div>
          </div>
          <div v-else class="empty">{{ sectorPanel.sentimentMeta?.missing_reason === 'no_matching_sector_sentiment' ? '所选行业暂无可用情绪样本' : '所选行业情绪数据不可用' }}</div>
        </div>
      </template>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onActivated, onUnmounted, nextTick, shallowRef } from 'vue'
import { useRouter } from 'vue-router'
import api from '../api/index.js'
import LoadingSpinner from '../components/LoadingSpinner.vue'
import MarketPulse from '../components/MarketPulse.vue'
import MarketOverview from '../components/market/MarketOverview.vue'
import MarketIndices from '../components/market/MarketIndices.vue'
import MarketSectors from '../components/market/MarketSectors.vue'
import MarketSentiment from '../components/market/MarketSentiment.vue'
import MarketBreadth from '../components/market/MarketBreadth.vue'
import MarketMacro from '../components/market/MarketMacro.vue'
import RetailHeatPanel from '../components/RetailHeatPanel.vue'
import { useChart } from '../composables/useChart.js'
import { predictTiming, getBreadth } from '../api/timing.js'
import { cssVar, fmtVol, fmtM } from '../utils/marketFormat.js'

const router = useRouter()
const loading = ref(true)
const error = ref('')
const indices = ref({})
const indexMeta = ref(null)
const gainers = ref([])
const losers = ref([])
const sectors = ref([])
const sectorMeta = ref(null)
const macro = ref(null)
const regime = ref(null)
const timingSignal = ref(null)
const breadthData = ref(null)
const sseChartData = ref([])
const sentOverview = ref({ today: {}, trend: [], news: [] })

const tabs = [
  { key: 'overview', label: '概览' },
  { key: 'indices', label: '指数' },
  { key: 'sectors', label: '板块' },
  { key: 'retail', label: '散户资金' },
  { key: 'sentiment', label: '情绪' },
  { key: 'breadth', label: '涨跌榜' },
  { key: 'macro', label: '宏观' },
]
const activeTab = ref('overview')

const riskLabel = computed(() => {
  if (!macro.value) return ''
  if (macro.value.risk_level === 'danger') return '宏观风险高'
  if (macro.value.risk_level === 'warning') return '宏观风险偏高'
  return ''
})

const pendingSignalsCount = ref(0)
async function fetchPendingSignals() {
  try {
    const r = await api.get('/rebalance/signals/pending')
    pendingSignalsCount.value = Array.isArray(r?.data) ? r.data.length : (Array.isArray(r) ? r.length : 0)
  } catch (e) { /* silent */ }
}

// ═══ 个股侧边面板 ═══
const stockTab = ref('kline')
const stockTabs = [
  { key: 'kline', label: 'K线' },
  { key: 'indicators', label: '技术指标' },
  { key: 'ai', label: 'AI分析' },
  { key: 'financial', label: '财务' },
  { key: 'news', label: '新闻' },
  { key: 'qlib', label: '量化预测' },
]
const stockPanel = ref({
  visible: false, loading: false, code: '', name: '',
  info: null, realtime: null,
  indicators: null, indicatorsLoading: false,
  financial: null, finLoading: false,
  news: null, newsLoading: false,
  qlibPred: null, qlibLoading: false,
  aiContent: '', aiBlocks: shallowRef([]), aiStreaming: false,
})

function openStockByCode(code) {
  if (!code) return
  closeSectorPanel()
  router.push(`/stock/${code}`)
}

function closeStock() {
  if (stockPanel.value._aiAborter) {
    try { stockPanel.value._aiAborter.abort() } catch {}
    stockPanel.value._aiAborter = null
  }
  stockPanel.value.aiStreaming = false
  stockPanel.value.visible = false
}

async function switchStockTab(key) {
  stockTab.value = key
  loadStockTab(key)
}

function loadStockTab(key) {
  const code = stockPanel.value.code
  if (key === 'kline') loadKline(code)
  else if (key === 'indicators') loadIndicators(code)
  else if (key === 'financial') loadFinancial(code)
  else if (key === 'news') loadNews(code)
  else if (key === 'qlib') loadQlibPred(code)
}

const { chartRef: stockChartRef, render: renderStock } = useChart()

async function loadKline(code) {
  try {
    const res = await api.get(`/stock/${code}/daily`)
    const data = res.data || []
    if (!data.length) return
    const dates = data.map(d => String(d.trade_date || d.date || '').slice(5, 10))
    await nextTick()
    const upC = cssVar('--up')
    const downC = cssVar('--down')
    const primaryC = cssVar('--primary')
    const warnC = cssVar('--warning')
    renderStock(() => ({
      tooltip: { trigger: 'axis' },
      grid: { left: '3%', right: '3%', bottom: '3%', containLabel: true },
      xAxis: { type: 'category', data: dates, axisLabel: { fontSize: 10 } },
      yAxis: { type: 'value', scale: true, axisLabel: { fontSize: 10 } },
      dataZoom: [{ type: 'inside' }, { type: 'slider', height: 20 }],
      series: [
        { type: 'candlestick', name: 'K线', data: data.map(d => [d.open, d.close, d.low, d.high]),
          itemStyle: { color: upC, color0: downC, borderColor: upC, borderColor0: downC } },
        { type: 'line', name: 'MA5', data: ma(data.map(d => d.close), 5), smooth: true, lineStyle: { color: warnC, width: 1 }, symbol: 'none' },
        { type: 'line', name: 'MA20', data: ma(data.map(d => d.close), 20), smooth: true, lineStyle: { color: primaryC, width: 1 }, symbol: 'none' },
      ],
    }))
  } catch { /* K线数据不可用 */ }
}

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

async function loadIndicators(code) {
  stockPanel.value.indicatorsLoading = true
  try {
    const res = await api.get(`/stock/${code}/indicators`)
    stockPanel.value.indicators = res.data || null
  } catch { stockPanel.value.indicators = null }
  stockPanel.value.indicatorsLoading = false
}

async function loadFinancial(code) {
  stockPanel.value.finLoading = true
  try {
    const res = await api.get(`/stock/${code}/financial`)
    stockPanel.value.financial = res.data || null
  } catch { stockPanel.value.financial = null }
  stockPanel.value.finLoading = false
}

async function loadNews(code) {
  stockPanel.value.newsLoading = true
  try {
    const res = await api.get(`/news/${code}?limit=50`)
    stockPanel.value.news = res.data || []
  } catch { stockPanel.value.news = [] }
  stockPanel.value.newsLoading = false
}

async function loadQlibPred(code) {
  stockPanel.value.qlibLoading = true
  try {
    const res = await api.get(`/qlib/predictions/${code}`)
    stockPanel.value.qlibPred = res.data || null
  } catch { stockPanel.value.qlibPred = null }
  stockPanel.value.qlibLoading = false
}

async function runAiAnalysis(mode) {
  stockPanel.value.aiContent = ''
  stockPanel.value.aiBlocks = []
  stockPanel.value.aiStreaming = true
  const blocks = []
  let aborter = null
  try {
    aborter = new AbortController()
    stockPanel.value._aiAborter = aborter
    const resp = await fetch(`/api/stock/${encodeURIComponent(stockPanel.value.code)}/analyze/stream?mode=${encodeURIComponent(mode)}`, {
      method: 'POST',
      signal: aborter.signal,
    })
    const reader = resp.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    let currentAgent = ''
    let currentText = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop() || ''
      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed.startsWith('data: ')) continue
        const payload = trimmed.slice(6)
        if (payload.startsWith('[AGENT:')) {
          if (currentAgent) blocks.push({ type: 'agent', agent: currentAgent, text: currentText })
          currentAgent = payload.slice(7, -1)
          currentText = ''
          blocks.push({ type: 'status', text: `分析中: ${currentAgent}` })
        } else if (payload.startsWith('[END:')) {
          if (currentAgent && currentText) {
            const idx = blocks.findIndex(b => b.type === 'status' && b.text.includes(currentAgent))
            if (idx >= 0) blocks.splice(idx, 1)
            blocks.push({ type: 'agent', agent: currentAgent, text: currentText })
          }
          currentAgent = ''
          currentText = ''
        } else if (payload.startsWith('[STATUS]')) {
          blocks.push({ type: 'status', text: payload.slice(8) })
        } else if (payload.startsWith('[ERROR]')) {
          blocks.push({ type: 'error', text: payload.slice(7) })
        } else if (payload.startsWith('[DONE]')) {
          // 完成
        } else {
          currentText += payload.replace(/\r/g, '\n')
        }
      }
      stockPanel.value.aiBlocks = [...blocks]
    }
    if (currentAgent && currentText) {
      const idx = blocks.findIndex(b => b.type === 'status' && b.text.includes(currentAgent))
      if (idx >= 0) blocks.splice(idx, 1)
      blocks.push({ type: 'agent', agent: currentAgent, text: currentText })
    }
    stockPanel.value.aiBlocks = [...blocks]
  } catch (e) {
    if (e.name === 'AbortError') return
    blocks.push({ type: 'error', text: `连接失败: ${e.message}` })
    stockPanel.value.aiBlocks = [...blocks]
  }
  stockPanel.value.aiStreaming = false
}

// ═══ 板块侧边面板 ═══
const sectorTab = ref('stocks')
const sectorTabs = [
  { key: 'stocks', label: '成分股' },
  { key: 'sr', label: '支撑/压力' },
  { key: 'dense', label: '密集成交区' },
  { key: 'sent', label: '行业情绪' },
]
const sectorPanel = ref({
  visible: false, loading: false, name: '', snapshotId: null,
  stocks: null, stocksMeta: null, membershipMeta: null, srScan: null, srMeta: null, srLoading: false,
  denseZones: null, denseMeta: null, dzLoading: false,
  sectorSent: null, sentimentMeta: null, sentLoading: false,
})
let sectorPanelRequestSeq = 0
let sectorTabRequestSeq = 0

function closeSectorPanel() {
  sectorPanelRequestSeq += 1
  sectorTabRequestSeq += 1
  sectorPanel.value.visible = false
  sectorPanel.value.loading = false
}

async function openSectorDetail(sector) {
  const requestSeq = ++sectorPanelRequestSeq
  sectorTabRequestSeq += 1
  const name = sector?.sector_name || sector?.['板块名称'] || sector?.name || ''
  const snapshotId = sector?.snapshot_id || null
  sectorPanel.value = { visible: true, loading: true, name, snapshotId, stocks: null, stocksMeta: null,
    membershipMeta: null, srScan: null, srMeta: null, srLoading: false,
    denseZones: null, denseMeta: null, dzLoading: false, sectorSent: null, sentimentMeta: null, sentLoading: false }
  sectorTab.value = 'stocks'
  try {
    const query = snapshotId ? `?snapshot_id=${encodeURIComponent(snapshotId)}` : ''
    const res = await api.get(`/sectors/${encodeURIComponent(name)}/stocks${query}`)
    if (requestSeq !== sectorPanelRequestSeq) return
    sectorPanel.value.stocks = res.data || []
    sectorPanel.value.stocksMeta = res.meta || null
    sectorPanel.value.membershipMeta = res.meta?.membership_envelope || null
  } catch (error) {
    if (requestSeq !== sectorPanelRequestSeq) return
    sectorPanel.value.stocks = []
    sectorPanel.value.stocksMeta = { data_envelope: { status: 'unavailable', missing_reason: error?.response?.data?.error || 'sector_request_failed' } }
    sectorPanel.value.membershipMeta = { status: 'unavailable', missing_reason: 'industry_snapshot_unavailable' }
  }
  if (requestSeq === sectorPanelRequestSeq) sectorPanel.value.loading = false
}

async function loadSectorTab(key) {
  const requestSeq = ++sectorTabRequestSeq
  const snapshotId = sectorPanel.value.snapshotId
  const name = sectorPanel.value.name
  if (key === 'stocks' && !sectorPanel.value.stocks) {
    try {
      const query = sectorPanel.value.snapshotId ? `?snapshot_id=${encodeURIComponent(sectorPanel.value.snapshotId)}` : ''
      const res = await api.get(`/sectors/${encodeURIComponent(name)}/stocks${query}`)
      if (requestSeq !== sectorTabRequestSeq || snapshotId !== sectorPanel.value.snapshotId) return
      sectorPanel.value.stocks = res.data || []
      sectorPanel.value.stocksMeta = res.meta || null
      sectorPanel.value.membershipMeta = res.meta?.membership_envelope || null
    } catch (error) {
      if (requestSeq !== sectorTabRequestSeq || snapshotId !== sectorPanel.value.snapshotId) return
      sectorPanel.value.stocks = []
      sectorPanel.value.stocksMeta = { data_envelope: { status: 'unavailable', missing_reason: error?.response?.data?.error || 'sector_request_failed' } }
      sectorPanel.value.membershipMeta = { status: 'unavailable', missing_reason: 'industry_snapshot_unavailable' }
    }
  } else if (key === 'sr') {
    sectorPanel.value.srLoading = true
    try {
      const query = sectorPanel.value.snapshotId ? `?snapshot_id=${encodeURIComponent(sectorPanel.value.snapshotId)}` : ''
      const res = await api.get(`/sectors/${encodeURIComponent(name)}/support-resistance${query}`)
      if (requestSeq !== sectorTabRequestSeq || snapshotId !== sectorPanel.value.snapshotId) return
      sectorPanel.value.srScan = res.data || []
      sectorPanel.value.srMeta = res.meta || null
    } catch (error) {
      if (requestSeq !== sectorTabRequestSeq || snapshotId !== sectorPanel.value.snapshotId) return
      sectorPanel.value.srScan = []
      sectorPanel.value.srMeta = { ...error?.response?.data?.meta,
        membership_status: error?.response?.data?.meta?.membership_status || 'unavailable',
        missing_reason: error?.response?.data?.error || 'sector_scan_unavailable' }
    }
    if (requestSeq === sectorTabRequestSeq && snapshotId === sectorPanel.value.snapshotId) sectorPanel.value.srLoading = false
  } else if (key === 'dense') {
    sectorPanel.value.dzLoading = true
    try {
      const query = sectorPanel.value.snapshotId ? `?snapshot_id=${encodeURIComponent(sectorPanel.value.snapshotId)}&max_stocks=80` : ''
      const res = await api.get(`/sectors/${encodeURIComponent(name)}/dense-zones${query}`)
      if (requestSeq !== sectorTabRequestSeq || snapshotId !== sectorPanel.value.snapshotId) return
      sectorPanel.value.denseZones = res.data || {}
      sectorPanel.value.denseMeta = res.meta || null
    } catch (error) {
      if (requestSeq !== sectorTabRequestSeq || snapshotId !== sectorPanel.value.snapshotId) return
      sectorPanel.value.denseZones = {}
      sectorPanel.value.denseMeta = { ...error?.response?.data?.meta,
        membership_status: error?.response?.data?.meta?.membership_status || 'unavailable',
        missing_reason: error?.response?.data?.error || 'sector_scan_unavailable' }
    }
    if (requestSeq === sectorTabRequestSeq && snapshotId === sectorPanel.value.snapshotId) sectorPanel.value.dzLoading = false
  } else if (key === 'sent' && !sectorPanel.value.sectorSent) {
    sectorPanel.value.sentLoading = true
    try {
      const res = await api.get(`/sentiment/sectors/rank?sector_name=${encodeURIComponent(name)}&sector_type=industry&snapshot_id=${encodeURIComponent(sectorPanel.value.snapshotId || '')}`)
      if (requestSeq !== sectorTabRequestSeq || snapshotId !== sectorPanel.value.snapshotId) return
      sectorPanel.value.sectorSent = res.data || []
      sectorPanel.value.sentimentMeta = res.meta || null
    } catch (error) {
      if (requestSeq !== sectorTabRequestSeq || snapshotId !== sectorPanel.value.snapshotId) return
      sectorPanel.value.sectorSent = []
      sectorPanel.value.sentimentMeta = { status: 'unavailable', missing_reason: error?.response?.data?.detail || 'sector_sentiment_request_failed' }
    }
    if (requestSeq === sectorTabRequestSeq && snapshotId === sectorPanel.value.snapshotId) sectorPanel.value.sentLoading = false
  }
}

function sentTag(score) {
  if (score == null) return ''
  if (score > 0.05) return 'tag-low'
  if (score < -0.05) return 'tag-high'
  return 'tag-mid'
}
function sentLabel(score) {
  if (score == null) return '--'
  if (score > 0.05) return '积极'
  if (score < -0.05) return '消极'
  return '中性'
}

async function loadAllData() {
  const pageSeq = ++pageRequestSeq
  const marketSeq = ++marketRequestSeq
  const results = await Promise.allSettled([
    api.get('/market/indices'),
    api.get('/market/top-gainers?n=20'),
    api.get('/market/top-losers?n=20'),
    api.get('/market/sectors'),
    api.get('/market/index-chart?index_name=' + encodeURIComponent('上证指数') + '&days=120'),
    api.get('/sentiment/market/overview?days=30'),
    api.get('/macro/overview'),
    api.get('/regime/current'),
  ])
  if (pageSeq !== pageRequestSeq) return
  if (marketSeq === marketRequestSeq) applyMarketSnapshot(results.slice(0, 4), marketSeq)
  const [chart, sentOv, macroOv, regOv] = results.slice(4)
  sseChartData.value = chart.status === 'fulfilled' && Array.isArray(chart.value.data) ? chart.value.data : []
  sentOverview.value = sentOv.status === 'fulfilled' ? sentOv.value.data || { today: {}, trend: [], news: [] } : { today: {}, trend: [], news: [] }
  macro.value = macroOv.status === 'fulfilled' ? macroOv.value.data || macroOv.value : null
  regime.value = regOv.status === 'fulfilled' ? regOv.value?.data || regOv.value : null

  const [sig, br] = await Promise.allSettled([predictTiming(), getBreadth(), fetchPendingSignals()])
  if (pageSeq === pageRequestSeq) {
    if (sig.status === 'fulfilled') timingSignal.value = sig.value?.data
    if (br.status === 'fulfilled') breadthData.value = br.value?.data
    loading.value = false
  }
}

let liveTimer = null
let liveLoading = false
let marketRequestSeq = 0
let pageRequestSeq = 0

function applyMarketSnapshot(results, seq) {
  if (seq !== marketRequestSeq) return
  const [idx, gain, lose, sec] = results
  indices.value = idx.status === 'fulfilled' ? idx.value.data || {} : {}
  indexMeta.value = idx.status === 'fulfilled' ? idx.value.meta || null : { freshness: 'unavailable' }
  gainers.value = gain.status === 'fulfilled' ? gain.value.data || [] : []
  losers.value = lose.status === 'fulfilled' ? lose.value.data || [] : []
  sectors.value = sec.status === 'fulfilled' ? sec.value.data || [] : []
  sectorMeta.value = sec.status === 'fulfilled'
    ? sec.value.meta || null
    : { data_envelope: { status: 'unavailable', missing_reason: 'sector_request_failed' } }
  const failed = results.filter(result => result.status === 'rejected').length
  error.value = failed ? `行情数据有${failed}项暂不可用` : ''
}

async function loadLiveData() {
  if (liveLoading) return
  liveLoading = true
  const seq = ++marketRequestSeq
  try {
    const results = await Promise.allSettled([
      api.get('/market/indices'),
      api.get('/market/top-gainers?n=20'),
      api.get('/market/top-losers?n=20'),
      api.get('/market/sectors'),
    ])
    applyMarketSnapshot(results, seq)
  } finally {
    liveLoading = false
  }
}

onMounted(() => {
  loadAllData()
  loadLiveData()
  liveTimer = setInterval(loadLiveData, 10000)
})
onActivated(() => loadLiveData())
onUnmounted(() => {
  marketRequestSeq += 1
  pageRequestSeq += 1
  sectorPanelRequestSeq += 1
  sectorTabRequestSeq += 1
  clearInterval(liveTimer)
})
</script>

<style scoped>
.mv-root {
  max-width: var(--content-max-width);
  margin: 0 auto;
  padding: 0 var(--content-padding);
}

/* ═══ Tab 导航 ═══ */
.mv-tabs {
  display: flex;
  gap: 4px;
  padding: 12px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 14px;
  overflow-x: auto;
}
.mv-tab {
  padding: 8px 18px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-secondary);
  font-size: var(--fs-sm);
  cursor: pointer;
  white-space: nowrap;
  transition: all var(--transition-fast);
}
.mv-tab:hover { border-color: var(--primary); color: var(--primary); }
.mv-tab.active { background: var(--primary); color: var(--on-primary); border-color: var(--primary); }

/* ═══ 侧边面板 (保留原逻辑) ═══ */
.stock-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,.4); z-index: 99; }
.stock-panel { position: fixed; top: 0; right: 0; width: 620px; max-width: 100vw; height: 100vh; background: var(--bg); z-index: 100; overflow-y: auto; padding: 20px; border-left: 1px solid var(--border); box-shadow: -4px 0 24px rgba(0,0,0,.3); }
.stock-panel-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.stock-panel-title { font-size: 20px; font-weight: 700; color: var(--text); }
.stock-panel-close { background: none; border: none; color: var(--text-muted); font-size: 22px; cursor: pointer; padding: 4px 8px; }
.stock-panel-close:hover { color: var(--text); }

.stock-quote-bar { display: flex; align-items: baseline; gap: 12px; padding: 12px 16px; background: var(--bg-secondary); border-radius: var(--radius-sm); margin-bottom: 14px; }
.sq-price { font-size: 26px; font-weight: 700; color: var(--text); font-family: var(--font-mono); }
.sq-chg { font-size: 16px; font-weight: 600; }
.sq-vol { margin-left: auto; font-size: var(--fs-sm); color: var(--text-muted); }

.stock-tabs { display: flex; gap: 4px; margin-bottom: 14px; flex-wrap: wrap; }
.stock-tab { padding: 6px 14px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: transparent; color: var(--text-secondary); font-size: var(--fs-sm); cursor: pointer; }
.stock-tab:hover { border-color: var(--primary); color: var(--primary); }
.stock-tab.active { background: var(--primary); color: var(--on-primary); border-color: var(--primary); }
.stock-tab-content { min-height: 300px; }

.ind-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; }
.ind-cell { display: flex; justify-content: space-between; padding: 8px 12px; background: var(--bg-secondary); border-radius: var(--radius-sm); font-size: var(--fs-sm); }
.ind-name { color: var(--text-muted); }
.ind-val { font-weight: 600; color: var(--text); font-family: var(--font-mono); }
.ind-section { margin-top: 14px; }
.ind-subtitle { font-size: var(--fs-sm); font-weight: 600; color: var(--text-secondary); margin-bottom: 6px; }
.ind-signal { padding: 4px 10px; margin: 3px 0; border-radius: 4px; font-size: var(--fs-sm); background: var(--bg-secondary); }
.ind-signal.signal-buy { border-left: 3px solid var(--up); }
.ind-signal.signal-sell { border-left: 3px solid var(--down); }

.ai-actions { display: flex; gap: 10px; flex-wrap: wrap; }
.ai-btn { padding: 8px 20px; border: 1px solid var(--primary); border-radius: var(--radius-sm); background: transparent; color: var(--primary); font-size: var(--fs-sm); cursor: pointer; }
.ai-btn:hover { background: var(--primary); color: var(--on-primary); }
.ai-btn-panel { border-color: var(--warning); color: var(--warning); }
.ai-btn-panel:hover { background: var(--warning); color: #fff; }
.ai-output { margin-top: 14px; max-height: 600px; overflow-y: auto; }
.ai-status { padding: 4px 0; font-size: var(--fs-sm); color: var(--text-muted); font-style: italic; }
.ai-error { padding: 6px 10px; background: rgba(239,68,68,.1); border-radius: 4px; color: var(--danger); font-size: var(--fs-sm); }
.ai-agent-block { margin: 10px 0; border: 1px solid var(--border); border-radius: var(--radius-sm); overflow: hidden; }
.ai-agent-name { padding: 6px 12px; background: var(--bg-secondary); font-size: var(--fs-sm); font-weight: 600; color: var(--primary); }
.ai-agent-text { padding: 10px 12px; margin: 0; font-size: var(--fs-sm); color: var(--text-secondary); white-space: pre-wrap; line-height: 1.7; max-height: 300px; overflow-y: auto; font-family: inherit; }

.fin-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; }
.fin-cell { display: flex; justify-content: space-between; padding: 8px 12px; background: var(--bg-secondary); border-radius: var(--radius-sm); font-size: var(--fs-sm); }
.fin-name { color: var(--text-muted); }
.fin-val { font-weight: 600; color: var(--text); font-family: var(--font-mono); }

.qlib-score { display: flex; align-items: baseline; gap: 14px; padding: 20px; background: var(--bg-secondary); border-radius: var(--radius-sm); }
.qlib-label { font-size: var(--fs-sm); color: var(--text-muted); }
.qlib-val { font-size: 28px; font-weight: 700; font-family: var(--font-mono); }
.qlib-meta { margin-top: 8px; padding: 0 12px; font-size: var(--fs-caption); color: var(--text-muted); }

.mv-news-row { display: flex; align-items: center; gap: 8px; padding: 6px 0; border-bottom: 1px solid rgba(42, 42, 80, .25); font-size: var(--fs-sm); }
.mv-news-row:last-child { border-bottom: none; }
.mv-news-title { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-secondary); }
.mv-news-date { color: var(--text-muted); font-size: var(--fs-caption); flex-shrink: 0; }

.sr-card { padding: 10px 12px; margin: 6px 0; background: var(--bg-secondary); border-radius: var(--radius-sm); font-size: var(--fs-sm); }
.sr-header { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
.sr-code { font-family: var(--font-mono); color: var(--primary); font-weight: 600; }
.sr-name { color: var(--text-secondary); flex: 1; }
.sr-type { padding: 2px 8px; border-radius: 4px; font-size: var(--fs-caption); font-weight: 600; }
.sr-type.signal-buy { background: rgba(34,197,94,.12); color: var(--up); }
.sr-type.signal-sell { background: rgba(239,68,68,.12); color: var(--down); }
.sr-detail { display: flex; gap: 16px; color: var(--text-muted); font-size: var(--fs-caption); }
.dz-summary { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
.mini-table { width: 100%; border-collapse: collapse; font-size: var(--fs-sm); }
.mini-table th { text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--border); color: var(--text-muted); font-size: var(--fs-caption); font-weight: 600; }
.mini-table td { padding: 5px 8px; border-bottom: 1px solid rgba(42,42,80,.3); color: var(--text-secondary); }
.td-mono { font-family: var(--font-mono); }

.empty { text-align: center; padding: 20px; color: var(--text-muted); font-size: var(--fs-sm); }
</style>
