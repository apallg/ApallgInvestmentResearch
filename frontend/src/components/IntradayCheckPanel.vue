<template>
  <div class="icp-root">
    <div class="icp-header" @click="expanded = !expanded">
      <div class="icp-header__left">
        <span class="icp-header__icon">{{ expanded ? '▾' : '▸' }}</span>
        <span class="icp-header__title">日内执行建议</span>
        <span v-if="buyCodes.length" class="icp-header__badge">买{{ buyCodes.length }}</span>
        <span v-if="sellSignals.length" class="icp-header__badge icp-header__badge--sell">卖{{ sellSignals.length }}</span>
        <span v-if="readyCount" class="icp-header__badge icp-header__badge--ready" role="status" aria-live="polite">条件通过 {{ readyCount }}</span>
        <span v-if="!buyCodes.length && !sellSignals.length" class="icp-header__badge icp-header__badge--empty">无调仓目标</span>
        <span v-if="source" class="icp-header__source">
          <span class="icp-dot" :class="sourceClass"></span>
          {{ sourceLabel }}
        </span>
        <span v-if="timestamp" class="icp-header__time">{{ timestamp }}</span>
        <span class="icp-header__time">{{ Math.round(refreshInterval / 1000) }}秒刷新</span>
      </div>
      <div class="icp-header__right">
        <span v-if="loading" class="icp-loading">⏳</span>
        <button v-if="(buyCodes.length || sellSignals.length) && !loading" class="icp-btn--sm" @click.stop="refresh">刷新</button>
        <span v-if="error" class="icp-error">{{ error }}</span>
      </div>
    </div>

    <div class="icp-phase">
      <span class="icp-phase__dot" :class="'icp-phase--' + phaseNow"></span>
      <span class="icp-phase__label">当前时段: {{ phaseInfo.label }}</span>
      <span class="icp-phase__rules">{{ phaseInfo.rules }}</span>
    </div>

    <div v-if="expanded" class="icp-body">
      <template v-if="buyCodes.length === 0 && sellSignals.length === 0">
        <div class="icp-empty">
          📋 当前无调仓信号。执行"完整调仓检查"后，这里会显示买入回调检测和卖出执行建议。
        </div>
      </template>

      <template v-else>
        <div v-if="readyCount" class="icp-ready-banner" role="status" aria-live="polite">
          <strong>行情与交易条件核验通过</strong>
          <span>{{ readyCodes.join('、') }} 仅为人工执行建议，不代表已下单；请通过本地委托流程再次核验</span>
        </div>

        <div v-if="sellResults.length" class="icp-sell">
          <div class="icp-sell__title">📉 卖出执行</div>
          <div v-for="item in sellResults" :key="'s' + item.code" class="icp-sell__row" :class="executionStateClass(item.execution)">
            <span class="icp-sell__code">{{ item.code }}</span>
            <span class="icp-sell__name">{{ item.name }}</span>
            <span class="icp-sell__urgency" :class="'icp-urg--' + item.urgency">{{ urgencyLabel(item.urgency) }}</span>
            <span v-if="item.pnl_pct != null" class="icp-sell__pnl" :class="item.pnl_pct >= 0 ? 'icp-up' : 'icp-down'">{{ (item.pnl_pct * 100).toFixed(1) }}%</span>
            <span v-if="item.suggested_shares" class="icp-sell__shares">调仓计划量 {{ item.suggested_shares }}股</span>
            <span v-if="item.execution" class="icp-card__exec" :class="'icp-exec--' + (item.execution.intent_action || item.execution.action)">
              {{ actionLabel(item.execution.intent_action || item.execution.action) }}
              <template v-if="item.execution.limit_price != null">@{{ item.execution.limit_price.toFixed(2) }}</template>
              <span class="icp-exec__reason">{{ item.execution.reason }}</span>
            </span>
            <span v-if="item.execution" class="icp-fill-signal" :class="'icp-fill-signal--' + item.execution.status">
              {{ item.execution.trigger_text }}
              <template v-if="item.execution.status === 'waiting' && item.execution.distance_pct != null"> · 相差 {{ (item.execution.distance_pct * 100).toFixed(2) }}%</template>
            </span>
            <span v-if="item.data_provenance" class="icp-sell__data-status">
              分钟数据 {{ item.analysis_status === 'verified' ? '已核验' : '未认证' }}
            </span>
          </div>
        </div>

        <div v-for="item in results" :key="item.code" class="icp-card" :class="executionStateClass(item.execution)">
          <div class="icp-card__summary" @click="toggleDetail(item.code)">
            <span class="icp-card__code">{{ item.code }}</span>
            <span class="icp-card__quality" :class="qualityColor(item.entry_quality_label)">
              {{ item.entry_quality_label }} {{ item.entry_quality_score?.toFixed(0) ?? '--' }}
            </span>
            <span class="icp-card__price">{{ item.current_price?.toFixed(2) }}<span v-if="item.entry_quality_label === '优秀'" class="icp-flash">💡</span></span>
            <span v-if="item.execution" class="icp-card__exec" :class="'icp-exec--' + (item.execution.intent_action || item.execution.action)">
              {{ actionLabel(item.execution.intent_action || item.execution.action) }}
              <template v-if="item.execution.limit_price != null">@{{ item.execution.limit_price.toFixed(2) }}</template>
              <span class="icp-exec__reason">{{ item.execution.reason }}</span>
            </span>
            <span v-if="item.execution" class="icp-fill-signal" :class="'icp-fill-signal--' + item.execution.status">
              {{ item.execution.trigger_text }}
              <template v-if="item.execution.status === 'waiting' && item.execution.distance_pct != null"> · 相差 {{ (item.execution.distance_pct * 100).toFixed(2) }}%</template>
            </span>
            <span v-if="item.nearest_support" class="icp-card__support">
              {{ item.nearest_support.name }} {{ item.nearest_support.price?.toFixed(2) }}
              <span :class="item.nearest_support.distance_pct >= 0 ? 'icp-up' : 'icp-down'">
                {{ item.nearest_support.distance_pct >= 0 ? '+' : '' }}{{ (item.nearest_support.distance_pct * 100).toFixed(1) }}%
              </span>
            </span>
            <span class="icp-card__risk" v-if="item.risk_warning">⚠ {{ item.risk_warning }}</span>
          </div>
          <div v-if="item.data_provenance" class="icp-card__data-status" role="status">
            分钟数据 {{ item.analysis_status === 'verified' ? '已核验' : '未认证' }} · {{ item.data_provenance.source || '来源未知' }}
          </div>

          <div v-if="detailCode === item.code" class="icp-card__detail">
            <div class="icp-detail-grid">
              <div class="icp-detail__signals">
                <strong v-if="item.signals?.length">回调信号</strong>
                <strong v-else style="color:var(--text-muted)">暂无回调信号</strong>
                <div v-for="sig in item.signals" :key="sig.type" class="icp-signal">
                  <span class="icp-signal__dot" :class="'icp-signal--s' + sig.strength"></span>
                  {{ sig.description }}
                </div>
              </div>
              <div class="icp-detail__supports">
                <strong>支撑位</strong>
                <div v-for="s in item.supports.slice(0, 5)" :key="s.name" class="icp-support-item">
                  <span>{{ s.name }}</span>
                  <span class="icp-support-item__price">{{ s.price?.toFixed(2) }}</span>
                  <span :class="s.distance_pct >= 0 ? 'icp-up' : 'icp-down'">
                    {{ s.distance_pct >= 0 ? '+' : '' }}{{ (s.distance_pct * 100).toFixed(1) }}%
                  </span>
                </div>
              </div>
            </div>
            <div v-if="item.recent_bars?.length" class="icp-chart-wrap">
              <div :ref="el => setChartRef(item.code, el)" class="icp-chart"></div>
            </div>
          </div>
        </div>

        <div v-if="unavailable.length" class="icp-unavailable">
          无日内数据: {{ unavailable.join(', ') }}
        </div>
      </template>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import * as rebalanceApi from '../api/rebalance.js'
import { blockUnverifiedMinuteExecution, hasVerifiedMinuteProvenance } from '../utils/minuteAnalysis.js'

const props = defineProps({
  buyCodes: { type: Array, default: () => [] },
  maData: { type: Object, default: null },
  sellSignals: { type: Array, default: () => [] },
  autoRefresh: { type: Boolean, default: true },
  refreshInterval: { type: Number, default: 5000 },
})

const emit = defineEmits(['update'])

const expanded = ref(false)
const loading = ref(false)
const error = ref('')
const results = ref([])
const sellResults = ref([])
const source = ref('')
const timestamp = ref('')
const unavailable = ref([])
const detailCode = ref(null)
const chartRefs = {}
let pollTimer = null
let visibilityBound = false
let requestGeneration = 0
const readyItems = computed(() => [
  ...results.value.map(item => ({ code: item.code, execution: item.execution })),
  ...sellResults.value.map(item => ({ code: item.code, execution: item.execution })),
].filter(item => item.execution?.can_execute))
const readyCount = computed(() => readyItems.value.length)
const readyCodes = computed(() => readyItems.value.map(item => item.code))

const PHASES = {
  open: { label: '早盘 9:30-10:00', rules: '高开>2%不追·挂前收价等回落·止损必市价' },
  mid: { label: '盘中', rules: '买入挂VWAP锚·止盈限价+0.2%·止损保留卖出意图' },
  close: { label: '尾盘 14:30-15:00', rules: '尾盘建议仍需通过行情、涨跌停与可卖量校验' },
  closed: { label: '非交易时段/休市', rules: '仅显示参考与待处理意图，不提供可执行资格' },
}
function computePhase() {
  const d = new Date()
  const t = d.getHours() * 60 + d.getMinutes()
  if (t >= 570 && t < 600) return 'open'
  if ((t >= 600 && t <= 690) || (t >= 780 && t < 870)) return 'mid'
  if (t >= 870 && t < 900) return 'close'
  return 'closed'
}
const phaseNow = ref(computePhase())
const phaseInfo = computed(() => PHASES[phaseNow.value] || PHASES.closed)
let phaseTimer = null

const sourceLabel = computed(() =>
  source.value === 'realtime' ? '实时行情' :
  source.value === 'ths' ? '同花顺实时' :
  source.value === 'ths_db' ? '同花顺缓存' : '日K参考')

const sourceClass = computed(() =>
  source.value === 'realtime' ? 'icp-dot--live' :
  source.value === 'ths' ? 'icp-dot--live' :
  source.value === 'ths_db' ? 'icp-dot--cache' : 'icp-dot--fallback')

function qualityColor(label) {
  return { '优秀': 'icp-q--great', '良好': 'icp-q--good', '一般': 'icp-q--ok', '观望': 'icp-q--poor', '日K参考': 'icp-q--poor' }[label] || 'icp-q--poor'
}

function actionLabel(action) {
  return { limit: '挂单', wait: '观望', market: '市价', fill: '兜底成交' }[action] || action
}

function urgencyLabel(u) {
  return { stop_loss: '止损', take_profit: '止盈', reduce: '减仓', sell: '卖出', add: '加仓', buy: '买入' }[u] || u
}

function executionStateClass(execution) {
  return execution?.status ? `icp-state--${execution.status}` : ''
}

function maskUnverifiedAnalysis(item) {
  const verified = item?.analysis_status === 'verified' && hasVerifiedMinuteProvenance(item?.data_provenance)
  if (verified) return item
  return {
    ...item,
    analysis_status: 'unverified',
    entry_quality_score: null,
    entry_quality_label: '分钟数据未认证',
    current_price: null,
    open_price: null,
    day_high: null,
    day_low: null,
    vwap: null,
    supports: [],
    nearest_support: null,
    signals: [],
    recent_bars: [],
    volume_ma: null,
    risk_warning: '分钟来源语义未核验，评分、信号、技术位和执行建议不可用',
    execution: blockUnverifiedMinuteExecution(item?.execution),
  }
}

function toggleDetail(code) {
  detailCode.value = detailCode.value === code ? null : code
  if (detailCode.value) {
    nextTick(() => renderMiniChart(code))
  }
}

function setChartRef(code, el) {
  if (el) chartRefs[code] = el
}

async function renderMiniChart(code) {
  await nextTick()
  const el = chartRefs[code]
  if (!el) return
  const item = results.value.find(r => r.code === code)
  if (!item?.recent_bars?.length) return

  const echarts = await import('echarts')
  let instance = echarts.getInstanceByDom(el)
  if (!instance) instance = echarts.init(el)
  else instance.resize()

  const bars = item.recent_bars
  const cats = bars.map(b => b.time)
  const ohlc = bars.map(b => [b.open, b.close, b.low, b.high])
  const supports = item.supports || []

  const markLines = supports.slice(0, 3).map((s, i) => ({
    name: s.name,
    yAxis: s.price,
    lineStyle: { color: ['#ef4444', '#f59e0b', '#8b5cf6'][i] || '#999', type: 'dashed', width: 1 },
    label: { formatter: s.name, fontSize: 10 }
  }))

  instance.setOption({
    grid: { left: 8, right: 8, top: 8, bottom: 4 },
    xAxis: { type: 'category', data: cats, axisLabel: { show: false }, axisTick: { show: false } },
    yAxis: { type: 'value', scale: true, axisLabel: { fontSize: 9 }, splitLine: { lineStyle: { color: '#eee', type: 'dashed' } } },
    series: [{
      type: 'candlestick',
      data: ohlc,
      itemStyle: { color: '#ef4444', color0: '#22c55e', borderColor: '#ef4444', borderColor0: '#22c55e' },
      markLine: { silent: true, symbol: 'none', data: markLines },
    }],
  }, { notMerge: true })
}

async function refresh() {
  const generation = ++requestGeneration
  if (!props.buyCodes.length && !props.sellSignals.length) {
    loading.value = false
    error.value = ''
    results.value = []
    sellResults.value = []
    source.value = ''
    timestamp.value = ''
    unavailable.value = []
    detailCode.value = null
    emit('update', { results: [], sell_results: [], timestamp: '', source: '', unavailable: [] })
    return
  }

  const buyCodes = [...props.buyCodes]
  const maData = props.maData ? { ...props.maData } : null
  const sellSignals = props.sellSignals.map(item => ({ ...item }))
  loading.value = true; error.value = ''
  results.value = []
  sellResults.value = []
  source.value = ''
  timestamp.value = ''
  unavailable.value = []
  detailCode.value = null
  emit('update', { results: [], sell_results: [], timestamp: '', source: '', unavailable: [] })
  try {
    const res = await rebalanceApi.checkIntraday(buyCodes, maData, sellSignals)
    if (generation !== requestGeneration) return
    const data = res?.data || res
    results.value = (Array.isArray(data.results) ? data.results : []).map(maskUnverifiedAnalysis)
    sellResults.value = (Array.isArray(data.sell_results) ? data.sell_results : []).map(maskUnverifiedAnalysis)
    source.value = data.source || ''
    timestamp.value = data.timestamp || ''
    phaseNow.value = data.market_phase || computePhase()
    unavailable.value = data.unavailable || []
    if ([...results.value, ...sellResults.value].some(item => item.execution?.can_execute)) expanded.value = true
    emit('update', data)
  } catch (e) {
    if (generation === requestGeneration) error.value = e.message || String(e)
  } finally {
    if (generation === requestGeneration) loading.value = false
  }
}

function onVisibilityChange() {
  if (document.hidden) {
    stopPolling()
  } else if (props.autoRefresh && expanded.value) {
    startPolling()
  }
}

function startPolling() {
  stopPolling()
  if (!props.autoRefresh || document.hidden || !expanded.value) return
  pollTimer = setTimeout(async () => {
    await refresh()
    startPolling()
  }, Math.max(3000, props.refreshInterval))
}

function stopPolling() {
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null }
}

watch(() => [props.buyCodes, props.sellSignals], async ([codes, sells]) => {
  if (codes?.length || sells?.length) {
    if (!expanded.value) expanded.value = true
    await refresh()
    startPolling()
  } else {
    await refresh()
    stopPolling()
  }
}, { deep: true })

watch(expanded, (v) => {
  if (props.autoRefresh) {
    if (v) startPolling()
    else stopPolling()
  }
})

onMounted(async () => {
  if (props.buyCodes.length || props.sellSignals.length) {
    expanded.value = true
    await refresh()
    startPolling()
  }
  if (!visibilityBound) {
    document.addEventListener('visibilitychange', onVisibilityChange)
    visibilityBound = true
  }
  phaseTimer = setInterval(() => { phaseNow.value = computePhase() }, 60000)
})

onUnmounted(() => {
  requestGeneration += 1
  stopPolling()
  if (phaseTimer) { clearInterval(phaseTimer); phaseTimer = null }
  if (visibilityBound) {
    document.removeEventListener('visibilitychange', onVisibilityChange)
    visibilityBound = false
  }
  import('echarts').then(({ default: echarts }) => {
    for (const el of Object.values(chartRefs)) {
      if (el) echarts.getInstanceByDom(el)?.dispose()
    }
  })
})
</script>

<style scoped>
.icp-root {
  margin: 4px 0 12px;
  border: 1px solid var(--border, #1e1e3a);
  border-radius: 8px;
  overflow: hidden;
  background: var(--bg, #08080f);
  display: flex;
  flex-direction: column;
  max-height: 38vh;
  flex-shrink: 0;
}
.icp-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  background: var(--bg-secondary, #0e0e1a);
  cursor: pointer;
  user-select: none;
  font-size: 13px;
  color: var(--text, #e4e4f0);
}
.icp-header__left { display: flex; align-items: center; gap: 8px; }
.icp-header__icon { font-size: 11px; color: var(--text-secondary, #8888a6); width: 12px; }
.icp-header__title { font-weight: 600; }
.icp-header__badge { background: var(--primary, #6b9fd4); color: #fff; font-size: 11px; padding: 0 6px; border-radius: 4px; }
.icp-header__badge--empty { background: var(--text-secondary, #8888a6); }
.icp-header__badge--ready { background: var(--warning, #f4a742); color: var(--bg, #08080f); animation: readyPulse 1.2s ease-in-out infinite; }
.icp-empty { padding: 16px 12px; color: var(--text-secondary, #8888a6); font-size: 13px; text-align: center; }
.icp-header__source { font-size: 11px; display: flex; align-items: center; gap: 4px; }
.icp-header__time { font-size: 11px; color: var(--text-secondary, #8888a6); }
.icp-header__right { display: flex; align-items: center; gap: 8px; }
.icp-loading { animation: spin 1s linear infinite; display: inline-block; }
@keyframes spin { to { transform: rotate(360deg); } }
.icp-error { color: var(--up, #ef4444); font-size: 11px; }

.icp-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
.icp-dot--live { background: #22c55e; }
.icp-dot--cache { background: #f4a742; }
.icp-dot--fallback { background: #555572; }

.icp-btn--sm {
  font-size: 11px; padding: 2px 8px; border: 1px solid var(--border, #2a2a4a);
  border-radius: 4px; background: var(--bg-elevated, #16162b);
  color: var(--text-secondary, #8888a6); cursor: pointer;
}
.icp-btn--sm:hover { background: var(--card-hover, #181830); color: var(--text, #e4e4f0); }

.icp-phase {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  padding: 6px 12px; background: var(--bg-secondary, #0e0e1a);
  border-bottom: 1px solid var(--border, #1e1e3a);
  font-size: 12px; color: var(--text-secondary, #8888a6);
}
.icp-phase__dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.icp-phase--open { background: #f4a742; box-shadow: 0 0 6px rgba(244,167,66,.6); }
.icp-phase--mid { background: var(--primary, #6b9fd4); box-shadow: 0 0 6px rgba(107,159,212,.6); }
.icp-phase--close { background: var(--up, #ef4444); box-shadow: 0 0 6px rgba(239,68,68,.6); }
.icp-phase--closed { background: var(--text-secondary, #8888a6); }
.icp-phase__label { font-weight: 600; color: var(--text, #e4e4f0); }
.icp-phase__rules { color: var(--text-secondary, #8888a6); }

.icp-body { padding: 10px 14px; overflow-y: auto; min-height: 0; }
.icp-card {
  border: 1px solid var(--border, #1e1e3a);
  border-radius: 6px;
  margin-bottom: 8px;
  overflow: hidden;
  background: var(--card-bg, #121224);
}
.icp-state--ready { border-color: var(--warning, #f4a742); box-shadow: inset 3px 0 0 var(--warning, #f4a742); }
.icp-ready-banner {
  display: flex; align-items: center; gap: 10px; margin-bottom: 10px; padding: 10px 12px;
  border: 1px solid rgba(244,167,66,.55); border-radius: 6px; background: rgba(244,167,66,.12);
  color: var(--text, #e4e4f0); font-size: 13px;
}
.icp-ready-banner strong { color: var(--warning, #f4a742); white-space: nowrap; }
.icp-fill-signal { font-size: 12px; font-weight: 600; }
.icp-fill-signal--ready { color: var(--warning, #f4a742); }
.icp-fill-signal--waiting { color: var(--primary, #6b9fd4); }
.icp-fill-signal--blocked { color: var(--text-secondary, #8888a6); }
@keyframes readyPulse { 0%, 100% { opacity: 1; } 50% { opacity: .65; } }
.icp-card__summary {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 12px; cursor: pointer;
  font-size: 14px;
  color: var(--text, #e4e4f0);
}
.icp-card__summary:hover { background: var(--card-hover, #181830); }
.icp-card__code { font-weight: 600; min-width: 60px; }
.icp-card__quality { font-size: 13px; padding: 2px 8px; border-radius: 4px; font-weight: 600; min-width: 60px; text-align: center; }
.icp-q--great { background: var(--up-muted, rgba(239,68,68,.12)); color: var(--up, #ef4444); border: 1px solid rgba(239, 68, 68, .35); }
.icp-q--good { background: rgba(244, 167, 66, .12); color: var(--warning, #f4a742); border: 1px solid rgba(244, 167, 66, .35); }
.icp-q--ok { background: var(--down-muted, rgba(34,197,94,.12)); color: var(--down, #22c55e); border: 1px solid rgba(34, 197, 94, .35); }
.icp-q--poor { background: rgba(136, 136, 166, .12); color: var(--text-secondary, #8888a6); border: 1px solid var(--border-light, #2a2a4a); }
.icp-card__price { font-weight: 500; font-size: 14px; }
.icp-card__exec { font-size: 13px; font-weight: 600; padding: 3px 10px; border-radius: 4px; display: inline-flex; align-items: center; gap: 4px; }
.icp-exec--limit { background: var(--primary-muted, rgba(107,159,212,.12)); color: var(--primary, #6b9fd4); border: 1px solid rgba(107, 159, 212, .35); }
.icp-exec--wait { background: rgba(136, 136, 166, .12); color: var(--text-secondary, #8888a6); border: 1px solid var(--border-light, #2a2a4a); }
.icp-exec--market { background: var(--up-muted, rgba(239,68,68,.12)); color: var(--up, #ef4444); border: 1px solid rgba(239, 68, 68, .35); }
.icp-exec--fill { background: rgba(244, 167, 66, .12); color: var(--warning, #f4a742); border: 1px solid rgba(244, 167, 66, .35); }
.icp-exec__reason { font-weight: 400; color: var(--text-secondary, #8888a6); }
.icp-flash { animation: flash 1.5s infinite; }
@keyframes flash { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
.icp-card__support { font-size: 12px; color: var(--text-secondary, #8888a6); }
.icp-card__risk { font-size: 11px; color: var(--down, #22c55e); }
.icp-card__data-status { padding: 0 12px 8px; font-size: 11px; color: var(--text-muted, #8888a6); }
.icp-sell__data-status { font-size: 11px; color: var(--text-muted, #8888a6); }
.icp-up { color: var(--up, #ef4444); }
.icp-down { color: var(--down, #22c55e); }

.icp-card__detail { border-top: 1px solid var(--border, #1e1e3a); padding: 8px 10px; color: var(--text, #e4e4f0); }
.icp-detail-grid { display: flex; gap: 20px; margin-bottom: 6px; font-size: 12px; }
.icp-detail__signals, .icp-detail__supports { flex: 1; }
.icp-detail__signals strong, .icp-detail__supports strong { display: block; margin-bottom: 4px; color: var(--text-secondary, #8888a6); }
.icp-signal { display: flex; align-items: center; gap: 4px; padding: 2px 0; }
.icp-signal__dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.icp-signal--s1 { background: #f4a742; }
.icp-signal--s2 { background: #ef4444; }
.icp-support-item { display: flex; gap: 8px; padding: 1px 0; }
.icp-support-item__price { font-family: monospace; }

.icp-chart-wrap { margin-top: 4px; }
.icp-chart { width: 100%; height: 160px; }

.icp-unavailable { font-size: 12px; color: var(--text-secondary, #8888a6); padding: 4px 0; }

.icp-header__badge--sell { background: var(--down, #22c55e); }

.icp-sell { margin-bottom: 10px; }
.icp-sell__title { font-size: 13px; font-weight: 600; color: var(--text-secondary, #8888a6); margin-bottom: 6px; }
.icp-sell__row {
  display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  padding: 9px 12px; border: 1px solid var(--border, #1e1e3a); border-radius: 6px; margin-bottom: 6px;
  background: var(--card-bg, #121224); font-size: 13px;
  color: var(--text, #e4e4f0);
}
.icp-sell__code { font-weight: 600; min-width: 60px; }
.icp-sell__name { color: var(--text-secondary, #8888a6); min-width: 56px; }
.icp-sell__urgency { font-size: 12px; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
.icp-urg--stop_loss { background: var(--up-muted, rgba(239,68,68,.12)); color: var(--up, #ef4444); border: 1px solid rgba(239, 68, 68, .35); }
.icp-urg--take_profit { background: var(--down-muted, rgba(34,197,94,.12)); color: var(--down, #22c55e); border: 1px solid rgba(34, 197, 94, .35); }
.icp-urg--normal, .icp-urg--reduce, .icp-urg--sell { background: var(--primary-muted, rgba(107,159,212,.12)); color: var(--primary, #6b9fd4); border: 1px solid rgba(107, 159, 212, .35); }
.icp-sell__pnl { font-weight: 600; min-width: 44px; }
.icp-sell__shares { color: var(--text-secondary, #8888a6); }
</style>
