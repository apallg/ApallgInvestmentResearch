<template>
  <div class="t0-root">
    <div class="t0-header">
      <div class="t0-header__left">
        <span class="t0-header__title">T+0 做T建议</span>
        <span class="t0-badge t0-badge--i">I 激进</span>
        <span class="t0-badge t0-badge--n">N 保守</span>
        <span v-if="source" class="t0-src">
          <span class="t0-dot" :class="source === 'ths' ? 't0-dot--ok' : 't0-dot--warn'"></span>
          {{ source === 'ths' ? '同花顺' : source }}
        </span>
        <span v-if="timestamp" class="t0-time">生成于 {{ timestamp }}</span>
      </div>
      <div class="t0-header__right">
        <span v-if="loading" class="t0-loading">⏳</span>
        <button class="t0-btn" :disabled="loading" @click="refresh">刷新</button>
        <span v-if="error" class="t0-error">{{ error }}</span>
      </div>
    </div>

    <div class="t0-note">
      提示：买入量受本地可用现金约束，卖出量受当日可卖股数约束；建议仅供参考，实际仓位与执行由项目风控规则决定。
    </div>
    <div v-if="sellableSource" class="t0-note" role="status" aria-atomic="true">
      账户 {{ accountId }} · 账本版本 {{ accountVersion }} · 卖出股数依据 {{ sellableSource === 'broker_reconciliation' ? '券商只读对账' : '本地账本 T+1' }}
    </div>
    <div v-if="dataStatus" class="t0-note" role="status" aria-atomic="true">
      {{ dataMode === 'historical_preview' ? '历史回看（不可执行）' : '实时数据' }} · 数据状态 {{ dataStatusLabel }} · 覆盖 {{ coverage.available ?? 0 }}/{{ coverage.requested ?? 0 }} · 决策日期 {{ asOf || '--' }} · 数据来源 {{ dataSourceLabel }} · 响应可用时间 {{ availableAt || '--' }}
    </div>
    <div v-if="semanticStatus === 'unverified'" class="t0-note t0-warn" role="status" aria-atomic="true">
      分钟量额单位/累计口径和价格基准尚未完成供应商语义核验；当前来源状态只表示本地数据门禁通过。
    </div>
    <div v-if="unavailable.length" class="t0-note t0-warn" role="status" aria-atomic="true">
      <div>暂不可计算：</div>
      <div v-for="code in unavailable" :key="code">{{ code }}：{{ unavailableDetails[code]?.reason || '原因未返回' }}</div>
    </div>

    <div v-if="results.length" class="t0-body">
      <div v-for="item in results" :key="item.code" class="t0-card">
        <div class="t0-card__top">
          <span class="t0-code">{{ item.code }}</span>
          <span class="t0-price">{{ fmt(item.current_price) }}</span>
          <span class="t0-meta">VWAP {{ fmt(item.vwap) }}</span>
          <span class="t0-meta">低 {{ fmt(item.day_low) }} / 高 {{ fmt(item.day_high) }}</span>
          <span class="t0-meta">ATR {{ fmt(item.atr_abs) }}</span>
          <span class="t0-meta">RSI {{ fmt(item.rsi, 1) }}</span>
          <span class="t0-meta" :class="{ 't0-warn': item.swing_factor < 1 }">
            波段因子 {{ (item.swing_factor * 100).toFixed(0) }}%
          </span>
        </div>
        <div v-if="item.data_provenance" class="t0-note t0-card__provenance">
          分钟数据观察时点 {{ item.data_provenance.observed_at || '未知' }} · 本地量额门禁 {{ item.data_provenance.quantity_status }} · 价格基准 {{ item.data_provenance.price_basis }} · 量/额单位 {{ item.data_provenance.unit?.volume }}/{{ item.data_provenance.unit?.amount }}
        </div>

        <div v-if="item.daily_flags && item.daily_flags.length" class="t0-flags">
          <span v-for="f in item.daily_flags" :key="f" class="t0-flag">{{ f }}</span>
        </div>

        <div class="t0-variants">
          <div v-for="v in [item.aggressive, item.conservative].filter(Boolean)" :key="v.name" class="t0-variant" :class="v.name === 'I' ? 't0-variant--i' : 't0-variant--n'">
            <div class="t0-variant__head">
              <span class="t0-variant__name">{{ v.label }} ({{ v.name }})</span>
              <span class="t0-state" :class="v.allowed ? 't0-state--yes' : 't0-state--no'">
                {{ v.allowed ? '可低吸' : semanticStatus === 'unverified' ? '不可执行' : '观望' }}
              </span>
            </div>
            <div class="t0-grid">
              <span>低吸价 <b>{{ fmt(v.entry_price) }}</b></span>
              <span>止盈 <b class="t0-up">{{ fmt(v.target_price) }}</b></span>
              <span>止损 <b class="t0-down">{{ fmt(v.stop_price) }}</b></span>
              <span>紧急 <b class="t0-down">{{ fmt(v.emergency_stop_price) }}</b></span>
              <span>建议买入量 <b>{{ v.suggested_shares ? v.suggested_shares + '股' : '--' }}</b></span>
              <span>建议卖出量 <b>{{ v.suggested_sell_shares ? v.suggested_sell_shares + '股' : '--' }}</b></span>
              <span>目标空间 <b>{{ fmt(v.target_edge_pct, 2) }}%</b></span>
            </div>
            <div v-if="v.flags && v.flags.length" class="t0-variant__flags">
              <span v-for="f in v.flags" :key="f" class="t0-flag">{{ f }}</span>
            </div>
          </div>
          <div v-if="!item.aggressive && !item.conservative" class="t0-unavailable">
            {{ semanticStatus === 'unverified' ? '分钟来源语义尚未核验，暂不展示分钟派生指标或做T建议。' : '当前缺少足够的5分钟行情，暂无法生成做T建议。' }}
          </div>
        </div>
      </div>
    </div>

    <div v-else-if="!loading" class="t0-empty">
      {{ unavailable.length ? '当前无可计算的做T标的，请核对上述持仓与数据状态。' : '当前无可做T标的（无持仓）。持仓出现后，这里会显示 I/N 两档做T建议。' }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { checkT0 } from '../api/rebalance.js'
import { hasVerifiedMinuteProvenance } from '../utils/minuteAnalysis.js'

const props = defineProps({
  codes: { type: Array, default: () => [] },
  refreshInterval: { type: Number, default: 10000 },
})

const results = ref([])
const source = ref('')
const timestamp = ref('')
const unavailable = ref([])
const unavailableDetails = ref({})
const dataStatus = ref('')
const dataMode = ref('')
const semanticStatus = ref('')
const coverage = ref({})
const dataSource = ref('')
const asOf = ref('')
const availableAt = ref('')
const sellableSource = ref('')
const accountId = ref('')
const accountVersion = ref(null)
const loading = ref(false)
const error = ref('')
let timer = null
let refreshGeneration = 0

const dataStatusLabels = {
  verified: '本地门禁通过',
  partial: '部分可用',
  unavailable: '不可用',
}
const dataSourceLabels = {
  ths_live_minute: '同花顺实时分钟线',
  ths_minute_history: '同花顺分钟历史库',
}

const dataStatusLabel = computed(() => dataStatusLabels[dataStatus.value] || dataStatus.value)
const dataSourceLabel = computed(() => dataSourceLabels[dataSource.value] || dataSource.value || '--')

function fmt(v, d = 2) {
  if (v == null || Number.isNaN(Number(v))) return '--'
  return Number(v).toFixed(d)
}

async function refresh() {
  const generation = ++refreshGeneration
  if (!props.codes?.length) {
    results.value = []
    unavailable.value = []
    unavailableDetails.value = {}
    dataStatus.value = ''
    dataMode.value = ''
    semanticStatus.value = ''
    coverage.value = {}
    dataSource.value = ''
    asOf.value = ''
    availableAt.value = ''
    source.value = ''
    sellableSource.value = ''
    accountId.value = ''
    accountVersion.value = null
    timestamp.value = ''
    error.value = ''
    loading.value = false
    return
  }
  const codes = [...props.codes]
  loading.value = true
  error.value = ''
  results.value = []
  unavailable.value = []
  unavailableDetails.value = {}
  dataStatus.value = ''
  dataMode.value = ''
  semanticStatus.value = ''
  coverage.value = {}
  dataSource.value = ''
  asOf.value = ''
  availableAt.value = ''
  source.value = ''
  sellableSource.value = ''
  accountId.value = ''
  accountVersion.value = null
  timestamp.value = ''
  try {
    const r = await checkT0(codes)
    if (generation !== refreshGeneration) return
    const data = r?.data ?? r ?? {}
    const incomingResults = Array.isArray(data.results) ? data.results : []
    const resultsVerified = incomingResults.length > 0 && incomingResults.every(item =>
      hasVerifiedMinuteProvenance(item?.data_provenance),
    )
    results.value = incomingResults.map(item => {
      if (hasVerifiedMinuteProvenance(item?.data_provenance)) return item
      return {
        ...item,
        current_price: null,
        vwap: null,
        day_high: null,
        day_low: null,
        atr_abs: null,
        rsi: null,
        intraday_ok: false,
        entry_threshold: null,
        low_position: null,
        aggressive: null,
        conservative: null,
        daily_flags: [...new Set([...(item?.daily_flags || []), '分钟来源语义未核验，不展示分钟派生指标或做T建议'])],
      }
    })
    unavailable.value = data.unavailable ?? []
    unavailableDetails.value = data.unavailable_details ?? {}
    dataStatus.value = data.data_status ?? ''
    dataMode.value = data.mode ?? ''
    semanticStatus.value = data.semantic_status === 'verified' && resultsVerified ? 'verified' : 'unverified'
    coverage.value = data.coverage ?? {}
    dataSource.value = data.data_source ?? ''
    asOf.value = data.as_of ?? ''
    availableAt.value = data.available_at ?? ''
    source.value = data.source ?? ''
    sellableSource.value = data.sellable_source ?? ''
    accountId.value = data.account_id ?? ''
    accountVersion.value = data.account_version ?? null
    timestamp.value = data.timestamp ?? ''
  } catch (e) {
    if (generation !== refreshGeneration) return
    error.value = e?.response?.data?.detail || e?.message || 'T+0获取失败'
    results.value = []
    unavailable.value = []
    unavailableDetails.value = {}
    dataStatus.value = 'unavailable'
    dataMode.value = ''
    semanticStatus.value = ''
    coverage.value = {}
    dataSource.value = ''
    asOf.value = ''
    availableAt.value = ''
    source.value = ''
    sellableSource.value = ''
    accountId.value = ''
    accountVersion.value = null
    timestamp.value = ''
  } finally {
    if (generation === refreshGeneration) loading.value = false
  }
}

function startTimer() {
  stopTimer()
  if (props.refreshInterval > 0) {
    timer = setInterval(refresh, props.refreshInterval)
  }
}

function stopTimer() {
  if (timer) {
    clearInterval(timer)
    timer = null
  }
}

onMounted(() => {
  refresh()
  startTimer()
})

onUnmounted(() => {
  refreshGeneration += 1
  stopTimer()
})

watch(() => props.codes, () => refresh())
watch(() => props.refreshInterval, startTimer)
</script>

<style scoped>
.t0-root {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  margin: 12px 0;
  background: #fff;
  font-size: 13px;
}
.t0-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 12px;
  border-bottom: 1px solid #f0f0f0;
}
.t0-header__left { display: flex; align-items: center; gap: 8px; }
.t0-header__title { font-weight: 600; color: #111827; }
.t0-badge { padding: 2px 8px; border-radius: 4px; font-size: 12px; color: #fff; }
.t0-badge--i { background: #2563eb; }
.t0-badge--n { background: #16a34a; }
.t0-src { display: inline-flex; align-items: center; gap: 4px; color: #6b7280; }
.t0-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.t0-dot--ok { background: #22c55e; }
.t0-dot--warn { background: #f59e0b; }
.t0-time { color: #9ca3af; }
.t0-btn { border: 1px solid #d1d5db; background: #f9fafb; border-radius: 4px; padding: 3px 10px; cursor: pointer; }
.t0-loading { color: #2563eb; }
.t0-error { color: #dc2626; }
.t0-note { padding: 6px 12px; color: #6b7280; font-size: 12px; background: #f9fafb; }
.t0-body { padding: 8px 12px 12px; display: flex; flex-direction: column; gap: 10px; }
.t0-card { border: 1px solid #eef0f3; border-radius: 6px; padding: 8px 10px; }
.t0-card__top { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.t0-code { font-weight: 700; color: #111827; }
.t0-price { font-weight: 700; color: #111827; }
.t0-meta { color: #6b7280; }
.t0-warn { color: #dc2626; font-weight: 600; }
.t0-flags { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0 2px; }
.t0-flag { background: #fef3c7; color: #92400e; border-radius: 3px; padding: 1px 6px; font-size: 11px; }
.t0-variants { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px; }
.t0-variant { border-radius: 6px; padding: 8px; }
.t0-variant--i { background: #eff6ff; border: 1px solid #bfdbfe; }
.t0-variant--n { background: #f0fdf4; border: 1px solid #bbf7d0; }
.t0-variant__head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.t0-variant__name { font-weight: 600; }
.t0-state { padding: 1px 8px; border-radius: 4px; font-size: 12px; }
.t0-state--yes { background: #dcfce7; color: #15803d; }
.t0-state--no { background: #fee2e2; color: #b91c1c; }
.t0-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px 8px; }
.t0-grid span { color: #4b5563; }
.t0-up { color: #dc2626; }
.t0-down { color: #16a34a; }
.t0-variant__flags { margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px; }
.t0-empty { padding: 18px; text-align: center; color: #9ca3af; }
@media (max-width: 640px) {
  .t0-variants { grid-template-columns: 1fr; }
}
</style>
