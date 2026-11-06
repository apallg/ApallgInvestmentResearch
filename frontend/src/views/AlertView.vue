<template>
  <div class="av-root">
    <div class="page-header">
      <h1 class="page-title">风险告警</h1>
    </div>
    <div v-if="error" class="error">{{ error }}</div>
    <div v-if="msg" class="msg msg--success">{{ msg }}</div>

    <div class="grid-4 mb-2">
      <MetricCard :value="stats.high" label="高风险" color="up" />
      <MetricCard :value="stats.mid" label="中风险" />
      <MetricCard :value="stats.low" label="低风险" color="down" />
    </div>

    <!-- 通知通道 + 定时扫描 -->
    <div class="card av-sched">
      <div class="av-sched-top">
        <span class="flex gap-2">
          <span v-if="channels.wecom" class="tag tag-low">企业微信</span>
          <span v-if="channels.coze" class="tag tag-mid">Coze</span>
          <span v-if="!channels.wecom && !channels.coze" class="av-no-channel">通知通道未配置</span>
        </span>
        <span class="av-sched-sep"></span>
        <span :class="['av-dot', sched.running ? 'av-dot--on' : 'av-dot--off']"></span>
        <span class="av-sched-title">定时扫描</span>
        <span v-if="sched.running" class="tag tag-low">运行中</span>
        <span v-else class="tag tag-high">已停止</span>
        <span class="av-sched-config">早盘 {{ sched.morning_time }} | 尾盘 {{ sched.afternoon_time }} | 阈值 {{ sched.default_threshold }}</span>
        <div class="av-sched-btns">
          <button v-if="!sched.running" class="btn-sm btn-success" @click="schedStart">启动</button>
          <button v-if="sched.running" class="btn-sm btn-danger" @click="schedStop">停止</button>
          <button class="btn-sm btn-outline" @click="schedTrigger" :disabled="schedTriggering">
            {{ schedTriggering ? '扫描中...' : '立即扫描' }}
          </button>
          <button class="btn-sm btn-outline" @click="testNotify" :disabled="testing">
            {{ testing ? '发送中...' : '测试通知' }}
          </button>
        </div>
      </div>
      <div v-if="sched.next_morning || sched.next_afternoon" class="av-sched-next">
        下次扫描:
        <span v-if="sched.next_morning">早盘 {{ fmtTime(sched.next_morning) }}</span>
        <span v-if="sched.next_morning && sched.next_afternoon"> | </span>
        <span v-if="sched.next_afternoon">尾盘 {{ fmtTime(sched.next_afternoon) }}</span>
      </div>
      <div v-if="sched.last_scan" class="av-sched-last">
        最近扫描: {{ fmtTime(sched.last_scan) }} — {{ sched.last_result }}
      </div>
    </div>

    <!-- 持仓扫描 -->
    <div v-if="holdingsLoadError" class="card empty" role="alert">持仓不可用：{{ holdingsLoadError }}。本次风险扫描不会按空持仓处理。</div>
    <div class="card" v-else-if="holdings.length">
      <div class="card-title">
        持仓扫描
        <span class="card-subtitle">{{ holdings.length }} 只</span>
      </div>
      <div class="av-scan-bar">
        <label class="av-check"><input type="checkbox" v-model="include.technical" /> 技术指标</label>
        <label class="av-check"><input type="checkbox" v-model="include.financial" /> 财务指标</label>
        <label class="av-check"><input type="checkbox" v-model="include.patterns" /> K线形态</label>
        <label class="av-check"><input type="checkbox" v-model="include.realtime" /> 实时行情</label>
        <span class="av-scan-note">日K线(必选)</span>
        <select v-model.number="scanThreshold" class="av-select-sm">
          <option :value="0">全部</option>
          <option :value="5">≥5</option>
          <option :value="7">≥7</option>
        </select>
        <button class="btn-sm btn-success" @click="batchScan" :disabled="scanning">
          {{ scanning ? '批量扫描中...' : '批量扫描' }}
        </button>
      </div>
      <table>
        <thead>
          <tr><th>代码</th><th>名称</th><th>持股</th><th>成本</th><th>通知阈值</th><th>最近风险</th><th>操作</th></tr>
        </thead>
        <tbody>
          <template v-for="h in holdings" :key="h.code">
          <tr>
            <td class="td-mono">{{ h.code }}</td>
            <td>{{ h.name }}</td>
            <td>{{ h.shares }}</td>
            <td>{{ Number(h.cost_price).toFixed(2) }}</td>
            <td>
              <select v-model.number="h.risk_threshold" @change="updateThreshold(h)" class="av-select-xs">
                <option v-for="n in 10" :key="n" :value="n">{{ n }}</option>
              </select>
            </td>
            <td>
              <span v-if="scannedCodes[h.code]"
                :class="['tag', riskTagClass(scannedCodes[h.code].risk_level)]"
                style="cursor:pointer"
                @click="toggleDetail(h.code)">
                {{ scanStatusLabel(scannedCodes[h.code]) }} {{ scannedCodes[h.code].risk_score == null ? '--' : scannedCodes[h.code].risk_score + '/10' }}
              </span>
              <span v-else class="text-muted">--</span>
            </td>
            <td>
              <button class="btn-sm btn-outline" @click="scanOne(h.code)" :disabled="singleScanning === h.code">
                {{ singleScanning === h.code ? '扫描中...' : '扫描' }}
              </button>
            </td>
          </tr>
          <tr v-if="expanded === h.code && scannedCodes[h.code]" class="av-detail-row">
            <td colspan="7">
              <div class="av-detail">
                <div class="av-detail-head">
                  <span>{{ scannedCodes[h.code].stock_name }}({{ h.code }}) 风险详情</span>
                  <button class="btn-sm btn-outline" @click="expanded = ''">收起</button>
                </div>
                <div class="av-detail-grid">
                  <div>
                    <div class="av-detail-label">风险评分</div>
                    <div class="av-detail-val">{{ scannedCodes[h.code].risk_score == null ? '--' : scannedCodes[h.code].risk_score + '/10' }}</div>
                  </div>
                  <div>
                    <div class="av-detail-label">风险等级</div>
                    <div class="av-detail-val">{{ scanStatusLabel(scannedCodes[h.code]) }}</div>
                  </div>
                  <div v-if="scannedCodes[h.code].notification_status">
                    <div class="av-detail-label">通知状态</div>
                    <div class="av-detail-val">{{ notificationStatusLabel(scannedCodes[h.code].notification_status) }}</div>
                  </div>
                  <div>
                    <div class="av-detail-label">当前价格</div>
                    <div class="av-detail-val">{{ scannedCodes[h.code].current_price || '--' }}</div>
                  </div>
                  <div>
                    <div class="av-detail-label">盈亏</div>
                    <div class="av-detail-val" :class="scannedCodes[h.code].profit_loss >= 0 ? 'text-red' : 'text-green'">
                      {{ scannedCodes[h.code].profit_loss?.toFixed(2) || '--' }}
                      ({{ scannedCodes[h.code].profit_loss_pct?.toFixed(2) || '--' }}%)
                    </div>
                  </div>
                </div>
                <div v-if="scannedCodes[h.code].failure_code" class="text-muted">
                  本轮扫描失败（{{ scannedCodes[h.code].failure_code }}），该持仓风险状态未知。
                </div>
                <div v-if="scannedCodes[h.code].suggestion" class="av-detail-section">
                  <div class="av-detail-label">操作建议</div>
                  <div class="av-detail-text">{{ scannedCodes[h.code].suggestion }}</div>
                </div>
                <div class="av-detail-section">
                  <div class="av-detail-label">分析详情</div>
                  <div class="av-detail-text av-detail-text--scroll">{{ scannedCodes[h.code].risk_detail }}</div>
                </div>
                <div v-if="scannedCodes[h.code].data_enabled" class="av-detail-sources">
                  数据源: {{ scannedCodes[h.code].data_enabled.join('、') }}
                  <span v-if="scannedCodes[h.code].data_skipped?.length"> | 跳过: {{ scannedCodes[h.code].data_skipped.join('、') }}</span>
                </div>
              </div>
            </td>
          </tr>
        </template>
        </tbody>
      </table>
    </div>
    <div v-else class="card empty">暂无持仓，请先在持仓管理中添加股票</div>

    <!-- 告警历史 -->
    <div class="av-filter">
      <select v-model="levelFilter" class="av-select-sm">
        <option value="">全部等级</option>
        <option value="高风险">高风险</option>
        <option value="中风险">中风险</option>
        <option value="低风险">低风险</option>
      </select>
    </div>

    <DataTable
      :rows="filtered"
      :columns="columns"
      row-key="id"
      empty-text="暂无告警记录"
    >
      <template #cell-stock_name="{ row }">{{ row.stock_name }}({{ row.stock_code }})</template>
      <template #cell-risk_level="{ value }">
        <span :class="['tag', riskTagClass(value)]">{{ value }}</span>
      </template>
      <template #cell-risk_score="{ value }">{{ value }}/10</template>
      <template #cell-risk_detail="{ value }">
        <span class="av-risk-cell" :title="value">{{ value?.slice(0, 80) }}</span>
      </template>
      <template #cell-created_at="{ value }">
        <span class="text-nowrap">{{ value?.slice(0,16) || '--' }}</span>
      </template>
    </DataTable>

    <div class="card">
      <div class="card-title flex-between">
        <span>风险扫描历史</span>
        <button class="btn-sm btn-outline" @click="loadScanRuns">刷新</button>
      </div>
      <div v-if="scanRunsError" class="text-muted">扫描记录刷新失败，以下可能是上次成功读取：{{ scanRunsError }}</div>
      <table v-if="scanRuns.length">
        <thead><tr><th>时间</th><th>时段</th><th>范围</th><th>已扫</th><th>未知</th><th>触发</th><th>状态</th><th>诊断</th></tr></thead>
        <tbody>
          <tr v-for="item in scanRuns" :key="item.scan_id">
            <td>{{ item.created_at?.slice(0, 16) || '--' }}</td>
            <td>{{ item.session_name }}</td>
            <td>{{ item.expected_count }}</td>
            <td>{{ item.scanned_count }}</td>
            <td>{{ item.unknown_count }}<span v-if="item.dynamic_unknown_count"> + {{ item.dynamic_unknown_count }} 动态价位不可用</span></td>
            <td>{{ item.triggered_count }}</td>
            <td>{{ scanRunStatusLabel(item.status) }}</td>
            <td>
              <details v-if="item.assessment?.unknown?.length || item.assessment?.dynamic_unknown?.length || item.assessment?.quote_data">
                <summary>查看明细</summary>
                <div v-if="item.assessment?.quote_data">
                  行情：{{ item.assessment.quote_data.data_envelope?.status || 'unknown' }}
                  · 来源 {{ item.assessment.quote_data.source || '--' }}
                  · 日期 {{ item.assessment.quote_data.as_of || '--' }}
                  · 覆盖 {{ item.assessment.quote_data.coverage?.accepted ?? 0 }}/{{ item.assessment.quote_data.coverage?.requested ?? 0 }}
                  · 观测 {{ item.assessment.quote_data.observed_at || '--' }}
                </div>
                <div v-for="entry in item.assessment?.unknown || []" :key="`unknown-${entry.code}`">{{ entry.code }}：{{ entry.reason }}</div>
                <div v-for="code in item.assessment?.dynamic_unknown || []" :key="`dynamic-${code}`">{{ code }}：动态价位不可用</div>
              </details>
              <span v-else>--</span>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else-if="!scanRunsError" class="text-muted">暂无扫描记录</div>
    </div>

    <div class="card">
      <div class="card-title flex-between">
        <span>通知送达队列</span>
        <button class="btn-sm btn-outline" @click="loadOutbox">刷新</button>
      </div>
      <div v-if="outboxError" class="text-muted">通知队列刷新失败，以下可能是上次成功读取：{{ outboxError }}</div>
      <table v-if="notificationOutbox.length">
        <thead><tr><th>队列号</th><th>告警/扫描</th><th>渠道</th><th>状态</th><th>尝试次数</th><th>渠道回执</th><th>最近错误</th><th>记录时间</th><th>人工处理</th></tr></thead>
        <tbody>
          <tr v-for="item in notificationOutbox" :key="item.id">
            <td class="td-mono">{{ item.id }}</td>
            <td>{{ item.scan_id || item.alert_id || '--' }}</td>
            <td>{{ item.delivery_channel || '--' }}</td>
            <td :title="item.resolution_reason || ''">{{ outboxStatusLabel(item) }}</td>
            <td>{{ item.attempt_count }}</td>
            <td :title="item.provider_response_json || ''">{{ item.provider_response_json?.slice(0, 100) || '--' }}</td>
            <td>{{ item.last_error || '--' }}</td>
            <td>{{ item.sent_at?.slice(0, 16) || item.updated_at?.slice(0, 16) || '--' }}</td>
            <td>
              <button v-if="item.status === 'delivery_unknown' && item.resolution_status !== 'operator_confirmed_sent'"
                class="btn-sm btn-outline" @click="beginOutboxResolution(item)">
                {{ item.resolution_status ? '再次核查' : '核查结果' }}
              </button>
              <span v-else-if="item.resolved_by">{{ item.resolved_by }} · {{ item.resolution_evidence }}</span>
              <span v-else class="text-muted">--</span>
            </td>
          </tr>
          <tr v-for="item in notificationOutbox.filter((row) => resolvingOutboxId === row.id)" :key="`resolve-${item.id}`">
            <td colspan="9">
              <div class="av-detail">
                <div class="av-detail-text">先核对渠道侧记录。只有确认未发送后才选择重排；结果未知时不要重排，以免重复通知。</div>
                <input v-model="resolutionReason" class="av-select-sm" maxlength="1000" placeholder="核查原因（至少10字）" />
                <input v-model="resolutionEvidence" class="av-select-sm" maxlength="255" placeholder="核查依据/工单引用" />
                <button class="btn-sm btn-outline" :disabled="resolvingOutbox" @click="resolveOutbox(item, 'confirmed_sent')">确认渠道已接受</button>
                <button class="btn-sm btn-danger" :disabled="resolvingOutbox" @click="resolveOutbox(item, 'confirmed_not_sent_requeue')">确认未发送并重排</button>
                <button class="btn-sm btn-outline" :disabled="resolvingOutbox" @click="resolvingOutboxId = null">取消</button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else-if="!outboxError" class="text-muted">暂无待发送或送达记录</div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted } from 'vue'
import api from '../api/index.js'
import MetricCard from '../components/MetricCard.vue'
import DataTable from '../components/DataTable.vue'
import { useHoldingsStore } from '../stores/holdings.js'

const SCANNED_MAX = 200

const holdingsStore = useHoldingsStore()
const holdingsLoadError = ref('')

const alerts = ref([])
const notificationOutbox = ref([])
const scanRuns = ref([])
const outboxError = ref('')
const scanRunsError = ref('')
const resolvingOutboxId = ref(null)
const resolutionReason = ref('')
const resolutionEvidence = ref('')
const resolvingOutbox = ref(false)
const stats = ref({ high: 0, mid: 0, low: 0 })
const channels = ref({ wecom: false, coze: false })
const holdings = ref([])
const scannedCodes = ref({})
const levelFilter = ref('')
const scanThreshold = ref(5)
const error = ref('')
const msg = ref('')
const testing = ref(false)
const scanning = ref(false)
const singleScanning = ref('')
const expanded = ref('')
let scanRequestId = 0
let latestBatchScanRequestId = 0
const singleScanRequestIds = new Map()
const include = reactive({
  technical: true,
  financial: true,
  patterns: true,
  realtime: true,
})

const columns = [
  { key: 'stock_name', label: '股票' },
  { key: 'risk_level', label: '等级' },
  { key: 'risk_score', label: '评分' },
  { key: 'risk_detail', label: '详情' },
  { key: 'created_at', label: '时间' },
]

const filtered = computed(() =>
  !levelFilter.value ? alerts.value : alerts.value.filter(a => a.risk_level === levelFilter.value)
)

const RISK_TAGS = { '高风险': 'tag-high', '中风险': 'tag-mid', '低风险': 'tag-low' }

function riskTagClass(level) {
  return RISK_TAGS[level] || 'tag'
}

function scanStatusLabel(item) {
  if (item.risk_status === 'failed') return '扫描失败'
  if (item.risk_status === 'disabled') return '扫描已关闭'
  if (item.risk_status === 'unknown') return '无法评分'
  return item.risk_level || '未知'
}

function notificationStatusLabel(status) {
  return {
    pending: '待发送',
    unavailable: '账本不可用，未排队',
    not_required: '未达通知阈值',
  }[status] || status
}

function outboxStatusLabel(item) {
  if (item.resolution_status === 'operator_confirmed_sent') return '人工核验：确认渠道已接受'
  if (item.resolution_status === 'operator_confirmed_not_sent_requeued') return '人工核验未发送，已重排'
  return {
    pending: '待发送',
    sending: '发送中',
    retry: '等待重试',
    sent: '已送达通道',
    failed: '重试耗尽',
    delivery_unknown: '发送结果待核实',
  }[item.status] || item.status
}

function beginOutboxResolution(item) {
  resolvingOutboxId.value = item.id
  resolutionReason.value = ''
  resolutionEvidence.value = ''
}

function scanRunStatusLabel(status) {
  return { complete: '完整', partial: '部分可用', degraded: '扫描不完整', empty: '空持仓' }[status] || status
}

function toggleDetail(code) {
  expanded.value = expanded.value === code ? '' : code
}

function _capScanned() {
  const keys = Object.keys(scannedCodes.value)
  if (keys.length > SCANNED_MAX) {
    const toRemove = keys.slice(0, keys.length - SCANNED_MAX)
    for (const k of toRemove) { delete scannedCodes.value[k] }
  }
}

async function updateThreshold(h) {
  try { await holdingsStore.updateThreshold(h.code, h.risk_threshold) } catch (e) { error.value = '阈值更新失败: ' + e.message }
}

async function loadHoldings() {
  try {
    const data = await holdingsStore.load()
    holdings.value = data.map(h => ({ ...h, risk_threshold: h.risk_threshold || 7 }))
    holdingsLoadError.value = ''
  } catch (e) {
    holdings.value = []
    holdingsLoadError.value = e.message || '读取失败'
    error.value = '持仓加载失败: ' + holdingsLoadError.value
  }
}

async function loadChannels() {
  try {
    const r = await api.get('/alerts/channels')
    channels.value = r.data || {}
  } catch (e) { error.value = '通知通道加载失败: ' + e.message }
}

async function testNotify() {
  testing.value = true; msg.value = ''; error.value = ''
  try {
    const r = await api.post('/alerts/test')
    msg.value = '测试通知已发送: ' + JSON.stringify(r.data)
  } catch (e) { error.value = e.message }
  testing.value = false
}

function getScanParams() {
  const params = {}
  for (const [k, v] of Object.entries(include)) {
    if (!v) params[k] = false
  }
  return Object.keys(params).length ? params : null
}

async function scanOne(code) {
  const requestId = ++scanRequestId
  singleScanRequestIds.set(code, requestId)
  singleScanning.value = code; msg.value = ''; error.value = ''
  try {
    const payload = { threshold: 0 }
    const inc = getScanParams()
    if (inc) payload.include = inc
    const r = await api.post(`/alerts/scan/${code}`, payload, { timeout: 180000 })
    if (r.data && singleScanRequestIds.get(code) === requestId && requestId > latestBatchScanRequestId) {
      scannedCodes.value[code] = r.data
      _capScanned()
      expanded.value = code
    }
    await loadData()
  } catch (e) {
    error.value = e.message
    if (singleScanRequestIds.get(code) === requestId && requestId > latestBatchScanRequestId) {
      const holding = holdings.value.find(item => item.code === code)
      scannedCodes.value[code] = {
        stock_code: code,
        stock_name: holding?.name || '',
        risk_level: '未知',
        risk_score: null,
        risk_status: 'failed',
        failure_code: 'request_failed',
      }
    }
  }
  singleScanning.value = ''
}

async function batchScan() {
  const requestId = ++scanRequestId
  latestBatchScanRequestId = requestId
  scanning.value = true; msg.value = ''; error.value = ''
  scannedCodes.value = {}
  expanded.value = ''
  try {
    const payload = { threshold: scanThreshold.value }
    const inc = getScanParams()
    if (inc) payload.include = inc
    const r = await api.post('/alerts/scan', payload, { timeout: 180000 })
    const results = r.data?.results || []
    const currentResults = {}
    for (const item of results) {
      const newerSingleScan = singleScanRequestIds.get(item.stock_code) > requestId
      currentResults[item.stock_code] = newerSingleScan
        ? scannedCodes.value[item.stock_code]
        : item
    }
    if (requestId === latestBatchScanRequestId) scannedCodes.value = currentResults
    expanded.value = ''
    msg.value = `本轮 ${r.data?.scan_id || '--'}：扫描 ${r.data?.total_count || 0} 只，触发风险 ${r.data?.count || 0} 条，无法评分 ${r.data?.unscored_count || 0} 只，失败 ${r.data?.failed_count || 0} 只，已关闭 ${r.data?.skipped_count || 0} 只`
    await loadData()
  } catch (e) { error.value = e.message }
  scanning.value = false
}

async function loadData() {
  const requestGeneration = ++alertsDataRequestGeneration
  try {
    const [a, s] = await Promise.all([
      api.get('/alerts/?limit=100'),
      api.get('/alerts/stats?limit=200'),
    ])
    if (requestGeneration !== alertsDataRequestGeneration) return
    alerts.value = a.data || []
    stats.value = s.data || { high: 0, mid: 0, low: 0 }
  } catch (e) {
    if (requestGeneration === alertsDataRequestGeneration) error.value = e.message
  }
  if (requestGeneration !== alertsDataRequestGeneration) return
  await loadOutbox()
  await loadScanRuns()
}

async function loadOutbox() {
  const requestGeneration = ++outboxRequestGeneration
  try {
    const r = await api.get('/alerts/outbox?limit=20')
    if (requestGeneration !== outboxRequestGeneration) return
    notificationOutbox.value = r.data || []
    outboxError.value = ''
  } catch (e) {
    if (requestGeneration === outboxRequestGeneration) outboxError.value = e.message
  }
}

async function resolveOutbox(item, resolution) {
  const reason = resolutionReason.value.trim()
  const evidenceRef = resolutionEvidence.value.trim()
  if (reason.length < 10 || evidenceRef.length < 3) {
    error.value = '请填写至少10字的核查原因和核查依据引用'
    return
  }
  if (resolution === 'confirmed_not_sent_requeue' && !window.confirm('确认渠道侧没有接收该通知？重排后系统会重新发送。')) return
  resolvingOutbox.value = true
  error.value = ''
  try {
    await api.post(`/alerts/outbox/${item.id}/resolve`, {
      resolution,
      reason,
      evidence_ref: evidenceRef,
    })
    msg.value = resolution === 'confirmed_sent' ? '已记录人工核验结果' : '已记录核验并重新排队'
    resolvingOutboxId.value = null
    await loadOutbox()
  } catch (e) {
    error.value = e.message
  } finally {
    resolvingOutbox.value = false
  }
}

async function loadScanRuns() {
  const requestGeneration = ++scanRunsRequestGeneration
  try {
    const r = await api.get('/alerts/scan-runs?limit=50')
    if (requestGeneration !== scanRunsRequestGeneration) return
    scanRuns.value = r.data || []
    scanRunsError.value = ''
  } catch (e) {
    if (requestGeneration === scanRunsRequestGeneration) scanRunsError.value = e.message
  }
}

const sched = reactive({
  running: false,
  morning_time: '',
  afternoon_time: '',
  default_threshold: 7,
  last_scan: null,
  last_result: null,
  next_morning: null,
  next_afternoon: null,
})
const schedTriggering = ref(false)
let schedTimer = null
let _schedRunning = null
let alertsDataRequestGeneration = 0
let outboxRequestGeneration = 0
let scanRunsRequestGeneration = 0
let schedulerStatusRequestGeneration = 0
let schedRefreshTimer = null

function fmtTime(s) {
  if (!s) return '--'
  return s.slice(0, 16).replace('T', ' ')
}

async function loadSchedStatus() {
  const requestGeneration = ++schedulerStatusRequestGeneration
  try {
    const r = await api.get('/scheduler/status')
    if (requestGeneration !== schedulerStatusRequestGeneration) return
    if (r.data) {
      const wasRunning = _schedRunning
      _schedRunning = r.data.running
      Object.assign(sched, r.data)
      if (!wasRunning && r.data.running) {
        await loadData()
      }
    }
  } catch (e) { /* 轮询静默 */ }
}

async function schedStart() {
  try { await api.post('/scheduler/start'); await loadSchedStatus() } catch (e) { error.value = '调度器启动失败: ' + e.message }
}
async function schedStop() {
  try { await api.post('/scheduler/stop'); await loadSchedStatus() } catch (e) { error.value = '调度器停止失败: ' + e.message }
}
async function schedTrigger() {
  schedTriggering.value = true
  try {
    await api.post('/scheduler/trigger')
    await loadSchedStatus()
    msg.value = '手动扫描已触发，请等待完成...'
    clearTimeout(schedRefreshTimer)
    schedRefreshTimer = setTimeout(async () => { await loadSchedStatus(); await loadData(); msg.value = '' }, 30000)
  } catch (e) { error.value = e.message }
  schedTriggering.value = false
}

onMounted(() => {
  loadHoldings(); loadChannels(); loadData(); loadSchedStatus()
  schedTimer = setInterval(loadSchedStatus, 30000)
})
onUnmounted(() => {
  alertsDataRequestGeneration += 1
  outboxRequestGeneration += 1
  scanRunsRequestGeneration += 1
  schedulerStatusRequestGeneration += 1
  clearInterval(schedTimer)
  clearTimeout(schedRefreshTimer)
})
</script>

<style scoped>
.av-root {
  max-width: var(--content-max-width);
  margin: 0 auto;
  padding: 0 var(--content-padding) var(--space-8);
}

/* ── 调度卡片 ── */
.av-sched { padding: 12px 16px; }
.av-sched-top {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.av-sched-sep { width: 1px; height: 20px; background: var(--border); }
.av-sched-title { font-weight: 600; font-size: var(--fs-body); color: var(--text); }
.av-sched-config { font-size: var(--fs-caption); color: var(--text-muted); }
.av-sched-btns { margin-left: auto; display: flex; gap: 6px; }
.av-sched-next { margin-top: 8px; font-size: var(--fs-caption); color: var(--text-muted); }
.av-sched-last { margin-top: 4px; font-size: var(--fs-caption); color: var(--text-muted); }
.av-no-channel { font-size: var(--fs-sm); color: var(--danger); }

.av-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.av-dot--on { background: var(--success); }
.av-dot--off { background: var(--danger); }

/* ── 扫描控制栏 ── */
.av-scan-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}
.av-check { display: flex; align-items: center; gap: 3px; font-size: var(--fs-caption); color: var(--text-secondary); cursor: pointer; }
.av-check input[type=checkbox] { width: auto; margin: 0; }
.av-scan-note { font-size: var(--fs-caption); color: var(--text-muted); margin-left: auto; }
.av-select-sm { padding: 3px 6px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg); color: var(--text); font-size: var(--fs-caption); font-family: var(--font); }
.av-select-xs { width: 52px; padding: 2px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg); color: var(--text); font-size: var(--fs-caption); }

/* ── 筛选 ── */
.av-filter { margin: 16px 0 8px; }

/* ── 风险详情单元格 ── */
.av-risk-cell {
  font-size: var(--fs-sm); display: block; max-width: 300px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

/* ── 详情展开行 ── */
.av-detail-row td { padding: 0; background: var(--bg-secondary); }
.av-detail { padding: 16px 20px; border-top: 1px solid var(--border); }
.av-detail-head {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 12px; font-weight: 600; font-size: var(--fs-base); color: var(--text);
}
.av-detail-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 12px; }
.av-detail-label { font-size: var(--fs-caption); color: var(--text-muted); margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.3px; }
.av-detail-val { font-size: var(--fs-base); font-weight: 600; color: var(--text); font-variant-numeric: tabular-nums; }
.av-detail-section { margin-top: 10px; }
.av-detail-text {
  font-size: var(--fs-sm); color: var(--text-secondary); margin-top: 4px;
  line-height: 1.6; background: var(--bg); padding: 10px 14px; border-radius: var(--radius-sm);
}
.av-detail-text--scroll { white-space: pre-wrap; max-height: 400px; overflow-y: auto; }
.av-detail-sources { font-size: var(--fs-caption); color: var(--text-muted); margin-top: 8px; }
</style>
