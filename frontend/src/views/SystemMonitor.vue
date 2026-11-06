<template>
  <div class="system-monitor">
    <h2>系统监控</h2>

    <div class="scheduler-grid">
      <SchedulerCard
        title="持仓风险扫描"
        :running="risk.running.value"
        :last-scan="risk.lastScan.value"
        :scan-count="risk.scanCount.value"
        :last-error="risk.lastError.value"
        :loading="risk.loading.value"
        @start="risk.start"
        @stop="risk.stop"
        @trigger="risk.trigger"
      />
      <SchedulerCard
        title="数据采集"
        :running="dataColl.running.value"
        :last-scan="dataColl.lastScan.value"
        :scan-count="dataColl.scanCount.value"
        :last-error="dataColl.lastError.value"
        :loading="dataColl.loading.value"
        @start="dataColl.start"
        @stop="dataColl.stop"
        @trigger="dataColl.trigger"
      />
      <SchedulerCard
        title="情绪分析"
        :running="sent.running.value"
        :last-scan="sent.lastScan.value"
        :scan-count="sent.scanCount.value"
        :last-error="sent.lastError.value"
        :loading="sent.loading.value"
        @start="sent.start"
        @stop="sent.stop"
        @trigger="sent.trigger"
      />
      <SchedulerCard
        title="冒烟测试"
        :running="smoke.running.value"
        :last-scan="smoke.lastScan.value"
        :scan-count="smoke.scanCount.value"
        :last-error="smoke.lastError.value"
        :loading="smoke.loading.value"
        @start="smoke.start"
        @stop="smoke.stop"
        @trigger="smoke.trigger"
      />
      <SchedulerCard
        title="公告采集"
        :running="notice.running.value"
        :last-scan="notice.lastScan.value"
        :scan-count="notice.scanCount.value"
        :last-error="notice.lastError.value"
        :loading="notice.loading.value"
        @start="notice.start"
        @stop="notice.stop"
        @trigger="notice.trigger"
      />
    </div>

    <div class="data-status-section">
      <h3>Redis 缓存
        <button class="preload-btn" @click="triggerPreload" :disabled="preloading">
          {{ preloading ? `预热中 ${preloadStatus}` : '缓存预热' }}
        </button>
        <span class="preload-hint" v-if="preloadDone">{{ preloadMsg }}</span>
      </h3>
      <div class="status-cards" v-if="redisStats">
        <div class="status-card">
          <span class="status-label">状态</span>
          <span class="status-value" :class="redisStats.available ? 'ok' : 'warn'">{{ redisStats.available ? '正常' : redisStats.error || '异常' }}</span>
        </div>
        <div class="status-card">
          <span class="status-label">内存占用</span>
          <span class="status-value ok">{{ redisStats.used_memory_human || '-' }}</span>
        </div>
        <div class="status-card">
          <span class="status-label">缓存键总数</span>
          <span class="status-value ok">{{ redisStats.total_keys ?? '-' }}</span>
        </div>
        <div class="status-card">
          <span class="status-label">快照缓存</span>
          <span class="status-value ok">{{ redisStats.snapshot_count ?? 0 }} 个</span>
        </div>
        <div class="status-card">
          <span class="status-label">日线缓存</span>
          <span class="status-value ok">{{ redisStats.daily_cache_count ?? 0 }} 个 ({{ redisStats.daily_stock_count ?? 0 }}只)</span>
          <span class="status-sub" v-if="redisStats.daily_date_start" style="font-size:12px;color:var(--text-muted)">{{ redisStats.daily_date_start }} ~ {{ redisStats.daily_date_end }}</span>
        </div>
        <div class="status-card" v-if="redisStats.snapshot_date_start">
          <span class="status-label">快照日期范围</span>
          <span class="status-value ok" style="font-size:12px;">{{ redisStats.snapshot_date_start }} ~ {{ redisStats.snapshot_date_end }}</span>
        </div>
      </div>
    </div>

    <div class="data-status-section">
      <h3>线程池
        <span class="health-badge" :class="threadHealth">{{ poolHealthText }}</span>
        <button class="preload-btn" @click="loadThreadPools" :disabled="threadLoading">
          {{ threadLoading ? '...' : '刷新' }}
        </button>
      </h3>
      <div class="thread-pool-grid" v-if="threadPools">
        <div class="thread-pool-card" v-for="p in threadPools" :key="p.kind">
          <div class="tp-header">
            <span class="tp-name">{{ p.kind === 'default' ? '通用线程池' : '轻量后台线程池' }}</span>
            <span class="tp-health" :class="p.health">{{ p.health === 'ok' ? '正常' : p.health === 'busy' ? '繁忙' : '高负载' }}</span>
          </div>
          <div class="tp-bar-wrap">
            <div class="tp-bar">
              <div class="tp-bar-fill" :class="p.health" :style="{width: p.saturation_pct + '%'}"></div>
            </div>
            <span class="tp-bar-label">{{ p.active_count }}/{{ p.max_workers }}</span>
          </div>
          <div class="tp-stats">
            <div class="tp-stat"><span class="tp-label">排队</span><span class="tp-value" :class="{warn: p.queued_count > 0}">{{ p.queued_count }}</span></div>
            <div class="tp-stat"><span class="tp-label">等待</span><span class="tp-value">{{ fmtMs(p.avg_wait_ms) }}</span></div>
            <div class="tp-stat"><span class="tp-label">执行</span><span class="tp-value">{{ fmtMs(p.avg_exec_ms) }}</span></div>
          </div>
          <div class="tp-footer">
            <span>完成 {{ fmtCount(p.total_completed) }}</span>
            <span v-if="p.total_failed" style="color:var(--danger)">失败 {{ p.total_failed }}</span>
          </div>
        </div>
      </div>
      <AppError v-if="threadError" :message="threadError" />
    </div>

    <div class="data-status-section">
      <h3>Forward Return 物化表
        <button class="preload-btn" @click="triggerFwdWarmup" :disabled="fwdWarming">
          {{ fwdWarming ? `预热中 ${fwdWarmStatus}` : '预热' }}
        </button>
        <span class="preload-hint" v-if="fwdStatus">{{ fwdStatus }}</span>
      </h3>
      <div class="status-cards" v-if="fwdInfo">
        <div class="status-card">
          <span class="status-label">状态</span>
          <span class="status-value" :class="fwdInfo.available ? 'ok' : 'warn'">{{ fwdInfo.available ? `就绪 (${fwdInfo.total_rows?.toLocaleString()} 行)` : '未预热' }}</span>
        </div>
        <div class="status-card" v-if="fwdInfo.date_from">
          <span class="status-label">日期范围</span>
          <span class="status-value ok">{{ fwdInfo.date_from?.slice(0,10) }} ~ {{ fwdInfo.date_to?.slice(0,10) }}</span>
        </div>
      </div>
    </div>

    <div class="data-status-section">
      <h3>量化训练数据</h3>
      <div class="status-cards">
        <div class="status-card" v-for="s in dataStatuses" :key="s.label">
          <span class="status-label">{{ s.label }}</span>
          <span class="status-value" :class="s.ok ? 'ok' : 'warn'">{{ s.text }}</span>
        </div>
        <div class="status-card" v-if="qlibGaps != null">
          <span class="status-label">数据缺口</span>
          <span class="status-value" :class="qlibGaps > 0 ? 'warn' : 'ok'">{{ qlibGaps > 0 ? `${qlibGaps} 处` : '无' }}</span>
        </div>
      </div>
    </div>

    <div class="data-status-section">
      <h3>基础设施健康
        <button class="preload-btn" @click="loadDataSourceHealth" :disabled="dsHealthLoading">{{ dsHealthLoading ? '...' : '刷新' }}</button>
      </h3>
      <div class="status-cards" v-if="dsHealth">
        <template v-for="(s, key) in dsHealth" :key="key">
          <div class="status-card" v-if="key !== 'factor_pool'">
            <span class="status-label">{{ { redis: '缓存数据库', mysql: '业务数据库', tushare: '专业行情源', baostock: '备用行情源', realtime_sources: '实时行情源' }[key] || key }}</span>
            <span class="status-value" :class="s.available ? 'ok' : 'warn'">{{ s.available ? (s.latency_ms != null ? `${s.latency_ms}ms` : '正常') : '异常' }}</span>
            <span class="status-sub" v-if="!s.available && s.error" style="font-size:11px;color:var(--danger)">{{ s.error }}</span>
          </div>
        </template>
        <div class="status-card" v-if="dsHealth.factor_pool">
          <span class="status-label">因子池</span>
          <span class="status-value ok">{{ dsHealth.factor_pool.total_factors ?? 0 }} 个</span>
        </div>
      </div>
      <AppError v-if="dsHealthError" :message="dsHealthError" />
    </div>

    <div class="async-tasks-section">
      <h3>异步任务 (最近20条)</h3>
      <table class="task-table" v-if="tasks.length">
        <thead>
          <tr><th>任务编号</th><th>描述</th><th>状态</th><th>更新时间</th></tr>
        </thead>
        <tbody>
          <tr v-for="t in tasks" :key="t.job_id">
            <td><code>{{ t.job_id }}</code></td>
            <td>{{ t.description }}</td>
            <td><span class="task-status" :class="t.status">{{ t.status }}</span></td>
            <td>{{ fmt(t.updated_at) }}</td>
          </tr>
        </tbody>
      </table>
      <AppEmpty v-else text="无异步任务" />
      <AppError v-if="tasksError" :message="tasksError" />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onActivated } from 'vue'
import { useScheduler } from '../composables/useScheduler.js'
import SchedulerCard from '../components/SchedulerCard.vue'
import AppEmpty from '../components/AppEmpty.vue'
import AppError from '../components/AppError.vue'
import api from '../api'

const risk = useScheduler('risk-scan')
const dataColl = useScheduler('data-collection')
const sent = useScheduler('sentiment')
const smoke = useScheduler('smoke-test')
const notice = useScheduler('notice')


const dataStatuses = ref([])
const redisStats = ref(null)
const tasks = ref([])
const tasksError = ref('')
const preloading = ref(false)
const preloadStatus = ref('')
const preloadDone = ref(false)
const preloadMsg = ref('')
const fwdWarming = ref(false)
const fwdWarmStatus = ref('')
const fwdInfo = ref(null)
const fwdStatus = ref('')

const threadPools = ref(null)
const threadLoading = ref(false)
const threadError = ref('')
const threadHealth = ref('ok')
const poolHealthText = ref('正常')

function poolLabel(kind) {
  return { default: '通用线程池', light: 'Light 后台' }[kind] || kind
}

async function loadThreadPools() {
  threadLoading.value = true
  threadError.value = ''
  try {
    const res = await api.get('/system/threads')
    const d = res?.data ?? res
    threadPools.value = Object.values(d?.pools ?? {})
    threadHealth.value = d?.health ?? 'ok'
    poolHealthText.value = d?.health === 'backpressure' ? '池饱和' : d?.health === 'busy' ? '忙碌中' : '正常'
  } catch (e) {
    threadError.value = e.message
  } finally {
    threadLoading.value = false
  }
}

async function triggerPreload() {
  preloading.value = true
  preloadDone.value = false
  preloadStatus.value = '...'
  try {
    const res = await api.post('/whitebox/preload-cache', { pool: 'a500' }, { timeout: 300000 })
    preloadDone.value = true
    preloadMsg.value = '已启动，预计2-3分钟完成'
    setTimeout(() => { preloadDone.value = false }, 8000)
  } catch (e) {
    preloadMsg.value = '启动失败: ' + e.message
    preloadDone.value = true
  } finally {
    preloading.value = false
    preloadStatus.value = ''
  }
}

async function triggerFwdWarmup() {
  fwdWarming.value = true
  fwdWarmStatus.value = '...'
  fwdStatus.value = ''
  try {
    const res = await api.post('/system/forward-returns/warm-up', {}, { timeout: 180000 })
    const d = res?.data ?? res
    fwdStatus.value = d?.success ? `预热完成 ${d.start}~${d.end}, 可右键查看` : '失败: ' + (d?.error || '未知')
    if (d?.success) await loadFwdStatus()
  } catch (e) {
    fwdStatus.value = '启动失败: ' + e.message
  } finally {
    fwdWarming.value = false
    fwdWarmStatus.value = ''
  }
}

async function loadFwdStatus() {
  try {
    const res = await api.get('/system/forward-returns/status')
    fwdInfo.value = res?.data ?? res
  } catch { fwdInfo.value = null }
}

async function loadRedisStats() {
  try {
    const res = await api.get('/system/health/redis-stats')
    redisStats.value = res?.data ?? res
  } catch { redisStats.value = null }
}

async function loadOverview() {
  await loadDataStatus()
  try {
    const res = await api.get('/system/tasks?limit=20')
    const d = res?.data ?? res
    tasks.value = Array.isArray(d) ? d : []
  } catch (e) {
    tasksError.value = e.message
  }
}

async function loadDataStatus() {
  try {
    const res = await api.get('/qlib/data/status')
    const d = res?.data ?? res
    dataStatuses.value = [
      { label: '日线数据覆盖', ok: (d?.stocks ?? 0) > 0, text: d?.stocks ? `${d.stocks} 只` : '-' },
      { label: '量化训练数据状态', ok: !!d?.synced, text: d?.synced ? '就绪' : '未就绪' },
      { label: '交易日范围', ok: !!d?.first_date, text: d?.first_date ? `${d.first_date} ~ ${d.last_date}` : '-' },
    ]
  } catch {
    dataStatuses.value = [{ label: '获取失败', ok: false, text: 'API异常' }]
  }
}

function fmt(ts) {
  if (!ts) return '-'
  try { return new Date(ts * 1000).toLocaleString('zh-CN') } catch { return String(ts) }
}

function fmtMs(v) {
  if (v == null) return '-'
  const n = Number(v)
  if (n >= 1000) return (n / 1000).toFixed(1) + 's'
  return Math.round(n) + 'ms'
}

function fmtCount(n) {
  if (n == null) return '0'
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M'
  if (n >= 1000) return (n / 1000).toFixed(1) + 'K'
  return String(n)
}

const dsHealth = ref(null)
const dsHealthLoading = ref(false)
const dsHealthError = ref('')
const qlibGaps = ref(null)

async function loadDataSourceHealth() {
  dsHealthLoading.value = true
  dsHealthError.value = ''
  try {
    const res = await api.get('/system/health/data-sources')
    dsHealth.value = res?.data ?? res
  } catch (e) { dsHealthError.value = e.message }
  dsHealthLoading.value = false
}

async function loadQlibGaps() {
  try {
    const res = await api.get('/qlib/data/gaps')
    const d = res?.data ?? res
    qlibGaps.value = Array.isArray(d) ? d.length : (d?.total || 0)
  } catch { qlibGaps.value = null }
}

function initAll() {
  risk.startPolling()
  dataColl.startPolling()
  sent.startPolling()
  smoke.startPolling()
  notice.startPolling()
  loadOverview()
  loadRedisStats()
  loadFwdStatus()
  loadThreadPools()
  loadDataSourceHealth()
  loadQlibGaps()
}
onMounted(() => initAll())
onActivated(() => { loadOverview(); loadRedisStats(); loadFwdStatus(); loadThreadPools(); loadDataSourceHealth(); loadQlibGaps() })
</script>

<style scoped>
.system-monitor { padding:28px 32px }
.system-monitor h2 { color:var(--text); font-size:var(--fs-2xl); font-weight:700; margin:0 0 24px 0; letter-spacing:0 }
.system-monitor h3 { color:var(--text-secondary); font-size:var(--fs-lg); font-weight:600; margin:24px 0 12px 0; display:flex; align-items:center; gap:12px }
.preload-btn { font-size:var(--fs-sm); padding:4px 14px; border:1px solid var(--primary); border-radius:4px; background:transparent; color:var(--primary); cursor:pointer; white-space:nowrap }
.preload-btn:hover:not(:disabled) { background:rgba(59,130,246,.1) }
.preload-btn:disabled { opacity:.6; cursor:not-allowed }
.preload-hint { font-size:var(--fs-sm); color:var(--success); font-weight:400 }
.scheduler-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr)); gap:16px }
.status-cards { display:flex; gap:12px; flex-wrap:wrap }
.status-card { flex:1; min-width:150px; background:var(--card-bg); border:1px solid var(--border); border-radius:var(--radius); padding:14px 18px }
.status-label { display:block; font-size:var(--fs-sm); color:var(--text-muted); margin-bottom:4px }
.status-value { font-size:var(--fs-xl); font-weight:700 }
.status-value.ok { color:var(--success) }
.status-value.warn { color:var(--warning) }
.task-table { width:100%; border-collapse:collapse; font-size:var(--fs-md); margin-top:8px }
.task-table th { text-align:left; padding:8px 12px; border-bottom:1px solid var(--border); color:var(--text-muted); font-weight:600; font-size:var(--fs-sm) }
.task-table td { padding:8px 12px; border-bottom:1px solid rgba(42,42,80,.4); color:var(--text-secondary) }
.task-table code { font-size:var(--fs-xs); background:rgba(59,130,246,.1); padding:2px 6px; border-radius:3px; color:var(--primary) }
.task-status { font-size:var(--fs-xs); padding:2px 8px; border-radius:10px }
.task-status.completed { background:rgba(34,197,94,.12); color:var(--success) }
.task-status.failed { background:rgba(239,68,68,.12); color:var(--danger) }
.task-status.running { background:rgba(59,130,246,.12); color:var(--primary) }
.task-status.pending { background:rgba(100,116,139,.1); color:var(--text-muted) }
/* 实盘策略卡片 (不使用 SchedulerCard 组件) */
.scheduler-card { background:var(--card-bg); border:1px solid var(--border); border-radius:var(--radius); padding:18px; min-width:240px }
.sch-indicator { display:inline-block; width:8px; height:8px; border-radius:50% }
.sch-indicator.on { background:var(--success) }
.sch-indicator.off { background:var(--text-muted) }
.sch-title { font-size:var(--fs-base); font-weight:600; color:var(--text); margin-left:8px }
.sch-body { display:flex; flex-direction:column; gap:4px; margin:14px 0 }
.sch-stat { display:flex; justify-content:space-between; font-size:var(--fs-sm) }
.sch-label { color:var(--text-muted) }
.sch-value { color:var(--text-secondary) }
.sch-actions { display:flex; gap:8px }
/* 线程池监控 */
.health-badge { font-size:var(--fs-sm); padding:2px 10px; border-radius:10px; font-weight:500 }
.health-badge.ok { background:rgba(34,197,94,.12); color:var(--success) }
.health-badge.busy { background:rgba(245,158,11,.12); color:var(--warning) }
.health-badge.backpressure { background:rgba(239,68,68,.12); color:var(--danger) }
.thread-pool-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:12px }
.thread-pool-card { background:var(--card-bg); border:1px solid var(--border); border-radius:var(--radius); padding:16px }
.tp-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px }
.tp-name { font-size:var(--fs-base); font-weight:600; color:var(--text) }
.tp-health { font-size:var(--fs-xs); padding:2px 8px; border-radius:8px; font-weight:500 }
.tp-health.ok { background:rgba(34,197,94,.12); color:var(--success) }
.tp-health.busy { background:rgba(245,158,11,.12); color:var(--warning) }
.tp-health.backpressure { background:rgba(239,68,68,.12); color:var(--danger) }
.tp-bar-wrap { display:flex; align-items:center; gap:8px; margin-bottom:12px }
.tp-bar { flex:1; height:6px; background:rgba(100,116,139,.15); border-radius:3px; overflow:hidden }
.tp-bar-fill { height:100%; border-radius:3px; transition:width .4s ease }
.tp-bar-fill.ok { background:var(--success) }
.tp-bar-fill.busy { background:var(--warning) }
.tp-bar-fill.backpressure { background:var(--danger) }
.tp-bar-label { font-size:var(--fs-xs); color:var(--text-muted); min-width:36px; text-align:right }
.tp-stats { display:flex; gap:16px; margin-bottom:10px }
.tp-stat { flex:1 }
.tp-label { display:block; font-size:var(--fs-xs); color:var(--text-muted) }
.tp-value { font-size:var(--fs-base); font-weight:600; color:var(--text-secondary) }
.tp-value.warn { color:var(--warning) }
.tp-footer { font-size:var(--fs-xs); color:var(--text-muted); display:flex; justify-content:space-between }
</style>
