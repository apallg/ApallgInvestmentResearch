<template>
  <div class="tuner-panel card">
    <div class="card-title">快速调优</div>
    <p class="tuner-desc">检测信号源有效性，优化权重组合，然后一键应用到选股。</p>

    <div v-if="tunerError" class="error">{{ tunerError }}</div>

    <!-- Phase 1 -->
    <div class="phase-section">
      <div class="phase-header" @click="showPhase1 = !showPhase1">
        <span class="phase-badge p1">Phase 1</span>
        <span>信号有效性检测</span>
        <span class="phase-arrow">{{ showPhase1 ? '▼' : '▶' }}</span>
      </div>
      <div v-if="showPhase1" class="phase-body">
        <div class="phase-form">
          <label>股票池 <select v-model="p1Pool"><option v-for="o in POOL_OPTIONS" :key="o.value" :value="o.value">{{ o.label }}</option></select></label>
          <label>回看周数 <input v-model.number="p1Weeks" type="number" min="4" max="52" /></label>
          <label>评估前K名 <input v-model.number="p1TopK" type="number" min="5" max="50" /></label>
          <label>起始日期 <input v-model="p1DateStart" type="date" /></label>
          <label>结束日期 <input v-model="p1DateEnd" type="date" /></label>
        </div>
        <button class="btn" :disabled="p1Running" @click="runPhase1">{{ p1Running ? '检测中...' : '开始检测' }}</button>

        <div v-if="p1Result" class="phase-result">
          <div class="result-summary">
            有效 {{ p1Result.summary?.valid_count || 0 }} /
            失效 {{ p1Result.summary?.invalid_count || 0 }} /
            无数据 {{ p1Result.summary?.no_data_count || 0 }}
          </div>
          <div class="select-row">
            <button class="btn-sm" @click="selectAllValid">全选有效</button>
            <button class="btn-sm" @click="clearSelection">清空</button>
            <span class="selected-count">已选 {{ selectedKeys.length }} 个信号</span>
          </div>
          <table class="p1-table">
            <thead>
              <tr><th>选</th><th>信号</th><th>状态</th><th>Rank IC</th><th>20日收益</th><th>20日胜率</th></tr>
            </thead>
            <tbody>
              <tr v-for="s in p1Result.signals" :key="s.key" :class="{ row: true, valid: s.status === '有效', nodata: s.status === '无数据' }">
                <td><input type="checkbox" :checked="selectedKeys.includes(s.key)" :disabled="s.status !== '有效'" @change="toggleKey(s.key)" /></td>
                <td>{{ s.label }}</td>
                <td><span :class="['tag', s.status === '有效' ? 'tag-low' : s.status === '失效' ? 'tag-high' : '']">{{ s.status }}</span></td>
                <td :class="s.metrics?.rank_ic > 0 ? 'up' : 'down'">{{ s.metrics?.rank_ic?.toFixed(4) || '--' }}</td>
                <td :class="s.metrics?.mean_20d > 0 ? 'up' : 'down'">{{ s.metrics?.mean_20d != null ? (s.metrics.mean_20d * 100).toFixed(2) + '%' : '--' }}</td>
                <td>{{ s.metrics?.hit_rate_20d != null ? (s.metrics.hit_rate_20d * 100).toFixed(1) + '%' : '--' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Phase 2 -->
    <div class="phase-section">
      <div class="phase-header" @click="showPhase2 = !showPhase2">
        <span class="phase-badge p2">Phase 2</span>
        <span>权重优化</span>
        <span class="phase-tip" v-if="selectedKeys.length && !p2Result">已选 {{ selectedKeys.length }} 个信号</span>
        <span class="phase-arrow">{{ showPhase2 ? '▼' : '▶' }}</span>
      </div>
      <div v-if="showPhase2" class="phase-body">
        <div class="phase-form">
          <label>候选数量 <input v-model.number="p2TopN" type="number" min="10" max="100" /></label>
          <label>回看周数 <input v-model.number="p2Weeks" type="number" min="4" max="52" /></label>
          <label>评估前K名 <input v-model.number="p2TopK" type="number" min="5" max="50" /></label>
        </div>
        <div v-if="!selectedKeys.length" class="msg">请先在 Phase 1 中选择有效信号</div>
        <button class="btn" :disabled="p2Running || !selectedKeys.length" @click="runPhase2">{{ p2Running ? '优化中...' : '开始优化' }}</button>
        <div v-if="p2Running && p2Progress" class="tuner-progress">
          {{ p2Progress.message || p2Progress.stage }}
          <span v-if="p2Progress.total"> · {{ p2Progress.current }}/{{ p2Progress.total }}</span>
          <span v-if="p2Progress.backing_mode === 'memmap'"> · 只读共享映射</span>
          <span v-if="p2Progress.mapped_mb != null"> · {{ p2Progress.mapped_mb }} MB</span>
        </div>

        <div v-if="p2Result" class="phase-result">
          <div class="weight-cards">
            <div v-for="w in optimizedWeights" :key="w.key" class="weight-card">
              <span class="wc-label">{{ w.label || w.key }}</span>
              <span class="wc-bar-bg"><span class="wc-bar" :style="{ width: (w.weight * 100) + '%' }"></span></span>
              <span class="wc-val">{{ (w.weight * 100).toFixed(1) }}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 操作栏 -->
    <div class="tuner-actions" v-if="p2Result">
      <button class="btn" @click="applyWeights">应用并重新选股</button>
      <span class="text-gray" style="font-size:var(--fs-sm)">将优化权重写入选股配置并自动重新筛选</span>
    </div>
    <div class="tuner-footer">
      <router-link to="/factors">前往因子管理 →</router-link>
      <span class="text-gray" style="font-size:var(--fs-xs)">深度因子分析、滚动筛选、因子池管理</span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onUnmounted } from 'vue'
import { POOL_OPTIONS } from '../constants/index.js'
import { startPhase1SignalTest, startPhase2WeightOpt, getTunerResult } from '../api/tuner.js'

const props = defineProps({
  currentPool: { type: String, default: 'all' },
})
const emit = defineEmits(['applyWeights'])

const tunerError = ref('')
const showPhase1 = ref(true)
const showPhase2 = ref(false)

const p1Pool = ref('all')
const p1Weeks = ref(12)
const p1TopK = ref(10)
const p1DateStart = ref('')
const p1DateEnd = ref('')
const p1Running = ref(false)
const p1Result = ref(null)
const selectedKeys = ref([])
let p1Timer = null

const p2TopN = ref(30)
const p2Weeks = ref(12)
const p2TopK = ref(10)
const p2Running = ref(false)
const p2Result = ref(null)
const p2Progress = ref(null)
let p2Timer = null

onUnmounted(() => {
  if (p1Timer) clearInterval(p1Timer)
  if (p2Timer) clearInterval(p2Timer)
})

const optimizedWeights = computed(() => {
  if (!p2Result.value?.configs?.length) return []
  const best = p2Result.value.configs[0]
  const weights = best.weights || best.source_weights || {}
  return Object.entries(weights)
    .filter(([, v]) => v > 0.001)
    .map(([k, v]) => ({ key: k, label: k, weight: v }))
    .sort((a, b) => b.weight - a.weight)
})

async function runPhase1() {
  p1Running.value = true
  tunerError.value = ''
  p1Result.value = null
  try {
    const res = await startPhase1SignalTest({
      pool: p1Pool.value, eval_weeks: p1Weeks.value, top_k: p1TopK.value,
      date_start: p1DateStart.value || null, date_end: p1DateEnd.value || null,
      skip_tech_subs: true, regime: null,
    })
    if (res.job_id) {
      p1Timer = setInterval(async () => {
        try {
          const r = await getTunerResult(res.job_id)
          if (r.status === 'done' || r.result) {
            clearInterval(p1Timer)
            p1Result.value = r.result || r
            p1Running.value = false
            showPhase2.value = true
          }
        } catch (e) { clearInterval(p1Timer); tunerError.value = e.message; p1Running.value = false }
      }, 3000)
    } else {
      p1Result.value = res
      p1Running.value = false
      showPhase2.value = true
    }
  } catch (e) { tunerError.value = e.message; p1Running.value = false }
}

function selectAllValid() {
  if (!p1Result.value?.signals) return
  selectedKeys.value = p1Result.value.signals.filter(s => s.status === '有效').map(s => s.key)
}
function clearSelection() { selectedKeys.value = [] }
function toggleKey(key) {
  const idx = selectedKeys.value.indexOf(key)
  if (idx >= 0) selectedKeys.value.splice(idx, 1)
  else selectedKeys.value.push(key)
}

async function runPhase2() {
  p2Running.value = true
  tunerError.value = ''
  p2Result.value = null
  p2Progress.value = null
  try {
    const res = await startPhase2WeightOpt({
      active_signals: selectedKeys.value, pool: p1Pool.value,
      top_n: p2TopN.value, eval_weeks: p2Weeks.value, top_k: p2TopK.value,
      date_start: p1DateStart.value || null, date_end: p1DateEnd.value || null,
      regime: null,
    })
    if (res.job_id) {
      p2Timer = setInterval(async () => {
        try {
          const r = await getTunerResult(res.job_id)
          p2Progress.value = r.progress || p2Progress.value
          if (r.status === 'done' || r.result) {
            clearInterval(p2Timer)
            p2Result.value = r.result || r
            p2Running.value = false
          }
        } catch (e) { clearInterval(p2Timer); tunerError.value = e.message; p2Running.value = false }
      }, 3000)
    } else {
      p2Result.value = res
      p2Running.value = false
    }
  } catch (e) { tunerError.value = e.message; p2Running.value = false }
}

function applyWeights() {
  if (!p2Result.value?.configs?.length) return
  const best = p2Result.value.configs[0]
  const weights = best.weights || best.source_weights || {}
  const skips = best.skips || {}
  const skipSignals = {}
  for (const k of Object.keys(weights)) {
    skipSignals[k] = skips[k] || false
  }
  emit('applyWeights', { weights, skipSignals })
}
</script>

<style scoped>
.tuner-panel { margin-top: 16px }
.tuner-desc { font-size: var(--fs-md); color: var(--text-muted); margin-bottom: 16px }
.phase-section { border: 1px solid var(--border); border-radius: var(--radius-sm); margin-bottom: 12px; overflow: hidden }
.phase-header { display: flex; align-items: center; gap: 10px; padding: 10px 14px; background: var(--bg-secondary); cursor: pointer; user-select: none; font-size: var(--fs-md); color: var(--text) }
.phase-header:hover { background: var(--card-hover) }
.phase-badge { font-size: var(--fs-xs); font-weight: 700; padding: 2px 8px; border-radius: 10px; color: #fff }
.phase-badge.p1 { background: var(--success) }
.phase-badge.p2 { background: var(--primary) }
.phase-arrow { margin-left: auto; font-size: var(--fs-xs); color: var(--text-muted) }
.phase-tip { font-size: var(--fs-xs); color: var(--primary); margin-left: auto; margin-right: 8px }
.phase-body { padding: 14px }
.phase-form { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 12px }
.phase-form label { display: flex; flex-direction: column; gap: 4px; font-size: var(--fs-sm); color: var(--text-muted) }
.phase-form select, .phase-form input { min-width: 100px }
.phase-result { margin-top: 14px }
.tuner-progress { margin-top: 10px; font-size: var(--fs-sm); color: var(--text-muted) }
.result-summary { font-size: var(--fs-md); color: var(--text-secondary); margin-bottom: 10px }
.select-row { display: flex; align-items: center; gap: 8px; margin-bottom: 10px }
.selected-count { font-size: var(--fs-sm); color: var(--text-muted) }
.btn-sm { padding: 3px 10px; font-size: var(--fs-xs); font-family: var(--font); background: var(--bg-secondary); border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-secondary); cursor: pointer }
.btn-sm:hover { border-color: var(--primary); color: var(--primary) }
.p1-table { font-size: var(--fs-sm) }
.p1-table th { font-size: var(--fs-xs) }
.p1-table td { padding: 5px 8px }
.row.valid { background: rgba(34,197,94,.03) }
.row.nodata { opacity: .4 }
.up { color: var(--up) }
.down { color: var(--down) }

.weight-cards { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px }
.weight-card { display: flex; align-items: center; gap: 8px; min-width: 200px; padding: 8px 10px; background: var(--bg-secondary); border-radius: var(--radius-sm) }
.wc-label { font-size: var(--fs-sm); color: var(--text); min-width: 60px }
.wc-bar-bg { flex: 1; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden }
.wc-bar { height: 100%; background: var(--primary); border-radius: 4px; transition: width .4s }
.wc-val { font-size: var(--fs-sm); color: var(--primary); font-weight: 600; min-width: 42px; text-align: right }

.tuner-actions { display: flex; align-items: center; gap: 12px; padding-top: 12px; border-top: 1px solid var(--border) }
.tuner-footer { display: flex; align-items: center; gap: 10px; margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border) }
.tuner-footer a { font-size: var(--fs-sm) }
</style>
