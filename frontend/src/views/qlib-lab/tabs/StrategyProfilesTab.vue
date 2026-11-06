<template>
  <section class="profile-page">
    <StrategyProfileBanner :profile="champion" title="生产正式策略" />
    <div v-if="error" class="profile-alert">{{ localizeStrategyText(error) }}</div>
    <div v-if="message" class="profile-ok">{{ message }}</div>

    <div class="profile-grid">
      <article class="profile-panel">
        <div class="panel-head"><h3>策略版本</h3><button @click="load">刷新</button></div>
        <div class="profile-list">
          <button v-for="row in profiles" :key="`${row.profile_id}-${row.version}`"
            :class="['profile-row', { active: selected === row }]" @click="selected = row">
            <span><strong>{{ row.name }}</strong> · 第{{ row.version }}版</span>
            <span :class="['status', row.status]">{{ strategyStatusText(row.status) }}</span>
            <code>{{ row.fingerprint.slice(0, 10) }}…</code>
          </button>
        </div>
      </article>

      <article class="profile-panel" v-if="selected">
        <div class="panel-head"><h3>{{ selected.name }} · 第{{ selected.version }}版</h3><button @click="copyFingerprint">复制指纹</button></div>
        <dl class="facts">
          <dt>策略标识</dt><dd>{{ selected.profile_id }}/{{ selected.version }}</dd>
          <dt>数据指纹</dt><dd><code>{{ selected.config?.data?.snapshot_id || '未绑定' }}</code></dd>
          <dt>因子指纹</dt><dd><code>{{ selected.config?.fingerprints?.factor || '未生成' }}</code></dd>
          <dt>模型指纹</dt><dd><code>{{ selected.config?.fingerprints?.model_refs || '未生成' }}</code></dd>
          <dt>主模型</dt><dd><code>{{ modelText(selected.config?.model?.refs?.[0]) }}</code></dd>
          <dt>影子模型</dt><dd>{{ selected.config?.model?.shadow_refs?.length || 0 }}个（仅观察，不改变选股）</dd>
          <dt>因子集</dt><dd>{{ localizeStrategyText(selected.config?.factors?.set_name || '未绑定') }} · {{ selected.config?.factors?.keys?.length || 0 }}个因子</dd>
          <dt>用途</dt><dd>{{ selected.config?.factors?.research_only ? '研究观察，不可晋级或新增买入' : '可按准入流程晋级' }}</dd>
          <dt>滚动模型</dt><dd>{{ selected.config?.model?.oos_schedule?.length || 0 }} 个独立样本外模型</dd>
          <dt>预测标签</dt><dd>次日开盘 → 第11个交易日开盘（预测10日，隔离11日）</dd>
          <dt>样本外窗口</dt><dd>{{ selected.config?.windows?.oos_start }} ～ {{ selected.config?.windows?.oos_end }}</dd>
          <dt>实盘回撤</dt><dd>{{ selected.runtime_health ? `${percent(selected.runtime_health.max_drawdown)}${selected.runtime_health.drawdown_breached ? ' · 已阻断新增买入' : ''}` : '尚无实盘净值记录' }}</dd>
        </dl>
        <div v-if="selected.blocked_reasons?.length" class="failures">
          <div v-for="reason in selected.blocked_reasons" :key="reason">× {{ localizeStrategyText(reason) }}</div>
        </div>
        <div class="actions">
          <button v-if="['draft','blocked'].includes(selected.status) && !selected.config?.factors?.research_only" @click="act('validate')">验证策略</button>
          <button v-if="selected.status === 'challenger'" @click="showEvaluate = true">提交5折样本外评估</button>
          <button v-if="selected.status === 'challenger' && selected.latest_evaluation?.passed" @click="act('paper')">开始模拟盘</button>
          <button v-if="selected.status === 'paper'" @click="showPaper = true">登记成熟周期</button>
          <button v-if="selected.status === 'paper'" @click="syncPaper">同步成熟周期</button>
          <button v-if="selected.status === 'paper'" :disabled="!selected.paper_run?.evidence?.passed" @click="confirmAct('activate')">激活正式策略</button>
          <button v-if="selected.status === 'superseded'" @click="confirmAct('rollback')">回滚到此版本</button>
          <button v-if="!selected.is_active && selected.status !== 'retired'" @click="confirmAct('retire')">退役</button>
        </div>
        <div v-if="selected.config?.factors?.research_only" class="evidence">该版本已在创建时核对模型、因子和指纹，只用于研究观察，无需执行生产准入验证。</div>
        <div v-if="selected.latest_evaluation" class="evidence">
          样本外评估：{{ selected.latest_evaluation.passed ? '通过' : '未通过' }} ·
          超额收益回撤比 {{ number(selected.latest_evaluation.metrics_json?.excess_calmar) }} ·
          重采样置信下界 {{ percent(selected.latest_evaluation.metrics_json?.bootstrap_lower_bound) }}
        </div>
        <div v-if="selected.latest_evaluation?.folds_json?.length" class="evidence-table-wrap">
          <table class="evidence-table"><thead><tr><th>样本外折</th><th>窗口</th><th>决策期</th><th>净超额</th><th>回撤</th><th>超额收益回撤比</th></tr></thead><tbody>
            <tr v-for="(fold, index) in selected.latest_evaluation.folds_json" :key="fold.job_id || index">
              <td>{{ index + 1 }}</td><td>{{ fold.window_start }} ～ {{ fold.window_end }}</td><td>{{ fold.decision_dates?.length || 0 }}</td>
              <td>{{ percent(fold.annual_excess) }}</td><td>{{ percent(fold.max_drawdown) }}</td><td>{{ number(fold.excess_calmar) }}</td>
            </tr>
          </tbody></table>
        </div>
        <div v-if="selected.paper_run" class="evidence">
          模拟运行：{{ selected.paper_run.evidence?.summary?.valid_periods || 0 }}/20 ·
          {{ localizeStrategyText(selected.paper_run.evidence?.failures?.join('；') || '准入条件已满足') }}
        </div>
        <div v-if="selected.paper_run?.periods?.length" class="evidence-table-wrap">
          <table class="evidence-table"><thead><tr><th>决策日</th><th>成熟</th><th>有效</th><th>净超额</th><th>换手</th><th>一致率</th></tr></thead><tbody>
            <tr v-for="period in selected.paper_run.periods" :key="period.decision_date">
              <td>{{ period.decision_date }}</td><td>{{ period.matured ? '是' : '否' }}</td><td>{{ period.valid ? '是' : '否' }}</td>
              <td>{{ percent(period.excess_return) }}</td><td>{{ percent(period.one_way_turnover) }}</td><td>{{ percent(period.signal_consistency) }}</td>
            </tr>
          </tbody></table>
        </div>
        <div v-if="profileDiff.length" class="evidence"><strong>与当前正式策略差异</strong><div v-for="item in profileDiff" :key="item">{{ localizeStrategyText(item) }}</div></div>
      </article>
    </div>

    <article class="profile-panel create-panel">
      <div class="panel-head"><h3>复制或创建草稿</h3><span>冻结内容修改后只能形成新版本</span></div>
      <div class="form-row">
        <label>策略标识<input v-model.trim="form.profile_id" /></label>
        <label>名称<input v-model.trim="form.name" /></label>
        <label>快照决策日<input v-model="form.snapshot_as_of" type="date" /></label>
        <button @click="create">创建</button>
      </div>
      <details class="json-details">
        <summary>高级技术配置</summary>
        <label class="json-label">内部配置字段<textarea v-model="configText" rows="15" spellcheck="false" /></label>
      </details>
      <p class="hint">主因子集必须冻结完整因子清单和折外证据；正式候选还必须绑定至少5个季度的独立滚动样本外模型。仅研究版本可使用影子模型，但不能用于新增买入。</p>
    </article>

    <div v-if="showEvaluate" class="modal" @click.self="showEvaluate=false">
      <div class="dialog"><h3>提交独立样本外折</h3><textarea v-model="jobIds" rows="6" placeholder="每行一个已完成回测任务编号" /><button @click="evaluate">提交</button><button @click="showEvaluate=false">取消</button></div>
    </div>
    <div v-if="showPaper" class="modal" @click.self="showPaper=false">
    <div class="dialog"><h3>登记已成熟模拟盘周期</h3><input v-model.number="paperForm.screening_save_id" type="number" placeholder="选股保存编号" /><input v-model="paperForm.decision_id" placeholder="决策编号" /><p class="hint">收益、基准、成交和成熟性由后端重算，不能手工填写。</p><button @click="recordPaper">计算并登记</button><button @click="showPaper=false">取消</button></div>
    </div>
  </section>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import StrategyProfileBanner from '../../../components/StrategyProfileBanner.vue'
import { localizeStrategyText, strategyStatusText } from '../../../utils/strategyTerms.js'
import {
  activateStrategyProfile, createStrategyProfile, evaluateStrategyProfile, getChampionProfile,
  listStrategyProfiles, recordPaperPeriod, retireStrategyProfile, rollbackStrategyProfile,
  startPaperProfile, syncPaperPeriods, validateStrategyProfile,
} from '../../../api/strategyProfiles.js'

const profiles = ref([]), champion = ref(null), selected = ref(null)
const error = ref(''), message = ref(''), showEvaluate = ref(false), showPaper = ref(false)
const jobIds = ref(''), paperForm = ref({ screening_save_id: null, decision_id: '' })
const form = ref({ profile_id: 'diagnostic_baseline', name: '诊断基线', snapshot_as_of: new Date().toISOString().slice(0, 10) })
const configText = ref(JSON.stringify({
  factors: { set_name: 'core_base_30', version: '请填写', keys: [], evidence: [] },
  model: { refs: [{ experiment_id: '', run_id: '' }], oos_schedule: [], fusion: 'single', weights: [1], label_mode: 'tradable_open_to_open', label_horizon: 10 },
  windows: { train_start: '', train_end: '', valid_start: '', valid_end: '', oos_start: '', oos_end: '' },
}, null, 2))
const profileDiff = computed(() => {
  if (!selected.value || !champion.value) return []
  if (selected.value.profile_id === champion.value.profile_id && selected.value.version === champion.value.version) return []
  const changed = []
  function walk(left, right, prefix = '') {
    for (const key of new Set([...Object.keys(left || {}), ...Object.keys(right || {})])) {
      const path = prefix ? `${prefix}.${key}` : key, a = left?.[key], b = right?.[key]
      if (a && b && typeof a === 'object' && typeof b === 'object' && !Array.isArray(a) && !Array.isArray(b)) walk(a, b, path)
      else if (JSON.stringify(a) !== JSON.stringify(b)) changed.push(`${path}: ${JSON.stringify(a)} → ${JSON.stringify(b)}`)
    }
  }
  walk(champion.value.config, selected.value.config)
  return changed
})

function unwrap(value) { return value && Object.prototype.hasOwnProperty.call(value, 'data') ? value.data : value }
async function load() {
  error.value = ''
  try {
    const [rows, active] = await Promise.all([listStrategyProfiles(), getChampionProfile()])
    profiles.value = unwrap(rows) || []
    champion.value = unwrap(active) || null
    if (selected.value) selected.value = profiles.value.find(row => row.profile_id === selected.value.profile_id && row.version === selected.value.version) || profiles.value[0]
    else selected.value = profiles.value[0] || null
  } catch (e) { error.value = e.message }
}
async function run(action) {
  error.value = ''; message.value = ''
  try { await action(); message.value = '操作完成'; await load() } catch (e) { error.value = typeof e.message === 'string' ? e.message : JSON.stringify(e.message) }
}
async function create() {
  await run(() => createStrategyProfile({ ...form.value, config: JSON.parse(configText.value), parent_version: selected.value?.version || null }))
}
async function act(type) {
  const { profile_id: id, version } = selected.value
  await run(() => type === 'validate' ? validateStrategyProfile(id, version) : startPaperProfile(id, version))
}
async function confirmAct(type) {
  const oldRef = champion.value ? `${champion.value.name} 第${champion.value.version}版\n${champion.value.fingerprint}` : '无'
  const newRef = `${selected.value.name} 第${selected.value.version}版\n${selected.value.fingerprint}`
  const prompt = type === 'retire' ? '确认永久退役该版本？' : `确认执行原子策略切换？\n\n当前正式策略：\n${oldRef}\n\n目标版本：\n${newRef}\n\n差异：\n${profileDiff.value.slice(0, 8).map(localizeStrategyText).join('\n') || '无可见差异'}`
  if (!window.confirm(prompt)) return
  const { profile_id: id, version } = selected.value
  await run(() => type === 'activate' ? activateStrategyProfile(id, version) : type === 'rollback' ? rollbackStrategyProfile(id, version) : retireStrategyProfile(id, version))
}
async function evaluate() {
  const ids = jobIds.value.split(/[,\n]/).map(x => x.trim()).filter(Boolean)
  const { profile_id: id, version } = selected.value
  await run(() => evaluateStrategyProfile(id, version, ids)); showEvaluate.value = false
}
async function recordPaper() {
  const { profile_id: id, version } = selected.value
  await run(() => recordPaperPeriod(id, version, paperForm.value)); showPaper.value = false
}
async function syncPaper() {
  const { profile_id: id, version } = selected.value
  await run(() => syncPaperPeriods(id, version))
}
function copyFingerprint() { navigator.clipboard.writeText(selected.value.fingerprint) }
function number(value) { return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : '--' }
function percent(value) { return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : '--' }
function modelText(ref) { return ref ? `${ref.experiment_id}/${ref.run_id}` : '未绑定' }
watch(selected, value => {
  if (!value) return
  form.value.profile_id = value.profile_id
  form.value.name = `${value.name} 副本`
  configText.value = JSON.stringify(value.config, null, 2)
})
onMounted(load)
</script>

<style scoped>
.profile-page { padding:14px 0; }.profile-grid { display:grid; grid-template-columns:minmax(260px, 0.8fr) minmax(420px, 1.5fr); gap:14px; }.profile-panel { border:1px solid var(--border); border-radius:10px; background:var(--surface); padding:16px; }.panel-head { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:12px; }.panel-head h3 { margin:0; }.profile-list { display:grid; gap:7px; max-height:520px; overflow:auto; }.profile-row { display:grid; grid-template-columns:1fr auto auto; gap:8px; text-align:left; align-items:center; padding:10px; border:1px solid var(--border); border-radius:7px; background:transparent; color:var(--text); }.profile-row.active { border-color:var(--primary); }.status { color:var(--text-muted); }.status.champion { color:var(--primary); }.status.blocked,.profile-alert,.failures { color:var(--up); }.facts { display:grid; grid-template-columns:100px minmax(0,1fr); gap:8px; }.facts dt { color:var(--text-muted); }.facts dd { margin:0; min-width:0; overflow-wrap:anywhere; }.actions,.form-row { display:flex; flex-wrap:wrap; gap:8px; margin:14px 0; }.evidence,.hint { color:var(--text-muted); font-size:12px; margin-top:8px; }.evidence-table-wrap{overflow:auto;margin-top:8px}.evidence-table{width:100%;border-collapse:collapse;font-size:12px}.evidence-table th,.evidence-table td{padding:6px;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap}.profile-ok { color:var(--down); }.create-panel { margin-top:14px; }.form-row label { display:grid; gap:4px; }.json-details{margin-top:10px}.json-details summary{cursor:pointer;color:var(--text-muted)}.json-label { display:grid; gap:6px;margin-top:8px }.json-label textarea,.dialog textarea { width:100%; font-family:var(--font-data); }.modal { position:fixed; inset:0; background:rgba(0,0,0,.45); display:grid; place-items:center; z-index:30; }.dialog { width:min(520px,90vw); background:var(--surface); border-radius:10px; padding:18px; display:grid; gap:10px; }code { overflow-wrap:anywhere; }button { cursor:pointer; }button:disabled { cursor:not-allowed; opacity:.45; }@media(max-width:900px){.profile-grid{grid-template-columns:1fr}}
</style>
