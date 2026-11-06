<template>
  <div class="qlib-lab">
    <!-- ═══ 头部 ═══ -->
    <header class="lab-header">
      <div class="lab-brand">
        <h1 class="lab-title">量化实验室</h1>
        <p class="lab-subtitle">智能模型训练 · 因子计算 · 精细化回测</p>
      </div>
    </header>

    <!-- ═══ 仪器状态栏 ═══ -->
    <div class="instrument-bar">
      <div class="inst-item">
        <span class="inst-value">{{ dataStatus.stocks || '--' }}</span>
        <span class="inst-label">量化标的</span>
        <span class="inst-dot" :class="dataStatus.synced ? 'ok' : 'warn'"></span>
      </div>
      <div class="inst-item">
        <span class="inst-value">{{ dataStatus.dates || '--' }}</span>
        <span class="inst-label">交易日</span>
      </div>
      <div class="inst-item">
        <span class="inst-value">{{ dataStatus.first_date ? dataStatus.first_date.slice(0,7) : '--' }}</span>
        <span class="inst-label">起始月份</span>
      </div>
      <div class="inst-sep"></div>
      <div class="inst-item">
        <span class="inst-value accent">{{ trainedModels.length }}</span>
        <span class="inst-label">已训练模型</span>
      </div>
      <div class="inst-item">
        <span class="inst-value accent" v-if="bestIC !== null">{{ (bestIC * 100).toFixed(2) }}%</span>
        <span class="inst-value muted" v-else>--</span>
        <span class="inst-label">最佳信息系数</span>
      </div>
      <div class="inst-item">
        <span class="inst-value accent" v-if="bestICIR !== null">{{ bestICIR.toFixed(2) }}</span>
        <span class="inst-value muted" v-else>--</span>
        <span class="inst-label">最佳信息系数稳定性</span>
      </div>
      <div class="inst-sep"></div>
      <div class="inst-item">
        <span class="inst-status" :class="schedulerStatuses.collection?.running ? 'live' : 'idle'">
          {{ schedulerStatuses.collection?.running ? '● 运行中' : '○ 待命' }}
        </span>
        <span class="inst-label">定时采集</span>
      </div>
    </div>

    <!-- ═══ Tab 导航 ═══ -->
    <nav class="lab-tabs" aria-label="量化实验室视图" role="tablist">
      <button
        v-for="t in tabs"
        :key="t.key"
        :class="{ active: activeTab === t.key }"
        role="tab"
        :aria-selected="activeTab === t.key"
        @click="activeTab = t.key"
      >{{ t.label }}</button>
    </nav>

    <TrainTab v-if="activeTab === 'train'" />

    <TrainedModelsTab v-if="activeTab === 'trained'" />

    <PredictTab v-if="activeTab === 'predict'" />

    <BacktestTab v-if="activeTab === 'backtest'" />

    <TimingTab v-if="activeTab === 'timing'" />

    <OptunaTab v-if="activeTab === 'optuna'" />

    <RollingTrainTab v-if="activeTab === 'rolling'" />

    <StrategyProfilesTab v-if="activeTab === 'profiles'" />

    <!-- ════════════════════════════════════════════
         
         ════════════════════════════════════════════ -->
      <!--
    
      
      
      
        
          
          
        </div>
        
          
            
            
            
            
          </div>
        </div>
      </div>
    </section>
      -->
  </div>
</template>

<script>
import { BacktestTab, TrainTab, TrainedModelsTab, PredictTab, TimingTab, OptunaTab, RollingTrainTab, StrategyProfilesTab } from './qlib-lab/tabs'
import qlibLabMixin from './qlib-lab/qlibLabMixin'
/*

export default {
  name: 'QlibLab',
  components: {
    BacktestTab,
    TrainTab,
    TrainedModelsTab,
    PredictTab,
    TimingTab,
    OptunaTab,
    RollingTrainTab,
    StrategyProfilesTab,
  },
  provide() {
    return { qlibLab: this }
  },
  data: () => ({
    _destroyed: false,
    activeTab: 'trained',
    tabs: [
      { key: 'train', label: '模型训练' },
      { key: 'timing', label: '择时模型' },
      { key: 'optuna', label: '自动调参' },
      { key: 'rolling', label: '滚动训练' },
      { key: 'trained', label: '已训练模型' },
      { key: 'predict', label: '模型预测' },
      { key: 'backtest', label: '独立回测' },
      
    ],
    dataStatus: { synced: false },
    schedulerStatuses: {},
    availableModels: [],
    trainCfg: {
      model_name: 'lightgbm', factor_set: 'Alpha158', instruments: 'csi300',
      train_dual: false, train_lgb_lstm: false, enable_crossover: false, enable_cross_sectional: false,
      label_norm: 'none', early_stop_metric: 'mse', param_profile: 'aggressive',
      use_reweighter: true, with_linear: false, with_catboost: false,
      use_curated: 'latest',
      train_start: '2022-01-01', train_end: '2024-12-31',
      valid_start: '2025-01-01', valid_end: '2025-06-30',
      test_start: '', test_end: '',
      topk: 50, n_drop: 5, label_horizon: 10, freq: 'day',
      custom_factor_keys: [], lgb_params: {}, xgb_params: {},
    },
    poolFactors: [],
    curatedSets: [],
    optunaParamsAvailable: '',
    paramDefs: [
      ['learning_rate', '学习率'],
      ['max_depth', '最大深度'],
      ['num_leaves', '叶节点数'],
      ['num_trees', '树数量'],
      ['subsample', '样本采样比例'],
      ['colsample_bytree', '特征采样比例'],
      ['reg_alpha', '一阶正则'],
      ['reg_lambda', '二阶正则'],
      ['min_child_samples', '子节点最少样本'],
      ['label_horizon', '择时周期'], ['threshold', '阈值'], ['n_features', '特征数'],
      ['fusion_weight', '融合权重'], ['lstm_seq_len', '时序窗口'],
    ],
    codePrefixOptions: CODE_PREFIX_OPTIONS,
    trainCodePrefixLabels: [],
    predictCodePrefixLabels: [],
    training: false, trainResult: null, trainError: '',
    trainedModels: [],
    loadingModels: false,
    selectedModel: null,
    modelSort: 'ic',
    modelFilter: 'all',
    modelTypeFilter: 'all',
    predictCfg: { instruments: 'csi300', start_time: '2024-06-01', end_time: '2024-12-31', topk: 30 },
    predicting: false, predictResult: null, predictError: '', showFactors: false,
    backtestCfg: { topk: 50, n_drop: 5, start_time: '2023-01-01', end_time: '2024-12-31' },
    backtesting: false, backtestResult: null, backtestError: '',
    optunaCuratedSets: [],
    optunaCfg: {
      model_name: 'lightgbm', search_dual: false,
      factor_set: 'Alpha158', use_curated: 'latest', instruments: 'csi300',
      train_start: '2022-01-01', train_end: '2024-12-31',
      valid_start: '2025-01-01', valid_end: '2025-06-30',
      test_start: '', test_end: '',
      label_horizon: 10, label_norm: 'none',
      n_trials: 50, timeout: 0,
    },
    optunaRunning: false, optunaResult: null, optunaMsg: '', optunaError: false,
    timingTrainCfg: {
      label_horizon: 10, threshold: 0.01,
      train_start: '', train_end: '',
      valid_start: '', valid_end: '',
      model_type: 'linear',
      task_type: 'binary',
    },
    timingTraining: false, timingTrainResult: null, timingTrainMsg: '', timingTrainError: false,
    regimeModelInfo: null,
    regimeStateLabels: ['高波快涨', '高波快跌', '高波震荡', '稳步上涨', '稳步下跌', '低波盘整'],
    timingPredictDate: '', timingPredicting: false, timingPredictResult: null, timingPredictError: '',
    timingRegimeResult: null, timingRegimeLoading: false,
    timingBTCfg: { start_date: '2020-01-01', end_date: '2026-07-01', benchmark: 'sh000300', model_name: '' },
    timingBTing: false, timingBTResult: null, timingBTMsg: '', timingBTError: false,
    timingShowTrain: true, timingShowRatio: false, ratioSaved: false,
    ratioConfig: { 0: 1.50, 1: 0.15, 2: 0.50, 3: 1.50, 4: 0.15, 5: 0.50 },
    ratioLabels: { 0: '高波快涨', 1: '高波快跌', 2: '高波震荡', 3: '稳步上涨', 4: '稳步下跌', 5: '低波盘整' },
    rollingCuratedSets: [],
    rollingCfg: {
      model_name: 'lightgbm_rank', factor_set: '__pool__', pool: 'csi300',
      train_dual: false, enable_crossover: false, enable_cross_sectional: false,
      label_norm: 'none', early_stop_metric: 'mse', param_profile: 'aggressive',
      use_reweighter: true, with_linear: false, with_catboost: false,
      use_curated: 'latest', label_horizon: 10,
      start_anchor: '2023-01-01', end_anchor: '',
      step_months: 3, train_years: 5, valid_months: 6,
      filter_mode: false,
    },
    rollingRunning: false, rollingResult: null, rollingMsg: '', rollingError: '', rollingPct: 0,
    rollingPrefixLabels: [],
    
    
  }),
  computed: {
    isModelStale() {
      if (!this.regimeModelInfo?.last_train_date) return false
      const d = new Date(this.regimeModelInfo.last_train_date)
      if (isNaN(d.getTime())) return false
      return (Date.now() - d.getTime()) > 30 * 24 * 3600 * 1000
    },
    activeCuratedSets() {
      const dirs = ['trend_momentum', 'volume_breakout', 'oversold_rebound', 'value_defense',
                    'core_base', 'trend_satellite', 'rebound_satellite']
      return (this.curatedSets || []).filter(s => dirs.includes(s.name))
    },
    topFactors() {
      return (this.predictResult?.feature_importance || []).slice(0, 5)
    },
    rollingFirstRound() {
      const offsetMonths = (dateStr, months) => {
        const d = new Date(dateStr); d.setMonth(d.getMonth() + months); return d.toISOString().slice(0, 10)
      }
      const offsetYears = (dateStr, years) => {
        const d = new Date(dateStr); d.setFullYear(d.getFullYear() + years); return d.toISOString().slice(0, 10)
      }
      const dayBefore = (dateStr) => {
        const d = new Date(dateStr); d.setDate(d.getDate() - 1); return d.toISOString().slice(0, 10)
      }
      const cfg = this.rollingCfg
      const step = cfg.step_months || 3
      const trainY = cfg.train_years || 5
      const validM = cfg.valid_months || 6
      const start = cfg.start_anchor || '2023-01-01'
      const end = cfg.end_anchor || new Date().toISOString().slice(0, 10)
      const validStart = offsetMonths(start, -validM)
      const trainStart = offsetYears(validStart, -(trainY - validM / 12))
      const testEnd = offsetMonths(start, step)
      let lastAnchor = ''; let nRounds = null
      try {
        let cur = new Date(start), endD = new Date(end), count = 0
        while (cur <= endD) { count++; cur.setMonth(cur.getMonth() + step) }
        lastAnchor = new Date(cur.getTime() - step * 30.44 * 86400 * 1000).toISOString().slice(0, 10)
        nRounds = count
      } catch {}
      return {
        train: trainStart + ' ~ ' + dayBefore(validStart),
        valid: validStart + ' ~ ' + dayBefore(start),
        test: start + ' ~ ' + dayBefore(testEnd),
        lastAnchor, nRounds,
      }
    },
    bestIC() {
      const models = this.trainedModels.filter(m => m.ic !== null && !Number.isNaN(m.ic))
      if (!models.length) return null
      return Math.max(...models.map(m => m.ic))
    },
    bestICIR() {
      const models = this.trainedModels.filter(m => m.icir !== null && !Number.isNaN(m.icir))
      if (!models.length) return null
      return Math.max(...models.map(m => m.icir))
    },
    dualCount() { return this.trainedModels.filter(m => m.is_dual).length },
    singleCount() { return this.trainedModels.filter(m => !m.is_dual).length },
    predictableCount() { return this.trainedModels.filter(m => m.has_predictions).length },
    sortedModels() {
      let list = [...this.trainedModels]
      if (this.modelTypeFilter === 'timing') {
        list = list.filter(m => ['timing', 'regime', 'regime_v2'].includes(m.model_type))
      } else if (this.modelTypeFilter === 'predict') {
        list = list.filter(m => !['timing', 'regime', 'regime_v2'].includes(m.model_type))
      }
      if (this.modelFilter === 'dual') list = list.filter(m => m.is_dual)
      else if (this.modelFilter === 'single') list = list.filter(m => !m.is_dual)
      else if (this.modelFilter === 'rolling') list = list.filter(m => m.is_rolling)
      else if (this.modelFilter === 'hasIC') list = list.filter(m => m.ic !== null)
      const cmp = (aVal, bVal) => {
        if (!Number.isFinite(aVal) && !Number.isFinite(bVal)) return 0
        return (bVal ?? -Infinity) - (aVal ?? -Infinity)
      }
      return list.sort((a, b) => {
        switch (this.modelSort) {
          case 'icir': return cmp(a.icir, b.icir)
          case 'date': return (b.trained_at || '').localeCompare(a.trained_at || '')
          case 'sharpe': return cmp(a.ls_sharpe, b.ls_sharpe)
          case 'ic':
          default: return cmp(a.ic, b.ic)
        }
      })
    },
    timingModels() {
      return (this.trainedModels || []).filter(m => m.model_type === 'timing')
    },
  },
  mounted() {
    this.loadDataStatus()
    this.loadModels()
    this.loadTrainedModels()
    this.loadSchedulerStatuses()
    this.loadOptunaParams()
    this.loadRegimeModelInfo()
  },
  beforeUnmount() {
    this._destroyed = true
  },
  methods: {
    async loadDataStatus() {
      try {
        const r = await qlibApi.getDataStatus()
        if (r?.data) this.dataStatus = r.data
      } catch(e) { this.dataStatus.synced = false }
    },
    async loadRegimeModelInfo() {
      try {
        const r = await getRegimeModelInfo()
        this.regimeModelInfo = r?.data || null
      } catch(e) { this.regimeModelInfo = null }
    },
    heatColor(v) {
      const a = Math.max(0, Math.min(1, Number(v) || 0))
      return `rgba(239, 68, 68, ${(a * 0.9).toFixed(2)})`
    },
    async loadModels() {
      try {
        const r = await qlibApi.getModels()
        if (r?.data) this.availableModels = r.data
      } catch(e) { this.availableModels = [] }
    },
    async onFactorSetChange() {
      if (this.trainCfg.factor_set === '__pool__') {
        try {
          const res = await api.get('/factor/pool/for-training')
          this.poolFactors = res.data?.factors || []
          if (!this.trainCfg.custom_factor_keys.length && this.poolFactors.length) {
            this.trainCfg.custom_factor_keys = this.poolFactors.filter(f => f.qlib_expression).map(f => f.key)
          }
        } catch { this.poolFactors = [] }
      }
      if (this.trainCfg.factor_set === '__curated__') {
        try {
          const r = await api.get('/factor/curated-sets')
          this.curatedSets = r.data || []
        } catch { this.curatedSets = [] }
      }
    },
    selectAllPoolFactors() {
      this.trainCfg.custom_factor_keys = this.poolFactors.filter(f => f.qlib_expression).map(f => f.key)
    },
    async startTraining() {
      this.training = true; this.trainResult = null; this.trainError = ''
      const startTime = Date.now()
      try {
        const cfg = { ...this.trainCfg }
        if (cfg.factor_set === '__pool__') {
          cfg.use_factor_pool = true
          cfg.factor_set = 'Alpha158'
          cfg.custom_factor_keys = null
        } else if (cfg.factor_set === '__curated__') {
          cfg.use_curated = cfg.use_curated || 'latest'
          cfg.use_factor_pool = false
          cfg.factor_set = 'Alpha158'
          cfg.custom_factor_keys = null
        } else {
          cfg.use_factor_pool = false
          cfg.use_curated = null
          cfg.custom_factor_keys = null
        }
        const labelToPrefixes = Object.fromEntries(this.codePrefixOptions.map(p => [p[1], p[0]]))
        const prefixes = this.trainCodePrefixLabels.flatMap(label => labelToPrefixes[label] || [])
        if (prefixes.length) cfg.code_prefixes = prefixes
        const r = await qlibApi.trainModel(cfg)
        const jobId = r.data.job_id
        this.trainError = `训练已启动: ${jobId}，轮询中...`
        for (let i = 0; i < 720; i++) {
          await pollSleep(10000)
          if (this._destroyed) return
          try {
            const job = await qlibApi.getTrainStatus(jobId)
            const status = job.data.status
            const elapsed = Math.floor((Date.now() - startTime) / 60000)
            if (status === 'completed') {
              this.trainResult = job.data.result
              this.trainError = ''
              this.loadTrainedModels()
              break
            } else if (status === 'error') {
              this.trainError = job.data.error || '训练失败'
              break
            } else {
              this.trainError = `训练中... (${elapsed}分钟) ${job.data.result?.message || ''}`
            }
          } catch(e) {
            if (e.response?.status === 404) {
              this.trainError = '任务已过期 (服务器可能重启过)，请重新提交训练'
              break
            }
            this.trainError = `轮询中... (网络抖动，继续等待)`
          }
        }
        if (!this.trainResult && !this.trainError.includes('训练失败') && !this.trainError.includes('任务已过期')) {
          this.trainError = '训练超时 (2小时)，任务仍在后台运行，稍后可查看结果'
        }
      } catch(e) { this.trainError = '启动失败: ' + (e.message || e) }
      this.training = false
    },
    async loadTrainedModels() {
      this.loadingModels = true
      try {
        const r = await qlibApi.getTrainedModels()
        if (r?.data) this.trainedModels = r.data
      } catch(e) {
        this.trainedModels = []
      }
      this.loadingModels = false
    },
    formatTime(iso) {
      if (!iso) return ''
      try {
        const d = new Date(iso)
        if (isNaN(d.getTime())) return String(iso).slice(0,16)
        return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`
      } catch { return String(iso).slice(0,16) }
    },
    async deleteModel(m) {
      const name = m.display_name || (m.model_class + ' ' + m.train_period)
      if (!confirm(`确定删除模型:\n\n${name}\n\n因子: ${m.factor_set}\n训练期: ${m.train_period}\n\n此操作不可撤销！`)) return
      try {
        if (m.model_type === 'timing' || m.model_type === 'regime' || m.model_type === 'regime_v2') {
          await deleteTimingModel(m.run_id)
        } else {
          await qlibApi.deleteTrainedModel(m.experiment_id, m.run_id)
        }
        if (this.selectedModel?.run_id === m.run_id) this.selectedModel = null
        await this.loadTrainedModels()
      } catch(e) { alert(`删除失败: ${e.message || e}`) }
    },
    selectModel(model) {
      if (this.selectedModel?.run_id === model.run_id) {
        this.selectedModel = null
      } else {
        this.selectedModel = model
        this.activeTab = 'predict'
      }
    },
    async toggleModelDisabled(m) {
      const action = m.disabled ? '恢复' : '停用'
      const name = m.display_name || (m.model_class + ' ' + m.train_period)
      if (!confirm(`确定${action}模型:\n\n${name}\n\n${action === '停用' ? '停用后所有使用该模型的地方将跳过该模型' : '恢复后可正常使用'}`)) return
      try {
        if (m.disabled) {
          await qlibApi.enableModel(m.experiment_id, m.run_id)
        } else {
          await qlibApi.disableModel(m.experiment_id, m.run_id)
        }
        if (this.selectedModel?.run_id === m.run_id && !m.disabled) {
          this.selectedModel = null
        }
        await this.loadTrainedModels()
      } catch(e) { alert(`${action}失败: ${e.message || e}`) }
    },
    toggleShowFactors() { this.showFactors = !this.showFactors },
    async runPrediction() {
      if (!this.selectedModel) {
        this.predictError = '请先选择一个已训练模型'
        return
      }
      this.predicting = true; this.predictResult = null; this.predictError = ''
      try {
        const cfg = {
          experiment_id: this.selectedModel.experiment_id,
          run_id: this.selectedModel.run_id,
          instruments: this.predictCfg.instruments,
          start_time: this.predictCfg.start_time,
          end_time: this.predictCfg.end_time,
          topk: this.predictCfg.topk,
        }
        const labelToPrefixes = Object.fromEntries(this.codePrefixOptions.map(p => [p[1], p[0]]))
        const prefixes = this.predictCodePrefixLabels.flatMap(label => labelToPrefixes[label] || [])
        if (prefixes.length) cfg.code_prefixes = prefixes
        const r = await qlibApi.predict(cfg)
        if (r?.data) this.predictResult = r.data
      } catch(e) {
        this.predictError = '预测失败: ' + (e.response?.data?.detail || e.message)
      }
      this.predicting = false
    },
    async runBacktest() {
      this.backtesting = true; this.backtestResult = null; this.backtestError = ''
      try {
        const r = await qlibApi.runBacktest(this.backtestCfg)
        this.backtestResult = r.data
      } catch(e) {
        this.backtestError = '独立回测需要服务器上的预测文件（predictions_file），当前界面未提供生成该文件的流程。' +
          '请改用训练任务的「自动回测」查看回测指标：' + (e.response?.data?.detail || e.message)
      }
      this.backtesting = false
    },

    
      try {
        
        
        
          try {
            
            
          
        }))
        
      
    },
    
      
    },

    async loadSchedulerStatuses() {
      try {
        const r = await qlibApi.getSchedulerStatus('collection/')
        this.schedulerStatuses = { collection: r?.data || {} }
      } catch(e) { this.schedulerStatuses = {} }
    },
    async startTimingTrain() {
      this.timingTraining = true; this.timingTrainResult = null; this.timingTrainMsg = '训练已启动...'; this.timingTrainError = false
      try {
        const r = await trainTiming(this.timingTrainCfg)
        const jobId = r.data.job_id
        for (let i = 0; i < 120; i++) {
          await pollSleep(5000)
          if (this._destroyed) return
          try {
            const job = await getTrainStatus(jobId)
            const status = job.data.status
            if (status === 'completed') {
              this.timingTrainResult = job.data.result
              this.timingTrainMsg = '训练完成'
              this.loadTrainedModels()
              break
            } else if (status === 'failed') {
              this.timingTrainMsg = '训练失败'
              this.timingTrainError = true
              break
            } else {
              this.timingTrainMsg = `训练中... (${i * 5}s)`
            }
          } catch { this.timingTrainMsg = '轮询中...' }
        }
      } catch(e) {
        this.timingTrainMsg = '启动失败: ' + (e.response?.data?.detail || e.message)
        this.timingTrainError = true
      }
      this.timingTraining = false
    },
    async runTimingPredict() {
      this.timingPredicting = true; this.timingPredictResult = null; this.timingRegimeResult = null; this.timingPredictError = ''
      try {
        const [timingR, regimeR] = await Promise.allSettled([
          predictTiming(this.timingPredictDate || null),
          predictRegime(this.timingPredictDate || null),
        ])
        if (timingR.status === 'fulfilled') this.timingPredictResult = timingR.value.data
        if (regimeR.status === 'fulfilled') this.timingRegimeResult = regimeR.value.data
        if (timingR.status === 'rejected' && regimeR.status === 'rejected') {
          this.timingPredictError = '预测失败: 无可用模型'
        }
      } catch(e) {
        this.timingPredictError = '预测失败: ' + (e.response?.data?.detail || e.message)
      }
      this.timingPredicting = false
    },
    async loadRatioConfig() {
      try {
        const r = await api.get('/timing/regime/ratio-config')
        if (r?.data) {
          this.ratioConfig = { ...r.data.ratios }
          this.ratioLabels = { ...r.data.labels }
          this.ratioSaved = false
        }
      } catch(e) { alert('加载失败: ' + (e.response?.data?.detail || e.message)) }
    },
    async saveRatioConfig() {
      try {
        const r = await api.post('/timing/regime/ratio-config', { ratios: this.ratioConfig })
        if (r?.data) {
          this.ratioSaved = true
          setTimeout(() => this.ratioSaved = false, 3000)
        }
      } catch(e) { alert('保存失败: ' + (e.response?.data?.detail || e.message)) }
    },
    resetRatioConfig() {
      this.ratioConfig = { 0: 1.50, 1: 0.15, 2: 0.50, 3: 1.50, 4: 0.15, 5: 0.50 }
      this.ratioSaved = false
    },
    async startTimingBT() {
      this.timingBTing = true; this.timingBTResult = null; this.timingBTMsg = '回测已启动...'; this.timingBTError = false
      try {
        const r = await backtestTiming(this.timingBTCfg)
        const jobId = r.data.job_id
        for (let i = 0; i < 120; i++) {
          await pollSleep(5000)
          if (this._destroyed) return
          try {
            const job = await getBacktestStatus(jobId)
            const status = job.data.status
            if (status === 'completed') {
              this.timingBTResult = job.data.result
              this.timingBTMsg = ''
              this.$nextTick(() => this.renderBTChart())
              break
            } else if (status === 'failed') {
              this.timingBTMsg = '回测失败'
              this.timingBTError = true
              break
            } else {
              this.timingBTMsg = `回测中... (${i * 5}s)`
            }
          } catch { this.timingBTMsg = '轮询中...' }
        }
      } catch(e) {
        this.timingBTMsg = '启动失败: ' + (e.response?.data?.detail || e.message)
        this.timingBTError = true
      }
      this.timingBTing = false
    },
    renderBTChart() {
      const el = this.$refs.timingBTChart
      if (!el || !this.timingBTResult?.dates?.length) return
      let chart = echarts.getInstanceByDom(el)
      if (!chart) chart = echarts.init(el)
      const dates = this.timingBTResult.dates
      chart.setOption({
        backgroundColor: 'transparent',
        grid: { left: 50, right: 16, top: 12, bottom: 28 },
        xAxis: { type: 'category', data: dates, axisLabel: { fontSize: 9, interval: Math.floor(dates.length / 6) } },
        yAxis: { type: 'value', axisLabel: { fontSize: 9, formatter: v => v.toFixed(1) } },
        series: [
          { name: '策略', type: 'line', data: this.timingBTResult.equity, smooth: true,
            lineStyle: { color: '#26a69a', width: 2 }, itemStyle: { color: '#26a69a' }, symbol: 'none' },
          { name: '持仓不动', type: 'line', data: this.timingBTResult.hold_equity, smooth: true,
            lineStyle: { color: '#888', width: 1, type: 'dashed' }, symbol: 'none' },
          { name: '基准', type: 'line', data: this.timingBTResult.benchmark_equity, smooth: true,
            lineStyle: { color: '#ff7043', width: 1 }, symbol: 'none' },
        ],
        legend: { bottom: 0, textStyle: { fontSize: 10, color: '#aaa' }, itemWidth: 16, itemHeight: 2 },
        tooltip: { trigger: 'axis' },
      }, true)
    },
    async onOptunaFactorSetChange() {
      if (this.optunaCfg.factor_set === '__curated__') {
        try {
          const r = await api.get('/factor/curated-sets')
          this.optunaCuratedSets = r.data || []
        } catch { this.optunaCuratedSets = [] }
      }
    },
    async startOptuna() {
      this.optunaRunning = true; this.optunaResult = null; this.optunaMsg = '搜索已启动...'; this.optunaError = false
      const startTime = Date.now()
      try {
        const cfg = { ...this.optunaCfg }
        if (cfg.factor_set === '__curated__') {
          cfg.use_curated = cfg.use_curated || 'latest'
          cfg.factor_set = 'Alpha158'
        } else {
          cfg.use_curated = null
        }
        const r = await qlibApi.optunaSearch(cfg)
        const jobId = r.data.job_id
        for (let i = 0; i < 360; i++) {
          await pollSleep(10000)
          if (this._destroyed) return
          try {
            const job = await qlibApi.getTrainStatus(jobId)
            const status = job.data.status
            const elapsed = Math.floor((Date.now() - startTime) / 60000)
            if (status === 'completed') {
              this.optunaResult = job.data.result
              this.optunaMsg = `搜索完成: ${elapsed}分钟, ${job.data.result?.n_trials || cfg.n_trials}轮`
              await this.loadTrainedModels()
              break
            } else if (status === 'error') {
              this.optunaMsg = job.data.error || '搜索失败'
              this.optunaError = true
              break
            } else {
              const phase = job.data.phase || ''
              const msg = job.data.message || ''
              this.optunaMsg = `搜索中... ${elapsed}分钟 | ${phase}: ${msg}`
            }
          } catch (e) {
            if (e.response?.status === 404) {
              this.optunaMsg = '任务已过期 (服务器可能重启过)，请重新提交'
              this.optunaError = true
              break
            }
            this.optunaMsg = `轮询中... (网络抖动)`
          }
        }
        if (!this.optunaResult && !this.optunaError) {
          this.optunaMsg = '搜索超时 (1小时)，任务仍在后台运行'
        }
      } catch(e) {
        this.optunaMsg = '启动失败: ' + (e.response?.data?.detail || e.message)
        this.optunaError = true
      }
      this.optunaRunning = false
    },
    applyOptunaParams() {
      if (this.optunaResult?.lgb_best_params) {
        this.trainCfg.lgb_params = { ...this.optunaResult.lgb_best_params }
        this.trainCfg.xgb_params = this.optunaResult.xgb_best_params ? { ...this.optunaResult.xgb_best_params } : {}
        this.activeTab = 'train'
        this.optunaMsg = '已应用最优参数到训练配置，切换到训练Tab'
      }
    },
    async loadOptunaParams() {
      try {
        const r = await api.get('/qlib/optuna-params')
        const sets = r?.data || []
        if (sets.length) {
          this.optunaParamsAvailable = sets.map(s => s.factor_set).join(', ')
        }
      } catch {}
    },
    async onRollingFactorSetChange() {
      if (this.rollingCfg.factor_set === '__curated__') {
        try {
          const r = await api.get('/factor/curated-sets')
          this.rollingCuratedSets = r.data || []
        } catch { this.rollingCuratedSets = [] }
      }
    },
    async startRollingTrain() {
      this.rollingRunning = true; this.rollingResult = null; this.rollingMsg = '滚动训练已启动...'; this.rollingError = ''; this.rollingPct = 0
      const startTime = Date.now()
      try {
        const cfg = { ...this.rollingCfg }
        if (cfg.factor_set === '__pool__') {
          cfg.use_factor_pool = true; cfg.use_curated = null
          cfg.factor_set = 'Alpha158'; cfg.custom_factor_keys = null
        } else if (cfg.factor_set === '__curated__') {
          cfg.use_curated = cfg.use_curated || 'latest'
          cfg.use_factor_pool = false
          cfg.factor_set = 'Alpha158'; cfg.custom_factor_keys = null
        } else {
          cfg.use_factor_pool = false; cfg.use_curated = null
          cfg.custom_factor_keys = null
        }
        const labelToPrefixes = Object.fromEntries(this.codePrefixOptions.map(p => [p[1], p[0]]))
        const prefixes = this.rollingPrefixLabels.flatMap(label => labelToPrefixes[label] || [])
        if (prefixes.length) cfg.code_prefixes = prefixes
        if (!cfg.end_anchor) cfg.end_anchor = ''
        if (!cfg.model_name) cfg.model_name = 'lightgbm_rank'
        const r = await qlibApi.rollingTrain(cfg)
        const jobId = r.data.job_id
        for (let i = 0; i < 1440; i++) {
          await pollSleep(10000)
          if (this._destroyed) return
          try {
            const job = await qlibApi.getTrainStatus(jobId)
            const status = job.data.status
            const elapsed = Math.floor((Date.now() - startTime) / 60000)
            if (status === 'completed') {
              this.rollingResult = job.data.result
              this.rollingMsg = ''
              this.rollingPct = 100
              break
            } else if (status === 'error') {
              this.rollingError = `训练失败: ${job.data.error}`
              break
            } else {
              const p = job.data.progress || 0
              const t = job.data.total || '?'
              const a = job.data.current_anchor || ''
              this.rollingMsg = `${elapsed}分钟 | 进度 ${p}/${t} | 锚点：${a}`
              this.rollingPct = t !== '?' ? Math.round((p / t) * 100) : 0
            }
          } catch (e) {
            if (e.response?.status === 404) {
              this.rollingError = '任务已过期 (服务器可能重启过)，请重新提交'
              break
            }
            this.rollingMsg = `轮询中... ${Math.floor((Date.now() - startTime) / 60000)}分钟`
          }
        }
      } catch(e) {
        this.rollingError = '启动失败: ' + (e.response?.data?.detail || e.message)
      }
      this.rollingRunning = false
    },
  },
}
*/
export default {
  name: 'QlibLab',
  components: {
    BacktestTab,
    TrainTab,
    TrainedModelsTab,
    PredictTab,
    TimingTab,
    OptunaTab,
    RollingTrainTab,
    StrategyProfilesTab,
  },
  mixins: [qlibLabMixin],
  provide() {
    return { qlibLab: this }
  },
}
</script>

<style>
/* ═══════════════════════════════════════════════
   QlibLab — Lab Bench
   量化实验室 · 仪器面板风格
   ═══════════════════════════════════════════════ */

.qlib-lab {
  max-width: var(--content-max-width);
  margin: 0 auto;
  padding: 0 var(--content-padding) var(--space-8);
}

/* ── 仪器状态栏 ── */
.instrument-bar {
  display: flex;
  align-items: center;
  gap: 0;
  padding: 10px 0;
  margin-bottom: var(--space-3);
  border-bottom: 1px solid var(--border);
  overflow-x: auto;
}
.inst-item {
  display: flex;
  align-items: baseline;
  gap: 6px;
  padding: 0 16px;
  flex-shrink: 0;
}
.inst-item:first-child { padding-left: 0; }
.inst-value {
  font-family: var(--font-data);
  font-size: var(--fs-lg);
  font-weight: 600;
  color: var(--text);
  font-variant-numeric: tabular-nums;
}
.inst-value.accent { color: var(--primary); }
.inst-value.muted { color: var(--text-muted); }
.inst-label {
  font-size: var(--fs-caption);
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.3px;
}
.inst-sep {
  width: 1px;
  height: 24px;
  background: var(--border);
  flex-shrink: 0;
}
.inst-status {
  font-size: var(--fs-sm);
  padding: 2px 8px;
  border-radius: var(--radius-full);
  font-weight: 500;
}
.inst-status.live { background: rgba(52,211,153,.12); color: var(--success); }
.inst-status.idle { background: rgba(100,116,139,.12); color: var(--text-muted); }
.inst-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.inst-dot.ok { background: var(--success); }
.inst-dot.warn { background: var(--warning); }

/* ── 头部 ── */
.lab-header { margin-bottom: 0; }
.lab-brand { }
.lab-title {
  font-size: var(--fs-2xl);
  font-weight: 700;
  color: var(--text);
  letter-spacing: 0;
}
.lab-subtitle {
  font-size: var(--fs-sm);
  color: var(--text-muted);
  margin-top: 2px;
}

/* ── Tab 导航 ── */
.lab-tabs {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: var(--space-5);
  overflow-x: auto;
}
.lab-tabs button {
  padding: 10px 18px;
  border: none;
  background: transparent;
  color: var(--text-muted);
  font-size: var(--fs-body);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  transition: all var(--transition-fast);
  font-family: var(--font);
  white-space: nowrap;
}
.lab-tabs button:hover { color: var(--text); }
.lab-tabs button.active { color: var(--primary); border-bottom-color: var(--primary); }

/* ── 卡片 ── */
.lab-card { padding-bottom: var(--space-8); }
.card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}
.card-title {
  font-size: var(--fs-xl);
  font-weight: 700;
  color: var(--text);
  margin-bottom: 16px;
}
.card-sub {
  font-size: var(--fs-body);
  font-weight: 400;
  color: var(--text-muted);
  margin-left: 8px;
}
.card-badge {
  font-size: var(--fs-caption);
  padding: 2px 8px;
  border-radius: var(--radius-full);
  font-weight: 500;
}
.card-badge.ok { background: rgba(52,211,153,.12); color: var(--success); }
.card-badge.warn { background: rgba(244,167,66,.12); color: var(--warning); }
.card-desc {
  font-size: var(--fs-sm);
  color: var(--text-muted);
  margin-bottom: 16px;
}

/* ── 步骤块 ── */
.step-block {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 16px;
  margin-bottom: 12px;
}
.step-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}
.step-num {
  width: 22px; height: 22px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--primary);
  color: #fff;
  border-radius: 50%;
  font-size: var(--fs-sm);
  font-weight: 700;
  flex-shrink: 0;
}
.step-title { font-size: var(--fs-lg); font-weight: 600; color: var(--text); }
.step-desc { font-size: var(--fs-sm); color: var(--text-muted); margin-left: 4px; }
.step-actions { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.step-collect { flex-direction: column; align-items: stretch; }

/* ── 提示 ── */
.hint-warn { margin-top: 4px; font-size: var(--fs-caption); color: var(--warning); }

/* ── 进度条 ── */
.progress-text { margin-top: 6px; font-size: var(--fs-sm); color: var(--text-muted); }
.train-progress { margin-top: 12px; }
.prog-header { display: flex; justify-content: space-between; font-size: var(--fs-sm); margin-bottom: 4px; }
.prog-title { color: var(--text); }
.prog-pct { color: var(--primary); font-weight: 600; }
.prog-bar { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
.prog-fill { height: 100%; background: var(--primary); border-radius: 2px; transition: width .3s; }

/* ── 数据质量 ── */
.quality-mini { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.check-results { margin-top: 6px; font-size: var(--fs-sm); }
.check-line { padding: 2px 0; color: var(--text-muted); }
.check-line.ok { color: var(--success); }
.check-line.fail { color: var(--danger); }
.check-detail { margin-top: 4px; }
.check-warn { color: var(--warning); }
.check-fail { color: var(--danger); }
.gap-tag { display: inline-block; margin-right: 8px; font-size: var(--fs-caption); }
.detail-summary { cursor: pointer; color: var(--warning); font-size: var(--fs-sm); }
.detail-summary:hover { color: var(--text); }
.detail-line { font-size: var(--fs-caption); color: var(--text-muted); padding: 1px 0; }

/* ── 定时任务卡片 ── */
.task-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
}
.task-card {
  padding: 14px;
  background: var(--card-bg);
  border-radius: var(--radius);
  border: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.task-top { display: flex; justify-content: space-between; align-items: center; }
.task-name { font-size: var(--fs-base); font-weight: 600; color: var(--text); }
.task-status { font-size: var(--fs-caption); padding: 2px 8px; border-radius: var(--radius-full); }
.task-status.live { background: rgba(52,211,153,.12); color: var(--success); }
.task-status.idle { background: rgba(100,116,139,.12); color: var(--text-muted); }
.task-info { font-size: var(--fs-sm); color: var(--text-secondary); }
.task-next { font-size: var(--fs-caption); color: var(--primary); }
.task-actions { margin-top: 4px; display: flex; gap: 6px; }
.fund-row { display: flex; align-items: center; gap: 8px; margin-top: 12px; }
.fund-label { font-size: var(--fs-body); color: var(--text-muted); }

/* ── Form 系统 ── */
.fm-block {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: var(--space-4);
  margin-bottom: var(--space-3);
}
.fm-block-title {
  font-size: var(--fs-caption);
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 10px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}
.form-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px 16px;
  margin-bottom: 8px;
}
.form-field {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: var(--fs-caption);
  color: var(--text-muted);
}
.form-field.check-label { flex-direction: row; align-items: center; gap: 6px; }
.form-field.check-label input[type=checkbox] { width: auto; margin: 0; }
.form-hint { font-size: var(--fs-sm); color: var(--text-muted); }
.form-hint.span-2 { grid-column: 1 / -1; }
.field {
  padding: 5px 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: var(--fs-sm);
}
.field:focus { border-color: var(--primary); }
.field-sm { width: 55px; }
.field-hint { font-size: var(--fs-caption); color: var(--text-muted); }
.field-label { font-size: var(--fs-body); color: var(--text-muted); }
.field-suffix { font-size: var(--fs-sm); color: var(--text-muted); }
.section-title {
  font-size: var(--fs-caption);
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin: 14px 0 8px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}

.option-row { display: flex; gap: 20px; flex-wrap: wrap; margin: 6px 0; }
.cb { display: flex; align-items: center; gap: 4px; font-size: var(--fs-sm); color: var(--text-secondary); cursor: pointer; }
.cb input[type=checkbox] { width: auto; margin: 0; }

.prefix-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin: 8px 0; }
.prefix-label { font-size: var(--fs-sm); color: var(--text-muted); white-space: nowrap; }

/* ── 通用 ── */
.msg { margin-bottom: 12px; }
.msg.error { color: var(--danger); }

.migrate-notice {
  padding: 24px 16px;
  font-size: var(--fs-body);
  color: var(--text-secondary);
  line-height: 1.8;
}
.migrate-notice a {
  color: var(--primary);
  font-weight: 600;
}
.text-muted { font-size: var(--fs-caption); color: var(--text-muted); }

/* ── 标签 ── */
.tag-sm {
  display: inline-block;
  font-size: var(--fs-caption);
  padding: 1px 7px;
  border-radius: var(--radius-full);
}
.tag-sm.ok { background: rgba(52,211,153,.12); color: var(--success); }
.tag-sm.warn { background: rgba(244,167,66,.12); color: var(--warning); }
.tag-sm.bad { background: rgba(245,85,85,.12); color: var(--danger); }
.tag-sm.muted { background: rgba(100,116,139,.12); color: var(--text-muted); }
.tag-sm.dual { background: rgba(107,159,212,.15); color: var(--primary); }
.tag-sm.single { background: rgba(100,116,139,.12); color: var(--text-secondary); }

/* ── 结果框 ── */
.result-box {
  margin-top: 16px;
  padding: 16px;
  background: var(--bg-secondary);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
}
.result-box h3 { margin: 0 0 12px; font-size: var(--fs-lg); color: var(--text); }
.result-box h4 { margin: 10px 0 4px; font-size: var(--fs-base); color: var(--text-secondary); }
.result-box table { width: 100%; border-collapse: collapse; }
.result-box td { padding: 4px 8px; border-bottom: 1px solid var(--border); color: var(--text-secondary); }

/* ── 指标行 ── */
.metrics-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }
.metric-item { text-align: center; min-width: 70px; }
.metric-value {
  font-size: 20px; font-weight: 700; color: var(--text);
  font-family: var(--font-mono);
}
.metric-value.good { color: var(--success); }
.metric-value.warn-text { color: var(--warning); }
.metric-label { font-size: var(--fs-caption); color: var(--text-muted); margin-top: 2px; }

/* ── 已训练模型 ── */
.trained-toolbar { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
.trained-count { font-size: var(--fs-body); color: var(--text-muted); }
.trained-sort { display: flex; align-items: center; gap: 6px; font-size: var(--fs-body); color: var(--text-muted); }

.trained-summary { display: flex; gap: 10px; margin-bottom: 18px; flex-wrap: wrap; }
.summary-item {
  flex: 1; min-width: 72px; text-align: center;
  padding: 12px 6px;
  background: var(--bg-secondary);
  border-radius: var(--radius);
  border: 1px solid var(--border);
}
.summary-val {
  font-size: 18px; font-weight: 700; color: var(--text);
  font-family: var(--font-mono);
}
.summary-val.good { color: var(--success); }
.summary-val.primary-text { color: var(--primary); }
.summary-label { font-size: var(--fs-caption); color: var(--text-muted); margin-top: 3px; }

/* 模型卡片 */
.trained-card {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 14px;
  background: var(--card-bg);
  overflow: hidden;
  transition: border-color var(--transition);
}
.trained-card:hover { border-color: var(--border-light); }
.trained-card.selected { border-color: var(--primary); box-shadow: 0 0 0 1px rgba(107,159,212,.25); }
.trained-card.disabled { opacity: .55; filter: grayscale(0.3); }
.trained-card.disabled .tc-name { text-decoration: line-through; }

.tc-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 16px;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap; gap: 8px;
}
.tc-title-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.tc-name { font-size: 15px; font-weight: 600; color: var(--text); }
.tc-meta {
  display: flex; align-items: center; gap: 8px;
  font-size: var(--fs-caption); color: var(--text-muted);
  font-family: var(--font-mono);
}

.tc-body { display: grid; grid-template-columns: 1fr 1fr; }
@media (max-width: 750px) { .tc-body { grid-template-columns: 1fr; } }

/* 关键指标条 */
.tc-kpi-strip {
  display: flex;
  gap: 4px;
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
  background: rgba(52, 211, 153, .03);
}
.tc-kpi {
  flex: 1;
  min-width: 70px;
  text-align: center;
  padding: 4px 8px;
}
.tc-kpi-val {
  font-family: var(--font-data);
  font-size: var(--fs-lg);
  font-weight: 700;
  color: var(--text);
  font-variant-numeric: tabular-nums;
}
.tc-kpi-val.good { color: var(--success); }
.tc-kpi-val.bad { color: var(--danger); }
.tc-kpi-label {
  font-size: var(--fs-caption);
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.tc-col { padding: 12px 16px; border-right: 1px solid var(--border); }
.tc-col:last-child { border-right: none; }
.col-title {
  font-size: var(--fs-sm); font-weight: 600; color: var(--text-muted);
  margin-bottom: 6px; padding-bottom: 4px;
  border-bottom: 1px solid var(--border);
  text-transform: uppercase; letter-spacing: 0.3px;
}

.tc-config .config-row { display: flex; justify-content: space-between; align-items: center; padding: 3px 0; font-size: var(--fs-sm); }
.config-key { color: var(--text-muted); flex-shrink: 0; margin-right: 8px; }
.config-val { color: var(--text-secondary); text-align: right; }
.config-val.strong { font-weight: 600; color: var(--text); }
.config-val.good { color: var(--success); }
.config-val.bad { color: var(--danger); }

.param-row { display: flex; justify-content: space-between; align-items: center; padding: 2px 0; font-size: var(--fs-sm); }
.param-key { color: var(--text-muted); font-family: var(--font-mono); font-size: var(--fs-caption); }
.param-val { color: var(--text); font-family: var(--font-mono); font-weight: 500; }
.dual-note { margin-top: 8px; padding-top: 6px; border-top: 1px dashed var(--border); }

.metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; }
.metric-cell { }
.metric-name { font-size: var(--fs-sm); color: var(--text-muted); }
.metric-num { font-weight: 600; font-family: var(--font-mono); font-size: var(--fs-sm); }
.metric-num.good { color: var(--success); }
.metric-num.bad { color: var(--danger); }
.metric-num.up { color: var(--up); }
.metric-num.down { color: var(--down); }

.tc-actions {
  display: flex; align-items: center; gap: 8px;
  padding: 10px 16px;
  border-top: 1px solid var(--border);
  background: var(--bg-secondary);
}

.type-badge {
  font-size: var(--fs-caption);
  padding: 1px 7px;
  border-radius: var(--radius-full);
  font-weight: 500;
}
.type-badge.dual { background: rgba(107,159,212,.15); color: var(--primary); }
.type-badge.single { background: rgba(100,116,139,.12); color: var(--text-secondary); }
.type-badge.timing { background: rgba(244,167,66,.15); color: var(--warning); }
.type-badge.regime { background: rgba(239,68,68,.12); color: var(--danger); }
.type-badge.regime-v2 { background: rgba(52,211,153,.12); color: var(--success); }
.type-badge.rolling { background: rgba(168,85,247,.12); color: #a855f7; }
.type-badge.expert { background: var(--primary-muted); color: var(--primary); }
.type-badge.subset { background: var(--bg-elevated); color: var(--warning); }
.type-badge.full-sample { background: var(--card-hover); color: var(--text-muted); }
.type-badge.pred { background: rgba(52,211,153,.12); color: var(--success); }

/* ── 模型预测 ── */
.selected-model-info {
  padding: 12px 14px;
  background: var(--primary-muted);
  border: 1px solid rgba(107,159,212,.15);
  border-radius: var(--radius-sm);
  font-size: var(--fs-base);
  color: var(--text);
}

.feature-importance-box {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  margin-bottom: 20px;
}
.feature-importance-box h3 { margin: 0 0 12px; font-size: var(--fs-lg); color: var(--text); }
.fi-grid { display: flex; flex-direction: column; gap: 6px; }
.fi-item { display: flex; align-items: center; gap: 10px; padding: 4px 0; }
.fi-rank { width: 28px; font-size: var(--fs-sm); font-weight: 700; color: var(--primary); text-align: right; flex-shrink: 0; }
.fi-name { width: 130px; font-size: var(--fs-sm); color: var(--text); font-family: var(--font-mono); flex-shrink: 0; }
.fi-bar { flex: 1; height: 14px; background: var(--border); border-radius: 7px; overflow: hidden; }
.fi-fill { height: 100%; background: var(--primary); border-radius: 7px; transition: width .3s; }
.fi-imp { width: 50px; font-size: var(--fs-sm); color: var(--text-secondary); text-align: right; flex-shrink: 0; font-family: var(--font-mono); }

.pred-group { margin-bottom: 20px; }
.pred-group h4 { margin: 0 0 8px; font-size: var(--fs-base); color: var(--text); }
.toggle-factors {
  display: inline-block; margin-left: 12px; padding: 2px 10px;
  background: var(--primary-muted); border-radius: var(--radius-sm);
  color: var(--primary); font-size: var(--fs-sm); cursor: pointer; font-weight: 400;
}
.toggle-factors:hover { background: rgba(107,159,212,.2); }

.pred-table { width: 100%; border-collapse: collapse; font-size: var(--fs-sm); }
.pred-table th { padding: 6px 8px; border-bottom: 1px solid var(--border); color: var(--text-muted); font-size: var(--fs-caption); text-align: left; font-weight: 600; }
.pred-table td { padding: 5px 8px; border-bottom: 1px solid rgba(42,42,80,.4); color: var(--text-secondary); }
.pred-table tbody tr:hover td { background: var(--primary-muted); }
.positive { color: var(--up) !important; }
.negative { color: var(--down) !important; }
.factor-th { max-width: 60px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.factor-td { font-size: var(--fs-caption); max-width: 60px; overflow: hidden; text-overflow: ellipsis; }




/* ── 按钮微调 ── */
.btn { padding: 8px 18px; background: var(--primary); color: var(--on-primary); border: none; border-radius: var(--radius-sm); cursor: pointer; font-size: var(--fs-body); font-weight: 600; font-family: var(--font); transition: all var(--transition-fast); }
.btn:hover { background: var(--primary-hover); }
.btn:disabled { opacity: .4; cursor: not-allowed; }
.btn.loading { opacity: .7; }
.btn.primary { background: var(--primary); color: var(--on-primary); }
.btn.danger { background: var(--danger); color: #fff; }
.btn.danger:hover { background: #e04444; }
.btn.warn { background: var(--warning); color: #1a1a2e; }
.btn.accent { background: var(--accent-cool); color: #1a1a2e; }
.btn-sm { padding: 4px 10px; font-size: var(--fs-caption); border: 1px solid var(--border); border-radius: var(--radius-sm); cursor: pointer; font-family: var(--font); background: var(--card-bg); color: var(--text); }
.btn-sm:hover { border-color: var(--primary); color: var(--primary); }
.btn-sm.go { background: var(--success); color: #fff; border-color: var(--success); }
.btn-sm.stop { background: var(--danger); color: #fff; border-color: var(--danger); }
.btn-sm.muted { background: transparent; color: var(--text-muted); }
.btn-sm.accent { background: var(--accent-cool); color: #1a1a2e; border-color: var(--accent-cool); }

/* ── 择时模型 ── */
.fm-block-badge {
  font-size: var(--fs-caption);
  padding: 2px 8px;
  border-radius: var(--radius-full);
  font-weight: 500;
  margin-left: 8px;
}
.fm-block-badge.up { background: rgba(239,68,68,.12); color: var(--up); }
.fm-block-badge.down { background: rgba(34,197,94,.12); color: var(--down); }

.tm-signal-row { display: flex; align-items: flex-end; gap: 12px; flex-wrap: wrap; }

.tm-signal-cards { display: flex; gap: 12px; margin-top: 12px; }
.tm-signal-card {
  flex: 1; min-width: 100px; text-align: center;
  padding: 16px 12px;
  background: var(--bg-secondary);
  border-radius: var(--radius);
  border: 1px solid var(--border);
}
.tm-signal-card--up { border-color: rgba(239,68,68,.25); }
.tm-signal-card--down { border-color: rgba(34,197,94,.25); }
.tm-signal-label {
  display: block;
  font-size: var(--fs-caption);
  color: var(--text-muted);
  margin-top: 4px;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}
.tm-signal-icon { font-size: 36px; }

.tm-metrics-strip {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  margin-top: 12px;
  padding: 12px 16px;
  background: var(--bg-secondary);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
}
.tm-metric { text-align: center; min-width: 60px; }
.tm-metric-val {
  font-family: var(--font-data);
  font-size: var(--fs-lg);
  font-weight: 700;
  color: var(--text);
  font-variant-numeric: tabular-nums;
}
.tm-metric-val.up { color: var(--up); }
.tm-metric-val.down { color: var(--down); }
.tm-metric-val--primary { color: var(--primary); }
.tm-metric-label {
  font-size: var(--fs-caption);
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.tm-regime { }
.tm-regime-row { display: flex; align-items: center; gap: 8px; padding: 3px 0; }
.tm-regime-name { width: 90px; font-size: var(--fs-sm); color: var(--text-secondary); flex-shrink: 0; }
.tm-regime-track { flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
.tm-regime-fill { height: 100%; background: var(--primary); border-radius: 3px; transition: width .4s; }
.tm-regime-pct { width: 45px; font-size: var(--fs-sm); color: var(--text-secondary); text-align: right; flex-shrink: 0; }
.tm-regime-foot { margin-top: 8px; padding-top: 6px; border-top: 1px solid var(--border); font-size: var(--fs-sm); color: var(--text-secondary); }
</style>
