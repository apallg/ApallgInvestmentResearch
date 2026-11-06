import * as echarts from 'echarts'
import qlibApi from '../../api/qlib'
import api from '../../api/index.js'
import {
  trainTiming,
  getTrainStatus,
  predictTiming,
  predictRegime,
  backtestTiming,
  getBacktestStatus,
  deleteTimingModel,
  getRegimeModelInfo,
} from '../../api/timing'
import { pollSleep } from '../../utils/poll.js'
import { CODE_PREFIX_OPTIONS } from './config'

export default {
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
      { key: 'profiles', label: '策略版本' },
    ],
    dataStatus: { synced: false },
    schedulerStatuses: {},
    availableModels: [],
    trainCfg: {
      model_name: 'lightgbm', factor_set: 'Alpha158', instruments: 'csi300',
      train_dual: false, train_lgb_lstm: false, enable_crossover: false, enable_cross_sectional: false,
      training_mode: 'full', expert_role: 'trend', label_mode: 'tradable_open_to_open',
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
    strategyProfiles: [],
    predictProfileKey: '',
    predictionJobId: localStorage.getItem('qlibPredictionJobId') || '',
    predictionStatus: '',
    modelSort: 'ic',
    modelFilter: 'all',
    modelTypeFilter: 'all',
    predictCfg: { instruments: 'csi300', start_time: '2024-06-01', end_time: '2024-12-31', decision_date: '2024-12-31', topk: 30 },
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
      label_horizon: 10, label_mode: 'tradable_open_to_open', label_norm: 'none',
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
      model_name: 'lightgbm_rank', factor_set: '__pool__', pool: 'stable_history_500d',
      train_dual: false, enable_crossover: false, enable_cross_sectional: false,
      label_norm: 'csrank', early_stop_metric: 'rank_ic', param_profile: 'conservative',
      use_reweighter: false, with_linear: false, with_catboost: false,
      use_curated: 'latest', label_horizon: 10, label_mode: 'tradable_open_to_open',
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
                    'core_base', 'trend_satellite', 'rebound_satellite',
                    'core_trend_satellite', 'core_rebound_satellite']
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
    fusionCount() { return this.trainedModels.filter(m => m.is_fusion).length },
    singleCount() { return this.trainedModels.filter(m => !m.is_dual && !m.is_fusion).length },
    predictableCount() { return this.trainedModels.filter(m => m.has_predictions).length },
    sortedModels() {
      let list = [...this.trainedModels]
      if (this.modelTypeFilter === 'timing') {
        list = list.filter(m => ['timing', 'regime', 'regime_v2'].includes(m.model_type))
      } else if (this.modelTypeFilter === 'predict') {
        list = list.filter(m => !['timing', 'regime', 'regime_v2'].includes(m.model_type))
      }
      if (this.modelFilter === 'dual') list = list.filter(m => m.is_dual)
      else if (this.modelFilter === 'fusion') list = list.filter(m => m.is_fusion)
      else if (this.modelFilter === 'single') list = list.filter(m => !m.is_dual && !m.is_fusion)
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
    selectedPredictProfile() {
      return this.strategyProfiles.find(row => `${row.profile_id}::${row.version}` === this.predictProfileKey) || null
    },
  },
  mounted() {
    this.loadDataStatus()
    this.loadModels()
    this.loadTrainedModels()
    this.loadSchedulerStatuses()
    this.loadOptunaParams()
    this.loadRegimeModelInfo()
    this.loadStrategyProfiles()
    if (this.predictionJobId) this.pollPrediction(this.predictionJobId)
  },
  beforeUnmount() {
    this._destroyed = true
  },
  methods: {
    async loadStrategyProfiles() {
      try {
        const result = await api.get('/strategy-profiles')
        this.strategyProfiles = (result?.data || []).filter(row =>
          ['challenger', 'paper', 'champion'].includes(row.status) || row.config?.factors?.research_only
        )
      } catch (e) {
        this.strategyProfiles = []
      }
    },
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
    onTrainingModeChange() {
      if (this.trainCfg.training_mode !== 'state_subset') {
        this.trainCfg.label_mode = 'tradable_open_to_open'
        return
      }
      this.trainCfg.train_lgb_lstm = true
      this.trainCfg.expert_role = this.trainCfg.expert_role || 'trend'
      this.trainCfg.label_horizon = 10
      this.trainCfg.label_mode = 'tradable_open_to_open'
      this.trainCfg.label_norm = 'csrank'
      this.trainCfg.param_profile = 'conservative'
    },
    selectAllPoolFactors() {
      this.trainCfg.custom_factor_keys = this.poolFactors.filter(f => f.qlib_expression).map(f => f.key)
    },
    async startTraining() {
      this.training = true; this.trainResult = null; this.trainError = ''
      const startTime = Date.now()
      try {
        const cfg = { ...this.trainCfg }
        if (!cfg.train_lgb_lstm) {
          cfg.training_mode = 'full'
          cfg.expert_role = null
          cfg.label_mode = 'tradable_open_to_open'
        }
        if (cfg.training_mode === 'state_subset') {
          cfg.train_lgb_lstm = true
          cfg.use_factor_pool = false
          cfg.use_curated = null
          cfg.factor_set = 'WhiteBoxHandler'
          cfg.custom_factor_keys = null
          cfg.label_horizon = 10
          cfg.label_mode = 'tradable_open_to_open'
          cfg.label_norm = 'csrank'
          cfg.param_profile = 'conservative'
        } else if (cfg.factor_set === '__pool__') {
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
              // 训练完成后按时间倒序展示, 新模型置顶 (否则 IC 排序时新模型沉底看不到)
              this.modelSort = 'date'
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
      if (!this.selectedPredictProfile && !this.selectedModel) {
        this.predictError = '请选择策略版本，或在研究模式下选择一个已训练模型'
        return
      }
      this.predicting = true; this.predictResult = null; this.predictError = ''
      try {
        let cfg
        if (this.selectedPredictProfile) {
          cfg = {
            profile_id: this.selectedPredictProfile.profile_id,
            profile_version: this.selectedPredictProfile.version,
            start_time: this.predictCfg.decision_date,
            end_time: this.predictCfg.decision_date,
          }
        } else {
          cfg = {
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
        }
        const r = await qlibApi.predict(cfg)
        const jobId = r?.data?.job_id
        if (!jobId) throw new Error('后端未返回预测任务ID')
        this.predictionJobId = jobId
        localStorage.setItem('qlibPredictionJobId', jobId)
        await this.pollPrediction(jobId)
      } catch(e) {
        this.predictError = '预测失败: ' + (e.message || e)
        this.predicting = false
      }
    },
    async pollPrediction(jobId) {
      this.predicting = true
      for (let i = 0; i < 720 && !this._destroyed; i++) {
        try {
          const response = await qlibApi.getPredictStatus(jobId)
          const task = response?.data
          if (!task) throw new Error('预测任务不存在')
          this.predictionStatus = task.progress?.message || task.status
          if (task.status === 'completed') {
            this.predictResult = task.result
            this.predictError = ''
            this.predicting = false
            localStorage.removeItem('qlibPredictionJobId')
            this.predictionJobId = ''
            return
          }
          if (['failed', 'error', 'cancelled', 'interrupted', 'stalled'].includes(task.status)) {
            this.predictError = task.error || task.progress?.message || `预测任务${task.status}`
            this.predicting = false
            localStorage.removeItem('qlibPredictionJobId')
            this.predictionJobId = ''
            return
          }
        } catch (e) {
          this.predictionStatus = `轮询暂时失败，继续等待: ${e.message || e}`
        }
        await pollSleep(5000)
      }
      this.predicting = false
      if (!this._destroyed) this.predictError = '预测轮询超时，任务仍可能在后台运行'
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
