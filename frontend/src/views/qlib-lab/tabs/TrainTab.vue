<template>
  <section class="lab-card">
    <h2 class="card-title">模型训练</h2>

    <div class="fm-block">
      <div class="fm-block-title">模型与因子</div>
      <div class="form-grid">
        <label class="form-field">模型 <select v-model="lab.trainCfg.model_name" class="field">
          <option v-for="m in lab.availableModels" :key="m.name" :value="m.name">{{ m.name }}</option>
        </select></label>
        <label class="form-field">股票池 <select v-model="lab.trainCfg.instruments" class="field">
          <option value="csi300">沪深300</option><option value="csi500">中证500</option><option value="csi300+csi500">沪深300与中证500</option><option value="all">全部股票</option><option value="stable_history_500d">稳定历史池500D</option>
        </select></label>
        <label class="form-field">因子集 <select v-model="lab.trainCfg.factor_set" @change="lab.onFactorSetChange" class="field">
          <option value="Alpha158">158因子集</option><option value="Alpha360">360因子集</option>
          <option value="__pool__">📦 因子池因子</option>
          <option value="__curated__">精选因子集</option>
        </select></label>
        <label class="form-field check-label">
          <input type="checkbox" v-model="lab.trainCfg.train_dual" />
          双树模型（按信息系数加权融合）
        </label>
        <label class="form-field check-label">
          <input type="checkbox" v-model="lab.trainCfg.train_lgb_lstm" />
          🧠 截面树与时序网络融合（各占一半）
        </label>
        <label v-if="lab.trainCfg.train_lgb_lstm" class="form-field">训练样本
          <select v-model="lab.trainCfg.training_mode" @change="lab.onTrainingModeChange" class="field">
            <option value="full">全样本训练</option>
            <option value="state_subset">状态子样本训练</option>
          </select>
        </label>
        <label v-if="lab.trainCfg.train_lgb_lstm && lab.trainCfg.training_mode === 'state_subset'" class="form-field">专家角色
          <select v-model="lab.trainCfg.expert_role" class="field">
            <option value="trend">趋势专家</option>
            <option value="rebound">反弹专家</option>
          </select>
        </label>
        <div v-if="lab.trainCfg.train_lgb_lstm && lab.trainCfg.training_mode === 'state_subset'" class="form-hint span-2">
          自动使用冻结的核心＋对应卫星因子，并锁定10日可成交标签、截面排名和保守超参；仅训练期适配度前50%的日期进入训练。
        </div>
        <div v-else-if="lab.trainCfg.train_lgb_lstm" class="form-hint span-2">
          💡 全样本训练建议：股票池选“稳定历史池500D”、超参方案选“保守”、标签标准化选“截面排名”
        </div>
      </div>
      <div v-if="lab.trainCfg.factor_set === '__pool__'" class="form-hint span-2" style="margin-top:4px">
        动态因子池活跃因子 {{ lab.poolFactors.length }} 个（训练时后端自动去同源）
      </div>
      <div v-if="lab.trainCfg.factor_set === '__curated__'" class="form-hint span-2" style="margin-top:4px">
        精选集 <select v-model="lab.trainCfg.use_curated" class="field field-sm">
          <option value="latest">最新版本</option>
          <option v-for="s in lab.activeCuratedSets" :key="s.name" :value="s.name">{{ s.name }} ({{ s.count }}因子, {{ s.created?.slice(0,10) }})</option>
        </select>
      </div>
    </div>

    <div class="fm-block">
      <div class="fm-block-title">高级选项</div>
      <div class="option-row">
        <label class="cb"><input type="checkbox" v-model="lab.trainCfg.enable_crossover" /> ✕ 因子交叉特征</label>
        <label class="cb"><input type="checkbox" v-model="lab.trainCfg.enable_cross_sectional" /> 📐 截面特征增强</label>
        <label class="cb"><input type="checkbox" v-model="lab.trainCfg.use_reweighter" /> ⚖ 流动性加权+次新剔除</label>
        <label class="cb"><input type="checkbox" v-model="lab.trainCfg.with_linear" /> 📐 岭回归线性基线</label>
        <label class="cb"><input type="checkbox" v-model="lab.trainCfg.with_catboost" /> 🐱 类别提升第三模型</label>
      </div>
      <div class="form-grid" style="margin-top:8px">
        <label class="form-field">标签标准化 <select v-model="lab.trainCfg.label_norm" :disabled="lab.trainCfg.training_mode === 'state_subset'" class="field">
          <option value="none">不标准化 (推荐)</option>
          <option value="cszscore">截面标准分</option>
          <option value="csrank">截面排名</option>
        </select></label>
        <label class="form-field">早停指标 <select v-model="lab.trainCfg.early_stop_metric" class="field">
          <option value="mse">均方误差（推荐）</option>
          <option value="rank_ic">排名信息系数</option>
        </select></label>
        <label class="form-field">超参方案 <select v-model="lab.trainCfg.param_profile" :disabled="lab.trainCfg.training_mode === 'state_subset'" class="field">
          <option value="default">默认 (depth=7/leaves=127)</option>
          <option value="conservative">保守 (depth=5/leaves=63)</option>
          <option value="aggressive">激进 (depth=8/leaves=200)</option>
          <option value="optuna" :disabled="!lab.optunaParamsAvailable">🔬 自动调参{{ lab.optunaParamsAvailable ? '（' + lab.optunaParamsAvailable + '）' : '（无）' }}</option>
        </select></label>
        <label class="form-field">标签周期 <input v-model.number="lab.trainCfg.label_horizon" :disabled="lab.trainCfg.training_mode === 'state_subset'" type="number" min="1" max="60" class="field field-sm" /> 日</label>
        <div class="form-hint span-2">统一使用可交易标签：决策日收盘后生成信号，下一交易日开盘买入，持有期末开盘退出。</div>
      </div>
    </div>

    <div class="fm-block">
      <div class="fm-block-title">数据范围</div>
      <div class="prefix-row">
        <span class="prefix-label">代码前缀:</span>
        <label class="cb" v-for="pf in lab.codePrefixOptions" :key="pf[0][0]">
          <input type="checkbox" :value="pf[1]" v-model="lab.trainCodePrefixLabels" /> {{ pf[1] }}
        </label>
        <span v-if="!lab.trainCodePrefixLabels.length" class="text-muted">不选=全部</span>
      </div>
      <div class="form-grid" style="margin-top:8px">
        <label class="form-field">训练开始 <input v-model="lab.trainCfg.train_start" class="field" /></label>
        <label class="form-field">训练结束 <input v-model="lab.trainCfg.train_end" class="field" /></label>
        <label class="form-field">验证开始 <input v-model="lab.trainCfg.valid_start" class="field" /></label>
        <label class="form-field">验证结束 <input v-model="lab.trainCfg.valid_end" class="field" /></label>
        <label class="form-field">测试开始 <input v-model="lab.trainCfg.test_start" class="field" /></label>
        <label class="form-field">测试结束 <input v-model="lab.trainCfg.test_end" class="field" /></label>
        <label class="form-field">持仓数量 <input v-model.number="lab.trainCfg.topk" type="number" class="field field-sm" /></label>
        <label class="form-field">每期剔除数量 <input v-model.number="lab.trainCfg.n_drop" type="number" class="field field-sm" /></label>
      </div>
    </div>
    <button @click="lab.startTraining" :disabled="lab.training" class="btn primary" :class="{ loading: lab.training }" style="margin-top:8px">
      {{ lab.training ? '训练中...' : '开始训练' }}
    </button>
    <div v-if="lab.trainResult" class="result-box">
      <div class="metrics-row">
        <div class="metric-item">
          <div class="metric-value good">{{ lab.trainResult.ic_mean != null ? (lab.trainResult.ic_mean * 100).toFixed(2) + '%' : '--' }}</div>
          <div class="metric-label">平均信息系数</div>
        </div>
        <div class="metric-item">
          <div class="metric-value good">{{ lab.trainResult.icir != null ? lab.trainResult.icir.toFixed(2) : '--' }}</div>
          <div class="metric-label">信息系数稳定性</div>
        </div>
        <div v-if="lab.trainResult.ric_mean != null" class="metric-item">
          <div class="metric-value">{{ (lab.trainResult.ric_mean * 100).toFixed(2) }}%</div>
          <div class="metric-label">排名信息系数</div>
        </div>
        <div v-if="lab.trainResult.lgb_ic_mean != null" class="metric-item">
          <div class="metric-value">{{ (lab.trainResult.lgb_ic_mean * 100).toFixed(2) }}%</div>
          <div class="metric-label">轻量提升树信息系数</div>
        </div>
        <div v-if="lab.trainResult.lstm_ic_mean != null" class="metric-item">
          <div class="metric-value">{{ (lab.trainResult.lstm_ic_mean * 100).toFixed(2) }}%</div>
          <div class="metric-label">时序网络信息系数</div>
        </div>
        <div v-if="lab.trainResult.xgb_ic_mean != null" class="metric-item">
          <div class="metric-value">{{ (lab.trainResult.xgb_ic_mean * 100).toFixed(2) }}%</div>
          <div class="metric-label">极端提升树信息系数</div>
        </div>
        <div v-if="lab.trainResult.degraded && lab.trainResult.degraded.length" class="metric-item">
          <div class="metric-value warn-text">降级</div>
          <div class="metric-label">融合状态</div>
        </div>
      </div>
      <div v-if="lab.trainResult.backtest?.report">
        <h3>回测结果</h3>
        <div v-for="(metrics, key) in lab.trainResult.backtest.report" :key="key">
          <h4>{{ key }}</h4>
          <table><tr v-for="(v, k) in metrics" :key="k"><td>{{ k }}</td><td>{{ v }}</td></tr></table>
        </div>
      </div>
    </div>
    <p v-if="lab.trainError" class="msg" :class="{ error: lab.trainError.includes('失败') || lab.trainError.includes('超时') }">{{ lab.trainError }}</p>

      <div class="fm-block" style="margin-top:16px;">
        <div class="fm-block-title">多模型集成（新增）</div>
        <div class="form-grid">
          <label class="form-field">模型
            <select v-model="ensembleModels" multiple class="field">
              <option v-for="m in lab.availableModels" :key="m.name" :value="m.name">{{ m.name }}</option>
            </select>
          </label>
          <label class="form-field">集成策略
            <select v-model="ensembleStrategy" class="field">
              <option value="ic">IC 加权</option>
              <option value="equal">等权平均</option>
            </select>
          </label>
        </div>
        <button @click="startEnsembleTrain" :disabled="ensembleRunning" class="btn primary" :class="{ loading: ensembleRunning }" style="margin-top:8px">
          {{ ensembleRunning ? '集成训练中...' : '开始集成训练' }}
        </button>
        <p v-if="ensembleMsg" class="msg" style="margin-top:8px">{{ ensembleMsg }}</p>
        <div v-if="ensembleResult" class="result-box" style="margin-top:8px">
          <div class="metrics-row">
            <div v-for="(v, name) in ensembleResult.sub_results" :key="name" class="metric-item">
              <div class="metric-value">{{ v.ic_mean != null ? (v.ic_mean * 100).toFixed(2) + '%' : '--' }}</div>
              <div class="metric-label">{{ name }} IC</div>
            </div>
            <div class="metric-item">
              <div class="metric-value">{{ ensembleResult.ensemble?.strategy || '--' }}</div>
              <div class="metric-label">集成策略</div>
            </div>
            <div class="metric-item">
              <div class="metric-value">{{ ensembleResult.prediction_count ?? '--' }}</div>
              <div class="metric-label">预测数</div>
            </div>
          </div>
          <div v-if="ensembleResult.ensemble?.weights" class="form-hint" style="margin-top:4px">
            权重: {{ JSON.stringify(ensembleResult.ensemble.weights) }}
          </div>
          <div v-if="ensembleResult.backtest?.report">
            <h3>集成回测结果</h3>
            <div v-for="(metrics, key) in ensembleResult.backtest.report" :key="key">
              <h4>{{ key }}</h4>
              <table><tr v-for="(v, k) in metrics" :key="k"><td>{{ k }}</td><td>{{ v }}</td></tr></table>
            </div>
          </div>
        </div>
      </div>
  </section>
</template>

<script>
import api from '../../../api'

export default {
  name: 'TrainTab',
  inject: ['qlibLab'],
  data() {
    return {
      ensembleModels: ['lightgbm', 'xgboost'],
      ensembleStrategy: 'ic',
      ensembleRunning: false,
      ensembleMsg: '',
      ensembleResult: null,
    }
  },
  computed: {
    lab() {
      return this.qlibLab
    },
  },
  methods: {
    async startEnsembleTrain() {
      this.ensembleRunning = true
      this.ensembleMsg = '集成训练已启动...'
      try {
        const cfg = {
          model_names: this.ensembleModels,
          ensemble_strategy: this.ensembleStrategy,
          factor_set: this.lab.trainCfg.factor_set,
          instruments: this.lab.trainCfg.instruments,
          train_start: this.lab.trainCfg.train_start,
          train_end: this.lab.trainCfg.train_end,
          valid_start: this.lab.trainCfg.valid_start,
          valid_end: this.lab.trainCfg.valid_end,
          test_start: this.lab.trainCfg.test_start,
          test_end: this.lab.trainCfg.test_end,
          label_horizon: this.lab.trainCfg.label_horizon,
          label_norm: this.lab.trainCfg.label_norm,
          use_factor_pool: this.lab.trainCfg.factor_set === '__pool__',
          use_curated: this.lab.trainCfg.factor_set === '__curated__' ? this.lab.trainCfg.use_curated : null,
          enable_cross_sectional: this.lab.trainCfg.enable_cross_sectional,
          enable_winsorize: true,
          enable_crossover: this.lab.trainCfg.enable_crossover,
          use_reweighter: this.lab.trainCfg.use_reweighter,
          auto_backtest: true,
          topk: this.lab.trainCfg.topk,
          n_drop: this.lab.trainCfg.n_drop,
        }
        const r = await api.post('/qlib/models/ensemble-train', cfg)
        const jobId = r?.data?.job_id || ''
        this.ensembleMsg = `集成训练已提交: ${jobId}，轮询中...`
        if (!jobId) throw new Error('未返回 job_id')
        const startTime = Date.now()
        for (let i = 0; i < 720; i++) {
          await new Promise(resolve => setTimeout(resolve, 10000))
          try {
            const job = await api.get(`/qlib/models/train/${jobId}`)
            const status = job.data.status
            const elapsed = Math.floor((Date.now() - startTime) / 60000)
            if (status === 'completed') {
              this.ensembleResult = job.data.result
              this.ensembleMsg = `集成训练完成（${elapsed} 分钟）`
              break
            } else if (status === 'error') {
              this.ensembleMsg = '集成训练失败: ' + (job.data.error || '未知错误')
              break
            } else {
              this.ensembleMsg = `集成训练中... (${elapsed}分钟) ${job.data.result?.message || ''}`
            }
          } catch (e) {
            if (e.response?.status === 404) {
              this.ensembleMsg = '任务已过期 (服务器可能重启过)，请重新提交'
              break
            }
            this.ensembleMsg = '轮询中... (网络抖动，继续等待)'
          }
        }
        if (!this.ensembleResult && !this.ensembleMsg.includes('失败') && !this.ensembleMsg.includes('任务已过期')) {
          this.ensembleMsg = '集成训练超时 (2小时)，任务仍在后台运行，稍后可查看结果'
        }
      } catch (e) {
        this.ensembleMsg = '启动失败: ' + (e.response?.data?.detail || e.message)
      } finally {
        this.ensembleRunning = false
      }
    },
  },
}
</script>
