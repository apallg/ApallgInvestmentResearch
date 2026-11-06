<template>
  <section class="lab-card">
    <h2 class="card-title">滚动时序训练
      <span class="card-sub">季度锚点滚动 · 每轮独立实验 · 全期样本外拼接评估</span>
    </h2>

    <div class="fm-block">
      <div class="fm-block-title">模型配置</div>
      <div class="form-grid">
        <label class="form-field">因子集 <select v-model="lab.rollingCfg.factor_set" @change="lab.onRollingFactorSetChange" class="field">
          <option value="Alpha158">158因子集</option><option value="Alpha360">360因子集</option>
          <option value="__pool__">📦 因子池因子</option>
          <option value="__curated__">精选因子集</option>
        </select></label>
        <label class="form-field">股票池 <select v-model="lab.rollingCfg.pool" class="field">
          <option value="csi300">沪深300</option><option value="csi500">中证500</option><option value="csi300+csi500">沪深300与中证500</option><option value="all">全部股票</option><option value="stable_history_500d">稳定历史池500D</option>
        </select></label>
        <div v-if="lab.rollingCfg.factor_set === '__pool__'" class="form-hint span-2">
          动态因子池活跃因子（训练时后端自动去同源）
        </div>
        <div v-if="lab.rollingCfg.factor_set === '__curated__'" class="form-hint span-2">
          精选集 <select v-model="lab.rollingCfg.use_curated" class="field field-sm">
            <option value="latest">最新版本</option>
            <option v-for="s in lab.rollingCuratedSets" :key="s.name" :value="s.name">{{ s.name }} ({{ s.count }}因子, {{ s.created?.slice(0,10) }})</option>
          </select>
        </div>
        <label class="form-field check-label span-2">
          <input type="checkbox" v-model="lab.rollingCfg.train_dual" />
          双树模型集成（按信息系数加权融合）
        </label>
        <label class="form-field check-label span-2">
          <input type="checkbox" v-model="lab.rollingCfg.enable_crossover" />
          ✕ 因子交叉特征
        </label>
        <label class="form-field">标签标准化 <select v-model="lab.rollingCfg.label_norm" class="field">
          <option value="none">不标准化（推荐）</option>
          <option value="cszscore">截面标准分</option>
          <option value="csrank">截面排名</option>
        </select></label>
        <label class="form-field">标签周期 <input v-model.number="lab.rollingCfg.label_horizon" type="number" min="1" max="60" class="field field-sm" /> 日</label>
        <div class="form-hint span-2">统一使用可交易标签：次日开盘 → 预测周期结束后的次日开盘。</div>
      </div>
    </div>

    <div class="fm-block">
      <div class="fm-block-title">高级选项</div>
      <div class="form-grid">
        <label class="form-field">早停指标 <select v-model="lab.rollingCfg.early_stop_metric" class="field">
          <option value="mse">均方误差（推荐）</option>
          <option value="rank_ic">排名信息系数</option>
        </select></label>
        <label class="form-field">超参方案 <select v-model="lab.rollingCfg.param_profile" class="field">
          <option value="default">默认（深度7、叶节点127）</option>
          <option value="conservative">保守（稳定池）</option>
          <option value="aggressive">激进（深度8、叶节点200）</option>
        </select></label>
      </div>
      <div class="option-row">
        <label class="cb"><input type="checkbox" v-model="lab.rollingCfg.use_reweighter" /> ⚖ 流动性加权+次新剔除</label>
        <label class="cb"><input type="checkbox" v-model="lab.rollingCfg.with_linear" /> 📐 岭回归线性基线</label>
        <label class="cb"><input type="checkbox" v-model="lab.rollingCfg.with_catboost" /> 🐱 类别提升第三模型</label>
      </div>
      <div class="prefix-row" style="margin-top:8px">
        <span class="prefix-label">代码前缀:</span>
        <label class="cb" v-for="pf in lab.codePrefixOptions" :key="pf[0][0]">
          <input type="checkbox" :value="pf[1]" v-model="lab.rollingPrefixLabels" /> {{ pf[1] }}
        </label>
        <span v-if="!lab.rollingPrefixLabels.length" class="text-muted">不选=全部</span>
      </div>
      <label v-if="lab.rollingCfg.factor_set === '__pool__'" class="filter-banner">
        <input type="checkbox" v-model="lab.rollingCfg.filter_mode" />
        <span style="font-weight:600;">🔍 嵌入式二次筛选</span>
        <span class="text-muted" style="font-size:12px;">— 追踪特征重要性，训练完成后自动淘汰低效因子</span>
      </label>
    </div>

    <div class="fm-block">
      <div class="fm-block-title">滚动窗口设置</div>
      <div class="form-grid">
        <label class="form-field">起始锚点 <input v-model="lab.rollingCfg.start_anchor" placeholder="2023-01-01" class="field" /></label>
        <label class="form-field">结束锚点 <input v-model="lab.rollingCfg.end_anchor" placeholder="空=今天" class="field" /></label>
        <label class="form-field">滚动步长 <select v-model.number="lab.rollingCfg.step_months" class="field">
          <option :value="1">1个月</option><option :value="3">3个月 (季度)</option><option :value="6">6个月</option>
        </select></label>
        <label class="form-field">训练窗口 <select v-model.number="lab.rollingCfg.train_years" class="field">
          <option :value="3">3年</option><option :value="5">5年</option><option :value="8">8年</option>
        </select></label>
        <label class="form-field">验证窗口 <select v-model.number="lab.rollingCfg.valid_months" class="field">
          <option :value="3">3个月</option><option :value="6">6个月</option><option :value="12">12个月</option>
        </select></label>
      </div>

      <div class="rolling-preview" v-if="lab.rollingFirstRound.nRounds !== null" style="margin-top:8px">
        <div class="preview-title">📐 第一轮示例</div>
        <div class="preview-dates">
          <span class="preview-tag train">训练 {{ lab.rollingFirstRound.train }}</span>
          <span class="preview-arrow">→</span>
          <span class="preview-tag valid">验证 {{ lab.rollingFirstRound.valid }}</span>
          <span class="preview-arrow">→</span>
          <span class="preview-tag test">测试 {{ lab.rollingFirstRound.test }}</span>
        </div>
        <div class="preview-count">
          预计 {{ lab.rollingFirstRound.nRounds }} 轮（{{ lab.rollingCfg.start_anchor || '2023-01-01' }} ~ {{ lab.rollingFirstRound.lastAnchor }}）
        </div>
      </div>
    </div>

    <button @click="lab.startRollingTrain" :disabled="lab.rollingRunning" class="btn primary" :class="{ loading: lab.rollingRunning }">
      {{ lab.rollingRunning ? '滚动训练中...' : '开始滚动训练' }}
    </button>

    <div v-if="lab.rollingRunning" class="train-progress">
      <div class="prog-header">
        <span class="prog-title">{{ lab.rollingMsg }}</span>
        <span class="prog-pct">{{ lab.rollingPct }}%</span>
      </div>
      <div class="prog-bar">
        <div class="prog-fill" :style="{ width: lab.rollingPct + '%' }"></div>
      </div>
    </div>
    <p v-if="lab.rollingError && !lab.rollingRunning" class="msg error">{{ lab.rollingError }}</p>

    <div v-if="lab.rollingResult" class="result-box">
      <h3>{{ lab.rollingResult.n_trained }} / {{ lab.rollingResult.n_planned }} 轮完成</h3>
      <div v-if="lab.rollingResult.oos && lab.rollingResult.oos.ic_mean != null" class="metrics-row">
        <div class="metric-item">
          <div class="metric-value good">{{ (lab.rollingResult.oos.ic_mean * 100).toFixed(2) }}%</div>
          <div class="metric-label">全期样本外平均信息系数</div>
        </div>
        <div class="metric-item">
          <div class="metric-value good">{{ (lab.rollingResult.oos.ric_mean * 100).toFixed(2) }}%</div>
          <div class="metric-label">全期样本外排名信息系数</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">{{ lab.rollingResult.oos.icir?.toFixed(2) || '--' }}</div>
          <div class="metric-label">全期样本外信息系数稳定性</div>
        </div>
        <div v-if="lab.rollingResult.oos.ic_win_rate_m != null" class="metric-item">
          <div class="metric-value">{{ (lab.rollingResult.oos.ic_win_rate_m * 100).toFixed(0) }}%</div>
          <div class="metric-label">月度信息系数胜率</div>
        </div>
      </div>
      <div v-if="lab.rollingResult.filter_result" class="filter-result">
        <strong class="good">✅ 因子池已更新</strong>
        <span>保留 {{ lab.rollingResult.filter_result.n_passed }} / 淘汰 {{ lab.rollingResult.filter_result.n_failed }}</span>
      </div>
    </div>
  </section>
</template>

<script>
export default {
  name: 'RollingTrainTab',
  inject: ['qlibLab'],
  computed: {
    lab() {
      return this.qlibLab
    },
  },
}
</script>
