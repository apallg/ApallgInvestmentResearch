<template>
  <section class="lab-card">
    <h2 class="card-title">自动超参数搜索</h2>
    <p class="card-desc">自动搜索最优超参数，支持单模型和双模型依次搜索后融合重训</p>

    <div class="fm-block">
      <div class="fm-block-title">搜索配置</div>
      <div class="form-grid">
        <label class="form-field">模型 <select v-model="lab.optunaCfg.model_name" class="field">
          <option value="lightgbm">轻量梯度提升树</option>
          <option value="xgboost">极端梯度提升树</option>
        </select></label>
        <label class="form-field">搜索模式 <select v-model="lab.optunaCfg.search_dual" class="field">
          <option :value="false">单模型搜索</option>
          <option :value="true">双模型依次搜索 → 融合重训</option>
        </select></label>
        <label class="form-field">因子集 <select v-model="lab.optunaCfg.factor_set" @change="lab.onOptunaFactorSetChange" class="field">
          <option value="Alpha158">158因子集</option><option value="Alpha360">360因子集</option>
          <option value="__curated__">精选因子集</option>
        </select></label>
        <label class="form-field">股票池 <select v-model="lab.optunaCfg.instruments" class="field">
          <option value="csi300">沪深300</option><option value="csi500">中证500</option><option value="csi300+csi500">沪深300与中证500</option><option value="all">全部股票</option><option value="stable_history_500d">稳定历史池500D</option>
        </select></label>
        <div v-if="lab.optunaCfg.factor_set === '__curated__'" class="form-hint span-2">
          精选集 <select v-model="lab.optunaCfg.use_curated" class="field field-sm">
            <option value="latest">最新版本</option>
            <option v-for="s in lab.optunaCuratedSets" :key="s.name" :value="s.name">{{ s.name }} ({{ s.count }}因子, {{ s.created?.slice(0,10) }})</option>
          </select>
        </div>
        <label class="form-field">标签标准化 <select v-model="lab.optunaCfg.label_norm" class="field">
          <option value="none">不标准化 (推荐)</option>
          <option value="cszscore">截面标准分</option>
          <option value="csrank">截面排名</option>
        </select></label>
        <label class="form-field">标签周期 <input v-model.number="lab.optunaCfg.label_horizon" type="number" min="1" max="60" class="field field-sm" /> 日</label>
        <div class="form-hint span-2">统一使用可交易标签：t+1开盘 → t+h+1开盘。</div>
      </div>
    </div>
    <div class="fm-block">
      <div class="fm-block-title">时间段 & 搜索参数</div>
      <div class="form-grid">
        <label class="form-field">训练开始 <input v-model="lab.optunaCfg.train_start" class="field" /></label>
        <label class="form-field">训练结束 <input v-model="lab.optunaCfg.train_end" class="field" /></label>
        <label class="form-field">验证开始 <input v-model="lab.optunaCfg.valid_start" class="field" /></label>
        <label class="form-field">验证结束 <input v-model="lab.optunaCfg.valid_end" class="field" /></label>
        <label class="form-field">测试开始 <input v-model="lab.optunaCfg.test_start" class="field" /></label>
        <label class="form-field">测试结束 <input v-model="lab.optunaCfg.test_end" class="field" /></label>
        <label class="form-field">搜索轮数 <input v-model.number="lab.optunaCfg.n_trials" type="number" min="10" max="500" class="field field-sm" />
          <span class="field-hint">推荐50-100</span>
        </label>
        <label class="form-field">超时(秒) <input v-model.number="lab.optunaCfg.timeout" type="number" min="0" class="field field-sm" />
          <span class="field-hint">0=不限</span>
        </label>
      </div>
    </div>

    <button @click="lab.startOptuna" :disabled="lab.optunaRunning" class="btn primary" :class="{ loading: lab.optunaRunning }">
      {{ lab.optunaRunning ? '搜索中...' : '开始超参搜索' }}
    </button>
    <p v-if="lab.optunaMsg" class="msg" :class="{ error: lab.optunaError }">{{ lab.optunaMsg }}</p>

    <div v-if="lab.optunaResult" class="result-box">
      <h3>搜索结果</h3>
      <div v-if="lab.optunaResult.lgb_best_params" class="metrics-row">
        <div class="metric-item">
          <div class="metric-value good">{{ (lab.optunaResult.lgb_best_ic * 100).toFixed(2) }}%</div>
          <div class="metric-label">轻量提升树最优信息系数</div>
        </div>
        <div class="metric-item">
          <div class="metric-value good">{{ (lab.optunaResult.xgb_best_ic * 100).toFixed(2) }}%</div>
          <div class="metric-label">极端提升树最优信息系数</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">{{ lab.optunaResult.n_trials }}</div>
          <div class="metric-label">搜索轮数</div>
        </div>
      </div>
      <div v-if="lab.optunaResult.ic_mean != null" class="metrics-row">
        <div class="metric-item">
          <div class="metric-value good">{{ (lab.optunaResult.ic_mean * 100).toFixed(2) }}%</div>
          <div class="metric-label">融合后信息系数</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">{{ lab.optunaResult.icir?.toFixed(2) || '--' }}</div>
          <div class="metric-label">融合后信息系数稳定性</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">{{ lab.optunaResult.lgb_weight?.toFixed(2) || '--' }}</div>
          <div class="metric-label">轻量提升树权重</div>
        </div>
        <div class="metric-item">
          <div class="metric-value">{{ lab.optunaResult.xgb_weight?.toFixed(2) || '--' }}</div>
          <div class="metric-label">极端提升树权重</div>
        </div>
      </div>
      <details v-if="lab.optunaResult.lgb_best_params" class="param-details">
        <summary class="detail-summary">最优参数详情</summary>
        <div class="param-detail-grid">
          <div>
            <h4>轻量梯度提升树</h4>
            <table><tr v-for="(v, k) in lab.optunaResult.lgb_best_params" :key="'lgb_'+k"><td>{{ k }}</td><td>{{ v }}</td></tr></table>
          </div>
          <div v-if="lab.optunaResult.xgb_best_params">
            <h4>极端梯度提升树</h4>
            <table><tr v-for="(v, k) in lab.optunaResult.xgb_best_params" :key="'xgb_'+k"><td>{{ k }}</td><td>{{ v }}</td></tr></table>
          </div>
        </div>
      </details>
      <button v-if="lab.optunaResult.lgb_best_params" class="btn success"
              @click="lab.applyOptunaParams()">📋 应用最优参数到训练配置</button>
    </div>
  </section>
</template>

<script>
export default {
  name: 'OptunaTab',
  inject: ['qlibLab'],
  computed: {
    lab() {
      return this.qlibLab
    },
  },
}
</script>
