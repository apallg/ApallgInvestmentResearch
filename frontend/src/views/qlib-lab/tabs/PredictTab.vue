<template>
  <section class="lab-card">
    <h2 class="card-title">模型预测</h2>
    <label class="form-field" style="max-width:520px;margin-bottom:16px;">策略版本
      <select v-model="lab.predictProfileKey" class="field" @change="useProfileDate">
        <option value="">研究模式（手工选择模型，不可用于晋级）</option>
        <option v-for="p in lab.strategyProfiles" :key="`${p.profile_id}::${p.version}`" :value="`${p.profile_id}::${p.version}`">{{ p.name }} 第{{ p.version }}版 · {{ p.config?.factors?.research_only ? '研究观察' : strategyStatusText(p.status) }}</option>
      </select>
    </label>
    <StrategyProfileBanner v-if="lab.selectedPredictProfile" :profile="lab.selectedPredictProfile" title="本次预测策略版本" />
    <div v-if="!lab.selectedPredictProfile && !lab.selectedModel" class="msg">
      正式预测请选择策略版本；研究模式需先在「已训练模型」页签中选择模型。
      <button @click="lab.activeTab = 'trained'" class="btn primary" style="margin-left:8px;">去选择</button>
    </div>
    <div v-if="lab.selectedPredictProfile || lab.selectedModel">
      <div v-if="!lab.selectedPredictProfile" class="selected-model-info">
        <strong>{{ localizeStrategyText(lab.selectedModel.display_name || lab.selectedModel.model_class) }}</strong>
        <span v-if="lab.selectedModel.is_dual" class="tag-sm dual">双树模型融合</span>
        <span v-else-if="lab.selectedModel.is_fusion" class="tag-sm dual">树模型与长短期记忆网络融合</span>
        <span v-else class="tag-sm single">单模型</span>
        <span class="tag-sm ok" v-if="lab.selectedModel.ic !== null">信息系数 {{ (lab.selectedModel.ic * 100).toFixed(2) }}%</span>
        <span class="tag-sm" v-if="lab.selectedModel.icir !== null">信息系数稳定性 {{ lab.selectedModel.icir.toFixed(2) }}</span><br>
        <span class="text-muted" style="font-size:12px;">因子：{{ localizeStrategyText(lab.selectedModel.factor_set) }} | 股票池：{{ localizeStrategyText(lab.selectedModel.instruments) }} | 训练：{{ lab.selectedModel.train_period }}</span>
      </div>
      <div v-else class="selected-model-info">策略版本已锁定模型、因子、稳定历史池500D、代码范围和候选容量；修改需复制为新草稿。</div>
      <div v-if="lab.selectedPredictProfile" class="form-grid">
        <label class="form-field">固定决策日 <input v-model="lab.predictCfg.decision_date" :min="lab.selectedPredictProfile.config?.availability?.first_decision_date || undefined" type="date" class="field"><small v-if="lab.selectedPredictProfile.config?.availability?.first_decision_date">可运行日期从 {{ lab.selectedPredictProfile.config.availability.first_decision_date }} 开始</small></label>
        <label class="form-field">股票池 <input value="稳定历史池500D" class="field" disabled></label>
        <label class="form-field">候选容量 <input :value="lab.selectedPredictProfile.config?.selection?.candidate_pool" class="field field-sm" disabled></label>
      </div>
      <template v-else>
        <div class="form-grid" style="margin-top:16px;"><label class="form-field">股票池 <select v-model="lab.predictCfg.instruments" class="field">
          <option value="csi300">沪深300</option><option value="csi500">中证500</option><option value="csi300+csi500">沪深300与中证500</option><option value="all">全部股票</option><option value="stable_history_500d">稳定历史池500D</option>
        </select></label></div>
        <div class="prefix-row"><span class="prefix-label">代码前缀:</span>
          <label class="cb" v-for="pf in lab.codePrefixOptions" :key="pf[0][0]"><input type="checkbox" :value="pf[1]" v-model="lab.predictCodePrefixLabels"> {{ pf[1] }}</label>
          <span v-if="!lab.predictCodePrefixLabels.length" class="text-muted">不选=全部</span>
        </div>
        <div class="form-grid">
          <label class="form-field">输出前几名 <input v-model.number="lab.predictCfg.topk" type="number" class="field field-sm"></label>
          <label class="form-field">开始日期 <input v-model="lab.predictCfg.start_time" class="field"></label>
          <label class="form-field">结束日期 <input v-model="lab.predictCfg.end_time" class="field"></label>
        </div>
      </template>
      <button @click="lab.runPrediction" :disabled="lab.predicting" class="btn primary" :class="{ loading: lab.predicting }">{{ lab.predicting ? '预测中...' : '生成预测' }}</button>
      <p v-if="lab.predicting && lab.predictionStatus" class="msg">任务 {{ lab.predictionJobId }} · {{ localizeStrategyText(lab.predictionStatus) }}</p>
      <p v-if="lab.predictError" class="msg error">{{ localizeStrategyText(lab.predictError) }}</p>
      <div v-if="lab.predictResult" class="result-box">
        <StrategyProfileBanner v-if="lab.predictResult.strategy_profile" :profile="resultProfile" title="结果绑定版本" />
        <p v-if="lab.predictResult.strategy_profile?.research_only" class="msg">该结果仅用于研究观察，不能用于正式调仓或新增买入。</p>
        <p v-for="warning in lab.predictResult.warnings || []" :key="warning" class="msg error">{{ warning }}</p>
        <p class="msg">分位分只表示同一日期内的横截面排序，不能跨日期比较，也不是预期收益或置信度。</p>
        <div v-if="lab.predictResult.feature_importance?.length" class="feature-importance-box">
          <h3>模型决策因子（前{{ lab.predictResult.feature_importance.length }}项）</h3>
          <div class="fi-grid"><div v-for="f in lab.predictResult.feature_importance" :key="f.rank" class="fi-item">
            <span class="fi-rank">#{{ f.rank }}</span><span class="fi-name">{{ f.name }}</span>
            <div class="fi-bar"><div class="fi-fill" :style="{ width: (f.importance / lab.predictResult.feature_importance[0].importance * 100) + '%' }"></div></div><span class="fi-imp">{{ f.importance }}</span>
          </div></div>
        </div>
        <h3>预测结果（共 {{ lab.predictResult.count }} 条）</h3>
        <div v-if="lab.predictResult.by_date"><div v-for="(stocks, date) in lab.predictResult.by_date" :key="date" class="pred-group">
          <h4>{{ date }}（前{{ stocks.length }}只）<span v-if="lab.predictResult.feature_importance" class="toggle-factors" @click="lab.toggleShowFactors()">{{ lab.showFactors ? '隐藏因子' : '显示因子' }}</span></h4>
          <table class="pred-table"><thead><tr>
            <th>排名</th><th>股票</th><th>模型输入</th><th>原始分</th><th>中性化分</th><th>同日分位分</th>
            <th v-if="lab.showFactors && lab.predictResult.feature_importance" v-for="f in lab.topFactors" :key="f.name" class="factor-th" :title="'排名 '+f.rank+'，重要性：'+f.importance">{{ f.name }}</th>
          </tr></thead><tbody><tr v-for="s in stocks" :key="s.instrument">
            <td>{{ s.rank }}</td><td>{{ s.instrument }}</td><td :class="scoreClass(s.model_input_score)">{{ formatScore(s.model_input_score) }}</td>
            <td :class="scoreClass(s.raw_score)">{{ formatScore(s.raw_score) }}</td><td :class="scoreClass(s.neutralized_score)">{{ formatScore(s.neutralized_score) }}</td><td :class="scoreClass(s.percentile_score)">{{ formatScore(s.percentile_score) }}</td>
            <td v-if="lab.showFactors && lab.predictResult.feature_importance" v-for="f in lab.topFactors" :key="f.name" class="factor-td" :class="scoreClass(s.factors?.[f.name])">{{ s.factors?.[f.name] ?? '-' }}</td>
          </tr></tbody></table>
        </div></div><p v-else>无按日期分组的预测数据</p>
        <div v-if="lab.predictResult.shadow_model_observations?.length" class="feature-importance-box">
          <h3>影子模型对照（不改变选股）</h3>
          <div v-for="item in lab.predictResult.shadow_model_observations" :key="item.model?.join('/')" class="msg">
            {{ localizeStrategyText(item.factor_set_name || '影子模型') }} · {{ item.status === 'completed' ? '已完成' : '失败' }}
            <span v-if="item.status === 'completed'"> · 前5名重合率：{{ Object.entries(item.top5_overlap || {}).map(([date, value]) => `${date} ${(value * 100).toFixed(0)}%`).join('；') }}</span>
            <span v-else> · {{ item.error }}</span>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script>
import StrategyProfileBanner from '../../../components/StrategyProfileBanner.vue'
import { localizeStrategyText, strategyStatusText } from '../../../utils/strategyTerms.js'

export default {
  name: 'PredictTab', components: { StrategyProfileBanner }, inject: ['qlibLab'],
  setup() { return { localizeStrategyText, strategyStatusText } },
  computed: {
    lab() { return this.qlibLab },
    resultProfile() { const p = this.lab.predictResult?.strategy_profile; return p ? { ...p, name: p.name || '预测结果策略', config: {} } : null },
  },
  methods: {
    useProfileDate() {
      const first = this.lab.selectedPredictProfile?.config?.availability?.first_decision_date
      if (first && (!this.lab.predictCfg.decision_date || this.lab.predictCfg.decision_date < first)) this.lab.predictCfg.decision_date = first
    },
    formatScore(value) { return Number.isFinite(Number(value)) ? Number(value).toFixed(6) : '不适用' },
    scoreClass(value) { return Number.isFinite(Number(value)) ? (Number(value) > 0 ? 'positive' : 'negative') : '' },
  },
}
</script>
