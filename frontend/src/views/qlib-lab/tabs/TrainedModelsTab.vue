<template>
  <section class="lab-card">
    <div class="card-head">
      <h2 class="card-title">已训练模型</h2>
      <span class="card-badge" v-if="lab.trainedModels.length">共 {{ lab.trainedModels.length }} 个</span>
    </div>

    <!-- 工具栏 -->
    <div class="trained-toolbar">
      <button @click="lab.loadTrainedModels" :disabled="lab.loadingModels" class="btn primary">
        {{ lab.loadingModels ? '加载中...' : '刷新列表' }}
      </button>
      <span class="trained-count" v-if="lab.trainedModels.length">
        共 <strong>{{ lab.trainedModels.length }}</strong> 个模型
        <span v-if="lab.sortedModels.length !== lab.trainedModels.length">（筛选后 {{ lab.sortedModels.length }} 个）</span>
      </span>
      <div class="trained-sort">
        <label>排序</label>
        <select v-model="lab.modelSort" class="field field-sm">
          <option value="ic">信息系数降序</option>
          <option value="icir">信息系数稳定性降序</option>
          <option value="date">训练时间 降序</option>
          <option value="sharpe">多空夏普比率降序</option>
        </select>
      </div>
      <div class="trained-sort">
        <label>类型</label>
        <select v-model="lab.modelTypeFilter" class="field field-sm">
          <option value="all">全部类型</option>
          <option value="timing">择时模型</option>
          <option value="predict">预测模型</option>
        </select>
      </div>
      <div class="trained-sort">
        <label>筛选</label>
        <select v-model="lab.modelFilter" class="field field-sm">
          <option value="all">全部</option>
          <option value="dual">双树模型</option>
          <option value="fusion">截面树与时序网络融合</option>
          <option value="single">单模型</option>
          <option value="rolling">滚动训练</option>
          <option value="hasIC">有信息系数数据</option>
        </select>
      </div>
    </div>

    <!-- 汇总统计 -->
    <div class="trained-summary" v-if="lab.trainedModels.length">
      <div class="summary-item">
        <div class="summary-val">{{ lab.trainedModels.length }}</div>
        <div class="summary-label">模型总数</div>
      </div>
      <div class="summary-item">
        <div class="summary-val">{{ lab.dualCount }}</div>
        <div class="summary-label">双树模型</div>
      </div>
      <div class="summary-item">
        <div class="summary-val">{{ lab.fusionCount }}</div>
        <div class="summary-label">截面与时序融合</div>
      </div>
      <div class="summary-item">
        <div class="summary-val">{{ lab.singleCount }}</div>
        <div class="summary-label">单模型</div>
      </div>
      <div class="summary-item">
        <div class="summary-val good">{{ lab.bestIC !== null ? (lab.bestIC * 100).toFixed(2) + '%' : '--' }}</div>
        <div class="summary-label">最佳信息系数</div>
      </div>
      <div class="summary-item">
        <div class="summary-val primary-text">{{ lab.bestICIR !== null ? lab.bestICIR.toFixed(2) : '--' }}</div>
        <div class="summary-label">最佳信息系数稳定性</div>
      </div>
      <div class="summary-item">
        <div class="summary-val">{{ lab.predictableCount }}</div>
        <div class="summary-label">可预测</div>
      </div>
    </div>

    <div v-if="lab.trainedModels.length === 0 && !lab.loadingModels" class="msg">暂无已训练模型，请先在"模型训练"页签中训练。</div>

    <!-- 模型卡片列表 -->
    <div v-for="(m, idx) in lab.sortedModels" :key="m.run_id" class="trained-card"
         :class="{ selected: lab.selectedModel?.run_id === m.run_id, disabled: m.disabled }">
      <div class="tc-header">
        <div class="tc-title-row">
          <span class="tc-name">{{ localizeStrategyText(m.display_name || (m.model_class + ' ' + m.train_period)) }}</span>
          <span v-if="m.is_dual" class="type-badge dual">双树模型</span>
          <span v-else-if="m.is_fusion" class="type-badge dual">截面与时序融合</span>
          <span v-else-if="m.model_type !== 'timing'" class="type-badge single">{{ localizeStrategyText(m.model_class || '单模型') }}</span>
          <span v-if="m.model_type === 'timing'" class="type-badge timing">择时</span>
          <span v-if="m.model_type === 'regime'" class="type-badge regime">市场状态第一版</span>
          <span v-if="m.model_type === 'regime_v2'" class="type-badge regime-v2">市场状态第二版</span>
          <span v-if="m.is_rolling" class="type-badge rolling">滚动训练</span>
          <span v-if="m.is_fusion && m.expert_role" class="type-badge expert">{{ m.expert_role === 'trend' ? '趋势专家' : '反弹专家' }}</span>
          <span v-if="m.is_fusion && m.is_state_subset" class="type-badge subset">状态子样本</span>
          <span v-else-if="m.is_fusion && m.training_mode === 'full'" class="type-badge full-sample">全样本</span>
          <span v-if="m.has_predictions" class="type-badge pred">已有预测</span>
        </div>
        <div class="tc-meta">
          <span>{{ m.run_id?.slice(0,8) }}</span>
          <span v-if="m.trained_at" class="tc-time">{{ lab.formatTime(m.trained_at) }}</span>
        </div>
      </div>

      <!-- 关键指标条 -->
      <div class="tc-kpi-strip">
        <div class="tc-kpi" v-if="m.ic !== null">
          <span class="tc-kpi-val" :class="m.ic > 0.02 ? 'good' : (m.ic > 0 ? '' : 'bad')">{{ m.model_type === 'regime_v2' ? m.ic?.toFixed(3) : (m.ic * 100).toFixed(2) + '%' }}</span>
          <span class="tc-kpi-label">{{ m.model_type === 'regime_v2' ? '波动率拟合度' : (m.model_type === 'regime' ? '准确率' : '信息系数') }}</span>
        </div>
        <div class="tc-kpi" v-if="m.icir !== null && m.model_type !== 'regime_v2'">
          <span class="tc-kpi-val good">{{ typeof m.icir === 'number' ? m.icir.toFixed(2) : m.icir }}</span>
          <span class="tc-kpi-label">{{ m.model_type === 'regime' ? '宏观分类能力' : '信息系数稳定性' }}</span>
        </div>
        <div class="tc-kpi" v-if="m.ric !== null">
          <span class="tc-kpi-val" :class="m.ric > 0.02 ? 'good' : ''">{{ m.model_type === 'regime_v2' ? m.ric?.toFixed(4) : (m.ric * 100).toFixed(2) + '%' }}</span>
          <span class="tc-kpi-label">{{ m.model_type === 'regime_v2' ? '波动率平均绝对误差' : '排名信息系数' }}</span>
        </div>
        <div class="tc-kpi" v-if="m.ls_sharpe != null">
          <span class="tc-kpi-val" :class="m.ls_sharpe > 0.5 ? 'good' : (m.ls_sharpe < 0 ? 'bad' : '')">{{ m.ls_sharpe.toFixed(2) }}</span>
          <span class="tc-kpi-label">多空夏普比率</span>
        </div>
        <div class="tc-kpi" v-if="m.q5_q1_spread != null">
          <span class="tc-kpi-val" :class="m.q5_q1_spread > 0.05 ? 'good' : ''">{{ (m.q5_q1_spread * 100).toFixed(2) }}%</span>
          <span class="tc-kpi-label">多空收益</span>
        </div>
        <div class="tc-kpi" v-if="m.ic_win_rate_m != null">
          <span class="tc-kpi-val" :class="m.ic_win_rate_m > 0.55 ? 'good' : ''">{{ (m.ic_win_rate_m * 100).toFixed(0) }}%</span>
          <span class="tc-kpi-label">月胜率</span>
        </div>
      </div>

      <div v-if="m.backtest_summary" class="tc-backtest-summary">
        <strong>手动回测</strong>
        <span>区间 {{ m.backtest_summary.test_period?.join(' 至 ') || '—' }}</span>
        <span>{{ m.backtest_summary.dates ?? m.backtest_summary.signal_dates?.length ?? 0 }} 个信号日</span>
        <span>组合收益 {{ formatPercent(m.backtest_summary.total_return) }}</span>
        <span>相对基准 {{ formatPercent(m.backtest_summary.excess_return) }}</span>
        <code>回测 {{ m.backtest_summary.run_id?.slice(0, 8) }}</code>
      </div>

      <div class="tc-body">
        <div class="tc-col tc-config">
          <div class="col-title">训练配置</div>
          <div class="config-row"><span class="config-key">因子集</span><span class="config-val strong">{{ m.factor_set || '-' }}</span></div>
          <div class="config-row"><span class="config-key">股票池</span><span class="config-val">{{ m.instruments || '-' }}</span></div>
          <div class="config-row"><span class="config-key">训练期</span><span class="config-val">{{ m.train_period || '-' }}</span></div>
          <div class="config-row"><span class="config-key">验证期</span><span class="config-val">{{ m.valid_period || '-' }}</span></div>
          <div class="config-row"><span class="config-key">测试期</span><span class="config-val">{{ m.test_period || '-' }}</span></div>
          <div class="config-row" v-if="m.label_period"><span class="config-key">标签周期</span><span class="config-val">{{ m.label_period }}日</span></div>
          <div class="config-row" v-if="m.label_mode"><span class="config-key">标签口径</span><span class="config-val" :class="m.label_mode === 'tradable_open_to_open' ? 'good' : 'bad'">{{ m.label_mode === 'tradable_open_to_open' ? '开盘到开盘' : '收盘到收盘（旧）' }}</span></div>
          <div class="config-row" v-if="m.label_norm"><span class="config-key">标准化</span><span class="config-val">{{ m.label_norm }}</span></div>
          <div class="config-row" v-if="m.early_stop_metric"><span class="config-key">早停指标</span><span class="config-val">{{ m.early_stop_metric }}</span></div>
          <div class="config-row" v-if="m.param_profile"><span class="config-key">超参方案</span><span class="config-val">{{ m.param_profile }}</span></div>
          <div class="config-row" v-if="m.is_fusion"><span class="config-key">融合模式</span><span class="config-val strong">截面树与时序网络（各占一半，按排名融合）</span></div>
          <div class="config-row" v-if="m.is_fusion"><span class="config-key">训练样本</span><span class="config-val strong">{{ m.is_state_subset ? '状态子样本' : '全样本' }}</span></div>
          <div class="config-row" v-if="m.expert_role"><span class="config-key">专家角色</span><span class="config-val">{{ m.expert_role === 'trend' ? '趋势专家' : '反弹专家' }}</span></div>
          <div class="config-row" v-if="m.is_state_subset"><span class="config-key">子样本日期</span><span class="config-val">{{ m.state_subset_n_dates ?? '-' }} / {{ m.state_subset_total_dates ?? '-' }}</span></div>
          <!-- 次要指标 -->
          <div class="col-title" style="margin-top:8px">更多指标</div>
          <div class="config-row" v-if="m.max_neg_ic_months != null"><span class="config-key">最长连负</span><span class="config-val" :class="m.max_neg_ic_months <= 3 ? 'good' : 'bad'">{{ m.max_neg_ic_months }}月</span></div>
          <div class="config-row" v-if="m.icir_yearly_min != null"><span class="config-key">年度最低稳定性</span><span class="config-val" :class="m.icir_yearly_min > 0.2 ? 'good' : ''">{{ m.icir_yearly_min.toFixed(2) }}</span></div>
          <div class="config-row" v-if="m.ic_decay_1m != null"><span class="config-key">信息系数月衰减</span><span class="config-val" :class="m.ic_decay_1m > -0.03 ? 'good' : ''">{{ (m.ic_decay_1m * 100).toFixed(2) }}%</span></div>
          <div class="config-row" v-if="m.ic_nonoverlap != null"><span class="config-key">去重叠信息系数</span><span class="config-val" :class="m.ic_nonoverlap > 0.02 ? 'good' : ''">{{ (m.ic_nonoverlap * 100).toFixed(2) }}%</span></div>
        </div>

        <div class="tc-col tc-params">
          <div class="col-title">模型超参</div>
          <template v-if="m.model_kwargs && Object.keys(m.model_kwargs).length">
            <template v-for="[key, label] in lab.paramDefs" :key="key">
              <div v-if="m.model_kwargs[key] != null" class="param-row">
                <span class="param-key">{{ label }}</span>
                <span class="param-val">{{ m.model_kwargs[key] }}</span>
              </div>
            </template>
          </template>
          <div v-else class="text-muted" style="font-size:12px;">默认参数</div>
          <div v-if="m.is_dual" class="dual-note">
            <div v-if="m.lgb_w != null" class="param-row">
              <span class="param-key">轻量提升树权重</span>
              <span class="param-val">{{ (m.lgb_w * 100).toFixed(1) }}%</span>
            </div>
            <div v-if="m.xgb_w != null" class="param-row">
              <span class="param-key">极端提升树权重</span>
              <span class="param-val">{{ (m.xgb_w * 100).toFixed(1) }}%</span>
            </div>
          </div>
          <div v-if="m.lgb_ic != null || m.xgb_ic != null || m.lstm_ic != null" class="dual-note">
            <div class="col-title">子模型</div>
            <div class="param-row" v-if="m.lgb_ic != null">
              <span class="param-key" v-if="m.model_type === 'regime' && m.model_kwargs?.model_type === 'linear'">弹性网络</span>
              <span class="param-key" v-else-if="m.model_type === 'regime'">轻量提升树</span>
              <span class="param-key" v-else-if="m.model_kwargs?.model_type === 'linear'">弹性网络</span>
              <span class="param-key" v-else>轻量提升树</span>
              <span class="param-val">{{ (m.lgb_ic * 100).toFixed(2) }}%</span>
            </div>
            <div class="param-row" v-if="m.xgb_ic != null">
              <span class="param-key" v-if="m.model_type === 'regime' && m.model_kwargs?.model_type === 'linear'">多层感知机</span>
              <span class="param-key" v-else-if="m.model_type === 'regime'">极端提升树</span>
              <span class="param-key" v-else-if="m.model_kwargs?.model_type === 'linear'">多层感知机</span>
              <span class="param-key" v-else>极端提升树</span>
              <span class="param-val">{{ (m.xgb_ic * 100).toFixed(2) }}%</span>
            </div>
            <div class="param-row" v-if="m.lstm_ic != null">
              <span class="param-key">时序网络</span>
              <span class="param-val">{{ (m.lstm_ic * 100).toFixed(2) }}%</span>
            </div>
            <div class="param-row" v-if="m.is_fusion && m.model_kwargs?.fusion_weight != null">
              <span class="param-key">融合权重</span>
              <span class="param-val">截面树 {{ (m.model_kwargs.fusion_weight * 100).toFixed(0) }}% / 时序网络 {{ ((1 - m.model_kwargs.fusion_weight) * 100).toFixed(0) }}%</span>
            </div>
          </div>
        </div>
      </div>

      <div class="tc-actions">
        <template v-if="m.model_type !== 'timing' && m.model_type !== 'regime' && m.model_type !== 'regime_v2'">
          <button @click="lab.selectModel(m)" class="btn primary" :disabled="m.disabled">
            {{ lab.selectedModel?.run_id === m.run_id ? '已选中' : '使用此模型' }}
          </button>
          <button v-if="m.has_predictions && lab.selectedModel?.run_id === m.run_id"
                  @click="lab.activeTab = 'predict'" class="btn info">
            📊 去预测
          </button>
          <button v-if="lab.selectedModel?.run_id === m.run_id"
                  @click="lab.activeTab = 'backtest'" class="btn accent">
            🔬 独立回测
          </button>
        </template>
        <button @click="lab.toggleModelDisabled(m)" class="btn warn" style="margin-right:4px;">
          {{ m.disabled ? '恢复' : '停用' }}
        </button>
        <button @click="lab.deleteModel(m)" class="btn danger" style="margin-left:auto;">
          删除
        </button>
      </div>
    </div>
  </section>
</template>

<script>
import { localizeStrategyText } from '../../../utils/strategyTerms.js'

export default {
  name: 'TrainedModelsTab',
  inject: ['qlibLab'],
  setup() { return { localizeStrategyText } },
  computed: {
    lab() {
      return this.qlibLab
    },
  },
  methods: {
    formatPercent(value) {
      return typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : '—'
    },
  },
}
</script>

<style scoped>
.tc-backtest-summary {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px 16px;
  margin: 0 0 14px;
  padding: 10px 12px;
  border: 1px solid var(--border, #334155);
  border-radius: 8px;
  color: var(--text-secondary, #cbd5e1);
  font-size: 12px;
}

.tc-backtest-summary strong {
  color: var(--text-primary, #f8fafc);
}
</style>
