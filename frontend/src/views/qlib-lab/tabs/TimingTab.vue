<template>
  <section class="lab-card">
    <h2 class="card-title">指数择时模型
      <span class="card-sub">
        {{ lab.timingTrainCfg.task_type === 'regime_v2' ? '梯度提升回归 · 径向基软映射' : (lab.timingTrainCfg.model_type === 'linear' ? '弹性网络与多层感知机 · 标准化与正则化' : '双树模型') }} · 市场宽度与隐状态特征
      </span>
    </h2>

    <!-- 当前模型状态 -->
    <div class="fm-block" v-if="lab.regimeModelInfo">
      <div class="fm-block-title">当前模型状态
        <span :style="{ color: lab.isModelStale ? 'var(--up)' : 'var(--down)', marginLeft: '8px', fontSize: '12px' }">
          {{ lab.isModelStale ? '⚠ 模型已陈旧，建议重训' : '✓ 模型新鲜' }}
        </span>
      </div>
      <div style="color:#888;font-size:12px;margin:4px 0">
        最后训练：<b style="color:#ddd">{{ lab.regimeModelInfo.last_train_date || '未知' }}</b>
        <template v-if="lab.regimeModelInfo.valid_acc != null">
          · 准确率 {{ (lab.regimeModelInfo.valid_acc * 100).toFixed(1) }}%
          · 前两类命中率 {{ (lab.regimeModelInfo.valid_top2 * 100).toFixed(1) }}%
        </template>
      </div>
      <div v-if="lab.regimeModelInfo.transition_matrix" style="margin-top:8px;display:inline-block">
        <div v-for="(row, i) in lab.regimeModelInfo.transition_matrix" :key="i" style="display:flex;align-items:center;gap:2px;height:22px">
          <span style="width:64px;font-size:11px;color:#999;flex-shrink:0">{{ lab.regimeStateLabels[i] }}</span>
          <span v-for="(v, j) in row" :key="j"
                :style="{ background: lab.heatColor(v), width: '34px', height: '20px', display: 'inline-block', textAlign: 'center', lineHeight: '20px', fontSize: '10px', color: v > 0.5 ? '#fff' : '#666', borderRadius: '2px' }"
                :title="`${lab.regimeStateLabels[i]} → ${lab.regimeStateLabels[j]}: ${(v*100).toFixed(1)}%`">
            {{ (v*100).toFixed(0) }}
          </span>
        </div>
      </div>
    </div>

    <!-- 模型训练 -->
    <div class="fm-block">
      <div class="fm-block-title">模型训练
        <span v-if="lab.timingTrainResult" class="fm-block-badge">分类能力={{ lab.timingTrainResult.fusion_auc?.toFixed(3) }}</span>
      </div>
      <div class="form-grid">
        <label class="form-field">任务类型 <select v-model="lab.timingTrainCfg.task_type" class="field">
          <option value="binary">择时 (涨跌)</option>
          <option value="regime">市场状态第一版（分类）</option>
          <option value="regime_v2">市场状态第二版（回归）</option>
        </select></label>
        <label v-if="lab.timingTrainCfg.task_type !== 'regime_v2'" class="form-field">模型类型 <select v-model="lab.timingTrainCfg.model_type" class="field">
          <option value="tree">双树模型</option>
          <option value="linear">弹性网络与多层感知机</option>
        </select></label>
        <label class="form-field">标签周期 <input v-model.number="lab.timingTrainCfg.label_horizon" type="number" min="1" max="20" class="field field-sm" /> 日</label>
        <label v-if="lab.timingTrainCfg.task_type === 'binary'" class="form-field">阈值 <input v-model.number="lab.timingTrainCfg.threshold" type="number" min="0" max="0.1" step="0.005" class="field field-sm" /></label>
      </div>
      <div class="form-grid" style="margin-top:8px">
        <label class="form-field">训练开始 <input v-model="lab.timingTrainCfg.train_start" class="field" /></label>
        <label class="form-field">训练结束 <input v-model="lab.timingTrainCfg.train_end" class="field" /></label>
        <label class="form-field">验证开始 <input v-model="lab.timingTrainCfg.valid_start" class="field" /></label>
        <label class="form-field">验证结束 <input v-model="lab.timingTrainCfg.valid_end" class="field" /></label>
      </div>
      <button @click="lab.startTimingTrain" :disabled="lab.timingTraining" class="btn" :class="{ loading: lab.timingTraining }" style="margin-top:8px">
        {{ lab.timingTraining ? '训练中...' : '训练择时模型' }}
      </button>
      <p v-if="lab.timingTrainMsg" class="msg" :class="{ error: lab.timingTrainError }" style="margin-top:8px">{{ lab.timingTrainMsg }}</p>

      <div v-if="lab.timingTrainResult" class="tm-metrics-strip">
        <template v-if="lab.timingTrainResult.task_type === 'regime_v2'">
          <div class="tm-metric">
            <span class="tm-metric-val" :class="lab.timingTrainResult.ret_r2 > 0 ? 'up' : 'down'">{{ lab.timingTrainResult.ret_r2?.toFixed(3) || '--' }}</span>
            <span class="tm-metric-label">收益拟合度</span>
          </div>
          <div class="tm-metric">
            <span class="tm-metric-val" :class="lab.timingTrainResult.vol_r2 > 0 ? 'up' : 'down'">{{ lab.timingTrainResult.vol_r2?.toFixed(3) || '--' }}</span>
            <span class="tm-metric-label">波动率拟合度</span>
          </div>
          <div class="tm-metric">
            <span class="tm-metric-val">{{ lab.timingTrainResult.ret_mae?.toFixed(4) || '--' }}</span>
            <span class="tm-metric-label">收益平均绝对误差</span>
          </div>
        </template>
        <template v-else>
          <div class="tm-metric">
            <span class="tm-metric-val">{{ lab.timingTrainResult.lgb_auc?.toFixed(3) || lab.timingTrainResult.en_auc?.toFixed(3) || '--' }}</span>
            <span class="tm-metric-label">{{ lab.timingTrainResult.lgb_auc != null ? '轻量提升树' : '弹性网络' }}</span>
          </div>
          <div class="tm-metric">
            <span class="tm-metric-val">{{ lab.timingTrainResult.xgb_auc?.toFixed(3) || lab.timingTrainResult.mlp_auc?.toFixed(3) || '--' }}</span>
            <span class="tm-metric-label">{{ lab.timingTrainResult.xgb_auc != null ? '极端提升树' : '多层感知机' }}</span>
          </div>
          <div class="tm-metric">
            <span class="tm-metric-val tm-metric-val--primary">{{ lab.timingTrainResult.fusion_auc?.toFixed(3) || '--' }}</span>
            <span class="tm-metric-label">融合结果</span>
          </div>
        </template>
        <div class="tm-metric">
          <span class="tm-metric-val">{{ lab.timingTrainResult.n_features || '--' }}</span>
          <span class="tm-metric-label">特征数</span>
        </div>
        <div class="tm-metric">
          <span class="tm-metric-val">{{ lab.timingTrainResult.n_train || '--' }}</span>
          <span class="tm-metric-label">样本量</span>
        </div>
      </div>
    </div>

    <!-- 当前信号 -->
    <div class="fm-block">
      <div class="fm-block-title">当前信号
        <span v-if="lab.timingPredictResult" class="fm-block-badge" :class="lab.timingPredictResult.signal === 'bullish' ? 'up' : 'down'">
          {{ lab.timingPredictResult.signal === 'bullish' ? '看多' : '看空' }}
        </span>
        <span v-if="lab.timingRegimeResult?.top_state" class="fm-block-badge">{{ stateName(lab.timingRegimeResult.top_state) }}</span>
      </div>
      <div class="tm-signal-row">
        <label class="form-field">日期 <input v-model="lab.timingPredictDate" placeholder="最新" class="field" style="width:110px" /></label>
        <button @click="lab.runTimingPredict" :disabled="lab.timingPredicting" class="btn">{{ lab.timingPredicting ? '...' : '获取信号' }}</button>
      </div>
      <p v-if="lab.timingPredictError" class="error" style="margin-top:8px">{{ lab.timingPredictError }}</p>

      <div v-if="lab.timingPredictResult" class="tm-signal-cards">
        <div class="tm-signal-card" :class="lab.timingPredictResult.signal === 'bullish' ? 'tm-signal-card--up' : 'tm-signal-card--down'">
          <span class="data-value data-value--xl">{{ (lab.timingPredictResult.position * 100).toFixed(0) }}%</span>
          <span class="tm-signal-label">二元仓位</span>
        </div>
        <div class="tm-signal-card">
          <span class="data-value data-value--xl">{{ (lab.timingPredictResult.prob * 100).toFixed(1) }}%</span>
          <span class="tm-signal-label">上涨概率</span>
        </div>
        <div class="tm-signal-card">
          <span class="tm-signal-icon">{{ lab.timingPredictResult.signal === 'bullish' ? '🟢' : lab.timingPredictResult.signal === 'bearish' ? '🔴' : '⚪' }}</span>
          <span class="tm-signal-label">{{ lab.timingPredictResult.as_of || '' }}</span>
        </div>
      </div>

      <div v-if="lab.timingRegimeResult?.probs" class="tm-regime" style="margin-top:12px">
        <div class="fm-block-title" style="margin-bottom:8px">市场状态概率</div>
        <div class="tm-regime-row" v-for="[name, prob] in Object.entries(lab.timingRegimeResult.probs)" :key="name">
          <span class="tm-regime-name">{{ stateName(name) }}</span>
          <div class="tm-regime-track"><div class="tm-regime-fill" :style="{ width: (prob * 100).toFixed(1) + '%' }"></div></div>
          <span class="tm-regime-pct">{{ (prob * 100).toFixed(1) }}%</span>
        </div>
        <div v-if="lab.timingRegimeResult.position_ratio != null" class="tm-regime-foot">
          最终仓位: <strong class="text-primary">{{ (lab.timingRegimeResult.position_ratio * 100).toFixed(0) }}%</strong>
          <span v-if="lab.timingRegimeResult.position_base != null">
            = 基础仓位 {{ (lab.timingRegimeResult.position_base * 100).toFixed(0) }}%
          </span>
          <span v-if="lab.timingRegimeResult.vol_scalar != null">
            × 波动缩放 {{ (lab.timingRegimeResult.vol_scalar * 100).toFixed(0) }}%
          </span>
          <span v-if="lab.timingRegimeResult.pred_vol != null" class="text-muted" style="margin-left:8px;font-size:var(--fs-sm)">
            预测波动: {{ (lab.timingRegimeResult.pred_vol * 100).toFixed(1) }}%
          </span>
        </div>
      </div>
    </div>

    <!-- 历史回测 -->
    <div class="fm-block">
      <div class="fm-block-title">历史回测
        <span v-if="lab.timingBTResult" class="fm-block-badge" :class="lab.timingBTResult.annual_return > 0 ? 'up' : 'down'">
          {{ (lab.timingBTResult.annual_return * 100).toFixed(1) }}% 年化
        </span>
      </div>
      <div class="tm-signal-row">
        <label class="form-field">模型 <select v-model="lab.timingBTCfg.model_name" class="field" style="width:180px">
          <option value="">默认 (最新)</option>
          <option v-for="m in lab.timingModels" :key="m.run_id" :value="m.run_id + '.pkl'">
            {{ m.display_name || m.run_id }}
          </option>
        </select></label>
        <label class="form-field">基准 <select v-model="lab.timingBTCfg.benchmark" class="field" style="width:110px">
          <option value="sh000300">沪深300</option>
          <option value="sh000001">上证</option>
          <option value="sh000905">中证500</option>
        </select></label>
        <span class="text-muted" style="font-size:var(--fs-sm)">{{ lab.timingBTCfg.start_date }} ~ {{ lab.timingBTCfg.end_date }}</span>
        <button @click="lab.startTimingBT" :disabled="lab.timingBTing" class="btn" :class="{ loading: lab.timingBTing }">
          {{ lab.timingBTing ? '回测中...' : '运行回测' }}
        </button>
      </div>
      <p v-if="lab.timingBTMsg" class="msg" :class="{ error: lab.timingBTError }" style="margin-top:8px">{{ lab.timingBTMsg }}</p>

      <div v-if="lab.timingBTResult" ref="timingBTChart" style="height:300px;margin-top:12px"></div>

      <div v-if="lab.timingBTResult" class="tm-metrics-strip" style="margin-top:12px">
        <div class="tm-metric">
          <span class="tm-metric-val" :class="lab.timingBTResult.total_return > 0 ? 'up' : 'down'">{{ (lab.timingBTResult.total_return * 100).toFixed(1) }}%</span>
          <span class="tm-metric-label">累计收益</span>
        </div>
        <div class="tm-metric">
          <span class="tm-metric-val">{{ (lab.timingBTResult.annual_return * 100).toFixed(1) }}%</span>
          <span class="tm-metric-label">年化收益</span>
        </div>
        <div class="tm-metric">
          <span class="tm-metric-val">{{ lab.timingBTResult.sharpe?.toFixed(2) || '--' }}</span>
          <span class="tm-metric-label">夏普</span>
        </div>
        <div class="tm-metric">
          <span class="tm-metric-val down">{{ (lab.timingBTResult.max_drawdown * 100).toFixed(1) }}%</span>
          <span class="tm-metric-label">最大回撤</span>
        </div>
        <div class="tm-metric">
          <span class="tm-metric-val">{{ (lab.timingBTResult.win_rate * 100).toFixed(0) }}%</span>
          <span class="tm-metric-label">胜率</span>
        </div>
        <div class="tm-metric">
          <span class="tm-metric-val" :class="lab.timingBTResult.excess_return >= 0 ? 'up' : 'down'">
            {{ lab.timingBTResult.excess_return != null ? (lab.timingBTResult.excess_return * 100).toFixed(1) + '%' : '--' }}
          </span>
          <span class="tm-metric-label">超额</span>
        </div>
      </div>
    </div>
  </section>
</template>

<script>
import * as echarts from 'echarts'

export default {
  name: 'TimingTab',
  inject: ['qlibLab'],
  computed: {
    lab() {
      return this.qlibLab
    },
  },
  watch: {
    'lab.timingBTResult': {
      handler() {
        this.$nextTick(() => this.renderChart())
      },
    },
  },
  methods: {
    stateName(value) {
      return ({ volatile_up: '高波快涨', volatile_down: '高波快跌', volatile_flat: '高波震荡', steady_up: '稳步上涨', steady_down: '稳步下跌', steady_flat: '低波盘整' })[value] || value
    },
    renderChart() {
      const el = this.$refs.timingBTChart
      if (!el || !this.lab.timingBTResult?.dates?.length) return
      let chart = echarts.getInstanceByDom(el)
      if (!chart) chart = echarts.init(el)
      const dates = this.lab.timingBTResult.dates
      chart.setOption({
        backgroundColor: 'transparent',
        grid: { left: 50, right: 16, top: 12, bottom: 28 },
        xAxis: { type: 'category', data: dates, axisLabel: { fontSize: 9, interval: Math.floor(dates.length / 6) } },
        yAxis: { type: 'value', axisLabel: { fontSize: 9, formatter: v => v.toFixed(1) } },
        series: [
          { name: '策略', type: 'line', data: this.lab.timingBTResult.equity, smooth: true,
            lineStyle: { color: '#26a69a', width: 2 }, itemStyle: { color: '#26a69a' }, symbol: 'none' },
          { name: '持仓不动', type: 'line', data: this.lab.timingBTResult.hold_equity, smooth: true,
            lineStyle: { color: '#888', width: 1, type: 'dashed' }, symbol: 'none' },
          { name: '基准', type: 'line', data: this.lab.timingBTResult.benchmark_equity, smooth: true,
            lineStyle: { color: '#ff7043', width: 1 }, symbol: 'none' },
        ],
        legend: { bottom: 0, textStyle: { fontSize: 10, color: '#aaa' }, itemWidth: 16, itemHeight: 2 },
        tooltip: { trigger: 'axis' },
      }, true)
    },
  },
}
</script>
