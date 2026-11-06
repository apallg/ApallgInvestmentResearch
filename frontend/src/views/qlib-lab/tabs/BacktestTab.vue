<template>
  <section class="lab-card">
    <h2 class="card-title">独立回测</h2>
    <div class="form-grid">
      <label class="form-field">持仓数量 <input v-model.number="lab.backtestCfg.topk" type="number" class="field field-sm" /></label>
      <label class="form-field">每期剔除数量 <input v-model.number="lab.backtestCfg.n_drop" type="number" class="field field-sm" /></label>
      <label class="form-field">开始 <input v-model="lab.backtestCfg.start_time" class="field" /></label>
      <label class="form-field">结束 <input v-model="lab.backtestCfg.end_time" class="field" /></label>
    </div>
    <button @click="lab.runBacktest" :disabled="lab.backtesting" class="btn primary" :class="{ loading: lab.backtesting }">
      {{ lab.backtesting ? '回测中...' : '运行回测' }}
    </button>
    <div v-if="lab.backtestError" class="form-error">{{ localizeStrategyText(lab.backtestError) }}</div>
    <div v-if="lab.backtestResult && lab.backtestResult.metrics" class="result-box">
      <table>
        <tr v-for="(v, k) in lab.backtestResult.metrics" :key="k"><td>{{ localizeStrategyText(k) }}</td><td>{{ v }}</td></tr>
      </table>
    </div>
      <div v-if="lab.backtestResult?.standard_report?.standard" class="result-box">
        <h3>标准风险指标</h3>
        <table>
          <tr v-for="(v, k) in lab.backtestResult.standard_report.standard.risk" :key="'risk_'+k"><td>{{ localizeStrategyText(k) }}</td><td>{{ v }}</td></tr>
          <tr v-for="(v, k) in lab.backtestResult.standard_report.standard.risk_without_cost" :key="'risk_wo_'+k"><td>{{ localizeStrategyText(k) }}（无成本）</td><td>{{ v }}</td></tr>
          <tr v-for="(v, k) in lab.backtestResult.standard_report.standard.risk_with_cost" :key="'risk_w_'+k"><td>{{ localizeStrategyText(k) }}（含成本）</td><td>{{ v }}</td></tr>
        </table>
      </div>
        <div v-if="lab.backtestResult?.standard_report?.curves?.dates?.length" ref="btChart" style="height:300px;margin-top:12px"></div>
  </section>
</template>

<script>
import * as echarts from 'echarts'
import { localizeStrategyText } from '../../../utils/strategyTerms.js'

export default {
  name: 'BacktestTab',
  inject: ['qlibLab'],
  setup() { return { localizeStrategyText } },
  computed: {
    lab() {
      return this.qlibLab
    },
  },
  watch: {
    'lab.backtestResult.standard_report.curves': {
      handler() {
        this.$nextTick(() => this.renderChart())
      },
      deep: true,
    },
  },
  methods: {
    renderChart() {
      const el = this.$refs.btChart
      const curves = this.lab.backtestResult?.standard_report?.curves
      if (!el || !curves?.dates?.length) return
      let chart = echarts.getInstanceByDom(el)
      if (!chart) chart = echarts.init(el)
      chart.setOption({
        backgroundColor: 'transparent',
        grid: { left: 60, right: 16, top: 16, bottom: 32 },
        tooltip: { trigger: 'axis' },
        legend: { bottom: 0, textStyle: { color: '#aaa' } },
        xAxis: { type: 'category', data: curves.dates, axisLabel: { color: '#888', fontSize: 10 } },
        yAxis: { type: 'value', axisLabel: { color: '#888', fontSize: 10 } },
        series: [
          { name: '策略累计', type: 'line', data: curves.cumulative_return, symbol: 'none', lineStyle: { color: '#26a69a' } },
          { name: '基准累计', type: 'line', data: curves.benchmark_return, symbol: 'none', lineStyle: { color: '#888', type: 'dashed' } },
          { name: '超额(无成本)', type: 'line', data: curves.excess_return_without_cost, symbol: 'none', lineStyle: { color: '#6b9fd4' } },
        ],
      }, true)
    },
  },
}
</script>
