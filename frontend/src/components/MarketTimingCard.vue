<template>
  <div class="card timing-card">
    <div class="card-title">
      市场择时信号
      <span class="update-time">{{ updateTime }}</span>
    </div>

    <div v-if="loading" class="loading-text">加载中...</div>

    <template v-if="!loading && displaySignal">
      <div class="timing-main">
        <div class="position-bar-wrap">
          <span class="bar-label">仓位建议</span>
          <div class="position-bar">
            <div class="bar-fill" :style="{width: ratioPct + '%', background: barColor}"></div>
          </div>
          <span class="bar-value" :style="{color: barColor}">{{ ratioPct }}%</span>
        </div>
        <div class="timing-detail">
          上涨概率 <strong>{{ probPct }}%</strong>
        </div>
        <slot name="actions"></slot>
      </div>

      <div class="breadth-grid" v-if="breadth">
        <div class="breadth-item">
          <span class="breadth-label">上涨家数</span>
          <span class="breadth-value up">{{ breadth.up_count?.toLocaleString() || '--' }}</span>
        </div>
        <div class="breadth-item">
          <span class="breadth-label">下跌家数</span>
          <span class="breadth-value down">{{ breadth.down_count?.toLocaleString() || '--' }}</span>
        </div>
        <div class="breadth-item">
          <span class="breadth-label">涨跌比</span>
          <span class="breadth-value">{{ breadth.up_ratio || '--' }}</span>
        </div>
        <div class="breadth-item">
          <span class="breadth-label">资金倾向</span>
          <span class="breadth-value">{{ fmtPercent(breadth.up_volume_ratio) }}</span>
        </div>
        <div class="breadth-item">
          <span class="breadth-label">{{ breadth.limit_count_status === 'unavailable' ? '涨停（规则未知）' : '涨停' }}</span>
          <span class="breadth-value up">{{ breadth.limit_up ?? '--' }}</span>
        </div>
        <div class="breadth-item">
          <span class="breadth-label">{{ breadth.limit_count_status === 'unavailable' ? '跌停（规则未知）' : '跌停' }}</span>
          <span class="breadth-value down">{{ breadth.limit_down ?? '--' }}</span>
        </div>
        <div class="breadth-item">
          <span class="breadth-label">中位涨跌</span>
          <span class="breadth-value" :class="breadth.pct_median == null ? '' : breadth.pct_median > 0 ? 'up' : breadth.pct_median < 0 ? 'down' : ''">{{ breadth.pct_median ?? '--' }}%</span>
        </div>
        <div class="breadth-item">
          <span class="breadth-label">振幅中位</span>
          <span class="breadth-value">{{ breadth.amplitude_median ?? '--' }}%</span>
        </div>
        <div class="breadth-item">
          <span class="breadth-label">未知 / 已入库涨跌覆盖</span>
          <span class="breadth-value">{{ breadth.unknown_count ?? '--' }} / {{ breadth.change_coverage == null ? '--' : `${(breadth.change_coverage * 100).toFixed(1)}%` }}</span>
        </div>
      </div>

      <div class="timing-footer" v-if="regime">
        市场状态 <span class="regime-tag" :class="regime.regime">{{ regimeLabel }}</span>
        <span class="confidence">置信度 {{ fmtPercent(regime.confidence) }}</span>
      </div>
    </template>

    <div v-if="!loading && !displaySignal" class="empty-text">
      暂无择时信号，请先训练模型
    </div>
  </div>
</template>

<script>
import { predictTiming, getBreadth } from '../api/timing.js'

export default {
  name: 'MarketTimingCard',
  props: {
    externalSignal: { type: Object, default: null },
    externalBreadth: { type: Object, default: null },
    noFetch: { type: Boolean, default: false },
    loading: { type: Boolean, default: false },
  },
  data() {
    return { signal: null, breadth: null, regime: null, updateTime: '' }
  },
  computed: {
    displaySignal() { return this.externalSignal || this.signal },
    displayBreadth() { return this.externalBreadth || this.breadth },
    probPct() { return ((this.displaySignal?.prob ?? 0) * 100).toFixed(0) },
    ratioPct() { return ((this.displaySignal?.position_ratio ?? 0) * 100).toFixed(0) },
    barColor() {
      const r = this.displaySignal?.position_ratio ?? 0
      if (r >= 0.8) return '#34d399'
      if (r >= 0.6) return '#84cc16'
      if (r >= 0.4) return '#f4a742'
      return '#f55555'
    },
    regimeLabel() {
      const map = { volatile_up: '高波快涨', volatile_down: '高波快跌', volatile_flat: '高波震荡', steady_up: '稳步上涨', steady_down: '稳步下跌', steady_flat: '低波盘整' }
      return map[this.regime?.regime] || this.regime?.regime || '--'
    }
  },
  async mounted() { if (!this.noFetch) await this.fetch() },
  methods: {
    fmtPercent(v) { return v != null ? (v * 100).toFixed(1) + '%' : '--' },
    async fetch() {
      this.loading = true
      this.updateTime = new Date().toLocaleString()
      const [sig, br] = await Promise.allSettled([predictTiming(), getBreadth()])
      if (sig.status === 'fulfilled') this.signal = sig.value?.data
      if (br.status === 'fulfilled') this.breadth = br.value?.data
      this.loading = false
    }
  }
}
</script>

<style scoped>
.timing-card { margin-bottom: 16px; }
.update-time { font-size: 12px; color: var(--text-muted, #9ca3af); font-weight: 400; margin-left: 12px; }
.timing-main { display: flex; align-items: center; gap: 24px; margin: 12px 0; flex-wrap: wrap; }
.position-bar-wrap { display: flex; align-items: center; gap: 10px; flex: 1; min-width: 200px; }
.bar-label { font-size: 13px; color: var(--text-secondary); min-width: 56px; }
.position-bar { flex: 1; height: 10px; background: var(--border); border-radius: 5px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 5px; transition: width 0.5s; }
.bar-value { font-size: 16px; font-weight: 700; min-width: 48px; }
.timing-detail { font-size: 13px; color: var(--text-secondary); }
.timing-detail strong { font-size: 18px; margin-left: 4px; }
.breadth-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 8px 0; }
.breadth-item { background: var(--bg-secondary); padding: 8px 10px; border-radius: 6px; display: flex; flex-direction: column; gap: 2px; }
.breadth-label { font-size: 11px; color: var(--text-muted); }
.breadth-value { font-size: 15px; font-weight: 600; color: var(--text); }
.breadth-value.up { color: var(--up); }
.breadth-value.down { color: var(--down); }
.timing-footer { margin-top: 8px; font-size: 13px; color: var(--text-secondary); display: flex; align-items: center; gap: 8px; }
.regime-tag { padding: 2px 10px; border-radius: 10px; font-size: 12px; font-weight: 600; color: #fff; }
.regime-tag.volatile_up { background: var(--up); }
.regime-tag.volatile_down { background: var(--down); }
.regime-tag.volatile_flat { background: #f97316; }
.regime-tag.steady_up { background: var(--primary); }
.regime-tag.steady_down { background: var(--text-muted); }
.regime-tag.steady_flat { background: var(--warning); }
.confidence { font-size: 11px; color: var(--text-muted); }
.loading-text, .empty-text { padding: 12px 0; color: var(--text-muted); font-size: 14px; }
</style>
