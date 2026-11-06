<template>
  <div class="pulse-strip">
    <!-- 市场状态 -->
    <div class="pulse-section">
      <span class="pulse-dot" :class="regimeClass"></span>
      <div class="pulse-block">
        <span class="pulse-regime">{{ regimeLabel }}</span>
        <span class="pulse-sub">{{ regimeSub }}</span>
      </div>
    </div>

    <span class="pulse-sep"></span>

    <!-- 择时仓位 -->
    <div class="pulse-section">
      <div class="pulse-block">
        <span class="pulse-label">仓位</span>
        <div class="pulse-bar-wrap">
          <div class="pulse-bar" :style="{ width: (ratioPct != null ? ratioPct : 0) + '%', background: barColor }"></div>
        </div>
        <span class="pulse-value" :style="{ color: barColor }">{{ ratioPct != null ? ratioPct + '%' : '--' }}</span>
      </div>
      <span class="pulse-sub">上涨概率 {{ probPct != null ? probPct + '%' : '--' }}</span>
    </div>

    <span class="pulse-sep"></span>

    <!-- 市场宽度 -->
    <div class="pulse-section">
      <div class="pulse-breadth">
        <span class="pulse-breadth-item">涨 <strong class="pulse-up">{{ fmt(breadth?.up_count) }}</strong></span>
        <span class="pulse-breadth-item">跌 <strong class="pulse-down">{{ fmt(breadth?.down_count) }}</strong></span>
        <span class="pulse-breadth-item">平 <strong>{{ fmt(breadth?.flat_count) }}</strong></span>
        <span class="pulse-breadth-item">未知 <strong>{{ fmt(breadth?.unknown_count) }}</strong></span>
        <span class="pulse-breadth-item">涨跌比 <strong>{{ breadth?.up_ratio ?? '--' }}</strong></span>
        <span class="pulse-breadth-item">已入库覆盖 <strong>{{ breadth?.change_coverage == null ? '--' : `${(breadth.change_coverage * 100).toFixed(1)}%` }}</strong></span>
        <span class="pulse-breadth-item">中位 <strong :class="breadth?.pct_median == null ? '' : breadth.pct_median >= 0 ? 'pulse-up' : 'pulse-down'">{{ fmtPct(breadth?.pct_median) }}</strong></span>
      </div>
    </div>

    <span class="pulse-sep"></span>

    <!-- 风险信号 -->
    <div class="pulse-section pulse-risk" v-if="riskLabel">
      <span class="pulse-risk-dot" :class="riskClass"></span>
      <span class="pulse-risk-text">{{ riskLabel }}</span>
    </div>
    <div class="pulse-section pulse-risk" v-else-if="pendingCount === 0">
      <span class="pulse-risk-dot safe"></span>
      <span class="pulse-risk-text">无异常</span>
    </div>

    <!-- 待处理信号 -->
    <div class="pulse-section pulse-risk" v-if="pendingCount > 0" style="margin-left:6px">
      <span class="pulse-risk-dot warn"></span>
      <span class="pulse-risk-text">{{ pendingCount }} 条待处理信号</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  regime: { type: Object, default: null },
  timing: { type: Object, default: null },
  breadth: { type: Object, default: null },
  riskLabel: { type: String, default: '' },
  pendingCount: { type: Number, default: 0 },
})

const regimeMap = {
  volatile_up: '高波快涨', volatile_down: '高波快跌', volatile_flat: '高波震荡',
  steady_up: '稳步上涨', steady_down: '稳步下跌', steady_flat: '低波盘整',
}
const regimeLabel = computed(() => regimeMap[props.regime?.regime] || props.regime?.regime || '加载中...')
const regimeSub = computed(() => {
  if (!props.regime) return ''
  const c = props.regime.confidence
  return c != null ? `置信 ${(c * 100).toFixed(0)}%` : ''
})
const regimeClass = computed(() => props.regime?.regime || '')

const probPct = computed(() => {
  const p = props.timing?.prob
  return p != null ? (p * 100).toFixed(0) : null
})
const ratioPct = computed(() => {
  const r = props.timing?.position_ratio
  return r != null ? (r * 100).toFixed(0) : null
})
const barColor = computed(() => {
  const r = props.timing?.position_ratio
  if (r == null) return '#6b7280'
  if (r >= 0.8) return '#34d399'
  if (r >= 0.6) return '#84cc16'
  if (r >= 0.4) return '#f4a742'
  return '#f55555'
})

const riskClass = computed(() => {
  const r = props.riskLabel || ''
  if (r.includes('危险') || r.includes('高')) return 'danger'
  if (r.includes('关注') || r.includes('偏高')) return 'warn'
  return 'safe'
})

function fmt(v) {
  if (v == null) return '--'
  return Number(v).toLocaleString()
}
function fmtPct(v) {
  if (v == null) return '--'
  return Number(v).toFixed(2) + '%'
}
</script>

<style scoped>
.pulse-strip {
  display: flex;
  align-items: center;
  gap: 0;
  padding: 12px 20px;
  margin-bottom: 14px;
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  font-size: var(--fs-sm);
  overflow: hidden;
}

.pulse-section {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 18px;
}
.pulse-section:first-child { padding-left: 0; }
.pulse-section:last-child { padding-right: 0; }

.pulse-sep {
  width: 1px;
  height: 36px;
  background: var(--border);
  flex-shrink: 0;
}

/* regime */
.pulse-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
  background: var(--text-muted);
}
.pulse-dot.volatile_up, .pulse-dot.steady_up { background: var(--up); }
.pulse-dot.volatile_down, .pulse-dot.steady_down { background: var(--down); }
.pulse-dot.ranging { background: var(--warning); }
.pulse-dot.volatile { background: #f97316; }

.pulse-block {
  display: flex;
  flex-direction: column;
  gap: 1px;
}
.pulse-regime {
  font-weight: 600;
  color: var(--text);
  font-size: var(--fs-base);
  white-space: nowrap;
}
.pulse-sub {
  font-size: var(--fs-xs);
  color: var(--text-muted);
  white-space: nowrap;
}

/* timing */
.pulse-label {
  color: var(--text-muted);
  font-size: var(--fs-xs);
  margin-right: 2px;
}
.pulse-bar-wrap {
  width: 80px; height: 6px;
  background: var(--border);
  border-radius: 3px;
  overflow: hidden;
}
.pulse-bar {
  height: 100%;
  border-radius: 3px;
  transition: width 0.6s;
}
.pulse-value {
  font-size: var(--fs-lg);
  font-weight: 700;
  min-width: 42px;
  font-variant-numeric: tabular-nums;
}

/* breadth */
.pulse-breadth {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}
.pulse-breadth-item {
  color: var(--text-muted);
  font-size: var(--fs-sm);
  white-space: nowrap;
}
.pulse-breadth-item strong {
  color: var(--text);
  margin-left: 3px;
  font-variant-numeric: tabular-nums;
}
.pulse-up { color: var(--up) !important; }
.pulse-down { color: var(--down) !important; }

/* risk */
.pulse-risk {
  margin-left: auto;
}
.pulse-risk-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
  background: var(--success);
}
.pulse-risk-dot.warn { background: var(--warning); }
.pulse-risk-dot.danger { background: var(--danger); }
.pulse-risk-dot.safe { background: var(--success); }
.pulse-risk-text {
  color: var(--text-secondary);
  font-size: var(--fs-sm);
  white-space: nowrap;
}
</style>
