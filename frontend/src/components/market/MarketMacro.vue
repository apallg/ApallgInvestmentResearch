<template>
  <div class="mm-root">
    <div class="card mm-risk-card" v-if="macro?.risk_score != null">
      <div class="card-title">宏观风险</div>
      <div class="mm-risk-bar" :class="'risk-' + macro.risk_level">
        <div class="mm-risk-track"><div class="mm-risk-fill" :style="{ width: (macro.risk_score * 100) + '%' }"></div></div>
        <div class="mm-risk-text">
          {{ macro.risk_level === 'danger' ? '⚠ 危险' : macro.risk_level === 'warning' ? '⚡ 关注' : '✓ 正常' }}
          ({{ (macro.risk_score * 100).toFixed(0) }})
        </div>
      </div>
      <ul class="mm-risk-signals" v-if="macro.risk_signals?.length">
        <li v-for="(s, i) in macro.risk_signals" :key="i">{{ s }}</li>
      </ul>
    </div>

    <div class="card">
      <div class="card-title">宏观指标全览</div>
      <div v-if="macro?.indicators" class="mm-grid">
        <div class="mm-item" v-for="(v, k) in macro.indicators" :key="k">
          <span class="mm-label">{{ k }}</span>
          <span :class="['mm-val', v?.trend || '']">{{ fmt(v?.value) }}</span>
        </div>
      </div>
      <div v-else class="empty">暂无宏观数据</div>
    </div>
  </div>
</template>

<script setup>
defineProps({ macro: { type: Object, default: null } })

function fmt(v) {
  if (v == null) return '--'
  const n = Number(v)
  return Math.abs(n) < 10 ? n.toFixed(3) : n.toFixed(2)
}
</script>

<style scoped>
.mm-root { display: flex; flex-direction: column; gap: 14px; }

.mm-risk-card { padding: 16px; }
.mm-risk-track { height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; margin-bottom: 8px; }
.mm-risk-fill { height: 100%; border-radius: 4px; transition: width .5s; }
.risk-warning .mm-risk-fill { background: var(--warning); }
.risk-danger .mm-risk-fill { background: var(--danger); }
.risk-normal .mm-risk-fill { background: var(--success); }
.mm-risk-text { font-size: var(--fs-sm); font-weight: 600; color: var(--text-secondary); }
.risk-danger .mm-risk-text { color: var(--danger); }
.mm-risk-signals { margin: 8px 0 0; padding-left: 16px; font-size: var(--fs-sm); color: var(--text-secondary); }
.mm-risk-signals li { padding: 2px 0; }

.mm-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
.mm-item { display: flex; flex-direction: column; gap: 2px; padding: 10px 12px; background: var(--bg-secondary); border-radius: var(--radius-sm); }
.mm-label { font-size: var(--fs-caption); color: var(--text-muted); }
.mm-val { font-family: var(--font-mono); font-size: var(--fs-md); font-weight: 600; color: var(--text-secondary); }
.mm-val.up { color: var(--up); }
.mm-val.down { color: var(--down); }

.empty { text-align: center; padding: 20px; color: var(--text-muted); font-size: var(--fs-sm); }
</style>
