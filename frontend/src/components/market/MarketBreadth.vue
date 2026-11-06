<template>
  <div class="mb-root">
    <div class="mb-summary" v-if="breadth">
      <div class="mb-sum-item">
        <span class="mb-sum-label">上涨家数</span>
        <span class="mb-sum-val up">{{ breadth.up_count ?? '--' }}</span>
      </div>
      <div class="mb-sum-item">
        <span class="mb-sum-label">下跌家数</span>
        <span class="mb-sum-val down">{{ breadth.down_count ?? '--' }}</span>
      </div>
      <div class="mb-sum-item">
        <span class="mb-sum-label">涨跌比（涨/跌）</span>
        <span class="mb-sum-val">{{ breadth.up_ratio ?? '--' }}</span>
      </div>
      <div class="mb-sum-item">
        <span class="mb-sum-label">平盘 / 未知</span>
        <span class="mb-sum-val">{{ breadth.flat_count ?? '--' }} / {{ breadth.unknown_count ?? '--' }}</span>
      </div>
      <div class="mb-sum-item">
        <span class="mb-sum-label">涨跌幅覆盖</span>
        <span class="mb-sum-val">{{ breadth.change_coverage == null ? '--' : `${(breadth.change_coverage * 100).toFixed(1)}%` }}</span>
      </div>
      <div class="mb-sum-item">
        <span class="mb-sum-label">中位涨跌</span>
        <span :class="['mb-sum-val', breadth.pct_median == null ? '' : breadth.pct_median >= 0 ? 'up' : 'down']">
          {{ fmtPct(breadth.pct_median) }}
        </span>
      </div>
    </div>
    <div class="mb-coverage-note" v-if="breadth">
      已入库样本 {{ breadth.total_stocks ?? '--' }} 只；有效涨跌幅 {{ breadth.valid_change_count ?? '--' }} 只。覆盖率以当日已入库行计算，不代表全市场应交易证券覆盖。
    </div>

    <div class="mb-tables">
      <DataTable title="涨幅榜 TOP20" :rows="gainers" :columns="columns" empty-text="暂无数据" />
      <DataTable title="跌幅榜 TOP20" :rows="losers" :columns="columns" empty-text="暂无数据" />
    </div>
  </div>
</template>

<script setup>
import DataTable from '../DataTable.vue'

defineProps({
  gainers: { type: Array, default: () => [] },
  losers: { type: Array, default: () => [] },
  breadth: { type: Object, default: null },
})

const columns = [
  { key: 'code', label: '代码' },
  { key: 'name', label: '名称' },
  { key: 'price', label: '最新价', format: v => Number(v || 0).toFixed(2) },
  { key: 'pct_chg', label: '涨跌幅', format: v => v == null || !Number.isFinite(Number(v)) ? '--' : ((Number(v) > 0 ? '+' : '') + Number(v).toFixed(2) + '%'),
    tdClass: row => row.pct_chg == null ? '' : row.pct_chg > 0 ? 'profit-up' : row.pct_chg < 0 ? 'profit-down' : '' },
]

function fmtPct(v) {
  if (v == null) return '--'
  return (v > 0 ? '+' : '') + Number(v).toFixed(2) + '%'
}
</script>

<style scoped>
.mb-root { display: flex; flex-direction: column; gap: 14px; }

.mb-summary { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; }
.mb-sum-item { display: flex; flex-direction: column; gap: 4px; padding: 12px 16px; background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius); }
.mb-sum-label { font-size: var(--fs-caption); color: var(--text-muted); }
.mb-sum-val { font-family: var(--font-mono); font-size: var(--fs-lg); font-weight: 600; color: var(--text); }
.mb-coverage-note { color: var(--text-muted); font-size: var(--fs-caption); }
.mb-sum-val.up { color: var(--up); }
.mb-sum-val.down { color: var(--down); }

.mb-tables { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 900px) { .mb-tables { grid-template-columns: 1fr; } }
@media (max-width: 900px) { .mb-summary { grid-template-columns: repeat(3, 1fr); } }
</style>
