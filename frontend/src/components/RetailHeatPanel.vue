<template>
  <div class="retail-heat card">
    <header class="rh-head">
      <span class="rh-title">散户热门板块</span>
      <span class="rh-date" v-if="tradeDate">{{ tradeDate }}</span>
      <span class="rh-legend">
        <i class="rh-dot rh-dot-retail"></i>散户流入
        <i class="rh-dot rh-dot-inst"></i>机构流入
      </span>
      <button class="rh-refresh" @click="load" :disabled="loading" title="刷新">{{ loading ? '…' : '↻' }}</button>
    </header>

    <div v-if="error" class="rh-state rh-state-error">{{ error }}</div>
    <div v-else-if="!items.length && !loading" class="rh-state">
      暂无数据 —— 需先采集概念板块成分与资金流，可在数据中心触发采集
    </div>

    <div v-else class="rh-list">
      <div class="rh-cols">
        <span>板块</span>
        <span class="rh-col-heat">热度</span>
        <span class="rh-col-flow">散户 ← 资金博弈 → 机构</span>
        <span class="rh-col-chg">涨幅</span>
      </div>
      <div v-for="(s, i) in items" :key="s.sector" class="rh-row">
        <span class="rh-rank" :class="'rank-' + (i + 1)">{{ i + 1 }}</span>
        <div class="rh-name">
          <span class="rh-sector">{{ s.sector }}</span>
          <span class="rh-count">{{ s.stock_count }}只</span>
        </div>
        <div class="rh-heat">{{ fmtHeat(s.heat_score) }}</div>
        <div class="rh-flow">
          <div class="rh-side">
            <div class="rh-bar rh-bar-retail" :style="{ width: retailW(s) + '%' }"></div>
          </div>
          <div class="rh-zero"></div>
          <div class="rh-side rh-side-inst">
            <div class="rh-bar rh-bar-inst" :style="{ width: instW(s) + '%' }"></div>
          </div>
        </div>
        <div class="rh-flow-nums">
          <span class="rh-num rh-num-retail">{{ fmtYi(s.sm_net) }}</span>
          <span class="rh-num rh-num-inst">{{ fmtYi(s.elg_net) }}</span>
        </div>
        <span class="rh-chg" :class="s.avg_pct_chg >= 0 ? 'up' : 'down'">{{ fmtPct(s.avg_pct_chg) }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import api from '../api/index.js'

const items = ref([])
const loading = ref(false)
const error = ref('')
const tradeDate = ref('')

const maxAbs = computed(() => {
  let m = 0
  for (const s of items.value) {
    m = Math.max(m, Math.abs(s.sm_net || 0), Math.abs(s.elg_net || 0))
  }
  return m || 1
})

function retailW(s) { return Math.min(100, (Math.abs(s.sm_net || 0) / maxAbs.value) * 100) }
function instW(s) { return Math.min(100, (Math.abs(s.elg_net || 0) / maxAbs.value) * 100) }

function fmtHeat(v) { return v == null ? '--' : v.toFixed(2) }
function fmtYi(v) {
  if (v == null) return '--'
  const yi = v / 10000
  return (yi > 0 ? '+' : '') + yi.toFixed(1) + '亿'
}
function fmtPct(v) { return v == null ? '--' : (v > 0 ? '+' : '') + v.toFixed(2) + '%' }

async function load() {
  loading.value = true
  error.value = ''
  try {
    const r = await api.get('/screening/retail-hot-sectors', { params: { top_n: 10 } })
    const data = r?.data || r
    tradeDate.value = data?.trade_date || ''
    items.value = data?.sectors || []
  } catch (e) {
    error.value = '加载失败：' + (e?.message || '网络错误')
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<style scoped>
.retail-heat { padding: 0; overflow: hidden; }

.rh-head {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 16px 12px;
  border-bottom: 1px solid var(--border);
}
.rh-title { font-weight: 700; font-size: var(--fs-lg); color: var(--text); }
.rh-date { color: var(--text-muted); font-size: var(--fs-xs); font-family: var(--font-mono); }
.rh-legend { margin-left: auto; display: flex; align-items: center; gap: 5px; color: var(--text-secondary); font-size: var(--fs-xs); }
.rh-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.rh-dot-retail { background: var(--up); margin-left: 6px; }
.rh-dot-inst { background: var(--down); margin-left: 10px; }
.rh-refresh {
  background: none; border: 1px solid var(--border); color: var(--text-secondary);
  border-radius: var(--radius-sm); width: 26px; height: 26px; cursor: pointer;
  font-size: var(--fs-sm); line-height: 1;
}
.rh-refresh:hover { border-color: var(--border-strong); color: var(--text); }

.rh-state { padding: 24px 16px; color: var(--text-muted); font-size: var(--fs-sm); text-align: center; }
.rh-state-error { color: var(--danger); }

.rh-list { padding: 4px 16px 10px; }
.rh-cols, .rh-row {
  display: grid;
  grid-template-columns: 20px 1fr 46px 1.2fr 120px 56px;
  gap: 10px;
  align-items: center;
}
.rh-cols { padding: 8px 0 4px; color: var(--text-muted); font-size: var(--fs-caption); }
.rh-col-flow { text-align: center; }
.rh-col-chg, .rh-col-heat { text-align: right; }

.rh-row { padding: 7px 0; border-top: 1px solid var(--border); }
.rh-row:hover { background: var(--card-hover); }

.rh-rank { color: var(--text-muted); font-size: var(--fs-xs); font-family: var(--font-mono); text-align: center; }
.rh-rank.rank-1 { color: var(--accent-warm); font-weight: 700; }
.rh-rank.rank-2 { color: var(--primary); font-weight: 600; }
.rh-rank.rank-3 { color: var(--accent-cool); }

.rh-name { display: flex; align-items: baseline; gap: 6px; min-width: 0; }
.rh-sector { color: var(--text); font-size: var(--fs-sm); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rh-count { color: var(--text-muted); font-size: var(--fs-caption); flex-shrink: 0; }

.rh-heat { color: var(--accent-warm); font-family: var(--font-mono); font-size: var(--fs-md); text-align: right; font-weight: 600; }

.rh-flow { display: flex; align-items: center; gap: 0; height: 14px; }
.rh-side { flex: 1; height: 100%; display: flex; align-items: center; }
.rh-side-retail { justify-content: flex-end; }
.rh-side-inst { justify-content: flex-start; }
.rh-bar { height: 6px; border-radius: 2px; transition: width .4s ease; }
.rh-bar-retail { background: var(--up); }
.rh-bar-inst { background: var(--down); }
.rh-zero { width: 1px; height: 12px; background: var(--border-strong); flex-shrink: 0; }

.rh-flow-nums { display: flex; flex-direction: column; gap: 1px; font-family: var(--font-mono); font-size: var(--fs-caption); }
.rh-num-retail { color: var(--up); }
.rh-num-inst { color: var(--down); }

.rh-chg { font-family: var(--font-mono); font-size: var(--fs-sm); text-align: right; }
.rh-chg.up { color: var(--up); }
.rh-chg.down { color: var(--down); }
</style>
