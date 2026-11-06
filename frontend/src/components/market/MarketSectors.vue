<template>
  <div class="ms-root">
    <div class="card">
      <div class="card-title">板块涨跌热力图</div>
      <div class="ms-caption">
        {{ meta?.as_of ? `日线统计截至 ${meta.as_of}` : '统计日期未知' }}；面积表示涨跌幅绝对值，仅纳入完整报价覆盖的行业，各取涨幅/跌幅前15。
      </div>
      <div ref="chartRef" class="ms-chart"></div>
      <div v-if="!rankedSectors.length" class="empty">当前无达到覆盖率要求的板块行情，不能生成涨跌排名</div>
    </div>

    <div class="card">
      <div class="card-title">板块一览 ({{ sectors.length }})</div>
      <div class="ms-caption">
        行业快照 {{ meta?.snapshot_id || '--' }}；统计日 {{ meta?.as_of || '--' }}；分类标准 {{ meta?.classification_standard || '--' }}。
      </div>
      <div v-if="sectors.length" class="ms-tags">
        <span v-for="s in sectors" :key="s['板块名称'] || s.name"
          :class="['ms-chip', sectorPct(s) > 0 ? 'up' : (sectorPct(s) < 0 ? 'down' : '')]"
          :title="`报价覆盖率 ${Math.round((s.coverage || 0) * 100)}%；${s.rank_eligible === false ? '未参与排名' : '完整覆盖，可参与排名'}`"
          @click="$emit('open-sector', s)">
          {{ s['板块名称'] || s.name }}
          <span class="ms-chip-pct">{{ formatSectorPct(s) }}</span>
        </span>
      </div>
      <div v-else class="empty">暂无数据</div>
    </div>
  </div>
</template>

<script setup>
import { watch, nextTick, computed } from 'vue'
import { useChart } from '../../composables/useChart.js'
import { cssVar } from '../../utils/marketFormat.js'

const props = defineProps({
  sectors: { type: Array, default: () => [] },
  meta: { type: Object, default: null },
})
defineEmits(['open-sector'])
const rankedSectors = computed(() => (props.sectors || []).filter(s => s.rank_eligible !== false && sectorPct(s) != null))

const { chartRef, render } = useChart()

function sectorPct(sector) {
  const value = sector['涨跌幅'] ?? sector.pct_chg
  return value == null || !Number.isFinite(Number(value)) ? null : Number(value)
}

function formatSectorPct(sector) {
  const value = sectorPct(sector)
  return value == null ? '--' : `${value > 0 ? '+' : ''}${value.toFixed(2)}%`
}

watch(() => props.sectors, async (sectors) => {
  await nextTick()
  const eligible = (sectors || []).filter(s => s.rank_eligible !== false && sectorPct(s) != null)
  const rising = eligible.filter(s => sectorPct(s) > 0).sort((a, b) => sectorPct(b) - sectorPct(a)).slice(0, 15)
  const falling = eligible.filter(s => sectorPct(s) < 0).sort((a, b) => sectorPct(a) - sectorPct(b)).slice(0, 15)
  const visible = [...rising, ...falling]
  const maxPct = Math.max(...visible.map(s => Math.abs(sectorPct(s))), 0.01)
  const treeData = visible.map(s => {
    const pct = sectorPct(s)
    const absPct = Math.abs(pct)
    const intensity = Math.min(1, absPct / (maxPct || 1))
    const alpha = 0.2 + intensity * 0.6
    const color = pct >= 0 ? `rgba(239, 68, 68, ${alpha.toFixed(2)})` : `rgba(34, 197, 94, ${alpha.toFixed(2)})`
    return { name: s['板块名称'] || s.name || '', value: Math.max(absPct, 0.001), pct, itemStyle: { color } }
  })
  await render(() => ({
    tooltip: { formatter: p => `${p.name}<br/>涨跌幅: ${p.data?.pct == null ? '--' : `${p.data.pct >= 0 ? '+' : ''}${p.data.pct.toFixed(2)}%`}` },
    series: [{
      type: 'treemap', data: treeData, width: '100%', height: '100%', roam: false, nodeClick: false,
      breadcrumb: { show: false },
      label: { show: true, formatter: p => `${p.name}\n${p.data?.pct == null ? '--' : `${p.data.pct >= 0 ? '+' : ''}${p.data.pct.toFixed(1)}%`}`, fontSize: 10, color: cssVar('--text') },
      itemStyle: { borderColor: cssVar('--bg'), borderWidth: 2, gapWidth: 2 },
      levels: [{ itemStyle: { borderWidth: 2, gapWidth: 2 } }],
    }],
  }))
}, { immediate: true })
</script>

<style scoped>
.ms-root { display: flex; flex-direction: column; gap: 14px; }
.ms-chart { height: 420px; }
.ms-caption { margin-top: -8px; color: var(--text-muted); font-size: var(--fs-caption); }

.ms-tags { display: flex; flex-wrap: wrap; gap: 6px; }
.ms-chip { display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; border-radius: var(--radius-sm); font-size: var(--fs-sm); background: var(--bg-secondary); color: var(--text-secondary); border: 1px solid var(--border); cursor: pointer; transition: filter var(--transition-fast); }
.ms-chip:hover { filter: brightness(1.15); }
.ms-chip.up { border-color: rgba(239, 68, 68, .25); color: var(--up); }
.ms-chip.down { border-color: rgba(34, 197, 94, .25); color: var(--down); }
.ms-chip-pct { font-size: var(--fs-caption); opacity: .8; }

.empty { text-align: center; padding: 20px; color: var(--text-muted); font-size: var(--fs-sm); }
</style>
