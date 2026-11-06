<template>
  <div class="msent-root">
    <div class="msent-summary">
      <div class="msent-sum-item">
        <span class="msent-sum-label">情绪指数</span>
        <span class="msent-sum-val" :style="{ color: scoreInfo.color }">{{ scoreInfo.text }}</span>
      </div>
      <div class="msent-sum-item">
        <span class="msent-sum-label">积极占比</span>
        <span class="msent-sum-val up">{{ posPct }}%</span>
      </div>
      <div class="msent-sum-item">
        <span class="msent-sum-label">消极占比</span>
        <span class="msent-sum-val down">{{ negPct }}%</span>
      </div>
      <div class="msent-sum-item">
        <span class="msent-sum-label">今日新闻</span>
        <span class="msent-sum-val">{{ sentOverview.today?.total_news || 0 }}条</span>
      </div>
    </div>

    <div class="card">
      <div class="card-title">市场情绪趋势 ({{ (sentOverview.trend || []).length }}天)</div>
      <div ref="chartRef" class="msent-chart"></div>
    </div>

    <div class="msent-cols">
      <div class="card">
        <div class="card-title">最新新闻</div>
        <div v-if="sentNews.length">
          <div v-for="(item, i) in sentNews.slice(0, 30)" :key="i" class="msent-news-row">
            <span :class="['tag', sentTag(item.sent_score)]">{{ sentLabel(item.sent_score) }}</span>
            <span class="ticker">{{ item.code }}</span>
            <span class="msent-news-title" :title="item.title">{{ item.title?.slice(0, 40) }}</span>
            <span class="msent-news-date">{{ item.date }}</span>
          </div>
        </div>
        <div v-else class="empty">暂无新闻</div>
      </div>

      <div class="card">
        <div class="card-title">新闻查询</div>
        <div class="msent-search">
          <input v-model="query.code" placeholder="输入代码 如 600519" @keyup.enter="search" class="msent-input" />
          <button class="btn btn-sm" @click="search" :disabled="query.loading">{{ query.loading ? '查询中...' : '查询' }}</button>
        </div>
        <div class="msent-results" v-if="query.searched">
          <LoadingSpinner v-if="query.loading" />
          <div v-else-if="!query.items.length" class="empty">暂无新闻</div>
          <div v-else v-for="(item, i) in query.items.slice(0, 50)" :key="i" class="msent-news-row">
            <span :class="['tag', sentTag(item.sent_score)]">{{ sentLabel(item.sent_score) }}</span>
            <span class="msent-news-title">{{ item.title?.slice(0, 32) }}</span>
            <span class="msent-news-date">{{ item.date || item.pub_date || '' }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import api from '../../api/index.js'
import LoadingSpinner from '../LoadingSpinner.vue'
import { useChart } from '../../composables/useChart.js'
import { cssVar } from '../../utils/marketFormat.js'

const props = defineProps({ sentOverview: { type: Object, default: () => ({}) } })

const sentOverview = computed(() => props.sentOverview || {})
const scoreInfo = computed(() => {
  const v = sentOverview.value?.today?.avg_score
  const num = Number(v || 0)
  return { text: v != null ? num.toFixed(4) : '--', color: num > 0 ? 'var(--up)' : num < -0.05 ? 'var(--down)' : '' }
})
const posPct = computed(() => ((sentOverview.value?.today?.pos_ratio || 0) * 100).toFixed(1))
const negPct = computed(() => ((sentOverview.value?.today?.neg_ratio || 0) * 100).toFixed(1))
const sentNews = computed(() => sentOverview.value?.news || [])

const query = ref({ code: '', items: [], loading: false, searched: false })

function sentTag(score) {
  if (score == null) return ''
  if (score > 0.05) return 'tag-low'
  if (score < -0.05) return 'tag-high'
  return 'tag-mid'
}
function sentLabel(score) {
  if (score == null) return '--'
  if (score > 0.05) return '积极'
  if (score < -0.05) return '消极'
  return '中性'
}

async function search() {
  const code = query.value.code.trim()
  if (!code) return
  query.value.items = []; query.value.loading = true; query.value.searched = true
  try {
    const res = await api.get(`/news/${code}?limit=50`)
    query.value.items = res.data || []
  } catch (e) {
    query.value.items = []
  } finally {
    query.value.loading = false
  }
}

const { chartRef, render } = useChart()

watch(() => props.sentOverview?.trend, async (trend) => {
  if (!trend?.length) return
  await nextTick()
  const upC = cssVar('--up')
  const warnC = cssVar('--warning')
  const primaryC = cssVar('--primary')
  const textMuted = cssVar('--text-muted')
  const borderColor = cssVar('--border')
  const sentData = trend.map(p => p.score ?? p.avg_score ?? 0)
  const ma10 = sentData.map((_, i) => i < 9 ? null : sentData.slice(i - 9, i + 1).reduce((a, b) => a + b, 0) / 10)
  await render(() => ({
    tooltip: { trigger: 'axis' },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'category', data: trend.map(p => String(p.date || '').slice(5, 10)), axisLabel: { fontSize: 11, color: textMuted } },
    yAxis: [
      { type: 'value', name: '情绪', axisLabel: { fontSize: 11, color: textMuted }, splitLine: { lineStyle: { color: borderColor } } },
      { type: 'value', name: '新闻数', axisLabel: { fontSize: 11, color: textMuted } },
    ],
    series: [
      { type: 'bar', name: '新闻数', yAxisIndex: 1, data: trend.map(p => p.news_count || 0), barWidth: '60%', itemStyle: { color: primaryC + '22', borderRadius: [3, 3, 0, 0] } },
      { type: 'line', name: '情绪指数', yAxisIndex: 0, data: sentData, smooth: true, lineStyle: { color: upC, width: 2.5 }, areaStyle: { color: upC + '10' }, symbol: 'circle', symbolSize: 4 },
      { type: 'line', name: '10日均线', yAxisIndex: 0, data: ma10, smooth: true, lineStyle: { color: warnC, width: 1.5, type: 'dashed' }, symbol: 'none' },
    ],
  }))
}, { immediate: true })
</script>

<style scoped>
.msent-root { display: flex; flex-direction: column; gap: 14px; }

.msent-summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
.msent-sum-item { display: flex; flex-direction: column; gap: 4px; padding: 12px 16px; background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius); }
.msent-sum-label { font-size: var(--fs-caption); color: var(--text-muted); }
.msent-sum-val { font-family: var(--font-mono); font-size: var(--fs-lg); font-weight: 600; color: var(--text); }
.msent-sum-val.up { color: var(--up); }
.msent-sum-val.down { color: var(--down); }

.msent-chart { height: 340px; }
.msent-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 900px) { .msent-cols { grid-template-columns: 1fr; } }

.msent-news-row { display: flex; align-items: center; gap: 8px; padding: 6px 0; border-bottom: 1px solid rgba(42, 42, 80, .25); font-size: var(--fs-sm); }
.msent-news-row:last-child { border-bottom: none; }
.msent-news-title { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-secondary); }
.msent-news-date { color: var(--text-muted); font-size: var(--fs-caption); flex-shrink: 0; }

.msent-search { display: flex; gap: 8px; padding-bottom: 12px; border-bottom: 1px solid var(--border); }
.msent-input { flex: 1; padding: 6px 10px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: var(--bg); color: var(--text); font-size: var(--fs-sm); outline: none; min-width: 0; }
.msent-input:focus { border-color: var(--primary); }
.msent-results { max-height: 600px; overflow-y: auto; }

.empty { text-align: center; padding: 20px; color: var(--text-muted); font-size: var(--fs-sm); }
</style>
