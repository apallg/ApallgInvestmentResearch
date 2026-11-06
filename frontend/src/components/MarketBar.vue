<template>
  <div class="market-bar" :class="barTone">
    <div class="bar-inner">
      <span class="bar-label">Apallg</span>

      <template v-if="!error">
        <!-- 指数 -->
        <span class="bar-item bar-index" v-if="indexName">
          {{ indexName }}
          <strong :class="indexChg >= 0 ? 'up' : 'down'">{{ fmtChg(indexChg) }}</strong>
        </span>

        <span class="bar-divider"></span>

        <!-- 涨跌家数 -->
        <span class="bar-item bar-breadth">
          涨<strong class="up">{{ breadth?.up_count ?? '--' }}</strong>
          <span class="bar-dim">/</span>
          跌<strong class="down">{{ breadth?.down_count ?? '--' }}</strong>
          <span class="bar-dim">/ 未知</span><strong>{{ breadth?.unknown_count ?? '--' }}</strong>
        </span>

        <span class="bar-divider"></span>

        <!-- 市场状态 -->
        <span class="bar-item bar-regime">
          <span class="bar-dot" :class="regimeKey"></span>
          {{ regimeLabel }}
        </span>

        <span class="bar-divider"></span>

        <!-- 建议仓位 -->
        <span class="bar-item bar-position">
          仓位
          <span class="bar-ratio-bar"><span class="bar-ratio-fill" :style="ratioStyle"></span></span>
          <strong>{{ ratioPct != null ? ratioPct + '%' : '--' }}</strong>
        </span>
      </template>

      <span class="bar-item bar-dim" v-else>数据加载中...</span>

      <span class="bar-spacer"></span>
      <span class="bar-item bar-kbd">Ctrl+K 搜股票</span>
      <button type="button" class="bar-item bar-time" aria-label="刷新顶部行情" @click="refresh" title="点击刷新">
        <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7" /></svg>
        {{ lastUpdate || '--:--' }} 更新
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import api from '../api/index.js'

const indexName = ref('')
const indexChg = ref(0)
const breadth = ref(null)
const regimeLabel = ref('')
const regimeKey = ref('')
const ratioPct = ref(0)
const error = ref(false)
const lastUpdate = ref('')

let _timer = null

const REGIME_CN = {
  volatile_up: '高波快涨', volatile_down: '高波快跌', volatile_flat: '高波震荡',
  steady_up: '稳步上涨', steady_down: '稳步下跌', steady_flat: '低波盘整',
}

const barTone = computed(() => {
  const k = regimeKey.value
  if (k === 'volatile_up' || k === 'steady_up') return 'tone-up'
  if (k === 'volatile_down' || k === 'steady_down') return 'tone-down'
  return 'tone-flat'
})

const ratioStyle = computed(() => ({
  width: ratioPct.value + '%',
  background: ratioPct.value >= 60 ? 'var(--up)' : ratioPct.value >= 35 ? 'var(--warning)' : 'var(--down)',
}))

function fmtChg(v) {
  if (v == null) return '--'
  return (v >= 0 ? '+' : '') + v.toFixed(2) + '%'
}

async function refresh() {
  try {
    const [regimeRes, breadthRes] = await Promise.all([
      api.get('/timing/regime/current').catch(() => null),
      api.get('/timing/breadth').catch(() => null),
    ])

    // axios 拦截器已剥离 HTTP response 包装, 但后端 L1 规范包了 {"data":...}
    const regimeData = regimeRes?.data || regimeRes
    const breadthData = breadthRes?.data || breadthRes

    if (regimeData) {
      regimeKey.value = regimeData.regime?.regime || ''
      regimeLabel.value = REGIME_CN[regimeKey.value] || regimeKey.value || '--'
      ratioPct.value = regimeData.position_ratio != null ? Math.round(regimeData.position_ratio * 100) : null
      if (regimeData.regime?.details?.index_change != null) {
        indexChg.value = regimeData.regime.details.index_change * 100
        indexName.value = '沪深300'
      }
    }
    if (breadthData) {
      breadth.value = breadthData
    }
    lastUpdate.value = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
    error.value = false
  } catch (e) {
    error.value = true
  }
}

onMounted(() => {
  refresh()
  _timer = setInterval(refresh, 60000)
})
onUnmounted(() => clearInterval(_timer))
</script>

<style scoped>
.market-bar {
  height: 44px;
  background: rgba(11, 18, 32, .82);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 20px;
  font-size: var(--fs-sm);
  color: var(--text-secondary);
  flex-shrink: 0;
  transition: border-color .6s;
  backdrop-filter: blur(14px);
}
.market-bar.tone-up   { border-bottom-color: rgba(239,68,68,.25); }
.market-bar.tone-down { border-bottom-color: rgba(34,197,94,.25); }

.bar-inner {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  max-width: 1400px;
  margin: 0 auto;
}

.bar-label {
  font-weight: 700;
  color: var(--primary);
  font-size: var(--fs-md);
  margin-right: 4px;
  letter-spacing: .5px;
  padding-right: 12px;
  border-right: 1px solid var(--border);
}

.bar-item {
  display: flex;
  align-items: center;
  gap: 5px;
  white-space: nowrap;
}
.bar-item strong { font-weight: 600; color: var(--text); }
.bar-item strong.up   { color: var(--up); }
.bar-item strong.down { color: var(--down); }
.bar-spacer { flex:1; }

.bar-divider {
  width: 1px;
  height: 16px;
  background: var(--border);
  flex-shrink: 0;
}

.bar-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--text-muted);
  flex-shrink: 0;
}
.bar-dot.volatile_up, .bar-dot.steady_up   { background: var(--up); }
.bar-dot.volatile_down, .bar-dot.steady_down { background: var(--down); }
.bar-dot.volatile_flat, .bar-dot.steady_flat { background: var(--warning); }

.bar-ratio-bar {
  width: 42px; height: 5px;
  background: var(--border);
  border-radius: 99px;
  overflow: hidden;
}
.bar-ratio-fill {
  height: 100%;
  border-radius: 2px;
  transition: width .6s, background .6s;
}

.bar-dim { color: var(--text-muted); }
.bar-time { border:0; background:transparent; color: var(--text-muted); cursor: pointer; font:inherit; font-size: var(--fs-xs); padding:4px; }
.bar-time svg { width:14px; height:14px; fill:none; stroke:currentColor; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }
.bar-time:hover, .bar-time:focus-visible { color: var(--text-secondary); }
.bar-kbd {
  color: var(--text-muted); font-size: var(--fs-xs);
  background: rgba(121,174,224,.08); padding: 4px 9px; border: 1px solid rgba(121,174,224,.14); border-radius: 4px;
}

@media (max-width: 760px) {
  .market-bar { padding: 0 12px; }
  .bar-inner { gap: 8px; }
  .bar-index, .bar-regime, .bar-position, .bar-divider, .bar-kbd { display: none; }
  .bar-label { margin-right:0; padding-right:8px; }
  .bar-time { font-size:0; min-width:32px; min-height:32px; justify-content:center; }
  .bar-time svg { width:16px; height:16px; }
}
</style>
