<template>
  <div class="scheduler-card" :class="{ running: running }">
    <div class="sch-header">
      <span class="sch-indicator" :class="running ? 'on' : 'off'" />
      <span class="sch-title">{{ title }}</span>
    </div>
    <div class="sch-body">
      <div class="sch-stat">
        <span class="sch-label">状态</span>
        <span class="sch-value">{{ running ? '运行中' : '已停止' }}</span>
      </div>
      <div class="sch-stat" v-if="lastScan">
        <span class="sch-label">上次扫描</span>
        <span class="sch-value">{{ formatTime(lastScan) }}</span>
      </div>
      <div class="sch-stat" v-if="scanCount > 0">
        <span class="sch-label">扫描次数</span>
        <span class="sch-value">{{ scanCount }}</span>
      </div>
    </div>
    <div class="sch-actions">
      <button v-if="!running" class="sch-btn start" :disabled="loading" @click="$emit('start')">启动</button>
      <button v-else class="sch-btn stop" :disabled="loading" @click="$emit('stop')">停止</button>
      <button class="sch-btn trigger" :disabled="loading" @click="$emit('trigger')">立即触发</button>
    </div>
    <div v-if="lastError" class="sch-error">{{ lastError }}</div>
  </div>
</template>

<script setup>
defineProps({
  title: { type: String, default: '' },
  running: { type: Boolean, default: false },
  lastScan: { type: [Number, String], default: null },
  scanCount: { type: Number, default: 0 },
  lastError: { type: String, default: '' },
  loading: { type: Boolean, default: false },
})
defineEmits(['start', 'stop', 'trigger'])

function formatTime(ts) {
  if (!ts) return '-'
  try {
    // 后端 last_scan 是 ISO 字符串; 若传的是 epoch 秒则转换
    const d = typeof ts === 'number' && ts < 1e12 ? new Date(ts * 1000) : new Date(ts)
    return Number.isNaN(d.getTime()) ? String(ts) : d.toLocaleString('zh-CN')
  } catch { return String(ts) }
}
</script>

<style scoped>
.scheduler-card { background:var(--card-bg); border:1px solid var(--border); border-radius:var(--radius); padding:18px; min-width:240px }
.scheduler-card.running { border-color:var(--success) }
.sch-header { display:flex; align-items:center; gap:8px; margin-bottom:12px }
.sch-indicator { width:8px; height:8px; border-radius:50% }
.sch-indicator.on { background:var(--success) }
.sch-indicator.off { background:var(--text-muted) }
.sch-title { font-size:var(--fs-base); font-weight:600; color:var(--text) }
.sch-body { display:flex; flex-direction:column; gap:4px; margin-bottom:14px }
.sch-stat { display:flex; justify-content:space-between; font-size:var(--fs-sm) }
.sch-label { color:var(--text-muted) }
.sch-value { color:var(--text-secondary) }
.sch-actions { display:flex; gap:8px }
.sch-btn { flex:1; padding:6px 0; border:none; border-radius:var(--radius-sm); font-size:var(--fs-sm); font-weight:500; cursor:pointer; font-family:var(--font) }
.sch-btn.start { background:var(--success); color:#fff }
.sch-btn.stop { background:var(--danger); color:#fff }
.sch-btn.trigger { background:var(--primary); color:var(--on-primary) }
.sch-btn:disabled { opacity:.5; cursor:not-allowed }
.sch-error { margin-top:8px; font-size:var(--fs-xs); color:var(--danger) }
</style>
