<template>
  <div class="app-error">
    <span class="app-error-icon">⚠️</span>
    <span class="app-error-text">{{ message }}</span>
    <button v-if="retryFn" class="app-error-retry" @click="retryFn">重试</button>
  </div>
</template>

<script setup>
defineProps({
  message: { type: String, default: '加载失败' },
  retryFn: { type: Function, default: null },
})
</script>

<style scoped>
.app-error { display:flex; align-items:center; gap:8px; padding:12px 16px; background:rgba(239,68,68,.1); border:1px solid rgba(239,68,68,.3); border-radius:8px; color:#ef4444; font-size:13px }
.app-error-icon { font-size:16px }
.app-error-text { flex:1 }
.app-error-retry { padding:4px 12px; border:1px solid rgba(239,68,68,.4); border-radius:4px; background:transparent; color:#ef4444; cursor:pointer; font-size:12px }
.app-error-retry:hover { background:rgba(239,68,68,.2) }
</style>
