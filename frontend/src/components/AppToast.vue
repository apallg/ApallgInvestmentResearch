<template>
  <teleport to="body">
    <transition-group name="toast" tag="div" class="toast-container">
      <div
        v-for="t in toasts" :key="t.id"
        class="toast" :class="'toast-' + t.type"
        @click="dismiss(t.id)"
      >
        <span class="toast-icon">{{ iconMap[t.type] }}</span>
        <span class="toast-msg">{{ t.msg }}</span>
      </div>
    </transition-group>
  </teleport>
</template>

<script setup>
import { ref } from 'vue'

const toasts = ref([])
let _id = 0

const iconMap = { success: '✓', warning: '⚠', error: '✕', info: 'ℹ' }

function show(msg, type = 'info', duration = 3000) {
  const id = ++_id
  toasts.value.push({ id, msg, type })
  if (duration > 0) setTimeout(() => dismiss(id), duration)
}

function dismiss(id) {
  toasts.value = toasts.value.filter(t => t.id !== id)
}

defineExpose({ show, dismiss })
</script>

<style scoped>
.toast-container {
  position: fixed; top: 52px; right: 20px; z-index: 9999;
  display: flex; flex-direction: column; gap: 8px; pointer-events: none;
}
.toast {
  display: flex; align-items: center; gap: 8px;
  padding: 10px 16px; border-radius: var(--radius-sm);
  font-size: var(--fs-md); color: var(--text); pointer-events: auto; cursor: pointer;
  backdrop-filter: blur(8px);
  box-shadow: 0 4px 16px rgba(0,0,0,.4);
  max-width: 400px;
}
.toast-success { background: rgba(34,197,94,.18); border: 1px solid rgba(34,197,94,.35); }
.toast-warning { background: rgba(245,158,11,.18); border: 1px solid rgba(245,158,11,.35); }
.toast-error   { background: rgba(239,68,68,.18); border: 1px solid rgba(239,68,68,.35); }
.toast-info    { background: rgba(107,159,212,.18); border: 1px solid rgba(107,159,212,.35); }
.toast-icon { font-size: 14px; flex-shrink: 0; }
.toast-msg  { line-height: 1.4; }

.toast-enter-active { transition: all .25s ease; }
.toast-leave-active { transition: all .2s ease; }
.toast-enter-from { opacity: 0; transform: translateX(40px); }
.toast-leave-to   { opacity: 0; transform: translateX(40px); }
</style>
