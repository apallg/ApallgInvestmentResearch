<template>
  <div class="app-empty">
    <div class="app-empty-icon">{{ icon }}</div>
    <p class="app-empty-text">{{ text }}</p>
    <button v-if="actionLabel" class="app-empty-action" @click="$emit('action')">
      {{ actionLabel }}
    </button>
  </div>
</template>

<script setup>
defineProps({
  text: { type: String, default: '暂无数据' },
  icon: { type: String, default: '📭' },
  actionLabel: { type: String, default: '' },
})
defineEmits(['action'])
</script>

<style scoped>
.app-empty { display:flex; flex-direction:column; align-items:center; justify-content:center; padding:48px 16px; color:var(--text-muted) }
.app-empty-icon { font-size:32px; margin-bottom:12px }
.app-empty-text { font-size:var(--fs-base); margin:0 }
.app-empty-action { margin-top:16px; padding:8px 20px; border:1px solid var(--border); border-radius:var(--radius-sm); background:var(--card-bg); color:var(--text); cursor:pointer; font-size:var(--fs-md); font-family:var(--font) }
.app-empty-action:hover { border-color:var(--primary); color:var(--primary) }
</style>
