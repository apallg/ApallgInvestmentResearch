<template>
  <div class="loading-spinner">
    <span class="spinner"></span>
    <span class="loading-text">{{ text }}</span>
  </div>
</template>
<script setup>
defineProps({
  text: { type: String, default: '加载中...' },
})
</script>
