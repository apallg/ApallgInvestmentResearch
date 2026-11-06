<template>
  <div class="card metric">
    <div class="value" :style="valueStyle">{{ displayValue }}</div>
    <div class="label"><slot name="label">{{ label }}</slot></div>
    <div :class="['delta', delta != null ? (delta > 0 ? 'up' : delta < 0 ? 'down' : '') : '']">
      {{ delta != null ? (delta > 0 ? '+' : '') + delta.toFixed(2) + '%' : ' ' }}
    </div>
  </div>
</template>
<script setup>
import { computed } from 'vue'
const props = defineProps({
  value: { default: '--' },
  label: { type: String, default: '' },
  delta: { default: null },
  color: { type: String, default: '' },
  customColor: { type: String, default: '' },
})
const displayValue = computed(() => {
  if (props.value == null) return '--'
  return props.value
})
const valueStyle = computed(() => {
  const style = {}
  if (props.customColor) style.color = props.customColor
  else if (props.color === 'up') style.color = 'var(--up)'
  else if (props.color === 'down') style.color = 'var(--down)'
  return style
})
</script>
