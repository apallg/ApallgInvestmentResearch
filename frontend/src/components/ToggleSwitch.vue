<template>
  <label class="toggle-switch" :class="{ 'is-on': modelValue }">
    <input
      type="checkbox"
      :checked="modelValue"
      @change="$emit('update:modelValue', ($event.target).checked)"
    />
    <span class="track">
      <span class="thumb"></span>
    </span>
    <span v-if="label" class="toggle-label">{{ label }}</span>
  </label>
</template>
<script setup>
defineProps({
  modelValue: { type: Boolean, default: false },
  label: { type: String, default: '' },
})
defineEmits(['update:modelValue'])
</script>
<style scoped>
.toggle-switch {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  user-select: none;
}
.toggle-switch input { display: none; }
.track {
  width: 38px; height: 20px;
  background: var(--border);
  border-radius: 10px;
  position: relative;
  transition: background 0.2s;
  flex-shrink: 0;
}
.is-on .track { background: var(--primary); }
.thumb {
  width: 16px; height: 16px;
  background: var(--text);
  border-radius: 50%;
  position: absolute;
  top: 2px; left: 2px;
  transition: transform 0.2s;
}
.is-on .thumb { transform: translateX(18px); }
.toggle-label { font-size: 13px; color: var(--text-secondary); }
</style>
