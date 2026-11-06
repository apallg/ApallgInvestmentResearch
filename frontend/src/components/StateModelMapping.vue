<template>
  <section class="state-model-map">
    <div class="state-model-map__head">
      <div>
        <div class="state-model-map__title">状态模型路由</div>
        <div class="state-model-map__desc">
          {{ useVolSwitch ? '按高低波动切换' : '按 6 类市场状态切换' }}；状态映射优先于单模型，全部留空时沿用自动匹配。
        </div>
      </div>
      <div class="state-model-map__actions">
        <button v-if="recommendedMap" type="button" class="state-model-map__clear" @click="$emit('update:modelValue', recommendedMap)">按建议填充</button>
        <button v-if="selectedCount" type="button" class="state-model-map__clear" @click="$emit('update:modelValue', {})">清空</button>
      </div>
    </div>

    <div class="state-model-map__grid">
      <label v-for="state in states" :key="state.key" class="state-model-map__row">
        <span class="state-model-map__label">
          <strong>{{ state.label }}</strong>
          <small>{{ state.hint }}</small>
        </span>
        <select :value="modelValue?.[state.key] || ''" @change="setModel(state.key, $event.target.value)">
          <option value="">自动匹配</option>
          <option v-for="model in availableModels" :key="modelId(model)" :value="modelId(model)">
            {{ modelLabel(model) }}
          </option>
        </select>
      </label>
    </div>

    <div class="state-model-map__status" :class="{ incomplete: selectedCount && !complete }">
      <span v-if="complete">已完成 {{ states.length }}/{{ states.length }} 个状态映射</span>
      <span v-else-if="selectedCount">已选 {{ selectedCount }}/{{ states.length }}，提交前需补全或清空</span>
      <span v-else>当前使用系统自动匹配</span>
    </div>
  </section>
</template>

<script setup>
import { computed, watch } from 'vue'
import { activeModelStates } from '../utils/modelRouting.js'

const props = defineProps({
  useVolSwitch: { type: Boolean, default: false },
  models: { type: Array, default: () => [] },
  modelValue: { type: Object, default: () => ({}) },
})
const emit = defineEmits(['update:modelValue'])

const states = computed(() => activeModelStates(props.useVolSwitch))
const availableModels = computed(() => props.models.filter(model =>
  !model.disabled && model.experiment_id !== 'timing' && model.run_id &&
  model.label_mode === 'tradable_open_to_open'))
const selectedCount = computed(() => states.value.filter(state => props.modelValue?.[state.key]).length)
const complete = computed(() => selectedCount.value === states.value.length)
const recommendedMap = computed(() => {
  const result = {}
  for (const state of states.value) {
    const token = state.direction === 'oversold_rebound' ? 'rebound' : state.direction.split('_')[0]
    const matches = availableModels.value.filter(model =>
      String(model.factor_label || model.factor_set || '').toLowerCase().includes(token))
    if (!matches.length) return null
    const best = matches.reduce((current, model) => {
      const currentScore = (current.is_fusion ? 1 : 0) * 100 + Number(current.ic ?? -99)
      const modelScore = (model.is_fusion ? 1 : 0) * 100 + Number(model.ic ?? -99)
      return modelScore > currentScore ? model : current
    })
    result[state.key] = modelId(best)
  }
  return result
})

watch(recommendedMap, mapping => {
  if (mapping && selectedCount.value === 0) emit('update:modelValue', mapping)
}, { immediate: true })

watch([() => props.models, () => props.modelValue], ([models, mapping]) => {
  if (!models.length || !mapping) return
  const allowed = new Set(availableModels.value.map(modelId))
  const next = { ...mapping }
  let changed = false
  for (const [key, value] of Object.entries(next)) {
    if (!allowed.has(value)) {
      delete next[key]
      changed = true
    }
  }
  if (changed) emit('update:modelValue', next)
}, { immediate: true, deep: true })

function modelId(model) {
  return `${model.experiment_id}/${model.run_id}`
}

function modelLabel(model) {
  const ic = model.ic == null ? '' : ` · IC ${Number(model.ic).toFixed(4)}`
  return `${model.display_name || model.model_class || model.run_id.slice(0, 8)}${ic}`
}

function setModel(key, value) {
  const next = { ...(props.modelValue || {}) }
  if (value) next[key] = value
  else delete next[key]
  emit('update:modelValue', next)
}
</script>

<style scoped>
.state-model-map { margin-top: 12px; padding: 14px; border: 1px solid var(--border); border-radius: 10px; background: var(--bg-secondary); }
.state-model-map__head { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; margin-bottom: 12px; }
.state-model-map__title { color: var(--text-primary); font-weight: 650; }
.state-model-map__desc { margin-top: 3px; color: var(--text-secondary); font-size: 12px; line-height: 1.5; }
.state-model-map__clear { min-height: 36px; padding: 0 12px; border: 1px solid var(--border); border-radius: 7px; background: transparent; color: var(--text-secondary); cursor: pointer; }
.state-model-map__actions { display: flex; gap: 8px; }
.state-model-map__grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.state-model-map__row { display: grid; grid-template-columns: minmax(92px, 0.7fr) minmax(180px, 1.3fr); align-items: center; gap: 10px; }
.state-model-map__label { display: flex; min-width: 0; flex-direction: column; }
.state-model-map__label strong { color: var(--text-primary); font-size: 13px; }
.state-model-map__label small { margin-top: 2px; color: var(--text-muted); font-size: 11px; }
.state-model-map select { width: 100%; min-height: 40px; padding: 0 10px; border: 1px solid var(--border); border-radius: 7px; background: var(--bg-card); color: var(--text-primary); }
.state-model-map__status { margin-top: 10px; color: var(--text-muted); font-size: 12px; }
.state-model-map__status.incomplete { color: var(--warning); }
@media (max-width: 900px) {
  .state-model-map__grid { grid-template-columns: 1fr; }
}
@media (max-width: 560px) {
  .state-model-map__row { grid-template-columns: 1fr; gap: 6px; }
}
</style>
