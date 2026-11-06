<template>
  <div class="sv-root">
    <div class="page-header">
      <h1 class="page-title">系统设置</h1>
      <div class="flex gap-2">
        <button class="btn btn-outline" @click="doReload" :disabled="loading">重载配置</button>
        <button class="btn" @click="saveAll" :disabled="!hasChanges || saving">
          {{ saving ? '保存中...' : '全部保存' }}
          <span v-if="changedCount" class="sv-badge">{{ changedCount }}</span>
        </button>
      </div>
    </div>

    <div v-if="error" class="error">{{ error }}</div>
    <div v-if="msg" class="msg msg--success">{{ msg }}</div>

    <div class="sv-body" v-if="!loading">
      <nav class="sv-nav">
        <a v-for="(g, idx) in groups" :key="g.name"
           :class="{ active: activeGroup === g.name }"
           @click.prevent="scrollToGroup(g.name, idx)"
           href="javascript:void(0)">
          {{ g.name }}
          <span v-if="groupChangeCount(g)" class="sv-nav-dot"></span>
        </a>
      </nav>

      <div class="sv-content">
        <div v-for="(g, idx) in groups" :key="g.name"
             :ref="el => groupRefs[idx] = el"
             class="card sv-group">
          <div class="sv-group-head">
            <h3 class="sv-group-title">{{ g.name }}</h3>
            <button class="btn-sm btn-outline"
                    @click="saveGroup(g.name)"
                    :disabled="!groupChangeCount(g) || saving">
              保存本组
            </button>
          </div>

          <div class="sv-fields">
            <div v-for="f in g.fields" :key="f.key"
                 class="sv-field"
                 :class="{ 'sv-field--changed': isChanged(f.key) }"
                 :title="f.description">
              <label class="sv-field-label">{{ f.label || f.key }}</label>

              <!-- bool -->
              <div v-if="f.type === 'bool'" class="sv-field-ctrl">
                <ToggleSwitch :modelValue="editValues[f.key] === true"
                              @update:modelValue="setField(f.key, $event)"
                              :label="editValues[f.key] === true ? '开启' : '关闭'" />
                <span class="sv-field-hint">默认: {{ f.default === 'true' ? '开启' : '关闭' }}</span>
              </div>

              <!-- select -->
              <div v-else-if="f.type === 'select'" class="sv-field-ctrl">
                <select :value="editValues[f.key]"
                        @change="setField(f.key, $event.target.value)">
                  <option v-for="opt in f.options" :key="opt" :value="opt">{{ opt }}</option>
                </select>
                <span class="sv-field-hint">默认: {{ f.default }}</span>
              </div>

              <!-- password -->
              <div v-else-if="f.type === 'password'" class="sv-field-ctrl">
                <div class="sv-pass-wrap">
                  <input :type="showPass[f.key] ? 'text' : 'password'"
                         :value="editValues[f.key]"
                         @input="setField(f.key, $event.target.value)"
                         :placeholder="f.default || '(空)'" />
                  <button class="sv-pass-toggle" :aria-label="showPass[f.key] ? '隐藏密码' : '显示密码'" @click="showPass[f.key] = !showPass[f.key]" type="button">
                    {{ showPass[f.key] ? '隐藏' : '显示' }}
                  </button>
                </div>
                <span class="sv-field-hint">默认: {{ f.sensitive && f.default ? '***' : (f.default || '(空)') }}</span>
              </div>

              <!-- int -->
              <div v-else-if="f.type === 'int'" class="sv-field-ctrl">
                <input type="number" step="1"
                       :value="editValues[f.key]"
                       @input="setField(f.key, $event.target.value)"
                       :placeholder="f.default" />
                <span class="sv-field-hint">默认: {{ f.default }}</span>
              </div>

              <!-- float -->
              <div v-else-if="f.type === 'float'" class="sv-field-ctrl">
                <input type="number" step="0.01"
                       :value="editValues[f.key]"
                       @input="setField(f.key, $event.target.value)"
                       :placeholder="f.default" />
                <span class="sv-field-hint">默认: {{ f.default }}</span>
              </div>

              <!-- string -->
              <div v-else class="sv-field-ctrl">
                <input type="text"
                       :value="editValues[f.key]"
                       @input="setField(f.key, $event.target.value)"
                       :placeholder="f.default || '(空)'" />
                <span class="sv-field-hint">默认: {{ f.default || '(空)' }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <LoadingSpinner v-else text="加载配置中..." />
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, nextTick } from 'vue'
import { fetchSettings, updateSettings, reloadSettings } from '../api/settings.js'
import ToggleSwitch from '../components/ToggleSwitch.vue'
import LoadingSpinner from '../components/LoadingSpinner.vue'

const groups = ref([])
const groupRefs = ref({})
const editValues = reactive({})
const originalValues = reactive({})
const showPass = reactive({})
const activeGroup = ref('')
const saving = ref(false)
const loading = ref(true)
const error = ref('')
const msg = ref('')

let origMap = {}

const changedKeys = computed(() => {
  const keys = []
  for (const k of Object.keys(editValues)) {
    if (String(editValues[k]) !== String(origMap[k])) keys.push(k)
  }
  return keys
})
const hasChanges = computed(() => changedKeys.value.length > 0)
const changedCount = computed(() => changedKeys.value.length)

function isChanged(key) {
  return changedKeys.value.includes(key)
}

function groupChangeCount(g) {
  return g.fields.filter(f => isChanged(f.key)).length
}

function setField(key, val) {
  editValues[key] = val
}

function scrollToGroup(name, idx) {
  activeGroup.value = name
  const el = groupRefs.value[idx]
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
}

async function loadSettings() {
  loading.value = true
  error.value = ''
  try {
    const res = await fetchSettings()
    const payload = res?.data ?? res
    groups.value = payload.groups || []

    for (const g of groups.value) {
      for (const f of g.fields) {
        let v = f.value
        if (f.type === 'bool') {
          v = String(v).toLowerCase() === 'true'
        }
        editValues[f.key] = v
        originalValues[f.key] = v
        if (f.type === 'password') {
          showPass[f.key] = false
        }
      }
    }
    origMap = { ...originalValues }

    if (groups.value.length > 0) {
      activeGroup.value = groups.value[0].name
    }
  } catch (e) {
    error.value = '加载配置失败: ' + e.message
  } finally {
    loading.value = false
  }
}

async function saveGroup(groupName) {
  const updates = {}
  const g = groups.value.find(x => x.name === groupName)
  if (!g) return

  for (const f of g.fields) {
    const cur = editValues[f.key]
    const orig = origMap[f.key]
    let curStr = f.type === 'bool' ? String(cur) : String(cur ?? '')
    let origStr = f.type === 'bool' ? String(orig) : String(orig ?? '')
    if (curStr !== origStr) {
      updates[f.key] = f.type === 'bool' ? cur : curStr
    }
  }

  if (!Object.keys(updates).length) return
  await doSave(updates)
}

async function saveAll() {
  const updates = {}
  for (const key of changedKeys.value) {
    updates[key] = editValues[key]
  }
  if (!Object.keys(updates).length) return
  await doSave(updates)
}

async function doSave(updates) {
  saving.value = true
  error.value = ''
  msg.value = ''
  try {
    const res = await updateSettings(updates)
    const data = res?.data ?? res
    if (data.errors && data.errors.length) {
      error.value = data.errors.join('; ')
    }
    if (data.changed && data.changed.length) {
      for (const k of data.changed) {
        origMap[k] = String(editValues[k] ?? '')
      }
      msg.value = `已保存 ${data.changed.length} 项配置`
      setTimeout(() => { msg.value = '' }, 3000)
    }
  } catch (e) {
    error.value = '保存失败: ' + e.message
  } finally {
    saving.value = false
  }
}

async function doReload() {
  error.value = ''
  msg.value = ''
  try {
    await reloadSettings()
    msg.value = '配置已从 .env 重新加载'
    setTimeout(() => { msg.value = '' }, 3000)
    await loadSettings()
  } catch (e) {
    error.value = '重载失败: ' + e.message
  }
}

onMounted(loadSettings)
</script>

<style scoped>
.sv-root {
  max-width: var(--content-max-width);
  margin: 0 auto;
  padding: 0 var(--content-padding) var(--space-8);
}

.sv-badge {
  background: var(--warning);
  color: #1a1a2e;
  font-size: var(--fs-caption);
  padding: 1px 6px;
  border-radius: var(--radius-full);
  font-weight: 700;
  margin-left: 4px;
}

/* ── Body: nav + content ── */
.sv-body {
  display: flex;
  gap: var(--space-6);
}

/* ── Nav ── */
.sv-nav {
  width: 150px;
  flex-shrink: 0;
  position: sticky;
  top: var(--space-5);
  align-self: flex-start;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.sv-nav a {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 7px 10px;
  color: var(--text-secondary);
  text-decoration: none;
  font-size: var(--fs-sm);
  border-radius: var(--radius-sm);
  transition: all var(--transition-fast);
}

.sv-nav a:hover {
  background: var(--primary-muted);
  color: var(--text);
}

.sv-nav a.active {
  background: rgba(107, 159, 212, .12);
  color: var(--primary);
  font-weight: 600;
}

.sv-nav-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--warning);
  flex-shrink: 0;
}

/* ── Content ── */
.sv-content {
  flex: 1;
  min-width: 0;
}

.sv-group {
  margin-bottom: 14px;
}

.sv-group-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}

.sv-group-title {
  font-size: var(--fs-lg);
  font-weight: 600;
  color: var(--text);
  margin: 0;
}

/* ── Fields ── */
.sv-fields {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.sv-field {
  display: flex;
  align-items: center;
  gap: var(--space-4);
  padding: 8px 10px;
  border-radius: var(--radius-sm);
  border-left: 3px solid transparent;
  transition: all var(--transition-fast);
}

.sv-field--changed {
  border-left-color: var(--warning);
  background: rgba(244, 167, 66, .04);
}

.sv-field-label {
  width: 170px;
  flex-shrink: 0;
  font-size: var(--fs-body);
  color: var(--text-secondary);
  word-break: break-all;
}

.sv-field-ctrl {
  flex: 1;
  display: flex;
  align-items: center;
  gap: var(--space-3);
}

.sv-field-ctrl input[type="text"],
.sv-field-ctrl input[type="number"],
.sv-field-ctrl select {
  flex: 1;
  max-width: 420px;
  padding: 6px 10px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text);
  font-size: var(--fs-body);
  font-family: var(--font-mono);
  outline: none;
  transition: border-color var(--transition-fast);
}

.sv-field-ctrl input:focus,
.sv-field-ctrl select:focus {
  border-color: var(--primary);
}

.sv-field--changed .sv-field-ctrl input,
.sv-field--changed .sv-field-ctrl select {
  border-color: rgba(244, 167, 66, .4);
}

.sv-field-hint {
  font-size: var(--fs-caption);
  color: var(--text-muted);
  white-space: nowrap;
}

/* ── Password ── */
.sv-pass-wrap {
  display: flex;
  flex: 1;
  max-width: 420px;
}

.sv-pass-wrap input {
  flex: 1;
  padding: 6px 10px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm) 0 0 var(--radius-sm);
  color: var(--text);
  font-size: var(--fs-body);
  font-family: var(--font-mono);
  outline: none;
}

.sv-pass-wrap input:focus {
  border-color: var(--primary);
}

.sv-pass-toggle {
  padding: 6px 8px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-left: none;
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  cursor: pointer;
  font-size: 14px;
}

/* ── Responsive ── */
@media (max-width: 860px) {
  .sv-body { flex-direction: column; }
  .sv-nav {
    width: 100%;
    position: static;
    flex-direction: row;
    flex-wrap: wrap;
    gap: 4px;
  }
  .sv-field {
    flex-direction: column;
    align-items: flex-start;
  }
  .sv-field-label { width: 100%; }
  .sv-field-ctrl { width: 100%; min-width: 0; flex-wrap: wrap; }
  .sv-field-ctrl input[type="text"],
  .sv-field-ctrl input[type="number"],
  .sv-field-ctrl select,
  .sv-pass-wrap { width: 100%; min-width: 0; max-width: none; }
  .sv-pass-wrap input { min-width: 0; }
  .sv-field-hint { flex-basis: 100%; white-space: normal; overflow-wrap: anywhere; }
}
</style>
