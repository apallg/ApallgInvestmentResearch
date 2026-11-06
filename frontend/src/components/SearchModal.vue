<template>
  <teleport to="body">
    <div v-if="open" class="search-overlay" @click.self="close">
      <div class="search-modal">
        <input
          ref="inputRef"
          v-model="query"
          class="search-input"
          placeholder="输入股票代码或名称..."
          @keydown.enter="pick(activeIdx)"
          @keydown.arrow-down.prevent="activeIdx = Math.min(activeIdx + 1, results.length - 1)"
          @keydown.arrow-up.prevent="activeIdx = Math.max(activeIdx - 1, 0)"
          @keydown.escape="close"
        />
        <div class="search-hint" v-if="!query">输入 代码 或 名称 搜索， 键导航</div>

        <div class="search-results" v-if="results.length">
          <div
            v-for="(r, i) in results" :key="r.code"
            class="search-row" :class="{ active: i === activeIdx }"
            @click="pick(i)" @mouseenter="activeIdx = i"
          >
            <span class="sr-code">{{ r.code }}</span>
            <span class="sr-name">{{ r.name }}</span>
            <span class="sr-type">{{ r.type || '' }}</span>
          </div>
        </div>

        <div class="search-empty" v-else-if="query && !loading">
          未找到 "{{ query }}" 相关股票
        </div>
      </div>
    </div>
  </teleport>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import api from '../api/index.js'

const open = ref(false)
const query = ref('')
const results = ref([])
const activeIdx = ref(0)
const loading = ref(false)
const inputRef = ref(null)
const router = useRouter()

let _debounce = null

watch(query, () => {
  activeIdx.value = 0
  if (!query.value.trim()) { results.value = []; return }
  clearTimeout(_debounce)
  _debounce = setTimeout(search, 200)
})

async function search() {
  const q = query.value.trim()
  if (!q) return
  loading.value = true
  try {
    const res = await api.get('/stock/search', { params: { q, limit: 10 } })
    results.value = Array.isArray(res?.data) ? res.data : (Array.isArray(res) ? res : [])
  } catch (e) {
    results.value = []
  }
  loading.value = false
}

function pick(idx) {
  const r = results.value[idx]
  if (!r) return
  close()
  router.push('/stock/' + r.code)
}

function show() {
  open.value = true
  query.value = ''
  results.value = []
  nextTick(() => inputRef.value?.focus())
}
function close() { open.value = false }

function onKeydown(e) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault()
    show()
  }
}

document.addEventListener('keydown', onKeydown)

defineExpose({ show, close })
</script>

<style scoped>
.search-overlay {
  position: fixed; inset: 0; z-index: 10000;
  background: rgba(0,0,0,.5);
  display: flex; justify-content: center; padding-top: 15vh;
}
.search-modal {
  width: 520px; max-height: 420px;
  background: var(--card-bg); border: 1px solid var(--border-light);
  border-radius: var(--radius-lg); box-shadow: 0 8px 40px rgba(0,0,0,.5);
  overflow: hidden; display: flex; flex-direction: column;
}
.search-input {
  width: 100%; padding: 14px 18px; border: none; border-bottom: 1px solid var(--border);
  font-size: 16px; color: var(--text); background: transparent; outline: none;
  font-family: var(--font);
}
.search-input::placeholder { color: var(--text-muted); }
.search-hint {
  padding: 8px 18px; font-size: var(--fs-xs); color: var(--text-muted);
}
.search-results { overflow-y: auto; }
.search-row {
  display: flex; align-items: center; gap: 14px;
  padding: 10px 18px; cursor: pointer; transition: background .1s;
}
.search-row.active, .search-row:hover { background: rgba(107,159,212,.1); }
.sr-code { font-family: var(--font-mono); font-size: var(--fs-md); color: var(--primary); min-width: 72px; }
.sr-name { font-size: var(--fs-md); color: var(--text); flex: 1; }
.sr-type { font-size: var(--fs-xs); color: var(--text-muted); }
.search-empty { padding: 24px 18px; color: var(--text-muted); text-align: center; font-size: var(--fs-md); }
</style>
