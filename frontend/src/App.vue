<template>
  <div v-if="isLoginPage">
    <router-view />
  </div>
  <div v-else class="layout">
    <aside :class="['sidebar', { 'sidebar--collapsed': collapsed }]">
      <button type="button" class="logo" :aria-label="collapsed ? '展开侧栏' : '收起侧栏'" @click="collapsed = !collapsed">
        <span class="logo-mark" aria-hidden="true">A</span>
        <span v-if="!collapsed" class="logo-copy">Apallg投研<small>A股量化平台</small></span>
        <span v-else class="logo-short">Ap</span>
      </button>
      <nav class="nav" aria-label="主导航">
        <router-link
          v-for="item in navItems" :key="item.path"
          :to="item.path"
          :class="{ active: isActive(item) }"
          :title="collapsed ? item.title : ''"
          @click="item.action ? (item.action(), $event.preventDefault()) : null"
        >
          <span class="nav-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" focusable="false">
              <path :d="iconPaths[item.icon]" />
            </svg>
          </span>
          <span v-if="!collapsed" class="nav-label">{{ item.title }}</span>
          <span v-if="!collapsed && item.badge" class="nav-badge">{{ item.badge }}</span>
        </router-link>
      </nav>
      <div class="sidebar-footer">
        <button class="lock-btn" type="button" aria-label="锁定屏幕" @click="doLock" :title="collapsed ? '锁定屏幕' : ''">
          <span class="control-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" focusable="false"><path d="M7 11V8a5 5 0 0 1 10 0v3M6 11h12v9H6z" /></svg>
          </span>
          <span v-if="!collapsed" class="lock-label">锁定屏幕</span>
        </button>
        <button class="toggle-btn" type="button" :aria-label="collapsed ? '展开侧栏' : '折叠侧栏'" @click="collapsed = !collapsed" :title="collapsed ? '展开' : '折叠'">
          <span class="control-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" focusable="false"><path :d="collapsed ? 'm9 6 6 6-6 6' : 'm15 6-6 6 6 6'" /></svg>
          </span>
        </button>
      </div>
    </aside>
    <main class="main">
      <MarketBar />
      <router-view v-slot="{ Component }">
        <keep-alive>
          <component :is="Component" />
        </keep-alive>
      </router-view>
    </main>
    <AppToast ref="toastRef" />
    <SearchModal ref="searchRef" />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import MarketBar from './components/MarketBar.vue'
import AppToast from './components/AppToast.vue'
import SearchModal from './components/SearchModal.vue'
import { setToastRef } from './composables/useToast.js'

const narrowViewport = typeof window === 'undefined' ? null : window.matchMedia('(max-width: 640px)')
const collapsed = ref(narrowViewport?.matches ?? false)
const route = useRoute()
const router = useRouter()
const toastRef = ref(null)
setTimeout(() => setToastRef(toastRef.value), 0)

const navItems = [
  { path: '/market',   title: '行情监控', icon: 'market' },
  { path: '/screening', title: '选股',      icon: 'screening' },
  { path: '/factors',   title: '因子管理',  icon: 'factors' },
  { path: '/qlib',      title: '量化实验室', icon: 'lab' },
  { path: '/data',      title: '数据中心',   icon: 'data' },
  { path: '/system',    title: '系统运维',  icon: 'system' },
  { path: '/settings',  title: '系统设置',  icon: 'settings' },
]

const iconPaths = {
  market: 'M4 19V5m0 14h16M8 16v-4m4 4V8m4 8V5m4 11v-7',
  screening: 'M12 3l2.6 5.3 5.9.9-4.3 4.2 1 5.9-5.2-2.8-5.2 2.8 1-5.9-4.3-4.2 5.9-.9L12 3z',
  factors: 'M7 7a3 3 0 1 0 0 6 3 3 0 0 0 0-6zm10 4a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM9.5 9.5l5 3M9.5 14.5l5-3',
  lab: 'M9 3v5.2L4.5 16a2.5 2.5 0 0 0 2.2 3.7h10.6a2.5 2.5 0 0 0 2.2-3.7L15 8.2V3M8 3h8M7 15h10',
  data: 'M4 5h16v14H4zM8 9h8M8 13h8M8 17h5',
  system: 'M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7zm0-5v2m0 13v2m9-8h-2m-14 0H3m15.4-6.4-1.4 1.4M7 16l-1.4 1.4m12.8 0L17 16M7 8 5.6 6.6',
  settings: 'M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zm0-5v2m0 14v2m9-9h-2M5 12H3m15.4-6.4-1.4 1.4M7 17l-1.4 1.4m12.8 0L17 17M7 7 5.6 5.6',
}

const isLoginPage = computed(() => route.path === '/login')
const isStockPage = computed(() => route.path.startsWith('/stock/'))

function handleViewportChange(event) {
  if (event.matches) collapsed.value = true
}

onMounted(() => narrowViewport?.addEventListener('change', handleViewportChange))
onUnmounted(() => narrowViewport?.removeEventListener('change', handleViewportChange))

function isActive(item) {
  if (item.path === '/market') return route.path === '/market' || route.path === '/sentiment' || route.path === '/sectors' || route.path.startsWith('/stock')
  return route.path.startsWith(item.path)
}

function doLock() {
  localStorage.removeItem('token')
  router.push('/login?locked=1')
}
</script>

<style scoped>
.layout { display:flex; min-height:100vh; max-width:100vw; overflow:hidden }
.sidebar {
  width:200px;
  background:linear-gradient(180deg, rgba(11,18,32,.98), rgba(8,14,25,.98)); border-right:1px solid var(--border);
  display:flex; flex-direction:column; transition:width .2s; position:sticky; top:0; height:100vh;
  overflow:hidden; flex-shrink:0; font-family:var(--font); box-shadow: 12px 0 32px rgba(0,0,0,.12);
}
.sidebar--collapsed { width:56px; }
.logo {
  width:100%; border:0; background:transparent; cursor:pointer; font-family:var(--font);
  min-height:74px; padding:16px 12px; border-bottom:1px solid var(--border);
  font-size:16px; font-weight:700; color:var(--primary); white-space:nowrap; text-align:center;
  display:flex; align-items:center; justify-content:center; gap:9px;
}
.logo-mark { width:27px; height:27px; display:grid; place-items:center; border:1px solid rgba(121,174,224,.6); border-radius:8px; color:var(--text); background:linear-gradient(145deg, rgba(121,174,224,.28), rgba(121,174,224,.06)); box-shadow:0 0 18px rgba(121,174,224,.12); font-size:14px; }
.logo-copy { display:block; text-align:left; line-height:1.15; }
.logo-copy small { display:block; font-size:var(--fs-xs); color:var(--text-muted); font-weight:400; margin-top:4px; letter-spacing:.3px; }
.logo-short { color:var(--text); font-size:14px; letter-spacing:0; }
.nav { flex:1; padding:8px 0 }
.nav a {
  display:flex; align-items:center; gap:10px; padding:11px 14px; margin:3px 8px; border-radius:7px; text-decoration:none;
  color:var(--text-secondary); font-size:var(--fs-md); transition:all .15s;
  border-left:2px solid transparent; white-space:nowrap;
}
.nav a:hover { background:rgba(121,174,224,.09); color:var(--text); transform:translateX(1px); }
.nav a.active { background:linear-gradient(90deg, rgba(121,174,224,.19), rgba(121,174,224,.07)); color:var(--text); border-left-color:var(--primary); box-shadow:inset 0 0 20px rgba(121,174,224,.04); }
.nav-icon { display:grid; place-items:center; width:22px; height:22px; flex-shrink:0; color:currentColor; }
.nav-icon svg { width:17px; height:17px; fill:none; stroke:currentColor; stroke-width:1.7; stroke-linecap:round; stroke-linejoin:round; }
.control-icon { display:grid; place-items:center; width:18px; height:18px; flex:none; }
.control-icon svg { width:16px; height:16px; fill:none; stroke:currentColor; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }
.nav-label { flex:1 }
.nav-badge { font-size:var(--fs-xs); background:var(--danger); color:#fff; padding:1px 6px; border-radius:10px }
.sidebar-footer { padding:10px 8px; border-top:1px solid var(--border) }
.lock-btn { width:100%; padding:6px 8px; border:none; background:transparent; color:var(--text-muted); font-size:13px; cursor:pointer; font-family:var(--font); display:flex; align-items:center; gap:6px; justify-content:center; margin-bottom:2px }
.lock-btn:hover { color:var(--warning); background:rgba(245,158,11,.08); border-radius:6px; }
.lock-label { font-size:12px }
.toggle-btn { width:100%; padding:6px; border:none; background:transparent; color:var(--text-muted); font-size:14px; cursor:pointer; font-family:var(--font) }
.toggle-btn:hover { color:var(--text); background:rgba(255,255,255,.04); border-radius:6px; }
.main { flex:1; overflow-y:auto; overflow-x:hidden; min-width:0 }
</style>
