import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  // ═══ 登录 ═══
  { path: '/login', name: 'login', component: () => import('../views/LoginView.vue'),
    meta: { title: '登录', noAuth: true } },

  // ═══ 6 个核心工作流 ═══
  { path: '/', redirect: '/market' },

  // 1. 行情监控 — 大盘+板块+个股搜索+情绪
  { path: '/market', name: 'market', component: () => import('../views/MarketView.vue'),
    meta: { title: '行情监控', icon: '📊' } },

  // 2. 选股 — 多因子筛选，独立于行情
  { path: '/screening', name: 'screening', component: () => import('../views/StockScreening.vue'),
    meta: { title: '选股', icon: '🎯' } },

  // 3. 因子管理 — 因子研发 + 因子池管理
  { path: '/factors', name: 'factors', component: () => import('../views/FactorManager.vue'),
    meta: { title: '因子管理', icon: '🧬' } },

  // 4. 交易风控 — 已合并到量化调仓 (持仓管理) + 风险告警 tab
  { path: '/trading', redirect: '/screening' },

  // 5. 量化实验室 — Qlib模型训练/预测/回测
  { path: '/qlib', name: 'qlib', component: () => import('../views/QlibLab.vue'),
    meta: { title: '量化实验室', icon: '🧠' } },

  // 5.5 数据中心 — 全量数据完整性监控与修复
  { path: '/data', name: 'data', component: () => import('../views/DataManager.vue'),
    meta: { title: '数据中心', icon: '📡' } },

  // 6. 系统运维 — 调度器+数据状态+异步任务+日志
  { path: '/system', name: 'system', component: () => import('../views/SystemMonitor.vue'),
    meta: { title: '系统运维', icon: '🖥' } },

  // 7. 系统设置 — .env 可视化管理
  { path: '/settings', name: 'settings', component: () => import('../views/SettingsView.vue'),
    meta: { title: '系统设置', icon: '⚙️' } },

  // ═══ 子路由 (交易风控内部 Tab) ═══
  { path: '/portfolio', redirect: '/trading' },
  { path: '/alerts', redirect: '/trading' },

  // ═══ 旧路由兼容 (重定向到新路径) ═══
  { path: '/strategy', redirect: '/qlib' },
  { path: '/backtest', redirect: '/qlib' },
  { path: '/lab', redirect: '/qlib' },
  { path: '/live', redirect: '/qlib' },
  { path: '/editor', redirect: '/qlib' },
  { path: '/whitebox', redirect: '/factors' },
  { path: '/scheduler', redirect: '/system' },
  { path: '/sentiment', redirect: '/market' },
  { path: '/sectors', redirect: '/market' },
  { path: '/stock', redirect: '/market' },
  { path: '/stock/:code', name: 'stock-detail', component: () => import('../views/StockDetail.vue'),
    meta: { title: '个股详情', icon: '📈' } },
]

function getToken() {
  return localStorage.getItem('token')
}

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.beforeEach((to, from, next) => {
  if (to.meta.noAuth) {
    return next()
  }
  if (!getToken()) {
    return next('/login')
  }
  next()
})

export default router
