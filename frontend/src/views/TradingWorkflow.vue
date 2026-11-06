<template>
  <div class="trading-workflow">
    <h2>交易风控</h2>
    <div class="tabs">
      <button v-for="tab in tabs" :key="tab.key"
        :class="['tab', { active: activeTab === tab.key }]"
        @click="activeTab = tab.key">
        {{ tab.label }}
      </button>
    </div>
    <div class="tab-content">
      <PortfolioView v-if="activeTab === 'portfolio'" />
      <AlertView v-else-if="activeTab === 'alerts'" />
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import PortfolioView from './PortfolioView.vue'
import AlertView from './AlertView.vue'

const activeTab = ref('portfolio')
const tabs = [
  { key: 'portfolio', label: '持仓管理' },
  { key: 'alerts', label: '风险告警' },
]
</script>

<style scoped>
.trading-workflow { height:100%; display:flex; flex-direction:column }
.trading-workflow h2 { color:var(--text); font-size:var(--fs-2xl); font-weight:700; margin:0 0 12px 0; padding:28px 32px 0; letter-spacing:0 }
.tabs { display:flex; gap:0; border-bottom:1px solid var(--border); padding:0 32px }
.tab { padding:10px 20px; border:none; background:transparent; color:var(--text-muted); font-size:var(--fs-md); cursor:pointer; border-bottom:2px solid transparent; margin-bottom:-1px; transition:all var(--transition); font-family:var(--font) }
.tab:hover { color:var(--text) }
.tab.active { color:var(--primary); border-bottom-color:var(--primary) }
.tab-content { flex:1 }
</style>
