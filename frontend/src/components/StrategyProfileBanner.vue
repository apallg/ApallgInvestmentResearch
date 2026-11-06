<template>
  <div class="profile-banner" :class="{ blocked: profile?.status === 'blocked' || profile?.runtime_health?.drawdown_breached }">
    <div>
      <span class="profile-label">{{ title }}</span>
      <strong v-if="profile?.profile_id">{{ profile.name }} · 第{{ profile.version }}版</strong>
      <strong v-else>尚无正式策略</strong>
      <span v-if="profile?.profile_id" class="profile-status">{{ strategyStatusText(profile.status) }}</span>
    </div>
    <button v-if="profile?.fingerprint" class="fingerprint" :title="profile.fingerprint" @click="copy">
      {{ profile.fingerprint.slice(0, 12) }}…
    </button>
    <span v-if="profile?.blocked_reasons?.length" class="profile-reason">{{ localizeStrategyText(profile.blocked_reasons.join('；')) }}</span>
    <span v-if="profile?.runtime_health?.drawdown_breached" class="profile-reason">
      实盘最大回撤 {{ percent(profile.runtime_health.max_drawdown) }}，已禁止新增买入
    </span>
  </div>
</template>

<script setup>
import { localizeStrategyText, strategyStatusText } from '../utils/strategyTerms.js'

const props = defineProps({ profile: { type: Object, default: null }, title: { type: String, default: '当前策略版本' } })
async function copy() {
  if (props.profile?.fingerprint) await navigator.clipboard.writeText(props.profile.fingerprint)
}
function percent(value) { return `${(Number(value || 0) * 100).toFixed(2)}%` }
</script>

<style scoped>
.profile-banner { display:flex; align-items:center; gap:12px; flex-wrap:wrap; padding:10px 14px; margin:10px 0 14px; border:1px solid var(--border); border-radius:8px; background:var(--surface); }
.profile-banner.blocked { border-color:var(--up); }
.profile-label { color:var(--text-muted); margin-right:8px; }
.profile-status { margin-left:8px; padding:2px 7px; border-radius:10px; background:var(--surface-2); color:var(--primary); font-size:12px; }
.fingerprint { border:0; background:transparent; color:var(--primary); cursor:pointer; font-family:var(--font-data); }
.profile-reason { width:100%; color:var(--up); font-size:12px; }
</style>
