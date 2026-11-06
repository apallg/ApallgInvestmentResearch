<template>
  <div class="login-page">
    <div class="login-card">
      <div class="login-header">
        <h1>Apallg投研</h1>
        <small>A股量化平台</small>
      </div>
      <div v-if="isLocked" class="locked-banner">
        屏幕已锁定，请重新登录
      </div>
      <form @submit.prevent="handleLogin" class="login-form">
        <div class="field">
          <label>账号</label>
          <input
            v-model="username"
            type="text"
            placeholder="请输入账号"
            autocomplete="username"
            :disabled="loading"
          />
        </div>
        <div class="field">
          <label>密码</label>
          <input
            v-model="password"
            type="password"
            placeholder="请输入密码"
            autocomplete="current-password"
            :disabled="loading"
          />
        </div>
        <p v-if="error" class="error-msg">{{ error }}</p>
        <button type="submit" class="login-btn" :disabled="loading">
          {{ loading ? '登录中...' : '登 录' }}
        </button>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRoute } from 'vue-router'
import api from '../api/index.js'

const route = useRoute()
const isLocked = computed(() => route.query.locked === '1')

const username = ref(localStorage.getItem('username') || '')
const password = ref('')
const loading = ref(false)
const error = ref('')

async function handleLogin() {
  error.value = ''
  if (!username.value || !password.value) {
    error.value = '请输入账号和密码'
    return
  }
  loading.value = true
  try {
    const res = await api.post('/auth/login', {
      username: username.value,
      password: password.value,
    })
    localStorage.setItem('token', res.token)
    localStorage.setItem('username', res.username)
    window.location.href = '/'
  } catch (e) {
    error.value = e.message || '登录失败'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
  background:
    radial-gradient(circle at 50% 0%, rgba(121,174,224,.18), transparent 32rem),
    linear-gradient(135deg, #070b14 0%, #0b1220 48%, #080d17 100%);
  font-family: var(--font, system-ui, sans-serif);
}
.login-page::before {
  content: '';
  position: absolute;
  inset: 0;
  opacity: .45;
  background-image: linear-gradient(rgba(121,174,224,.025) 1px, transparent 1px), linear-gradient(90deg, rgba(121,174,224,.025) 1px, transparent 1px);
  background-size: 32px 32px;
  mask-image: linear-gradient(to bottom, black, transparent 80%);
}
.login-card {
  position: relative;
  z-index: 1;
  background: linear-gradient(145deg, rgba(16,26,43,.96), rgba(9,16,29,.97));
  border: 1px solid var(--border, #2a2d3a);
  border-radius: 14px;
  padding: 42px 38px 38px;
  width: 380px;
  max-width: 90vw;
  box-shadow: 0 24px 70px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.04);
}
.login-header {
  text-align: center;
  margin-bottom: 32px;
}
.login-header h1 {
  margin: 0;
  font-size: 24px;
  color: var(--text, #e0e0e0);
  letter-spacing: 0;
}
.login-header small {
  color: var(--text-muted, #8b8fa3);
  font-size: 13px;
  letter-spacing: .5px;
}
.locked-banner {
  background: rgba(245,158,11,.12);
  border: 1px solid rgba(245,158,11,.3);
  color: var(--warning, #f59e0b);
  padding: 10px 14px;
  border-radius: 6px;
  font-size: 14px;
  text-align: center;
  margin-bottom: 20px;
}
.login-form .field {
  margin-bottom: 18px;
}
.login-form label {
  display: block;
  margin-bottom: 6px;
  color: var(--text-secondary, #a0a4b8);
  font-size: 14px;
}
.login-form input {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border, #2a2d3a);
  border-radius: 6px;
  background: var(--bg-primary, #0f1117);
  color: var(--text, #e0e0e0);
  font-size: 14px;
  outline: none;
  box-sizing: border-box;
  min-height: 42px;
  transition: border-color .15s, box-shadow .15s, background .15s;
}
.login-form input:focus {
  border-color: var(--primary, #6b9fd4);
  box-shadow: var(--focus-ring);
  background: rgba(7,11,20,.88);
}
.error-msg {
  color: var(--danger, #ef4444);
  font-size: 13px;
  margin: 0 0 12px;
}
.login-btn {
  width: 100%;
  min-height: 44px;
  padding: 11px;
  border: none;
  border-radius: 6px;
  background: linear-gradient(135deg, var(--primary, #6b9fd4), #5c8fc2);
  color: #fff;
  font-size: 15px;
  cursor: pointer;
  box-shadow: 0 8px 18px rgba(121,174,224,.16);
  transition: transform .15s, box-shadow .15s, filter .15s;
}
.login-btn:hover { filter: brightness(1.08); box-shadow: 0 10px 24px rgba(121,174,224,.24); transform: translateY(-1px); }
.login-btn:active { transform: translateY(0); }
.login-btn:disabled { opacity: 0.6; cursor: not-allowed }
</style>
