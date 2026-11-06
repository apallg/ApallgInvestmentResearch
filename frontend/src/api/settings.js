import api from './index'

export function fetchSettings() {
  return api.get('/settings/settings')
}

export function updateSettings(updates) {
  return api.put('/settings/settings', updates)
}

export function reloadSettings() {
  return api.post('/settings/settings/reload')
}
