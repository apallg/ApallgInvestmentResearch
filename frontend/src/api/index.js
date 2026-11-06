import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
})

api.interceptors.request.use(config => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  res => res.data,
  err => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('username')
      if (window.location.pathname !== '/login') {
        window.location.href = '/login'
      }
      throw new Error('登录已过期，请重新登录')
    }
    const detail = err.response?.data?.detail
    const error = err.response?.data?.error
    const msg = detail || error || err.message
    if (Array.isArray(msg)) {
      const parts = msg.map(x => typeof x === 'string' ? x : (x.msg || JSON.stringify(x)))
      const e = new Error(parts.join('; '))
      e.status = err.response?.status
      e.commitOutcome = err.response?.headers?.['x-commit-outcome']
      throw e
    }
    const e = new Error(msg)
    e.status = err.response?.status
    e.commitOutcome = err.response?.headers?.['x-commit-outcome']
    throw e
  }
)

export default api
