import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [
    vue(),
    {
      name: 'server-timeout',
      configureServer(server) {
        server.httpServer.timeout = 1200000
        server.httpServer.keepAliveTimeout = 600000
        server.httpServer.headersTimeout = 600001
      },
    },
    {
      name: 'silence-proxy-errors',
      configureServer(server) {
        const orig = server.config.logger.error
        server.config.logger.error = (msg, options) => {
          if (typeof msg === 'string' && msg.includes('ECONNREFUSED') && msg.includes('/api/')) return
          return orig(msg, options)
        }
      },
    },
  ],
  build: {
    target: 'es2020',
    cssCodeSplit: true,
    rollupOptions: {
      output: {
        manualChunks: {
          echarts: ['echarts'],
          vendor: ['vue', 'vue-router', 'axios'],
        },
      },
    },
  },
  server: {
    port: 5174,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://127.0.0.1:8001',
        changeOrigin: true,
        timeout: 1200000,
        proxyTimeout: 1200000,
        configure: (proxy) => {
          proxy.on('error', (err, _req, res) => {
            if (res && !res.headersSent) {
              res.writeHead(503, { 'Content-Type': 'application/json' })
            }
            if (res) {
              res.end(JSON.stringify({ detail: '后端启动中，请稍后刷新...' }))
            }
          })
        },
      },
    },
  },
})
