import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api/metrics': {
        target: 'http://100.81.50.51:8082',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/metrics/, '/metrics')
      },
      '/api/control': {
        target: 'http://100.81.50.51:8082',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/control/, '/control')
      },
      '/api/stream': {
        target: 'http://100.81.50.51:8090',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/stream/, '/stream.mjpg')
      }
    }
  }
})
