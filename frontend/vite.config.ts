import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      "/api/rand": "http://127.0.0.1:5000",
      "/ckpt/act": "http://127.0.0.1:5000",
      "/model/load":  {
        target: "http://127.0.0.1:5000",
        timeout: 10000
      },
      "/model/predict": "http://127.0.0.1:5000",
      "/model/run": "http://127.0.0.1:5000",
    }
  }
})
