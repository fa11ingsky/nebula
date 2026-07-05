import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Cross-origin isolation headers, mirroring public/_headers (which only takes effect on
// the deployed Netlify site) - without these here too, `npm run dev`/`npm run preview`
// can't exercise SharedArrayBuffer/crossOriginIsolated locally, since that file is never
// read by Vite's own dev/preview servers.
const crossOriginIsolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  server: {
    headers: crossOriginIsolationHeaders,
  },
  preview: {
    headers: crossOriginIsolationHeaders,
  },
  worker: {
    // simulation.worker.ts now dynamically imports the threaded WASM gravity build (see
    // gravity.ts's initGravityWasm), which itself spawns further pthread worker instances
    // via `new Worker(new URL(...))` - that nested code-splitting isn't representable in
    // Vite's default 'iife' worker output format (a single self-contained function can't
    // reference separate chunks), so the worker bundle needs to be emitted as real ES
    // modules instead. Requires the worker itself to be constructed with { type: 'module' }
    // (see Particles.vue) - already the case here.
    format: 'es',
  },
})
