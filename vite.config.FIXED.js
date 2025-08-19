// vite.config.js
// Copy and paste this entire file to D:\Dev\kha\tori_ui_svelte\vite.config.js

import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  esbuild: {
    target: 'es2020'  // THIS IS THE KEY FIX FOR THE "Unexpected ?" ERROR
  },
  server: {
    port: 5173,
    strictPort: false
  },
  optimizeDeps: {
    exclude: ['@webgpu/types']
  }
});
