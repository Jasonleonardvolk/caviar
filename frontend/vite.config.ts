import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import path from 'path';
import autoPrecache from './plugins/autoPrecache';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    svelte(),
    
    // Auto-generate precache manifest for Service Worker
    autoPrecache({
      include: [
        'public/wasm/**/*.{wasm,js}',
        'public/hybrid/wgsl/**/*.wgsl',
        'public/assets/**/*.{ktx2,ktx,png,jpg,jpeg,webp}',
        'public/assets/**/*.{bin,json}',
        'public/assets/**/*.{mp4,webm,av1,ivf}'
      ],
      extra: ['/'],
      manifestName: 'precache.manifest.json'
    })
  ],
  
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@hybrid': path.resolve(__dirname, './hybrid'),
      '@lib': path.resolve(__dirname, './lib'),
      '@components': path.resolve(__dirname, './components'),
      '@shaders': path.resolve(__dirname, './shaders'),
      '@wgsl': path.resolve(__dirname, './hybrid/wgsl')
    }
  },
  
  server: {
    port: 5173,
    host: true,
    cors: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true
      }
    }
  },
  
  build: {
    target: 'esnext',
    minify: 'terser',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['svelte'],
          'webgpu': ['@webgpu/types'],
          'codecs': ['ktx-parse', 'mp4box']
        }
      }
    },
    assetsInlineLimit: 0, // Don't inline assets
    chunkSizeWarningLimit: 2000
  },
  
  optimizeDeps: {
    include: ['svelte', 'ktx-parse', 'mp4box'],
    exclude: ['@webgpu/types']
  },
  
  define: {
    'import.meta.env.BUILD_TIME': JSON.stringify(new Date().toISOString()),
    'import.meta.env.VERSION': JSON.stringify(process.env.npm_package_version)
  },
  
  worker: {
    format: 'es',
    plugins: []
  },
  
  // Enable WASM support
  assetsInclude: ['**/*.wasm', '**/*.wgsl', '**/*.ktx2', '**/*.ivf']
});
