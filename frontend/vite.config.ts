import { defineConfig } from 'vite';
import { sveltekit } from '@sveltejs/kit/vite';
import path from 'path';

export default defineConfig({
  plugins: [
    sveltekit()
  ],
  resolve: {
    alias: {
      // SvelteKit-native src mapping
      '@': path.resolve(__dirname, './src'),
      '@app': path.resolve(__dirname, './src'),
      // Support legacy roots that exist in your repo
      '@lib': path.resolve(__dirname, './lib'),
      '@components': path.resolve(__dirname, './components'),
      '@shaders': path.resolve(__dirname, './shaders'),
      '@wgsl': path.resolve(__dirname, './hybrid/wgsl'),
      '@hybrid': path.resolve(__dirname, './hybrid'),
      // Also map src/lib for SvelteKit convention
      '$lib': path.resolve(__dirname, './src/lib')
    }
  },
  server: {
    port: 5173,
    host: true,
    cors: true,
    proxy: {
      // Keep FastAPI available for any backend services
      '/ws': { 
        target: 'ws://localhost:8000', 
        ws: true 
      },
      '/api/back': { 
        target: 'http://localhost:8000', 
        changeOrigin: true 
      }
    }
  },
  build: {
    target: 'esnext',
    minify: 'terser',
    sourcemap: true,
    assetsInlineLimit: 0,
    chunkSizeWarningLimit: 2000,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['svelte'],
          webgpu: ['@webgpu/types'],
          codecs: ['ktx-parse', 'mp4box']
        }
      }
    }
  },
  optimizeDeps: {
    include: ['svelte', 'ktx-parse', 'mp4box'],
    exclude: ['@webgpu/types']
  },
  define: {
    'import.meta.env.BUILD_TIME': JSON.stringify(new Date().toISOString()),
    'import.meta.env.VERSION': JSON.stringify(process.env.npm_package_version)
  },
  // Enable WASM support
  assetsInclude: ['**/*.wasm', '**/*.wgsl', '**/*.ktx2', '**/*.ivf']
});