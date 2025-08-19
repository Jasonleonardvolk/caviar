import { defineConfig } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: '.',
  publicDir: 'public',
  define: {
    __IRIS_WAVE__: process.env.VITE_IRIS_ENABLE_WAVE === '1'
  },
  server: {
    port: 5173,
    open: '/',
    fs: {
      strict: false
    }
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true
  },
  optimizeDeps: {
    include: ['onnxruntime-web'],
    force: true
  },
  cacheDir: '.vite-cache'
});