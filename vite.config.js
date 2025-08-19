import { defineConfig } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: path.resolve(__dirname, 'frontend/public'),
  server: {
    port: 5173,
    open: '/tests/quilt_display.html',
    fs: {
      strict: false,
      allow: [
        path.resolve(__dirname, 'frontend'),
        path.resolve(__dirname, 'configs'),
        path.resolve(__dirname)
      ]
    }
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'frontend/lib')
    }
  }
});