// Fix for vite.config.js - Replace the proxy section with this
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
    plugins: [sveltekit()],
    server: {
        port: 5173,
        strictPort: false,
        proxy: {
            '/api': {
                target: 'http://127.0.0.1:8002',  // Use IPv4 only
                changeOrigin: true,
                timeout: 60000,  // 60 second timeout
                configure: (proxy, options) => {
                    // Suppress connection refused spam
                    proxy.on('error', (err, req, res) => {
                        if (err.code === 'ECONNREFUSED') {
                            console.log('⏳ Waiting for API server...');
                        } else {
                            console.error('Proxy error:', err.message);
                        }
                    });
                    
                    // Don't retry so aggressively
                    proxy.on('proxyReq', (proxyReq, req, res) => {
                        proxyReq.setHeader('X-No-Retry', 'true');
                    });
                }
            },
            '/upload': {
                target: 'http://127.0.0.1:8002',  // Use IPv4 only
                changeOrigin: true,
                timeout: 300000  // 5 minutes for uploads
            },
            '/ws': {
                target: 'ws://127.0.0.1:8002',  // Use IPv4 only
                ws: true,
                changeOrigin: true
            }
        }
    }
});
