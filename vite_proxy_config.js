// Add to vite.config.js in the tori_ui_svelte directory
// This reduces the proxy error spam

export default {
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8002',
        changeOrigin: true,
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            // Suppress connection refused errors during startup
            if (err.code === 'ECONNREFUSED') {
              console.log('⏳ Waiting for API server...');
            }
          });
          proxy.on('proxyReq', (proxyReq, req, res) => {
            // Add retry logic
            proxyReq.setHeader('X-Retry-After', '1000');
          });
        }
      }
    }
  }
}