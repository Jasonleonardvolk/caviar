// Fix for Node.js EventEmitter warning in frontend
// Add to the beginning of your vite dev server configuration

// In package.json scripts section, modify the dev script:
{
  "scripts": {
    "dev": "NODE_OPTIONS='--max-old-space-size=4096 --trace-warnings' vite dev"
  }
}

// Or create a vite plugin to handle this:
// vite-plugin-proxy-retry.js

export function proxyRetryPlugin() {
  let retryCount = 0;
  const maxRetries = 5;
  
  return {
    name: 'proxy-retry',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const originalWrite = res.write;
        const originalEnd = res.end;
        
        res.write = function(...args) {
          retryCount = 0; // Reset on successful write
          return originalWrite.apply(res, args);
        };
        
        res.end = function(...args) {
          retryCount = 0; // Reset on successful end
          return originalEnd.apply(res, args);
        };
        
        next();
      });
    }
  };
}