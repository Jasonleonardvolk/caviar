// vite.config.js
import { sveltekit } from "file:///C:/Users/jason/Desktop/tori/kha/tori_ui_svelte/node_modules/@sveltejs/kit/src/exports/vite/index.js";
import { defineConfig } from "file:///C:/Users/jason/Desktop/tori/kha/tori_ui_svelte/node_modules/vite/dist/node/index.js";
var vite_config_default = defineConfig({
  plugins: [sveltekit()],
  css: {
    postcss: "./postcss.config.js"
  },
  server: {
    port: 5173,
    host: true,
    hmr: {
      overlay: false
      // Disable error overlay
    },
    fs: {
      allow: [".."]
    },
    proxy: {
      // 🔧 FIXED: Correct port routing to match enhanced launcher (8002)
      "/upload": {
        target: "http://127.0.0.1:8002",
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/upload/, "/api/upload"),
        timeout: 6e5,
        // allow 10 min for slow uploads
        proxyTimeout: 6e5,
        // same for proxy side
        headers: {
          "Connection": "keep-alive"
        }
      },
      // 🧠 EXPLICIT SOLITON MEMORY ROUTES
      "/api/soliton": {
        target: "http://127.0.0.1:8002",
        changeOrigin: true,
        secure: false,
        ws: false,
        configure: (proxy, _options) => {
          let retryCount = 0;
          const maxRetries = 5;
          const retryDelay = 500;
          proxy.on("error", (err, req, res) => {
            if (err.code === "ECONNREFUSED" && retryCount < maxRetries) {
              retryCount++;
              console.log(`\u{1F9E0} Soliton proxy retry ${retryCount}/${maxRetries} after ${retryDelay}ms...`);
              setTimeout(() => {
                proxy.web(req, res, { target: "http://127.0.0.1:8002" });
              }, retryDelay);
            } else {
              console.log("\u{1F9E0} Soliton proxy error:", err);
              retryCount = 0;
            }
          });
          proxy.on("proxyReq", (proxyReq, req, _res) => {
            console.log("\u{1F9E0} Soliton request:", req.method, req.url);
          });
          proxy.on("proxyRes", (proxyRes, req, _res) => {
            console.log("\u{1F9E0} Soliton response:", proxyRes.statusCode, req.url);
            retryCount = 0;
          });
        }
      },
      "/api": {
        target: "http://127.0.0.1:8002",
        changeOrigin: true,
        secure: false,
        ws: true,
        timeout: 6e5,
        // allow 10 min for slow uploads
        proxyTimeout: 6e5,
        // same for proxy side
        headers: {
          "Connection": "keep-alive"
        },
        configure: (proxy, _options) => {
          let retryCount = 0;
          const maxRetries = 5;
          const retryDelay = 500;
          proxy.on("error", (err, req, res) => {
            if (err.code === "ECONNREFUSED" && retryCount < maxRetries) {
              retryCount++;
              console.warn(`\u26A0\uFE0F API proxy retry ${retryCount}/${maxRetries} after ${retryDelay}ms...`);
              setTimeout(() => {
                proxy.web(req, res, { target: "http://127.0.0.1:8002" });
              }, retryDelay);
            } else {
              console.warn("\u26A0\uFE0F API proxy error (backend may not be ready):", err.message);
              retryCount = 0;
            }
          });
          proxy.on("proxyRes", (proxyRes, req, _res) => {
            retryCount = 0;
          });
        }
      }
    }
  },
  build: {
    target: "esnext",
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ["svelte"]
        }
      }
    }
  },
  optimizeDeps: {
    include: []
  },
  define: {
    __VERSION__: JSON.stringify(process.env.npm_package_version || "1.0.0"),
    __TORI_THEME__: JSON.stringify("light"),
    "import.meta.env.VITE_API_URL": JSON.stringify(process.env.VITE_API_URL || "http://127.0.0.1:8002")
  }
});
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcuanMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJDOlxcXFxVc2Vyc1xcXFxqYXNvblxcXFxEZXNrdG9wXFxcXHRvcmlcXFxca2hhXFxcXHRvcmlfdWlfc3ZlbHRlXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ZpbGVuYW1lID0gXCJDOlxcXFxVc2Vyc1xcXFxqYXNvblxcXFxEZXNrdG9wXFxcXHRvcmlcXFxca2hhXFxcXHRvcmlfdWlfc3ZlbHRlXFxcXHZpdGUuY29uZmlnLmpzXCI7Y29uc3QgX192aXRlX2luamVjdGVkX29yaWdpbmFsX2ltcG9ydF9tZXRhX3VybCA9IFwiZmlsZTovLy9DOi9Vc2Vycy9qYXNvbi9EZXNrdG9wL3Rvcmkva2hhL3RvcmlfdWlfc3ZlbHRlL3ZpdGUuY29uZmlnLmpzXCI7aW1wb3J0IHsgc3ZlbHRla2l0IH0gZnJvbSAnQHN2ZWx0ZWpzL2tpdC92aXRlJztcbmltcG9ydCB7IGRlZmluZUNvbmZpZyB9IGZyb20gJ3ZpdGUnO1xuXG5leHBvcnQgZGVmYXVsdCBkZWZpbmVDb25maWcoe1xuICBwbHVnaW5zOiBbc3ZlbHRla2l0KCldLFxuICBjc3M6IHtcbiAgICBwb3N0Y3NzOiAnLi9wb3N0Y3NzLmNvbmZpZy5qcydcbiAgfSxcbiAgc2VydmVyOiB7XG4gICAgcG9ydDogNTE3MyxcbiAgICBob3N0OiB0cnVlLFxuICAgIGhtcjoge1xuICAgICAgb3ZlcmxheTogZmFsc2UgIC8vIERpc2FibGUgZXJyb3Igb3ZlcmxheVxuICAgIH0sXG4gICAgZnM6IHtcbiAgICAgIGFsbG93OiBbJy4uJ11cbiAgICB9LFxuICAgIHByb3h5OiB7XG4gICAgICAvLyBcdUQ4M0RcdUREMjcgRklYRUQ6IENvcnJlY3QgcG9ydCByb3V0aW5nIHRvIG1hdGNoIGVuaGFuY2VkIGxhdW5jaGVyICg4MDAyKVxuICAgICAgJy91cGxvYWQnOiB7XG4gICAgICAgIHRhcmdldDogXCJodHRwOi8vMTI3LjAuMC4xOjgwMDJcIixcbiAgICAgICAgY2hhbmdlT3JpZ2luOiB0cnVlLFxuICAgICAgICBzZWN1cmU6IGZhbHNlLFxuICAgICAgICByZXdyaXRlOiAocGF0aCkgPT4gcGF0aC5yZXBsYWNlKC9eXFwvdXBsb2FkLywgJy9hcGkvdXBsb2FkJyksXG4gICAgICAgIHRpbWVvdXQ6IDYwMF8wMDAsICAgICAgICAvLyBhbGxvdyAxMCBtaW4gZm9yIHNsb3cgdXBsb2Fkc1xuICAgICAgICBwcm94eVRpbWVvdXQ6IDYwMF8wMDAsICAgLy8gc2FtZSBmb3IgcHJveHkgc2lkZVxuICAgICAgICBoZWFkZXJzOiB7XG4gICAgICAgICAgJ0Nvbm5lY3Rpb24nOiAna2VlcC1hbGl2ZSdcbiAgICAgICAgfVxuICAgICAgfSxcbiAgICAgIC8vIFx1RDgzRVx1RERFMCBFWFBMSUNJVCBTT0xJVE9OIE1FTU9SWSBST1VURVNcbiAgICAgICcvYXBpL3NvbGl0b24nOiB7XG4gICAgICAgIHRhcmdldDogXCJodHRwOi8vMTI3LjAuMC4xOjgwMDJcIixcbiAgICAgICAgY2hhbmdlT3JpZ2luOiB0cnVlLFxuICAgICAgICBzZWN1cmU6IGZhbHNlLFxuICAgICAgICB3czogZmFsc2UsXG4gICAgICAgIGNvbmZpZ3VyZTogKHByb3h5LCBfb3B0aW9ucykgPT4ge1xuICAgICAgICAgIC8vIEFkZCByZXRyeSBsb2dpYyBmb3IgY29sZCBib290XG4gICAgICAgICAgbGV0IHJldHJ5Q291bnQgPSAwO1xuICAgICAgICAgIGNvbnN0IG1heFJldHJpZXMgPSA1O1xuICAgICAgICAgIGNvbnN0IHJldHJ5RGVsYXkgPSA1MDA7IC8vIG1zXG4gICAgICAgICAgXG4gICAgICAgICAgcHJveHkub24oJ2Vycm9yJywgKGVyciwgcmVxLCByZXMpID0+IHtcbiAgICAgICAgICAgIGlmIChlcnIuY29kZSA9PT0gJ0VDT05OUkVGVVNFRCcgJiYgcmV0cnlDb3VudCA8IG1heFJldHJpZXMpIHtcbiAgICAgICAgICAgICAgcmV0cnlDb3VudCsrO1xuICAgICAgICAgICAgICBjb25zb2xlLmxvZyhgXHVEODNFXHVEREUwIFNvbGl0b24gcHJveHkgcmV0cnkgJHtyZXRyeUNvdW50fS8ke21heFJldHJpZXN9IGFmdGVyICR7cmV0cnlEZWxheX1tcy4uLmApO1xuICAgICAgICAgICAgICBzZXRUaW1lb3V0KCgpID0+IHtcbiAgICAgICAgICAgICAgICAvLyBUcmlnZ2VyIHJldHJ5IGJ5IHJlLWVtaXR0aW5nIHRoZSByZXF1ZXN0XG4gICAgICAgICAgICAgICAgcHJveHkud2ViKHJlcSwgcmVzLCB7IHRhcmdldDogXCJodHRwOi8vMTI3LjAuMC4xOjgwMDJcIiB9KTtcbiAgICAgICAgICAgICAgfSwgcmV0cnlEZWxheSk7XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICBjb25zb2xlLmxvZygnXHVEODNFXHVEREUwIFNvbGl0b24gcHJveHkgZXJyb3I6JywgZXJyKTtcbiAgICAgICAgICAgICAgcmV0cnlDb3VudCA9IDA7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfSk7XG4gICAgICAgICAgcHJveHkub24oJ3Byb3h5UmVxJywgKHByb3h5UmVxLCByZXEsIF9yZXMpID0+IHtcbiAgICAgICAgICAgIGNvbnNvbGUubG9nKCdcdUQ4M0VcdURERTAgU29saXRvbiByZXF1ZXN0OicsIHJlcS5tZXRob2QsIHJlcS51cmwpO1xuICAgICAgICAgIH0pO1xuICAgICAgICAgIHByb3h5Lm9uKCdwcm94eVJlcycsIChwcm94eVJlcywgcmVxLCBfcmVzKSA9PiB7XG4gICAgICAgICAgICBjb25zb2xlLmxvZygnXHVEODNFXHVEREUwIFNvbGl0b24gcmVzcG9uc2U6JywgcHJveHlSZXMuc3RhdHVzQ29kZSwgcmVxLnVybCk7XG4gICAgICAgICAgICByZXRyeUNvdW50ID0gMDsgLy8gUmVzZXQgb24gc3VjY2Vzc2Z1bCByZXNwb25zZVxuICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgICB9LFxuICAgICAgJy9hcGknOiB7XG4gICAgICAgIHRhcmdldDogXCJodHRwOi8vMTI3LjAuMC4xOjgwMDJcIixcbiAgICAgICAgY2hhbmdlT3JpZ2luOiB0cnVlLFxuICAgICAgICBzZWN1cmU6IGZhbHNlLFxuICAgICAgICB3czogdHJ1ZSxcbiAgICAgICAgdGltZW91dDogNjAwXzAwMCwgICAgICAgIC8vIGFsbG93IDEwIG1pbiBmb3Igc2xvdyB1cGxvYWRzXG4gICAgICAgIHByb3h5VGltZW91dDogNjAwXzAwMCwgICAvLyBzYW1lIGZvciBwcm94eSBzaWRlXG4gICAgICAgIGhlYWRlcnM6IHtcbiAgICAgICAgICAnQ29ubmVjdGlvbic6ICdrZWVwLWFsaXZlJ1xuICAgICAgICB9LFxuICAgICAgICBjb25maWd1cmU6IChwcm94eSwgX29wdGlvbnMpID0+IHtcbiAgICAgICAgICAvLyBBZGQgcmV0cnkgbG9naWMgZm9yIGNvbGQgYm9vdFxuICAgICAgICAgIGxldCByZXRyeUNvdW50ID0gMDtcbiAgICAgICAgICBjb25zdCBtYXhSZXRyaWVzID0gNTtcbiAgICAgICAgICBjb25zdCByZXRyeURlbGF5ID0gNTAwOyAvLyBtc1xuICAgICAgICAgIFxuICAgICAgICAgIHByb3h5Lm9uKCdlcnJvcicsIChlcnIsIHJlcSwgcmVzKSA9PiB7XG4gICAgICAgICAgICBpZiAoZXJyLmNvZGUgPT09ICdFQ09OTlJFRlVTRUQnICYmIHJldHJ5Q291bnQgPCBtYXhSZXRyaWVzKSB7XG4gICAgICAgICAgICAgIHJldHJ5Q291bnQrKztcbiAgICAgICAgICAgICAgY29uc29sZS53YXJuKGBcdTI2QTBcdUZFMEYgQVBJIHByb3h5IHJldHJ5ICR7cmV0cnlDb3VudH0vJHttYXhSZXRyaWVzfSBhZnRlciAke3JldHJ5RGVsYXl9bXMuLi5gKTtcbiAgICAgICAgICAgICAgc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICAgICAgICAgICAgLy8gVHJpZ2dlciByZXRyeSBieSByZS1lbWl0dGluZyB0aGUgcmVxdWVzdFxuICAgICAgICAgICAgICAgIHByb3h5LndlYihyZXEsIHJlcywgeyB0YXJnZXQ6IFwiaHR0cDovLzEyNy4wLjAuMTo4MDAyXCIgfSk7XG4gICAgICAgICAgICAgIH0sIHJldHJ5RGVsYXkpO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgY29uc29sZS53YXJuKCdcdTI2QTBcdUZFMEYgQVBJIHByb3h5IGVycm9yIChiYWNrZW5kIG1heSBub3QgYmUgcmVhZHkpOicsIGVyci5tZXNzYWdlKTtcbiAgICAgICAgICAgICAgcmV0cnlDb3VudCA9IDA7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfSk7XG4gICAgICAgICAgcHJveHkub24oJ3Byb3h5UmVzJywgKHByb3h5UmVzLCByZXEsIF9yZXMpID0+IHtcbiAgICAgICAgICAgIHJldHJ5Q291bnQgPSAwOyAvLyBSZXNldCBvbiBzdWNjZXNzZnVsIHJlc3BvbnNlXG4gICAgICAgICAgfSk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH0sXG4gIGJ1aWxkOiB7XG4gICAgdGFyZ2V0OiAnZXNuZXh0JyxcbiAgICBzb3VyY2VtYXA6IHRydWUsXG4gICAgcm9sbHVwT3B0aW9uczoge1xuICAgICAgb3V0cHV0OiB7XG4gICAgICAgIG1hbnVhbENodW5rczoge1xuICAgICAgICAgIHZlbmRvcjogWydzdmVsdGUnXVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9LFxuICBvcHRpbWl6ZURlcHM6IHtcbiAgICBpbmNsdWRlOiBbXVxuICB9LFxuICBkZWZpbmU6IHtcbiAgICBfX1ZFUlNJT05fXzogSlNPTi5zdHJpbmdpZnkocHJvY2Vzcy5lbnYubnBtX3BhY2thZ2VfdmVyc2lvbiB8fCAnMS4wLjAnKSxcbiAgICBfX1RPUklfVEhFTUVfXzogSlNPTi5zdHJpbmdpZnkoJ2xpZ2h0JyksXG4gICAgXCJpbXBvcnQubWV0YS5lbnYuVklURV9BUElfVVJMXCI6IEpTT04uc3RyaW5naWZ5KHByb2Nlc3MuZW52LlZJVEVfQVBJX1VS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
