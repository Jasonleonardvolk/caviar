/* TORI Holographic Display Service Worker v4
 * Enhanced with WebGPU shader and ONNX model caching
 * Optimized for iOS 26 Safari PWA deployment
 */

const CACHE_NAME = 'tori-holographic-v4';
const SHADER_CACHE = 'tori-shaders-v1';
const MODEL_CACHE = 'tori-models-v1';
const MANIFEST_URL = '/precache.manifest.json';

// Critical assets that MUST be cached for offline operation
const CRITICAL_ASSETS = [
  '/',
  '/index.html',
  '/manifest.webmanifest'
];

// Shader paths - these are critical for WebGPU rendering
const SHADER_ASSETS = [
  '/hybrid/wgsl/propagation.wgsl',
  '/hybrid/wgsl/fftShift.wgsl',
  '/hybrid/wgsl/butterflyStage.wgsl',
  '/hybrid/wgsl/bitReversal.wgsl',
  '/hybrid/wgsl/normalize.wgsl',
  '/hybrid/wgsl/transpose.wgsl',
  '/hybrid/wgsl/multiViewSynthesis.wgsl',
  '/hybrid/wgsl/lightFieldComposer.wgsl',
  '/hybrid/wgsl/phaseOcclusion.wgsl',
  '/hybrid/wgsl/velocityField.wgsl',
  '/hybrid/wgsl/wavefieldEncoder.wgsl',
  '/hybrid/wgsl/wavefieldEncoder_optimized.wgsl',
  '/hybrid/wgsl/hybridWavefieldBlend.wgsl',
  '/hybrid/wgsl/lenticularInterlace.wgsl',
  '/hybrid/wgsl/multiDepthWaveSynth.wgsl',
  '/hybrid/wgsl/topologicalOverlay.wgsl',
  '/hybrid/wgsl/avatarShader.wgsl'
];

// ONNX models for neural holography
const MODEL_ASSETS = [
  '/models/waveop_fno_v1.onnx',
  '/models/depth_estimator.onnx'
];

// Install event - cache all critical assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      try {
        // Open all caches
        const [mainCache, shaderCache, modelCache] = await Promise.all([
          caches.open(CACHE_NAME),
          caches.open(SHADER_CACHE),
          caches.open(MODEL_CACHE)
        ]);
        
        // Cache critical assets
        console.log('[ServiceWorker] Caching critical assets');
        await mainCache.addAll(CRITICAL_ASSETS);
        
        // Cache shaders with error handling for each
        console.log('[ServiceWorker] Caching WebGPU shaders');
        const shaderPromises = SHADER_ASSETS.map(async (url) => {
          try {
            const response = await fetch(url);
            if (response.ok) {
              await shaderCache.put(url, response);
              console.log('[ServiceWorker] Cached shader:', url);
            }
          } catch (err) {
            console.warn('[ServiceWorker] Failed to cache shader:', url, err);
          }
        });
        await Promise.all(shaderPromises);
        
        // Cache ONNX models (may be large, handle carefully)
        console.log('[ServiceWorker] Caching ONNX models');
        const modelPromises = MODEL_ASSETS.map(async (url) => {
          try {
            const response = await fetch(url);
            if (response.ok) {
              await modelCache.put(url, response);
              console.log('[ServiceWorker] Cached model:', url);
            }
          } catch (err) {
            console.warn('[ServiceWorker] Failed to cache model:', url, err);
          }
        });
        await Promise.all(modelPromises);
        
        // Try to fetch and cache manifest if available
        try {
          const manifestResponse = await fetch(MANIFEST_URL, { cache: 'no-store' });
          if (manifestResponse.ok) {
            const { files } = await manifestResponse.json();
            console.log('[ServiceWorker] Caching manifest files:', files.length);
            
            // Cache manifest files (non-critical, so failures are OK)
            const manifestPromises = files.map(async (url) => {
              try {
                const response = await fetch(url);
                if (response.ok) {
                  await mainCache.put(url, response);
                }
              } catch (err) {
                // Silent fail for non-critical manifest files
              }
            });
            await Promise.all(manifestPromises);
          }
        } catch (err) {
          console.log('[ServiceWorker] No manifest or manifest caching failed (OK in dev)');
        }
        
        await self.skipWaiting();
      } catch (error) {
        console.error('[ServiceWorker] Install error:', error);
        // Still install even if some caching fails
        await self.skipWaiting();
      }
    })()
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      // Get all cache names
      const cacheNames = await caches.keys();
      
      // Delete old caches but keep our current ones
      const currentCaches = [CACHE_NAME, SHADER_CACHE, MODEL_CACHE];
      await Promise.all(
        cacheNames
          .filter(name => name.startsWith('tori-') && !currentCaches.includes(name))
          .map(name => {
            console.log('[ServiceWorker] Deleting old cache:', name);
            return caches.delete(name);
          })
      );
      
      // Take control of all clients
      await self.clients.claim();
      console.log('[ServiceWorker] Activated and claimed clients');
    })()
  );
});

// Fetch event - intelligent caching strategy based on resource type
self.addEventListener('fetch', (event) => {
  const request = event.request;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }
  
  // Skip WebSocket requests
  if (url.protocol === 'ws:' || url.protocol === 'wss:') {
    return;
  }
  
  // Skip Chrome extension requests
  if (url.protocol === 'chrome-extension:') {
    return;
  }
  
  // Skip API calls that shouldn't be cached
  if (url.pathname.startsWith('/api/') && !url.pathname.includes('/quilt/')) {
    return;
  }
  
  event.respondWith(
    (async () => {
      // Determine which cache to use based on resource type
      let cacheName = CACHE_NAME;
      if (url.pathname.endsWith('.wgsl')) {
        cacheName = SHADER_CACHE;
      } else if (url.pathname.endsWith('.onnx')) {
        cacheName = MODEL_CACHE;
      }
      
      // Try cache first for all resources
      const cachedResponse = await caches.match(request);
      
      if (cachedResponse) {
        // For shaders and models, return cache immediately (they don't change often)
        if (cacheName === SHADER_CACHE || cacheName === MODEL_CACHE) {
          return cachedResponse;
        }
        
        // For other resources, return cache and update in background
        const fetchPromise = fetch(request)
          .then(async (networkResponse) => {
            if (networkResponse && networkResponse.status === 200) {
              const cache = await caches.open(cacheName);
              cache.put(request, networkResponse.clone());
            }
            return networkResponse;
          })
          .catch(() => cachedResponse);
        
        return cachedResponse;
      }
      
      // Not in cache, try network
      try {
        const networkResponse = await fetch(request);
        
        // Cache successful responses
        if (networkResponse && networkResponse.status === 200) {
          // Clone the response before caching
          const responseToCache = networkResponse.clone();
          
          // Determine cache strategy based on content type
          const contentType = networkResponse.headers.get('content-type') || '';
          
          // Always cache WebGPU shaders
          if (url.pathname.endsWith('.wgsl') || contentType.includes('wgsl')) {
            const cache = await caches.open(SHADER_CACHE);
            cache.put(request, responseToCache);
            console.log('[ServiceWorker] Cached shader:', url.pathname);
          }
          // Always cache ONNX models
          else if (url.pathname.endsWith('.onnx') || contentType.includes('onnx')) {
            const cache = await caches.open(MODEL_CACHE);
            cache.put(request, responseToCache);
            console.log('[ServiceWorker] Cached model:', url.pathname);
          }
          // Cache other successful responses
          else {
            const cache = await caches.open(cacheName);
            cache.put(request, responseToCache);
          }
        }
        
        return networkResponse;
      } catch (error) {
        console.error('[ServiceWorker] Fetch failed:', url.pathname, error);
        
        // Return offline page for navigation requests
        if (request.destination === 'document') {
          const offlineResponse = await caches.match('/offline.html');
          if (offlineResponse) {
            return offlineResponse;
          }
        }
        
        // Return a basic error response
        return new Response('Network error - offline', {
          status: 503,
          statusText: 'Service Unavailable',
          headers: new Headers({
            'Content-Type': 'text/plain'
          })
        });
      }
    })()
  );
});

// Message event - handle commands from the app
self.addEventListener('message', (event) => {
  const { type, data } = event.data || {};
  
  switch (type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;
      
    case 'CLEAR_CACHE':
      event.waitUntil(
        (async () => {
          await Promise.all([
            caches.delete(CACHE_NAME),
            caches.delete(SHADER_CACHE),
            caches.delete(MODEL_CACHE)
          ]);
          // Recreate empty caches
          await Promise.all([
            caches.open(CACHE_NAME),
            caches.open(SHADER_CACHE),
            caches.open(MODEL_CACHE)
          ]);
        })()
      );
      break;
      
    case 'CACHE_URLS':
      if (data && data.urls) {
        event.waitUntil(
          caches.open(CACHE_NAME).then(cache => {
            return cache.addAll(data.urls);
          })
        );
      }
      break;
      
    case 'CACHE_SHADERS':
      if (data && data.shaders) {
        event.waitUntil(
          caches.open(SHADER_CACHE).then(cache => {
            return cache.addAll(data.shaders);
          })
        );
      }
      break;
      
    case 'CACHE_MODELS':
      if (data && data.models) {
        event.waitUntil(
          caches.open(MODEL_CACHE).then(cache => {
            return cache.addAll(data.models);
          })
        );
      }
      break;
      
    case 'GET_CACHE_STATUS':
      event.waitUntil(
        (async () => {
          const [mainKeys, shaderKeys, modelKeys] = await Promise.all([
            caches.open(CACHE_NAME).then(c => c.keys()),
            caches.open(SHADER_CACHE).then(c => c.keys()),
            caches.open(MODEL_CACHE).then(c => c.keys())
          ]);
          
          // Send cache status back to client
          const clients = await self.clients.matchAll();
          clients.forEach(client => {
            client.postMessage({
              type: 'CACHE_STATUS',
              data: {
                main: mainKeys.length,
                shaders: shaderKeys.length,
                models: modelKeys.length,
                ready: true
              }
            });
          });
        })()
      );
      break;
  }
});

// Background sync for quilt/holographic data
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-holograms') {
    event.waitUntil(syncHolographicAssets());
  }
});

async function syncHolographicAssets() {
  try {
    // Sync any updated shaders
    const shaderCache = await caches.open(SHADER_CACHE);
    for (const shaderUrl of SHADER_ASSETS) {
      try {
        const response = await fetch(shaderUrl, { cache: 'no-cache' });
        if (response.ok) {
          await shaderCache.put(shaderUrl, response);
        }
      } catch (err) {
        // Continue on error
      }
    }
    
    console.log('[ServiceWorker] Holographic assets synced');
  } catch (error) {
    console.error('[ServiceWorker] Sync failed:', error);
  }
}

// Periodic background sync (if supported)
self.addEventListener('periodicsync', (event) => {
  if (event.tag === 'update-shaders') {
    event.waitUntil(syncHolographicAssets());
  }
});

console.log('[ServiceWorker] v4 loaded - WebGPU shader caching enabled');
