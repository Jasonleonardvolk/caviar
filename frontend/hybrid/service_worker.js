/* frontend/hybrid/service_worker.js */
const SW_VERSION = 'v1';
const CACHE_NAME = `gaea_cache_${SW_VERSION}`;

// Adjust as needed
const PRECACHE_URLS = [
  '/',                                // app shell
  '/public/tests/waveop_dashboard.html',
  '/public/tests/schrodinger_bench.html',
  '/public/models/waveop_fno_v1.onnx',
  '/lib/webgpu/generated/shaderSources.ts', // will be intercepted as text
];

// Cache-first for shaders/models; network-first for HTML
self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE_NAME);
    await cache.addAll(PRECACHE_URLS);
    self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.map((k) => (k === CACHE_NAME ? null : caches.delete(k))));
    self.clients.claim();
  })());
});

function isAsset(req) {
  const u = new URL(req.url);
  return (
    u.pathname.endsWith('.onnx') ||
    u.pathname.endsWith('.wgsl') ||
    u.pathname.includes('/generated/shaderSources') ||
    u.pathname.endsWith('.js') ||
    u.pathname.endsWith('.mjs')
  );
}

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Bypass cross-origin
  if (self.location.origin !== url.origin) return;

  // HTML: network-first
  if (request.mode === 'navigate' || request.headers.get('accept')?.includes('text/html')) {
    event.respondWith((async () => {
      try {
        const net = await fetch(request);
        const cache = await caches.open(CACHE_NAME);
        cache.put(request, net.clone());
        return net;
      } catch {
        const cache = await caches.open(CACHE_NAME);
        const cached = await cache.match(request);
        return cached ?? new Response('<h1>Offline</h1>', { headers: { 'Content-Type': 'text/html' } });
      }
    })());
    return;
  }

  // Shaders / models / JS: cache-first
  if (isAsset(request)) {
    event.respondWith((async () => {
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(request);
      if (cached) return cached;
      const net = await fetch(request);
      cache.put(request, net.clone());
      return net;
    })());
  }
});
