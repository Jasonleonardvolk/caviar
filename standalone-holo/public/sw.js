// sw.js - ultra-minimal cache-first with network update
const CACHE = 'holo-cache-v1';
const ASSETS = [
  '/',
  '/index.html',
  '/manifest.webmanifest'
  // You can add built assets after build (dist paths).
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(self.clients.claim());
});
self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (url.origin === location.origin) {
    e.respondWith(
      caches.match(e.request).then(cached => {
        const fetchP = fetch(e.request).then(resp => {
          if (resp && resp.status === 200) {
            const copy = resp.clone();
            caches.open(CACHE).then(c => c.put(e.request, copy));
          }
          return resp;
        }).catch(() => cached);
        return cached || fetchP;
      })
    );
  }
});