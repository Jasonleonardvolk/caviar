/**
 * Service Worker Registration Helper
 * Handles SW registration with proper error handling and update notifications
 */

export interface SWRegistrationOptions {
  onUpdate?: (registration: ServiceWorkerRegistration) => void;
  onSuccess?: (registration: ServiceWorkerRegistration) => void;
  onError?: (error: Error) => void;
  immediate?: boolean;
}

export async function registerSW(options: SWRegistrationOptions = {}): Promise<ServiceWorkerRegistration | undefined> {
  if (!('serviceWorker' in navigator)) {
    console.log('[SW] Service Workers not supported');
    return undefined;
  }

  // Check if we're on localhost or HTTPS
  const isLocalhost = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
  const isHttps = location.protocol === 'https:';
  
  if (!isLocalhost && !isHttps) {
    console.warn('[SW] Service Workers require HTTPS (or localhost)');
    return undefined;
  }

  try {
    // Wait for window load unless immediate registration requested
    if (!options.immediate && document.readyState !== 'complete') {
      await new Promise(resolve => window.addEventListener('load', resolve));
    }

    // Register the service worker
    const registration = await navigator.serviceWorker.register('/service-worker.js', {
      scope: '/',
      updateViaCache: 'none' // Always check for updates
    });

    console.log('[SW] Registration successful:', registration.scope);

    // Check for updates every hour
    setInterval(() => {
      registration.update();
    }, 60 * 60 * 1000);

    // Handle updates
    registration.addEventListener('updatefound', () => {
      const newWorker = registration.installing;
      if (!newWorker) return;

      newWorker.addEventListener('statechange', () => {
        if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
          // New service worker available
          console.log('[SW] New content available');
          if (options.onUpdate) {
            options.onUpdate(registration);
          } else {
            // Default update notification
            if (confirm('New version available! Reload to update?')) {
              newWorker.postMessage({ type: 'SKIP_WAITING' });
              window.location.reload();
            }
          }
        }
      });
    });

    // Handle controller change (new SW activated)
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      console.log('[SW] Controller changed, reloading...');
      window.location.reload();
    });

    // Success callback
    if (options.onSuccess) {
      options.onSuccess(registration);
    }

    return registration;
  } catch (error) {
    console.error('[SW] Registration failed:', error);
    if (options.onError) {
      options.onError(error as Error);
    }
    return undefined;
  }
}

/**
 * Unregister all service workers
 */
export async function unregisterSW(): Promise<boolean> {
  if (!('serviceWorker' in navigator)) {
    return false;
  }

  try {
    const registrations = await navigator.serviceWorker.getRegistrations();
    await Promise.all(registrations.map(r => r.unregister()));
    console.log('[SW] All service workers unregistered');
    return true;
  } catch (error) {
    console.error('[SW] Unregistration failed:', error);
    return false;
  }
}

/**
 * Clear all caches
 */
export async function clearSWCache(): Promise<boolean> {
  if (!('caches' in window)) {
    return false;
  }

  try {
    const cacheNames = await caches.keys();
    await Promise.all(cacheNames.map(name => caches.delete(name)));
    console.log('[SW] All caches cleared');
    
    // Tell SW to clear its cache too
    if (navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({ type: 'CLEAR_CACHE' });
    }
    
    return true;
  } catch (error) {
    console.error('[SW] Cache clear failed:', error);
    return false;
  }
}

/**
 * Cache specific URLs programmatically
 */
export async function cacheUrls(urls: string[]): Promise<void> {
  if (!navigator.serviceWorker.controller) {
    console.warn('[SW] No active service worker');
    return;
  }

  navigator.serviceWorker.controller.postMessage({
    type: 'CACHE_URLS',
    urls
  });
}

/**
 * Check if service worker is supported and active
 */
export function isSWActive(): boolean {
  return 'serviceWorker' in navigator && navigator.serviceWorker.controller !== null;
}

/**
 * Get service worker registration status
 */
export async function getSWStatus(): Promise<{
  supported: boolean;
  registered: boolean;
  active: boolean;
  scope?: string;
  updateAvailable: boolean;
}> {
  const supported = 'serviceWorker' in navigator;
  
  if (!supported) {
    return { supported: false, registered: false, active: false, updateAvailable: false };
  }

  try {
    const registration = await navigator.serviceWorker.getRegistration();
    
    if (!registration) {
      return { supported: true, registered: false, active: false, updateAvailable: false };
    }

    return {
      supported: true,
      registered: true,
      active: registration.active !== null,
      scope: registration.scope,
      updateAvailable: registration.waiting !== null
    };
  } catch {
    return { supported: true, registered: false, active: false, updateAvailable: false };
  }
}

// Auto-register on import if not in development
if (import.meta.env.PROD) {
  registerSW();
}
