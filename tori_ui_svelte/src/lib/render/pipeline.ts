import { resolveCaps } from '../device/deviceMatrix';
import { detectiPhoneModelFromUA } from '../device/ua';
import { ThermalGovernor } from '../runtime/thermalGovernor';
import { telemetry } from '../stores/telemetry';

const gov = new ThermalGovernor({ targetFps: 60 });

declare const navigator: any; // for SSR guards
function getUA(): string { try { return navigator?.userAgent ?? ""; } catch { return ""; } }

// Placeholder for your existing local hologram function
async function localHologramPropagate(frame: Float32Array, options: any): Promise<Float32Array> {
  // TODO: Replace with your actual hologram propagation implementation
  return frame;
}

// Server assist function - always routes through proxy
async function serverAssist(frame: Float32Array, options: any): Promise<Float32Array> {
  const url = '/api/render';
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ frame: Array.from(frame), ...options })
  });
  
  if (!response.ok) throw new Error('Server assist failed');
  const data = await response.json();
  return new Float32Array(data.field);
}

// Update telemetry helper
function updateTelemetry(updates: any) {
  telemetry.update((t) => ({ ...t, ...updates }));
}

// Log significant events
function logEvent(event: string) {
  telemetry.update((t) => ({
    ...t,
    events: [...t.events.slice(-9), event] // Keep last 10 events
  }));
}

// Conservative assist wrapper with timeout: never block the frame loop
async function assistOrLocal(
  local: () => Promise<Float32Array>,
  assist: () => Promise<Float32Array>,
  budgetMs = 250
): Promise<Float32Array> {
  // Always compute local first (keeps UI responsive)
  const outLocal = await local();
  
  // Set up timeout for assist
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), budgetMs);
  
  try {
    // Try server assist with timeout
    const assistPromise = assist();
    const timeoutPromise = new Promise<null>((_, reject) => {
      ctrl.signal.addEventListener('abort', () => reject(new Error('Timeout')));
    });
    
    const outAssist = await Promise.race([assistPromise, timeoutPromise]);
    clearTimeout(timer);
    
    if (outAssist) {
      return outAssist as Float32Array;
    }
  } catch (e) {
    clearTimeout(timer);
    console.warn('Assist failed, falling back to local:', e);
  }
  
  // **FAIL-CLOSED**: Always return local result on any failure
  return outLocal;
}

export async function renderFrame(frame: Float32Array, sceneMeta: any) {
  const model = detectiPhoneModelFromUA(getUA());
  const caps = resolveCaps(model);
  let N = caps.maxN;
  let assistState = 'local';

  // First, always compute local result for timing
  const t0 = performance.now();
  const outLocal = await localHologramPropagate(frame, { N, caps, sceneMeta });
  const t1 = performance.now();
  const frameMs = t1 - t0;

  gov.tick(frameMs);
  const advice = gov.advise(N);

  // Update telemetry with frame time
  updateTelemetry({ frameMsEMA: frameMs, N, zModes: caps.zernikeModes });

  // Handle thermal degradation
  if (advice.action !== 'hold') {
    if (advice.action === 'decrease') {
      assistState = 'degraded';
      logEvent(`N downshift: ${N} → ${advice.nextN}`);
      N = advice.nextN;
    }
    
    if (!caps.serverFallback) {
      // No server assist available, just use degraded local
      updateTelemetry({ N, assist: assistState });
      return await localHologramPropagate(frame, { N, caps, sceneMeta });
    }
  }

  // Use server assist if: 
  // 1. Device tier allows it (caps.serverFallback)
  // 2. Frame time exceeds budget OR thermal pressure
  const allowAssist = caps.serverFallback;
  const needAssist = frameMs > ((1000/60) + 1.0) || advice.action === 'decrease';
  
  if (allowAssist && needAssist) {
    // Use the fail-closed wrapper with timeout protection
    const localFn = () => Promise.resolve(outLocal);  // Already computed
    const assistFn = () => serverAssist(frame, { N, caps, sceneMeta });
    
    const result = await assistOrLocal(localFn, assistFn);
    
    // Update telemetry based on what actually happened
    if (result === outLocal) {
      updateTelemetry({ assist: 'local' });
      logEvent('Using local (assist unavailable or timed out)');
    } else {
      updateTelemetry({ assist: 'remote' });
      logEvent('Assist flip: local → remote');
    }
    
    return result;
  }
  
  // Default: return the local result we already computed
  updateTelemetry({ assist: assistState });
  return outLocal;
}