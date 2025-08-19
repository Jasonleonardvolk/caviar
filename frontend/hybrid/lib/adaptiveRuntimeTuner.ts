/**
 * Periodically re-measures frame timing and nudges settings to maintain target FPS.
 * Works alongside your AdaptiveRenderer instance.
 */
import { AdaptiveRenderer } from './adaptiveRenderer';

export function startRuntimeTuner(renderer: AdaptiveRenderer, periodMs=10000) {
  let stop = false;
  const loop = async () => {
    while (!stop) {
      try {
        const settings = await renderer.initialize(); // reuse measurement path
        console.log('[RuntimeTuner] settings', settings);
      } catch (e) {
        console.warn('[RuntimeTuner] measurement failed', e);
      }
      await new Promise(r => setTimeout(r, periodMs));
    }
  };
  loop();
  return () => { stop = true; };
}
