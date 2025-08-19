// ${IRIS_ROOT}\frontend\apps\phone_client\src\main.ts
import { SensorManager } from '../../hybrid/lib/sensors/SensorManager';
import { mountHeadTrackingSettings } from '../../hybrid/ui/HeadTrackingSettings';
import { mountHeadTrackingPrompt } from '../../hybrid/ui/HeadTrackingPrompt';
// import { setParallaxApply } from '../../hybrid/lib/parallaxController'; // optional camera direct wire

async function boot() {
  const canvas = (document.querySelector('canvas') as HTMLCanvasElement) || document.createElement('canvas');

  // Manager controls provider + enablement
  const mgr = new SensorManager(canvas);

  // Gentle prompt (one-time) to upgrade beyond Mouse -> FaceID/WebXR/DOF
  mountHeadTrackingPrompt({
    async onEnable() { await mgr.enable(); return mgr.getState().enabled; },
    storageKey: 'ht_prompt_v1_phone', snoozeDays: 30
  });

  // Settings gear (user-driven, not pesky)
  mountHeadTrackingSettings({ manager: mgr });

  // Optionally auto-enable if user accepted before
  if (mgr.getState().enabled) await mgr.enable();
}

document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', boot) : boot();
