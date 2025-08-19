// ${IRIS_ROOT}\frontend\apps\pc_donor\src\main.ts
import { SensorManager } from '../../hybrid/lib/sensors/SensorManager';
import { mountHeadTrackingSettings } from '../../hybrid/ui/HeadTrackingSettings';
import { mountHeadTrackingPrompt } from '../../hybrid/ui/HeadTrackingPrompt';

async function boot() {
  const canvas = (document.querySelector('canvas') as HTMLCanvasElement) || document.createElement('canvas');
  const mgr = new SensorManager(canvas);

  mountHeadTrackingPrompt({
    async onEnable() { await mgr.enable(); return mgr.getState().enabled; },
    storageKey: 'ht_prompt_v1_pc', snoozeDays: 30
  });

  mountHeadTrackingSettings({ manager: mgr });
  if (mgr.getState().enabled) await mgr.enable();
}

document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', boot) : boot();
