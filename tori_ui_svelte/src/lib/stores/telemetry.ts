import { writable } from 'svelte/store';

export type AssistState = 'local' | 'remote' | 'degraded';

export const telemetry = writable({ 
  frameMsEMA: 0, 
  N: 256, 
  zModes: 12, 
  assist: 'local' as AssistState,
  events: [] as string[] 
});