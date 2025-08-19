// ${IRIS_ROOT}\frontend\hybrid\lib\sensors\SensorManager.ts
import type { ISensorProvider } from './ISensorProvider';
import { selectBestSensorProvider } from './selectBestSensorProvider';
import { MouseProvider } from './MouseProvider';
import type { Pose } from '../tracking/types';
import { ingestInput } from '../parallaxController';

type ProviderPref = 'auto' | 'FaceIDDepth(iOS)' | 'WebXR(inline)' | 'DeviceOrientation' | 'Mouse';

export type SensorState = {
  enabled: boolean;
  providerName: string;
  providerPref: ProviderPref;
  supported: Record<ProviderPref, boolean>;
};

const LS_ENABLED = 'ht_enabled_v1';
const LS_PROVIDER = 'ht_provider_pref_v1';

export class SensorManager {
  private canvas: HTMLCanvasElement;
  private current: ISensorProvider | null = null;
  private enabled = false;
  private providerPref: ProviderPref = 'auto';
  private listeners: Array<(s: SensorState) => void> = [];

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    // restore prefs
    try {
      const e = localStorage.getItem(LS_ENABLED);
      if (e != null) this.enabled = e === '1';
      const p = localStorage.getItem(LS_PROVIDER) as ProviderPref | null;
      if (p) this.providerPref = p;
    } catch {}
  }

  onChange(fn: (s: SensorState) => void) { this.listeners.push(fn); }
  private emit(state: Partial<SensorState> = {}) {
    const s = this.getState();
    const merged = { ...s, ...state };
    this.listeners.forEach(fn => fn(merged));
  }

  getState(): SensorState {
    return {
      enabled: this.enabled,
      providerName: this.current?.name ?? 'none',
      providerPref: this.providerPref,
      supported: {
        'auto': true,
        'FaceIDDepth(iOS)': /iPhone|iPad/i.test(navigator.userAgent),
        'WebXR(inline)': !!(navigator as any).xr,
        'DeviceOrientation': 'DeviceOrientationEvent' in window,
        'Mouse': true,
      }
    };
  }

  async enable(): Promise<void> {
    if (this.enabled) return;
    this.enabled = true;
    localStorage.setItem(LS_ENABLED, '1');
    await this.startProvider();
    this.emit();
  }

  async disable(): Promise<void> {
    if (!this.enabled) return;
    this.enabled = false;
    localStorage.setItem(LS_ENABLED, '0');
    if (this.current) { try { this.current.stop(); } catch {} this.current = null; }
    this.emit();
  }

  async setProviderPref(pref: ProviderPref): Promise<void> {
    this.providerPref = pref;
    localStorage.setItem(LS_PROVIDER, pref);
    if (this.enabled) await this.startProvider();
    this.emit();
  }

  async startProvider(): Promise<void> {
    if (!this.enabled) return;
    // stop current
    if (this.current) { try { this.current.stop(); } catch {} this.current = null; }

    const provider = await this.resolveProvider();
    this.current = provider;
    await provider.start((pose: Pose) => ingestInput(pose));
    window.addEventListener('beforeunload', () => provider.stop(), { once: true });
  }

  private async resolveProvider(): Promise<ISensorProvider> {
    // Honor forced preference if supported, else fall back to auto chain
    const pref = this.providerPref;
    const byName: Record<ProviderPref, () => Promise<ISensorProvider>> = {
      'FaceIDDepth(iOS)': async () => {
        const { FaceIdDepthProvider } = await import('./face/FaceIdDepthProvider.ios');
        return new FaceIdDepthProvider();
      },
      'WebXR(inline)': async () => {
        const { WebXRHeadProvider } = await import('./WebXRHeadProvider');
        return new WebXRHeadProvider();
      },
      'DeviceOrientation': async () => {
        const { DeviceOrientationProvider } = await import('./DeviceOrientationProvider');
        return new DeviceOrientationProvider();
      },
      'Mouse': async () => new MouseProvider(this.canvas),
      'auto': async () => selectBestSensorProvider(this.canvas),
    };

    const forced = await byName[pref]();
    try {
      if (pref !== 'auto' && (await forced.isSupported())) return forced;
    } catch { /* ignore */ }
    return byName['auto']();
  }
}
