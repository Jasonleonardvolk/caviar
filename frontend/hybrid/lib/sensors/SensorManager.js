import { selectBestSensorProvider } from './selectBestSensorProvider';
import { MouseProvider } from './MouseProvider';
import { ingestInput } from '../parallaxController';
const LS_ENABLED = 'ht_enabled_v1';
const LS_PROVIDER = 'ht_provider_pref_v1';
export class SensorManager {
    constructor(canvas) {
        this.current = null;
        this.enabled = false;
        this.providerPref = 'auto';
        this.listeners = [];
        this.canvas = canvas;
        // restore prefs
        try {
            const e = localStorage.getItem(LS_ENABLED);
            if (e != null)
                this.enabled = e === '1';
            const p = localStorage.getItem(LS_PROVIDER);
            if (p)
                this.providerPref = p;
        }
        catch { }
    }
    onChange(fn) { this.listeners.push(fn); }
    emit(state = {}) {
        const s = this.getState();
        const merged = { ...s, ...state };
        this.listeners.forEach(fn => fn(merged));
    }
    getState() {
        return {
            enabled: this.enabled,
            providerName: this.current?.name ?? 'none',
            providerPref: this.providerPref,
            supported: {
                'auto': true,
                'FaceIDDepth(iOS)': /iPhone|iPad/i.test(navigator.userAgent),
                'WebXR(inline)': !!navigator.xr,
                'DeviceOrientation': 'DeviceOrientationEvent' in window,
                'Mouse': true,
            }
        };
    }
    async enable() {
        if (this.enabled)
            return;
        this.enabled = true;
        localStorage.setItem(LS_ENABLED, '1');
        await this.startProvider();
        this.emit();
    }
    async disable() {
        if (!this.enabled)
            return;
        this.enabled = false;
        localStorage.setItem(LS_ENABLED, '0');
        if (this.current) {
            try {
                this.current.stop();
            }
            catch { }
            this.current = null;
        }
        this.emit();
    }
    async setProviderPref(pref) {
        this.providerPref = pref;
        localStorage.setItem(LS_PROVIDER, pref);
        if (this.enabled)
            await this.startProvider();
        this.emit();
    }
    async startProvider() {
        if (!this.enabled)
            return;
        // stop current
        if (this.current) {
            try {
                this.current.stop();
            }
            catch { }
            this.current = null;
        }
        const provider = await this.resolveProvider();
        this.current = provider;
        await provider.start((pose) => ingestInput(pose));
        window.addEventListener('beforeunload', () => provider.stop(), { once: true });
    }
    async resolveProvider() {
        // Honor forced preference if supported, else fall back to auto chain
        const pref = this.providerPref;
        const byName = {
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
            if (pref !== 'auto' && (await forced.isSupported()))
                return forced;
        }
        catch { /* ignore */ }
        return byName['auto']();
    }
}
