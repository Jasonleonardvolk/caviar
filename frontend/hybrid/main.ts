import { isWebGPUAvailable, isWASMAvailable } from './lib/deviceDetect';
import { handleRenderError } from './lib/errorHandler';

(async () => {
  try {
    const boot = async () => {
      if (isWebGPUAvailable()) {
        const mod = await import('./webgpuRenderer');
        await mod.init();
        return 'webgpu';
      } else if (isWASMAvailable()) {
        const mod = await import('./wasmFallbackRenderer');
        await mod.init();
        // Show Reduced Quality banner
        const evt = new CustomEvent('reducedQuality', { detail: { mode: 'WASM' } });
        window.dispatchEvent(evt);
        return 'wasm';
      } else {
        alert("Your device does not support advanced holographic rendering.");
        return 'none';
      }
    };
    const mode = await boot();
    console.log('[HybridBoot] mode=', mode);
  } catch (err:any) {
    handleRenderError(err);
  }
})();
