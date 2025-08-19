// ${IRIS_ROOT}\frontend\lib\webgpu\wave\NullWaveBackend.ts
import type { WaveBackend, WaveRenderParams, WaveBackendConfig } from './WaveBackend';

/**
 * No-op implementation of WaveBackend for production IRIS 1.0
 * This keeps the interface stable while removing all wave compute overhead.
 */
export class NullWaveBackend implements WaveBackend {
  private _enabled = false;
  
  constructor(config?: WaveBackendConfig) {
    // Intentionally empty - no resources allocated
    if (config) {
      // Silently ignore config in null backend
    }
  }
  
  async ready(): Promise<void> {
    // No-op: instantly ready since we don't load any shaders or create buffers
    return Promise.resolve();
  }
  
  renderPattern(params?: WaveRenderParams): void {
    // No-op: no wave processing in production
    // This prevents any GPU compute overhead
    if (params) {
      // Silently ignore params
    }
  }
  
  dispose(): void {
    // No-op: no resources to clean up
  }
  
  isEnabled(): boolean {
    return this._enabled; // Always false for null backend
  }
}

/**
 * Factory function for consistent instantiation
 */
export function createNullWaveBackend(config?: WaveBackendConfig): WaveBackend {
  return new NullWaveBackend(config);
}
