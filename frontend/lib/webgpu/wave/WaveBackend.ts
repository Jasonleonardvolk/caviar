// ${IRIS_ROOT}\frontend\lib\webgpu\wave\WaveBackend.ts
/**
 * Stable interface for wave-based holographic rendering backends.
 * In IRIS 1.0, this is stubbed out with NullWaveBackend.
 * Labs/future builds can swap in FftWaveBackend or other implementations.
 */
export interface WaveBackend {
  /**
   * Initialize the backend (load shaders, create buffers, etc.)
   */
  ready(): Promise<void>;
  
  /**
   * Render a holographic pattern (FFT/Angular Spectrum/GS)
   * In production, this is a no-op. In Labs, it computes wave propagation.
   */
  renderPattern(params?: WaveRenderParams): void;
  
  /**
   * Cleanup resources
   */
  dispose?(): void;
  
  /**
   * Check if wave processing is actually enabled
   */
  isEnabled(): boolean;
}

export interface WaveRenderParams {
  wavelength?: number;      // nm, default 550 (green)
  propagationDistance?: number; // meters
  iterations?: number;      // GS iterations
  target?: GPUTexture;      // output texture
  timestamp?: number;       // for animation
}

export type WaveBackendConfig = {
  device?: GPUDevice;
  format?: GPUTextureFormat;
  resolution?: [number, number];
  enableFFT?: boolean;
  enableGS?: boolean;
  enablePropagation?: boolean;
}
