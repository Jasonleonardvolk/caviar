// ${IRIS_ROOT}\frontend\lib\webgpu\engine.ts
/**
 * Main WebGPU engine with conditional wave processing support
 * IRIS 1.0 ships with wave processing disabled by default
 */

import type { WaveBackend } from './wave/WaveBackend';

// Declare the build-time flag
declare const __IRIS_WAVE__: boolean | undefined;

export interface EngineConfig {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  format?: GPUTextureFormat;
  enableWaveProcessing?: boolean; // Runtime override
}

export class WebGPUEngine {
  protected device: GPUDevice;
  protected canvas: HTMLCanvasElement;
  private context: GPUCanvasContext;
  private wave: WaveBackend | null = null;
  private format: GPUTextureFormat;
  
  constructor(private config: EngineConfig) {
    this.device = config.device;
    this.canvas = config.canvas;
    this.format = config.format || 'bgra8unorm';
    
    // Setup canvas context
    const ctx = config.canvas.getContext('webgpu');
    if (!ctx) throw new Error('Failed to get WebGPU context');
    this.context = ctx;
    
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'premultiplied',
    });
  }
  
  /**
   * Initialize the engine with optional wave processing
   */
  async initialize(): Promise<void> {
    // Conditional wave backend loading based on build flag and runtime config
    const waveEnabled = (typeof __IRIS_WAVE__ !== 'undefined' ? __IRIS_WAVE__ : false) 
                        && (this.config.enableWaveProcessing !== false);
    
    if (waveEnabled) {
      console.log('[WebGPUEngine] Wave processing enabled - loading FFT backend');
      try {
        // Dynamic import for tree-shaking when disabled
        const { FftWaveBackend } = await import('./wave/FftWaveBackend');
        this.wave = new FftWaveBackend({
          device: this.device,
          format: this.format,
          resolution: [this.config.canvas.width, this.config.canvas.height]
        });
        await this.wave.ready();
        console.log('[WebGPUEngine] FFT wave backend initialized');
      } catch (error) {
        console.warn('[WebGPUEngine] Failed to load FFT backend, falling back to null backend:', error);
        const { NullWaveBackend } = await import('./wave/NullWaveBackend');
        this.wave = new NullWaveBackend();
      }
    } else {
      console.log('[WebGPUEngine] Wave processing disabled - using null backend');
      const { NullWaveBackend } = await import('./wave/NullWaveBackend');
      this.wave = new NullWaveBackend();
    }
  }
  
  /**
   * Render a frame with optional wave processing
   */
  renderFrame(timestamp: number): void {
    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();
    
    // Render wave pattern if backend is available
    if (this.wave) {
      this.wave.renderPattern({
        timestamp,
        target: this.context.getCurrentTexture()
      });
    }
    
    // Regular rendering pipeline continues here
    // ... your existing render code ...
    
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear' as GPULoadOp,
        storeOp: 'store' as GPUStoreOp,
      }]
    });
    
    // ... render your scene ...
    
    renderPass.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }
  
  /**
   * Render method for compatibility
   */
  render(deltaMs?: number): void {
    this.renderFrame(performance.now());
  }
  
  /**
   * Check if wave processing is enabled
   */
  isWaveProcessingEnabled(): boolean {
    return this.wave?.isEnabled() || false;
  }
  
  /**
   * Clean up resources
   */
  dispose(): void {
    if (this.wave) {
      this.wave.dispose?.();
      this.wave = null;
    }
    this.context.unconfigure();
  }
}

/**
 * Factory function to create engine with proper initialization
 */
export async function createWebGPUEngine(config: EngineConfig): Promise<WebGPUEngine> {
  const engine = new WebGPUEngine(config);
  await engine.initialize();
  return engine;
}

/**
 * Build WebGPU renderer (alias for compatibility)
 */
export async function buildWebGPU(opts: any): Promise<any> {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error('WebGPU not supported');
  
  const device = await adapter.requestDevice();
  
  const engine = await createWebGPUEngine({
    device,
    canvas: opts.canvas,
    format: 'bgra8unorm',
    enableWaveProcessing: opts.enableWaveProcessing
  });
  
  return {
    type: 'webgpu',
    engine,
    device,
    render: (scene: any) => engine.render(),
    dispose: () => engine.dispose(),
    setQuality: () => {},
    getCapabilities: () => ({
      maxTextureSize: 8192,
      maxTextureLayers: 256,
      supportsCompute: true,
      supportsTimestampQuery: device.features.has('timestamp-query'),
      supportsSubgroups: false,
      supportsF16: device.features.has('shader-f16'),
      supportedFeatures: device.features
    })
  };
}
