/**
 * Type definitions for renderer configurations
 */

export interface RendererConfig {
  views: number;
  resolution: [number, number];
  samples: number;
  backend: 'webgpu' | 'wasm' | 'webgl2' | 'webgl' | 'cpu';
}

export interface BaseRenderer {
  config: RendererConfig;
  initialize(): Promise<void>;
  render(): void;
  destroy(): void;
}
