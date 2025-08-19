// D:\Dev\kha\frontend\types\renderer.d.ts
// Minimal public surface used by callers. Expand later if needed.

export type RenderBackend = 'webgpu' | 'webgl2' | 'cpu';

export interface RendererOptions {
  backend?: RenderBackend;
  enableFoveation?: boolean;
  enableMultiView?: boolean;
  enablePhaseLUT?: boolean;
}

export interface AdapterInfo {
  vendor?: string;
  architecture?: string;
  device?: string;
  // Raw WebGPU
  wgAdapterName?: string;
  wgVendorId?: number;
  wgArchitecture?: string;
}

export interface RendererCaps {
  supportsWebGPU: boolean;
  supportsTimestampQuery: boolean;
  maxWorkgroupInvocations: number;
  maxStorageBufferBindingSize: number;
}

export interface Renderer {
  readonly canvas: HTMLCanvasElement;
  readonly backend: RenderBackend;
  readonly caps: RendererCaps;

  init(canvas: HTMLCanvasElement, opts?: Partial<RendererOptions>): Promise<void>;
  render(deltaMs?: number): void;
  resize(width: number, height: number): void;
  dispose(): void;
}

// Additional exports that were missing
export type InitOpts = RendererOptions;
export type RendererHandle = Renderer;
