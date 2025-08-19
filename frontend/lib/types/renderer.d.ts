// Core renderer type definitions

export interface InitOpts {
  canvas: HTMLCanvasElement;
  mode?: 'webgpu' | 'webgl' | 'cpu' | 'auto';
  preferredDevice?: 'discrete' | 'integrated' | 'cpu';
  enableProfiling?: boolean;
  enableDebug?: boolean;
  maxTextureSize?: number;
  powerPreference?: 'high-performance' | 'low-power';
  viewMode?: 'mono' | 'stereo' | 'holographic';
  quality?: 'low' | 'medium' | 'high' | 'ultra';
}

export interface RendererHandle {
  type: 'webgpu' | 'webgl' | 'cpu' | 'hybrid';
  render(scene: any): Promise<void>;
  dispose(): void;
  setQuality(quality: 'low' | 'medium' | 'high' | 'ultra'): void;
  getCapabilities(): RendererCapabilities;
  device?: GPUDevice;
  context?: GPUCanvasContext | WebGLRenderingContext;
}

export interface RendererCapabilities {
  maxTextureSize: number;
  maxTextureLayers: number;
  supportsCompute: boolean;
  supportsTimestampQuery: boolean;
  supportsSubgroups: boolean;
  supportsF16: boolean;
  supportedFeatures: Set<string>;
}

export interface HolographicCapability {
  supported: boolean;
  maxViews: number;
  maxResolution: [number, number];
  features: string[];
}

// Export commonly used types
export type RenderMode = 'webgpu' | 'webgl' | 'cpu' | 'auto';
export type QualityLevel = 'low' | 'medium' | 'high' | 'ultra';
export type ViewMode = 'mono' | 'stereo' | 'holographic';
