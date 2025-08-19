/// <reference lib="dom" />
/// <reference lib="dom.iterable" />
/// <reference types="vite/client" />
/// <reference types="@webgpu/types" />

export {};

declare global {
  interface Navigator {
    gpu: GPU;
  }
  interface WorkerNavigator {
    gpu: GPU;
  }
  interface ImportMetaEnv {
    VITE_MODE?: string;
    VITE_API_BASE?: string;
  }
  
  // Ensure WebGPU types are globally available
  const GPU: GPU;
  const GPUAdapter: GPUAdapter;
  const GPUDevice: GPUDevice;
  const GPUBuffer: GPUBuffer;
  const GPUTexture: GPUTexture;
  const GPUTextureFormat: GPUTextureFormat;
  const GPUBufferUsage: typeof GPUBufferUsage;
  const GPUTextureUsage: typeof GPUTextureUsage;
  const GPUShaderStage: typeof GPUShaderStage;
  const GPUCanvasContext: GPUCanvasContext;
  const GPURenderPipeline: GPURenderPipeline;
  const GPUComputePipeline: GPUComputePipeline;
  const GPUBindGroup: GPUBindGroup;
  const GPUQuerySet: GPUQuerySet;
}
