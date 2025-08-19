/// <reference types="@webgpu/types" />

/**
 * Global WebGPU Type Definitions
 * This ensures WebGPU types are available throughout the project
 */

// Augment the global scope with WebGPU types
declare global {
  // Ensure Navigator has gpu property
  interface Navigator {
    readonly gpu: GPU;
  }
  
  interface WorkerNavigator {
    readonly gpu: GPU;
  }

  // Import all WebGPU types into global scope
  type GPU = import('@webgpu/types').GPU;
  type GPUAdapter = import('@webgpu/types').GPUAdapter;
  type GPUDevice = import('@webgpu/types').GPUDevice;
  type GPUBuffer = import('@webgpu/types').GPUBuffer;
  type GPUTexture = import('@webgpu/types').GPUTexture;
  type GPUTextureView = import('@webgpu/types').GPUTextureView;
  type GPUSampler = import('@webgpu/types').GPUSampler;
  type GPUBindGroupLayout = import('@webgpu/types').GPUBindGroupLayout;
  type GPUBindGroup = import('@webgpu/types').GPUBindGroup;
  type GPUPipelineLayout = import('@webgpu/types').GPUPipelineLayout;
  type GPUShaderModule = import('@webgpu/types').GPUShaderModule;
  type GPUComputePipeline = import('@webgpu/types').GPUComputePipeline;
  type GPURenderPipeline = import('@webgpu/types').GPURenderPipeline;
  type GPUCommandEncoder = import('@webgpu/types').GPUCommandEncoder;
  type GPUComputePassEncoder = import('@webgpu/types').GPUComputePassEncoder;
  type GPURenderPassEncoder = import('@webgpu/types').GPURenderPassEncoder;
  type GPURenderBundle = import('@webgpu/types').GPURenderBundle;
  type GPURenderBundleEncoder = import('@webgpu/types').GPURenderBundleEncoder;
  type GPUQueue = import('@webgpu/types').GPUQueue;
  type GPUQuerySet = import('@webgpu/types').GPUQuerySet;
  type GPUCanvasContext = import('@webgpu/types').GPUCanvasContext;
  
  // Ensure the usage flags are available as values
  const GPUBufferUsage: {
    readonly MAP_READ: 0x0001;
    readonly MAP_WRITE: 0x0002;
    readonly COPY_SRC: 0x0004;
    readonly COPY_DST: 0x0008;
    readonly INDEX: 0x0010;
    readonly VERTEX: 0x0020;
    readonly UNIFORM: 0x0040;
    readonly STORAGE: 0x0080;
    readonly INDIRECT: 0x0100;
    readonly QUERY_RESOLVE: 0x0200;
  };

  const GPUTextureUsage: {
    readonly COPY_SRC: 0x01;
    readonly COPY_DST: 0x02;
    readonly TEXTURE_BINDING: 0x04;
    readonly STORAGE_BINDING: 0x08;
    readonly RENDER_ATTACHMENT: 0x10;
  };

  const GPUShaderStage: {
    readonly VERTEX: 0x1;
    readonly FRAGMENT: 0x2;
    readonly COMPUTE: 0x4;
  };

  const GPUMapMode: {
    readonly READ: 0x1;
    readonly WRITE: 0x2;
  };

  const GPUColorWrite: {
    readonly RED: 0x1;
    readonly GREEN: 0x2;
    readonly BLUE: 0x4;
    readonly ALPHA: 0x8;
    readonly ALL: 0xF;
  };
}

export {};
