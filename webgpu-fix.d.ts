/// <reference types="@webgpu/types" />

interface Navigator {
  readonly gpu: GPU;
}

interface GPUBufferUsage {
  readonly MAP_READ: number;
  readonly MAP_WRITE: number;
  readonly COPY_SRC: number;
  readonly COPY_DST: number;
  readonly INDEX: number;
  readonly VERTEX: number;
  readonly UNIFORM: number;
  readonly STORAGE: number;
  readonly INDIRECT: number;
  readonly QUERY_RESOLVE: number;
}

interface GPUTextureUsage {
  readonly COPY_SRC: number;
  readonly COPY_DST: number;
  readonly TEXTURE_BINDING: number;
  readonly STORAGE_BINDING: number;
  readonly RENDER_ATTACHMENT: number;
}

interface GPUShaderStage {
  readonly VERTEX: number;
  readonly FRAGMENT: number;
  readonly COMPUTE: number;
}

interface GPUColorWrite {
  readonly RED: number;
  readonly GREEN: number;
  readonly BLUE: number;
  readonly ALPHA: number;
  readonly ALL: number;
}

interface GPUMapMode {
  readonly READ: number;
  readonly WRITE: number;
}

declare const GPUBufferUsage: GPUBufferUsage;
declare const GPUTextureUsage: GPUTextureUsage;
declare const GPUShaderStage: GPUShaderStage;
declare const GPUColorWrite: GPUColorWrite;
declare const GPUMapMode: GPUMapMode;
