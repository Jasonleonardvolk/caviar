// ${IRIS_ROOT}\frontend\lib\webgpu\pipelines\parallaxComposite.ts
import type { Caps } from '../caps';

export type CompositePipelines = {
  module: GPUShaderModule;
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
};

export async function createParallaxComposite(device: GPUDevice, caps: Caps): Promise<CompositePipelines> {
  const code = caps.subgroups
    ? await (await fetch('/shaders/parallaxComposite.subgroup.wgsl')).text()
    : await (await fetch('/shaders/parallaxComposite.basic.wgsl')).text();

  const module = device.createShaderModule({ code, label: caps.subgroups ? 'composite-subgroup' : 'composite-basic' });

  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { viewDimension: '2d-array' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ]
  });

  const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const pipeline = device.createComputePipeline({
    layout: pl,
    compute: { module, entryPoint: 'main' }
  });

  return { module, pipeline, bindGroupLayout: bgl };
}
