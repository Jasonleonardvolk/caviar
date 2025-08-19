// ${IRIS_ROOT}\frontend\lib\webgpu\pipelines\viewBlendOcclusion.ts
export type ViewBlendPipelines = {
  module: GPUShaderModule;
  pipeline: GPUComputePipeline;
  bgl: GPUBindGroupLayout;
};

export async function createViewBlendOcclusion(device: GPUDevice): Promise<ViewBlendPipelines> {
  const code = await (await fetch('/shaders/viewBlendOcclusion.wgsl')).text();
  const module = device.createShaderModule({ code, label: 'view-blend-occlusion' });
  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { viewDimension: '2d-array' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ]
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    compute: { module, entryPoint: 'main' }
  });
  return { module, pipeline, bgl };
}
