// ${IRIS_ROOT}\frontend\lib\webgpu\pipelines\jointBilateralUpsample.ts
export type JbuPipelines = {
  module: GPUShaderModule;
  pipeline: GPUComputePipeline;
  bgl: GPUBindGroupLayout;
};

export async function createJBU(device: GPUDevice): Promise<JbuPipelines> {
  const code = await (await fetch('/shaders/jointBilateralUpsample.wgsl')).text();
  const module = device.createShaderModule({ code, label: 'jbu' });
  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ]
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    compute: { module, entryPoint: 'main' }
  });
  return { module, pipeline, bgl };
}
