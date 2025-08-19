// ${IRIS_ROOT}\frontend\lib\webgpu\effects\motionBlend.ts
export async function createMotionBlend(device: GPUDevice) {
  const code = await (await fetch('/shaders/motionBlend.wgsl')).text();
  const module = device.createShaderModule({ code, label: 'motionBlend' });
  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: {} },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: {} },
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
