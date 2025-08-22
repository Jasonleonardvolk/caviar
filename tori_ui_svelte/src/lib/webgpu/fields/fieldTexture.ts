export function createComplexFieldTextures(device: GPUDevice, w: number, h: number) {
  const fmt = 'rgba32float'; // or 'rgba16float' if you validate precision
  const src = device.createTexture({ size: {width:w, height:h}, format: fmt, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const dst = device.createTexture({ size: {width:w, height:h}, format: fmt, usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
  return { src, dst };
}