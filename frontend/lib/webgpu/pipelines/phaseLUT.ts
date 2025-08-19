// frontend\lib\webgpu\pipelines\phaseLUT.ts
// Pipeline for applying phase correction using a Look-Up Table (LUT)

export interface PhaseLUTParams {
  width: number;
  height: number;
  lutWidth: number;
  lutHeight: number;
  gain?: number;
  maxCorrection?: number;
  scaleX?: number;
  scaleY?: number;
  offsetX?: number;
  offsetY?: number;
}

export class PhaseLUTPipeline {
  private device: GPUDevice;
  private pipeline!: GPUComputePipeline;
  private paramsBuffer: GPUBuffer;
  private lutTexture?: GPUTexture;
  private lutSampler: GPUSampler;

  constructor(device: GPUDevice) {
    this.device = device;
    
    // Create sampler for LUT texture
    this.lutSampler = device.createSampler({
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      minFilter: 'linear',
      magFilter: 'linear',
    });

    // Create params buffer (64 bytes)
    this.paramsBuffer = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  async init(shaderModule: GPUShaderModule) {
    this.pipeline = await this.device.createComputePipelineAsync({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });
  }

  updateParams(params: PhaseLUTParams) {
    const buffer = new ArrayBuffer(64);
    const view = new DataView(buffer);
    
    // Pack params according to struct layout
    view.setUint32(0, params.width, true);
    view.setUint32(4, params.height, true);
    view.setUint32(8, params.lutWidth, true);
    view.setUint32(12, params.lutHeight, true);
    view.setFloat32(16, params.gain ?? 1.0, true);
    view.setFloat32(20, params.maxCorrection ?? Math.PI, true);
    view.setFloat32(24, params.scaleX ?? (params.lutWidth - 1) / (params.width - 1), true);
    view.setFloat32(28, params.scaleY ?? (params.lutHeight - 1) / (params.height - 1), true);
    view.setFloat32(32, params.offsetX ?? 0, true);
    view.setFloat32(36, params.offsetY ?? 0, true);
    // Padding at 40-63

    this.device.queue.writeBuffer(this.paramsBuffer, 0, new Uint8Array(buffer.buffer).buffer);
  }

  setLUTTexture(texture: GPUTexture) {
    this.lutTexture = texture;
  }

  createLUTFromData(data: Float32Array, width: number, height: number): GPUTexture {
    const texture = this.device.createTexture({
      size: [width, height, 1],
      format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.device.queue.writeTexture(
      { texture },
      data.buffer as ArrayBuffer,
      { bytesPerRow: width * 4 },
      [width, height, 1]
    );

    this.lutTexture = texture;
    return texture;
  }

  apply(
    commandEncoder: GPUCommandEncoder,
    reBuf: GPUBuffer,
    imBuf: GPUBuffer,
    width: number,
    height: number
  ) {
    if (!this.lutTexture) {
      throw new Error('LUT texture not set. Call setLUTTexture() or createLUTFromData() first.');
    }

    // Create bind group with correct binding layout
    const bg0 = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: reBuf } },
        { binding: 1, resource: { buffer: imBuf } },
        { binding: 2, resource: { buffer: this.paramsBuffer } },
        { binding: 3, resource: this.lutTexture.createView() },
        { binding: 4, resource: this.lutSampler },
      ],
    });

    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bg0);
    
    // Dispatch with 8x8 workgroups
    const workgroupsX = Math.ceil(width / 8);
    const workgroupsY = Math.ceil(height / 8);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY);
    
    pass.end();
  }
}
