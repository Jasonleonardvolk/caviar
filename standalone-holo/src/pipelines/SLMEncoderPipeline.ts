// SLMEncoderPipeline.ts - Encode complex fields to SLM patterns
import { phase_only_encode_wgsl, lee_offaxis_encode_wgsl } from '../shaderSources';

export type EncodingMode = 'phase_only' | 'lee_offaxis';

export class SLMEncoderPipeline {
  private device: GPUDevice;
  private phaseOnlyPipeline: GPUComputePipeline | null = null;
  private leeOffaxisPipeline: GPUComputePipeline | null = null;
  
  constructor(device: GPUDevice) {
    this.device = device;
  }
  
  async init() {
    // Create phase-only pipeline
    const phaseModule = this.device.createShaderModule({
      code: phase_only_encode_wgsl
    });
    
    this.phaseOnlyPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: phaseModule,
        entryPoint: 'main'
      }
    });
    
    // Create Lee off-axis pipeline
    const leeModule = this.device.createShaderModule({
      code: lee_offaxis_encode_wgsl
    });
    
    this.leeOffaxisPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: leeModule,
        entryPoint: 'main'
      }
    });
  }
  
  async encodeToTexture(
    amp: Float32Array,
    phi: Float32Array,
    width: number,
    height: number,
    mode: EncodingMode = 'phase_only'
  ): Promise<GPUTexture> {
    if (!this.phaseOnlyPipeline || !this.leeOffaxisPipeline) {
      await this.init();
    }
    
    if (mode === 'phase_only') {
      return this.encodePhaseOnly(amp, phi, width, height);
    } else {
      return this.encodeLeeOffaxis(amp, phi, width, height);
    }
  }
  
  private async encodePhaseOnly(
    amp: Float32Array,
    phi: Float32Array,
    width: number,
    height: number
  ): Promise<GPUTexture> {
    const pipeline = this.phaseOnlyPipeline!;
    const size = width * height;
    
    // Output will be double width for phase-only encoding
    const outputWidth = width * 2;
    const outputHeight = height;
    const outputSize = outputWidth * outputHeight;
    
    // Create buffers
    const ampBuffer = this.device.createBuffer({
      size: amp.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    const phiBuffer = this.device.createBuffer({
      size: phi.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    const phaseOutBuffer = this.device.createBuffer({
      size: outputSize * 4, // Float32
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    
    const paramsBuffer = this.device.createBuffer({
      size: 16, // 4 u32 values
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    // Write data
    this.device.queue.writeBuffer(ampBuffer, 0, amp.slice());
    this.device.queue.writeBuffer(phiBuffer, 0, phi.slice());
    
    const params = new Uint32Array([width, height, 0, 0]);
    this.device.queue.writeBuffer(paramsBuffer, 0, params.slice());
    
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: ampBuffer } },
        { binding: 1, resource: { buffer: phiBuffer } },
        { binding: 2, resource: { buffer: phaseOutBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } }
      ]
    });
    
    // Run compute
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(
      Math.ceil(width / 16),
      Math.ceil(height / 16),
      1
    );
    passEncoder.end();
    
    // Create output texture
    const texture = this.device.createTexture({
      size: [outputWidth, outputHeight],
      format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING
    });
    
    // Copy buffer to texture
    commandEncoder.copyBufferToTexture(
      {
        buffer: phaseOutBuffer,
        bytesPerRow: outputWidth * 4,
        rowsPerImage: outputHeight
      },
      { texture },
      [outputWidth, outputHeight]
    );
    
    this.device.queue.submit([commandEncoder.finish()]);
    
    return texture;
  }
  
  private async encodeLeeOffaxis(
    amp: Float32Array,
    phi: Float32Array,
    width: number,
    height: number
  ): Promise<GPUTexture> {
    const pipeline = this.leeOffaxisPipeline!;
    const size = width * height;
    
    // Create buffers
    const ampBuffer = this.device.createBuffer({
      size: amp.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    const phiBuffer = this.device.createBuffer({
      size: phi.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    const holoOutBuffer = this.device.createBuffer({
      size: size * 4, // Float32
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    
    const paramsBuffer = this.device.createBuffer({
      size: 32, // 8 values (2 u32, 2 u32 padding, 4 f32)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    // Write data
    this.device.queue.writeBuffer(ampBuffer, 0, amp.slice());
    this.device.queue.writeBuffer(phiBuffer, 0, phi.slice());
    
    // Set Lee hologram parameters
    const params = new ArrayBuffer(32);
    const view = new DataView(params);
    view.setUint32(0, width, true);
    view.setUint32(4, height, true);
    view.setUint32(8, 0, true); // binary: 0 for grayscale
    view.setUint32(12, 0, true); // padding
    view.setFloat32(16, 10.0, true); // fx_cycles
    view.setFloat32(20, 10.0, true); // fy_cycles
    view.setFloat32(24, 0.5, true); // bias
    view.setFloat32(28, 0.5, true); // scale
    
    this.device.queue.writeBuffer(paramsBuffer, 0, params.slice());
    
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: ampBuffer } },
        { binding: 1, resource: { buffer: phiBuffer } },
        { binding: 2, resource: { buffer: holoOutBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } }
      ]
    });
    
    // Run compute
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(
      Math.ceil(width / 16),
      Math.ceil(height / 16),
      1
    );
    passEncoder.end();
    
    // Create output texture
    const texture = this.device.createTexture({
      size: [width, height],
      format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING
    });
    
    // Copy buffer to texture
    commandEncoder.copyBufferToTexture(
      {
        buffer: holoOutBuffer,
        bytesPerRow: width * 4,
        rowsPerImage: height
      },
      { texture },
      [width, height]
    );
    
    this.device.queue.submit([commandEncoder.finish()]);
    
    return texture;
  }
  
  // Convert phase texture to RGBA for display
  async phaseToRGBA(phaseTexture: GPUTexture): Promise<GPUTexture> {
    const [width, height] = [phaseTexture.width, phaseTexture.height];
    
    const rgbaTexture = this.device.createTexture({
      size: [width, height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
    });
    
    // Simple shader to convert phase to color
    const shaderCode = `
      @group(0) @binding(0) var phase_tex: texture_2d<f32>;
      @group(0) @binding(1) var phase_sampler: sampler;
      
      struct VertexOutput {
        @builtin(position) pos: vec4<f32>,
        @location(0) uv: vec2<f32>,
      }
      
      @vertex
      fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
        var output: VertexOutput;
        // Full screen triangle
        let x = f32(i32(vertex_index) - 1);
        let y = f32(i32(vertex_index & 1u) * 2 - 1);
        output.pos = vec4<f32>(x, y, 0.0, 1.0);
        output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
        return output;
      }
      
      const PI: f32 = 3.14159265359;
      
      @fragment
      fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
        let phase = textureSample(phase_tex, phase_sampler, input.uv).r;
        
        // Map phase to hue (HSV to RGB)
        let hue = (phase / (2.0 * PI)) * 360.0;
        let h = hue / 60.0;
        let c = 1.0;
        let x = c * (1.0 - abs((h % 2.0) - 1.0));
        
        var rgb: vec3<f32>;
        if (h < 1.0) {
          rgb = vec3<f32>(c, x, 0.0);
        } else if (h < 2.0) {
          rgb = vec3<f32>(x, c, 0.0);
        } else if (h < 3.0) {
          rgb = vec3<f32>(0.0, c, x);
        } else if (h < 4.0) {
          rgb = vec3<f32>(0.0, x, c);
        } else if (h < 5.0) {
          rgb = vec3<f32>(x, 0.0, c);
        } else {
          rgb = vec3<f32>(c, 0.0, x);
        }
        
        return vec4<f32>(rgb, 1.0);
      }
    `;
    
    const module = this.device.createShaderModule({ code: shaderCode });
    
    const pipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module,
        entryPoint: 'vs_main'
      },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [{ format: 'rgba8unorm' }]
      },
      primitive: { topology: 'triangle-list' }
    });
    
    const sampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear'
    });
    
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: phaseTexture.createView() },
        { binding: 1, resource: sampler }
      ]
    });
    
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: rgbaTexture.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store'
      }]
    });
    
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.draw(3);
    passEncoder.end();
    
    this.device.queue.submit([commandEncoder.finish()]);
    
    return rgbaTexture;
  }
}