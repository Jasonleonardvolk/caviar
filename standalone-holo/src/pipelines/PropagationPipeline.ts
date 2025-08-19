// PropagationPipeline.ts - FFT-based wave propagation for holographic rendering
export class PropagationPipeline {
  private device: GPUDevice;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  
  // Cached textures for the field
  private fieldTexture: GPUTexture | null = null;
  private outputTexture: GPUTexture | null = null;
  
  constructor(device: GPUDevice) {
    this.device = device;
  }
  
  async init() {
    // Load the propagation shader
    const shaderCode = await this.loadPropagationShader();
    
    const shaderModule = this.device.createShaderModule({
      code: shaderCode
    });
    
    this.pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'angular_spectrum_propagate'
      }
    });
    
    this.bindGroupLayout = this.pipeline.getBindGroupLayout(0);
  }
  
  private async loadPropagationShader(): Promise<string> {
    // For now, use a simplified version
    // In production, load from frontend/lib/webgpu/shaders/propagation.wgsl
    return `
      struct PropagationParams {
        distance: f32,
        wavelength: f32,
        pixel_size: f32,
        amplitude_scale: f32,
        width: u32,
        height: u32,
        _pad0: u32,
        _pad1: u32,
      }
      
      @group(0) @binding(0) var<uniform> params: PropagationParams;
      @group(0) @binding(1) var<storage, read> field_real: array<f32>;
      @group(0) @binding(2) var<storage, read> field_imag: array<f32>;
      @group(0) @binding(3) var<storage, read_write> output_real: array<f32>;
      @group(0) @binding(4) var<storage, read_write> output_imag: array<f32>;
      
      const PI: f32 = 3.14159265359;
      
      @compute @workgroup_size(8, 8, 1)
      fn angular_spectrum_propagate(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let x = global_id.x;
        let y = global_id.y;
        
        if (x >= params.width || y >= params.height) {
          return;
        }
        
        let idx = y * params.width + x;
        
        // Get input field
        let real = field_real[idx];
        let imag = field_imag[idx];
        
        // Simple propagation (placeholder - should use FFT)
        // For now, just apply a phase shift based on distance
        let k = 2.0 * PI / params.wavelength;
        let phase_shift = k * params.distance;
        
        // Apply phase shift (complex multiplication by exp(i*phase))
        let cos_phase = cos(phase_shift);
        let sin_phase = sin(phase_shift);
        
        output_real[idx] = real * cos_phase - imag * sin_phase;
        output_imag[idx] = real * sin_phase + imag * cos_phase;
      }
    `;
  }
  
  async propagate(
    ampField: Float32Array,
    phiField: Float32Array,
    width: number,
    height: number,
    distance: number = 100.0,
    wavelength: number = 0.000532 // 532nm green laser
  ): Promise<{ real: Float32Array, imag: Float32Array }> {
    if (!this.pipeline) {
      await this.init();
    }
    
    const size = width * height;
    
    // Convert amp/phi to complex field (real + imaginary)
    const fieldReal = new Float32Array(size);
    const fieldImag = new Float32Array(size);
    
    for (let i = 0; i < size; i++) {
      const amp = ampField[i];
      const phi = phiField[i];
      fieldReal[i] = amp * Math.cos(phi);
      fieldImag[i] = amp * Math.sin(phi);
    }
    
    // Create buffers
    const paramsBuffer = this.device.createBuffer({
      size: 32, // 8 * 4 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    const fieldRealBuffer = this.device.createBuffer({
      size: fieldReal.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    const fieldImagBuffer = this.device.createBuffer({
      size: fieldImag.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    const outputRealBuffer = this.device.createBuffer({
      size: fieldReal.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    
    const outputImagBuffer = this.device.createBuffer({
      size: fieldImag.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    
    // Write data
    const params = new ArrayBuffer(32);
    const view = new DataView(params);
    view.setFloat32(0, distance, true);
    view.setFloat32(4, wavelength, true);
    view.setFloat32(8, 1.0, true); // pixel_size
    view.setFloat32(12, 1.0, true); // amplitude_scale
    view.setUint32(16, width, true);
    view.setUint32(20, height, true);
    
    this.device.queue.writeBuffer(paramsBuffer, 0, params);
    this.device.queue.writeBuffer(fieldRealBuffer, 0, fieldReal);
    this.device.queue.writeBuffer(fieldImagBuffer, 0, fieldImag);
    
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout!,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: fieldRealBuffer } },
        { binding: 2, resource: { buffer: fieldImagBuffer } },
        { binding: 3, resource: { buffer: outputRealBuffer } },
        { binding: 4, resource: { buffer: outputImagBuffer } }
      ]
    });
    
    // Run compute
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.pipeline!);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(
      Math.ceil(width / 8),
      Math.ceil(height / 8),
      1
    );
    passEncoder.end();
    
    // Read back results
    const readRealBuffer = this.device.createBuffer({
      size: fieldReal.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    
    const readImagBuffer = this.device.createBuffer({
      size: fieldImag.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    
    commandEncoder.copyBufferToBuffer(outputRealBuffer, 0, readRealBuffer, 0, fieldReal.byteLength);
    commandEncoder.copyBufferToBuffer(outputImagBuffer, 0, readImagBuffer, 0, fieldImag.byteLength);
    
    this.device.queue.submit([commandEncoder.finish()]);
    
    // Map and read
    await readRealBuffer.mapAsync(GPUMapMode.READ);
    await readImagBuffer.mapAsync(GPUMapMode.READ);
    
    const real = new Float32Array(readRealBuffer.getMappedRange());
    const imag = new Float32Array(readImagBuffer.getMappedRange());
    
    const resultReal = new Float32Array(real);
    const resultImag = new Float32Array(imag);
    
    readRealBuffer.unmap();
    readImagBuffer.unmap();
    
    return { real: resultReal, imag: resultImag };
  }
  
  // Convert complex field back to amp/phi
  complexToAmpPhi(real: Float32Array, imag: Float32Array): { amp: Float32Array, phi: Float32Array } {
    const size = real.length;
    const amp = new Float32Array(size);
    const phi = new Float32Array(size);
    
    for (let i = 0; i < size; i++) {
      amp[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
      phi[i] = Math.atan2(imag[i], real[i]);
    }
    
    return { amp, phi };
  }
}