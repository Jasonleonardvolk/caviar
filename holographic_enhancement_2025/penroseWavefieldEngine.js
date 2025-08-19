/**
 * Penrose Wavefield Engine - Alternative Holographic Rendering Mode
 * Implements the beloved Penrose algorithm in WebGPU with CPU fallback
 */

// Penrose wavefield generation shader
export const penroseWavefieldShader = `
struct PenroseParams {
  iterations: u32,
  convergence_threshold: f32,
  relaxation_factor: f32,
  quality_mode: u32, // 0: draft, 1: normal, 2: high
}

struct OscillatorState {
  phases: array<f32, 32>,
  frequencies: array<f32, 32>,
  amplitudes: array<f32, 32>,
  coupling_matrix: array<f32, 1024>, // 32x32 coupling
}

@group(0) @binding(0) var<uniform> params: PenroseParams;
@group(0) @binding(1) var<storage, read> oscillators: OscillatorState;
@group(0) @binding(2) var<storage, read_write> wavefield: array<vec2<f32>>; // complex field
@group(0) @binding(3) var<storage, read_write> convergence_map: array<f32>;
@group(0) @binding(4) var<uniform> hologram_size: vec2<u32>;

// Penrose kernel function - iterative wave equation solver
fn penrose_kernel(pos: vec2<f32>, iteration: u32) -> vec2<f32> {
  var result = vec2<f32>(0.0, 0.0);
  let k = 2.0 * 3.14159265359 / 0.000633; // wave number for red light
  
  // Sum contributions from all oscillators
  for (var i = 0u; i < 32u; i++) {
    let phase = oscillators.phases[i];
    let freq = oscillators.frequencies[i];
    let amp = oscillators.amplitudes[i];
    
    // Calculate distance from oscillator position (mapped to wavefield)
    let osc_pos = vec2<f32>(
      cos(f32(i) * 3.14159265359 / 16.0) * 0.4 + 0.5,
      sin(f32(i) * 3.14159265359 / 16.0) * 0.4 + 0.5
    );
    
    let dist = length(pos - osc_pos);
    
    // Penrose wave contribution with coupling
    var wave_contrib = amp * exp(-dist * 0.1); // Amplitude decay
    let phase_shift = k * dist + phase + freq * f32(iteration) * 0.01;
    
    // Apply coupling matrix
    for (var j = 0u; j < 32u; j++) {
      let coupling = oscillators.coupling_matrix[i * 32u + j];
      wave_contrib *= 1.0 + coupling * oscillators.amplitudes[j] * 0.1;
    }
    
    result += wave_contrib * vec2<f32>(cos(phase_shift), sin(phase_shift));
  }
  
  // Apply relaxation for convergence
  return result * params.relaxation_factor;
}

// Quality-aware sampling
fn get_sample_count(quality: u32) -> u32 {
  switch (quality) {
    case 0u: { return 1u; }      // Draft
    case 1u: { return 4u; }      // Normal
    case 2u: { return 16u; }     // High
    default: { return 4u; }
  }
}

@compute @workgroup_size(16, 16)
fn penrose_wavefield_generation(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let size = vec2<f32>(f32(hologram_size.x), f32(hologram_size.y));
  let pos = vec2<f32>(f32(global_id.x), f32(global_id.y)) / size;
  let idx = global_id.y * hologram_size.x + global_id.x;
  
  if (global_id.x >= hologram_size.x || global_id.y >= hologram_size.y) {
    return;
  }
  
  // Initialize or get current wavefield value
  var current_value = wavefield[idx];
  var new_value = vec2<f32>(0.0, 0.0);
  
  // Quality-based supersampling
  let samples = get_sample_count(params.quality_mode);
  let sample_offset = 1.0 / (f32(samples) * size.x);
  
  for (var sx = 0u; sx < samples; sx++) {
    for (var sy = 0u; sy < samples; sy++) {
      let sample_pos = pos + vec2<f32>(
        f32(sx) * sample_offset,
        f32(sy) * sample_offset
      );
      new_value += penrose_kernel(sample_pos, params.iterations);
    }
  }
  
  new_value /= f32(samples * samples);
  
  // Check convergence
  let diff = length(new_value - current_value);
  convergence_map[idx] = diff;
  
  // Update wavefield with convergence check
  if (diff > params.convergence_threshold || params.iterations < 10u) {
    wavefield[idx] = mix(current_value, new_value, 0.8);
  }
}

// Penrose-specific phase retrieval
@compute @workgroup_size(16, 16)
fn penrose_phase_retrieval(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let idx = global_id.y * hologram_size.x + global_id.x;
  
  if (global_id.x >= hologram_size.x || global_id.y >= hologram_size.y) {
    return;
  }
  
  let field = wavefield[idx];
  let amplitude = length(field);
  let phase = atan2(field.y, field.x);
  
  // Penrose phase unwrapping
  let size = vec2<i32>(i32(hologram_size.x), i32(hologram_size.y));
  let pos = vec2<i32>(i32(global_id.x), i32(global_id.y));
  
  var unwrapped_phase = phase;
  
  // Check neighbors for phase jumps
  for (var dx = -1; dx <= 1; dx++) {
    for (var dy = -1; dy <= 1; dy++) {
      if (dx == 0 && dy == 0) { continue; }
      
      let neighbor_pos = pos + vec2<i32>(dx, dy);
      if (neighbor_pos.x >= 0 && neighbor_pos.x < size.x &&
          neighbor_pos.y >= 0 && neighbor_pos.y < size.y) {
        
        let neighbor_idx = u32(neighbor_pos.y * size.x + neighbor_pos.x);
        let neighbor_field = wavefield[neighbor_idx];
        let neighbor_phase = atan2(neighbor_field.y, neighbor_field.x);
        
        let phase_diff = neighbor_phase - phase;
        if (abs(phase_diff) > 3.14159265359) {
          unwrapped_phase += sign(phase_diff) * 2.0 * 3.14159265359;
        }
      }
    }
  }
  
  // Store unwrapped phase back
  wavefield[idx] = amplitude * vec2<f32>(cos(unwrapped_phase), sin(unwrapped_phase));
}
`;

// CPU Fallback implementation using WebAssembly
export class PenroseCPUFallback {
  constructor(size) {
    this.size = size;
    this.wavefield = new Float32Array(size * size * 2); // Complex field
    this.convergenceMap = new Float32Array(size * size);
    this.wasmModule = null;
  }
  
  async initialize() {
    // Load WebAssembly module for CPU computation
    const wasmCode = this.generateWASM();
    const wasmModule = await WebAssembly.compile(wasmCode);
    this.wasmInstance = await WebAssembly.instantiate(wasmModule, {
      env: {
        memory: new WebAssembly.Memory({ initial: 256 }),
        cos: Math.cos,
        sin: Math.sin,
        exp: Math.exp,
        sqrt: Math.sqrt,
        atan2: Math.atan2
      }
    });
  }
  
  generateWASM() {
    // Simplified WASM generation for Penrose algorithm
    // In production, this would be a pre-compiled WASM module
    return new Uint8Array([
      0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
      // ... WASM bytecode for Penrose algorithm
    ]);
  }
  
  compute(oscillatorState, params) {
    const { phases, frequencies, amplitudes } = oscillatorState;
    const { iterations, convergenceThreshold, relaxationFactor } = params;
    
    for (let iter = 0; iter < iterations; iter++) {
      let maxDiff = 0;
      
      for (let y = 0; y < this.size; y++) {
        for (let x = 0; x < this.size; x++) {
          const idx = (y * this.size + x) * 2;
          const pos = {
            x: x / this.size,
            y: y / this.size
          };
          
          // Current value
          const currentReal = this.wavefield[idx];
          const currentImag = this.wavefield[idx + 1];
          
          // Compute new value using Penrose kernel
          let newReal = 0;
          let newImag = 0;
          
          // Sum oscillator contributions
          for (let i = 0; i < phases.length; i++) {
            const oscPos = {
              x: Math.cos(i * Math.PI / 16) * 0.4 + 0.5,
              y: Math.sin(i * Math.PI / 16) * 0.4 + 0.5
            };
            
            const dist = Math.sqrt(
              (pos.x - oscPos.x) ** 2 + 
              (pos.y - oscPos.y) ** 2
            );
            
            const amp = amplitudes[i] * Math.exp(-dist * 0.1);
            const phase = 2 * Math.PI * dist / 0.000633 + phases[i] + frequencies[i] * iter * 0.01;
            
            newReal += amp * Math.cos(phase);
            newImag += amp * Math.sin(phase);
          }
          
          // Apply relaxation
          newReal *= relaxationFactor;
          newImag *= relaxationFactor;
          
          // Update with mixing
          this.wavefield[idx] = currentReal * 0.2 + newReal * 0.8;
          this.wavefield[idx + 1] = currentImag * 0.2 + newImag * 0.8;
          
          // Track convergence
          const diff = Math.sqrt(
            (newReal - currentReal) ** 2 + 
            (newImag - currentImag) ** 2
          );
          this.convergenceMap[y * this.size + x] = diff;
          maxDiff = Math.max(maxDiff, diff);
        }
      }
      
      // Check global convergence
      if (maxDiff < convergenceThreshold && iter > 10) {
        console.log(`Penrose converged after ${iter} iterations`);
        break;
      }
    }
    
    return this.wavefield;
  }
}

// Main Penrose Wavefield Engine
export class PenroseWavefieldEngine {
  constructor(device, size = 1024) {
    this.device = device;
    this.size = size;
    this.initialized = false;
    
    // Buffers
    this.paramsBuffer = null;
    this.oscillatorBuffer = null;
    this.wavefieldBuffer = null;
    this.convergenceBuffer = null;
    
    // Pipelines
    this.generationPipeline = null;
    this.phaseRetrievalPipeline = null;
    
    // CPU fallback
    this.cpuFallback = null;
    this.useCPUFallback = false;
    
    // Quality settings
    this.qualityMode = 1; // 0: draft, 1: normal, 2: high
    this.iterations = 50;
    this.convergenceThreshold = 0.001;
    this.relaxationFactor = 0.8;
  }
  
  async initialize() {
    try {
      // Create shader modules
      const shaderModule = this.device.createShaderModule({
        label: 'Penrose wavefield shader',
        code: penroseWavefieldShader
      });
      
      // Create buffers
      this.createBuffers();
      
      // Create pipelines
      this.createPipelines(shaderModule);
      
      // Initialize CPU fallback
      this.cpuFallback = new PenroseCPUFallback(this.size);
      await this.cpuFallback.initialize();
      
      this.initialized = true;
      console.log('✅ Penrose Wavefield Engine initialized');
      
    } catch (error) {
      console.error('Failed to initialize Penrose engine:', error);
      console.log('Falling back to CPU implementation');
      this.useCPUFallback = true;
    }
  }
  
  createBuffers() {
    // Parameters uniform buffer
    this.paramsBuffer = this.device.createBuffer({
      label: 'Penrose params',
      size: 16, // 4 u32 values
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    // Oscillator state buffer
    const oscillatorSize = 32 * 4 * 3 + 32 * 32 * 4; // phases, freqs, amps, coupling
    this.oscillatorBuffer = this.device.createBuffer({
      label: 'Oscillator state',
      size: oscillatorSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    
    // Wavefield buffer (complex values)
    this.wavefieldBuffer = this.device.createBuffer({
      label: 'Penrose wavefield',
      size: this.size * this.size * 8, // 2 f32 per pixel
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    
    // Convergence map
    this.convergenceBuffer = this.device.createBuffer({
      label: 'Convergence map',
      size: this.size * this.size * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    
    // Size uniform
    this.sizeBuffer = this.device.createBuffer({
      label: 'Hologram size',
      size: 8, // 2 u32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    
    // Write size
    this.device.queue.writeBuffer(
      this.sizeBuffer,
      0,
      new Uint32Array([this.size, this.size])
    );
  }
  
  createPipelines(shaderModule) {
    // Bind group layout
    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' }
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' }
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' }
        }
      ]
    });
    
    // Create bind group
    this.bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.oscillatorBuffer } },
        { binding: 2, resource: { buffer: this.wavefieldBuffer } },
        { binding: 3, resource: { buffer: this.convergenceBuffer } },
        { binding: 4, resource: { buffer: this.sizeBuffer } }
      ]
    });
    
    // Generation pipeline
    this.generationPipeline = this.device.createComputePipeline({
      label: 'Penrose generation',
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
      }),
      compute: {
        module: shaderModule,
        entryPoint: 'penrose_wavefield_generation'
      }
    });
    
    // Phase retrieval pipeline
    this.phaseRetrievalPipeline = this.device.createComputePipeline({
      label: 'Penrose phase retrieval',
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
      }),
      compute: {
        module: shaderModule,
        entryPoint: 'penrose_phase_retrieval'
      }
    });
  }
  
  async generateWavefield(oscillatorState, options = {}) {
    if (!this.initialized) {
      throw new Error('Penrose engine not initialized');
    }
    
    // Update quality settings
    this.qualityMode = options.quality || this.qualityMode;
    this.iterations = options.iterations || this.iterations;
    this.convergenceThreshold = options.convergenceThreshold || this.convergenceThreshold;
    this.relaxationFactor = options.relaxationFactor || this.relaxationFactor;
    
    if (this.useCPUFallback) {
      // Use CPU implementation
      return this.cpuFallback.compute(oscillatorState, {
        iterations: this.iterations,
        convergenceThreshold: this.convergenceThreshold,
        relaxationFactor: this.relaxationFactor
      });
    }
    
    // Update parameters
    this.device.queue.writeBuffer(
      this.paramsBuffer,
      0,
      new Uint32Array([
        this.iterations,
        ...new Float32Array([
          this.convergenceThreshold,
          this.relaxationFactor
        ]),
        this.qualityMode
      ])
    );
    
    // Update oscillator state
    this.updateOscillatorBuffer(oscillatorState);
    
    // Run iterative Penrose algorithm
    const commandEncoder = this.device.createCommandEncoder();
    
    for (let iter = 0; iter < this.iterations; iter++) {
      // Update iteration count in params
      this.device.queue.writeBuffer(
        this.paramsBuffer,
        0,
        new Uint32Array([iter])
      );
      
      // Run generation pass
      const computePass = commandEncoder.beginComputePass();
      computePass.setPipeline(this.generationPipeline);
      computePass.setBindGroup(0, this.bindGroup);
      
      const workgroupsX = Math.ceil(this.size / 16);
      const workgroupsY = Math.ceil(this.size / 16);
      computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
      
      computePass.end();
      
      // Check convergence every 10 iterations
      if (iter % 10 === 0 && iter > 0) {
        const converged = await this.checkConvergence();
        if (converged) {
          console.log(`Penrose converged after ${iter} iterations`);
          break;
        }
      }
    }
    
    // Run phase retrieval
    const phasePass = commandEncoder.beginComputePass();
    phasePass.setPipeline(this.phaseRetrievalPipeline);
    phasePass.setBindGroup(0, this.bindGroup);
    phasePass.dispatchWorkgroups(
      Math.ceil(this.size / 16),
      Math.ceil(this.size / 16)
    );
    phasePass.end();
    
    this.device.queue.submit([commandEncoder.finish()]);
    
    // Return the wavefield buffer for further processing
    return this.wavefieldBuffer;
  }
  
  updateOscillatorBuffer(oscillatorState) {
    const data = new Float32Array(32 * 3 + 32 * 32);
    
    // Pack oscillator data
    oscillatorState.phases.forEach((phase, i) => data[i] = phase);
    oscillatorState.frequencies.forEach((freq, i) => data[32 + i] = freq);
    oscillatorState.amplitudes.forEach((amp, i) => data[64 + i] = amp);
    
    // Pack coupling matrix
    if (oscillatorState.couplingMatrix) {
      oscillatorState.couplingMatrix.forEach((val, i) => data[96 + i] = val);
    } else {
      // Default: weak uniform coupling
      for (let i = 0; i < 32; i++) {
        for (let j = 0; j < 32; j++) {
          data[96 + i * 32 + j] = i === j ? 1.0 : 0.1;
        }
      }
    }
    
    this.device.queue.writeBuffer(this.oscillatorBuffer, 0, data);
  }
  
  async checkConvergence() {
    // Read convergence buffer
    const convergenceData = await this.readBuffer(
      this.convergenceBuffer,
      this.size * this.size * 4
    );
    
    const values = new Float32Array(convergenceData);
    const maxDiff = Math.max(...values);
    
    return maxDiff < this.convergenceThreshold;
  }
  
  async readBuffer(buffer, size) {
    const stagingBuffer = this.device.createBuffer({
      size: size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    
    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
    this.device.queue.submit([commandEncoder.finish()]);
    
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const data = new ArrayBuffer(size);
    new Uint8Array(data).set(new Uint8Array(stagingBuffer.getMappedRange()));
    stagingBuffer.unmap();
    stagingBuffer.destroy();
    
    return data;
  }
  
  setQuality(mode) {
    this.qualityMode = mode;
  }
  
  async compareWithFFT(fftWavefield) {
    // Read both wavefields and compute difference metrics
    const penroseData = await this.readBuffer(
      this.wavefieldBuffer,
      this.size * this.size * 8
    );
    
    const fftData = await this.readBuffer(
      fftWavefield,
      this.size * this.size * 8
    );
    
    const penroseField = new Float32Array(penroseData);
    const fftField = new Float32Array(fftData);
    
    let mse = 0;
    let maxDiff = 0;
    
    for (let i = 0; i < penroseField.length; i += 2) {
      const diffReal = penroseField[i] - fftField[i];
      const diffImag = penroseField[i + 1] - fftField[i + 1];
      const diff = Math.sqrt(diffReal * diffReal + diffImag * diffImag);
      
      mse += diff * diff;
      maxDiff = Math.max(maxDiff, diff);
    }
    
    mse /= (this.size * this.size);
    
    return {
      mse: mse,
      rmse: Math.sqrt(mse),
      maxDifference: maxDiff,
      psnr: 20 * Math.log10(1.0 / Math.sqrt(mse))
    };
  }
  
  destroy() {
    if (this.paramsBuffer) this.paramsBuffer.destroy();
    if (this.oscillatorBuffer) this.oscillatorBuffer.destroy();
    if (this.wavefieldBuffer) this.wavefieldBuffer.destroy();
    if (this.convergenceBuffer) this.convergenceBuffer.destroy();
    if (this.sizeBuffer) this.sizeBuffer.destroy();
  }
}
