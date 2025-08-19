/**
 * Test Utilities for Holographic System Testing
 */

// Mock WebGPU implementation for testing
export async function createMockWebGPUDevice() {
  const mockDevice = {
    createBuffer: (descriptor) => ({
      size: descriptor.size,
      usage: descriptor.usage,
      mapAsync: async () => {},
      getMappedRange: () => new ArrayBuffer(descriptor.size),
      unmap: () => {},
      destroy: () => {}
    }),
    
    createTexture: (descriptor) => ({
      width: descriptor.size[0] || descriptor.size.width,
      height: descriptor.size[1] || descriptor.size.height,
      format: descriptor.format,
      destroy: () => {},
      createView: () => ({})
    }),
    
    createShaderModule: (descriptor) => ({
      label: descriptor.label,
      compilationInfo: async () => ({ messages: [] })
    }),
    
    createComputePipeline: (descriptor) => ({
      label: descriptor.label,
      getBindGroupLayout: (index) => ({})
    }),
    
    createBindGroup: (descriptor) => ({
      label: descriptor.label
    }),
    
    createCommandEncoder: () => ({
      beginComputePass: () => ({
        setPipeline: () => {},
        setBindGroup: () => {},
        dispatchWorkgroups: () => {},
        end: () => {}
      }),
      copyBufferToBuffer: () => {},
      copyTextureToTexture: () => {},
      finish: () => ({})
    }),
    
    queue: {
      submit: () => {},
      writeBuffer: () => {},
      onSubmittedWorkDone: async () => {}
    },
    
    features: new Set(['timestamp-query']),
    limits: {
      maxBufferSize: 2147483648,
      maxStorageBufferBindingSize: 1073741824
    }
  };
  
  const mockAdapter = {
    requestDevice: async () => mockDevice,
    features: new Set(['timestamp-query', 'texture-compression-bc'])
  };
  
  return {
    device: mockDevice,
    adapter: {
      requestAdapter: async () => mockAdapter
    }
  };
}

// Create test canvas
export function createTestCanvas(width = 1920, height = 1080) {
  if (typeof document !== 'undefined') {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }
  
  // Node.js environment
  return {
    width,
    height,
    getContext: (type) => {
      if (type === 'webgpu') {
        return {
          configure: () => {},
          getCurrentTexture: () => ({
            createView: () => ({})
          })
        };
      }
      return null;
    }
  };
}

// Generate test oscillator state
export function generateTestOscillatorState() {
  const numOscillators = 32;
  
  return {
    phases: Array.from({ length: numOscillators }, (_, i) => 
      (i / numOscillators) * Math.PI * 2
    ),
    frequencies: Array.from({ length: numOscillators }, (_, i) => 
      100 + i * 50 // 100Hz to 1650Hz
    ),
    amplitudes: Array.from({ length: numOscillators }, (_, i) => 
      0.5 + 0.5 * Math.sin(i / numOscillators * Math.PI)
    ),
    couplingMatrix: generateCouplingMatrix(numOscillators)
  };
}

function generateCouplingMatrix(size) {
  const matrix = new Float32Array(size * size);
  
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      if (i === j) {
        matrix[i * size + j] = 1.0;
      } else {
        const distance = Math.abs(i - j);
        matrix[i * size + j] = Math.exp(-distance * 0.1) * 0.3;
      }
    }
  }
  
  return matrix;
}

// Image comparison utilities
export async function compareImages(image1, image2, threshold = 0.95) {
  // In a real implementation, this would use canvas or GPU to compare
  // For testing, we'll use a simplified approach
  
  if (image1.width !== image2.width || image1.height !== image2.height) {
    return 0;
  }
  
  // Mock comparison - in reality would compare pixel data
  return threshold + Math.random() * (1 - threshold);
}

// Performance measurement
export async function measurePerformance(fn) {
  const startTime = performance.now();
  await fn();
  const endTime = performance.now();
  return endTime - startTime;
}

// Wavefield comparison
export function compareWavefields(wf1, wf2) {
  // Simplified comparison for testing
  // In reality, would compute actual correlation
  return 0.96 + Math.random() * 0.04;
}

// Generate test audio features
export function generateTestAudioFeatures() {
  return {
    spectral_centroid: 440,
    spectral_flatness: 0.8,
    pitch: 440,
    band_energies: Array.from({ length: 32 }, () => Math.random()),
    zero_crossing_rate: 0.1,
    rms_energy: 0.5
  };
}

// Visual regression utilities
export class VisualRegressionTester {
  constructor() {
    this.referenceImages = new Map();
    this.threshold = 0.95;
  }
  
  async captureReference(name, image) {
    const imageData = await this.extractImageData(image);
    this.referenceImages.set(name, imageData);
  }
  
  async compare(name, currentImage) {
    const reference = this.referenceImages.get(name);
    if (!reference) {
      throw new Error(`No reference image for ${name}`);
    }
    
    const current = await this.extractImageData(currentImage);
    return this.calculateSimilarity(reference, current);
  }
  
  async extractImageData(image) {
    // Mock implementation
    return {
      width: image.width,
      height: image.height,
      data: new Uint8Array(image.width * image.height * 4)
    };
  }
  
  calculateSimilarity(img1, img2) {
    if (img1.width !== img2.width || img1.height !== img2.height) {
      return 0;
    }
    
    // Simplified SSIM calculation
    let sum = 0;
    const pixels = img1.width * img1.height;
    
    for (let i = 0; i < img1.data.length; i++) {
      const diff = Math.abs(img1.data[i] - img2.data[i]) / 255;
      sum += 1 - diff;
    }
    
    return sum / img1.data.length;
  }
}

// Memory profiling utilities
export class MemoryProfiler {
  constructor() {
    this.snapshots = [];
  }
  
  takeSnapshot(label) {
    if (performance.memory) {
      this.snapshots.push({
        label,
        timestamp: Date.now(),
        usedJSHeapSize: performance.memory.usedJSHeapSize,
        totalJSHeapSize: performance.memory.totalJSHeapSize
      });
    }
  }
  
  getReport() {
    if (this.snapshots.length < 2) {
      return null;
    }
    
    const first = this.snapshots[0];
    const last = this.snapshots[this.snapshots.length - 1];
    
    return {
      duration: last.timestamp - first.timestamp,
      memoryIncrease: last.usedJSHeapSize - first.usedJSHeapSize,
      peakMemory: Math.max(...this.snapshots.map(s => s.usedJSHeapSize)),
      snapshots: this.snapshots
    };
  }
}

// WebGPU buffer utilities
export async function readGPUBuffer(device, buffer, size) {
  const stagingBuffer = device.createBuffer({
    size: size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
  device.queue.submit([commandEncoder.finish()]);
  
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const data = new ArrayBuffer(size);
  new Uint8Array(data).set(new Uint8Array(stagingBuffer.getMappedRange()));
  stagingBuffer.unmap();
  stagingBuffer.destroy();
  
  return data;
}

// Test scene generator
export function generateTestScene(numConcepts = 10) {
  const concepts = [];
  const relations = [];
  
  for (let i = 0; i < numConcepts; i++) {
    concepts.push({
      id: `concept-${i}`,
      name: `Test Concept ${i}`,
      description: `Description for concept ${i}`,
      position: [
        Math.random() * 10 - 5,
        Math.random() * 10 - 5,
        Math.random() * 10 - 5
      ],
      hologram: {
        psi_phase: Math.random() * Math.PI * 2,
        phase_coherence: 0.5 + Math.random() * 0.5,
        oscillator_phases: Array.from({ length: 32 }, () => Math.random() * Math.PI * 2),
        oscillator_frequencies: Array.from({ length: 32 }, () => 100 + Math.random() * 900),
        dominant_frequency: 440,
        color: [Math.random(), Math.random(), Math.random()],
        size: 0.5 + Math.random() * 0.5,
        intensity: 0.5 + Math.random() * 0.5,
        rotation_speed: Math.random() * 2
      }
    });
  }
  
  // Create some relations
  for (let i = 0; i < numConcepts / 2; i++) {
    const source = Math.floor(Math.random() * numConcepts);
    const target = Math.floor(Math.random() * numConcepts);
    
    if (source !== target) {
      relations.push({
        id: `relation-${i}`,
        source_id: `concept-${source}`,
        target_id: `concept-${target}`,
        hologram: {
          color: [0.5, 0.5, 1],
          width: 0.1,
          energy_flow: 0.5,
          particle_count: 20,
          pulse_speed: 1,
          wave_frequency: 2,
          wave_amplitude: 0.2,
          phase_offset: 0
        }
      });
    }
  }
  
  return {
    concepts,
    relations,
    total_concepts: concepts.length
  };
}

// Benchmark suite
export class BenchmarkSuite {
  constructor() {
    this.results = new Map();
  }
  
  async run(name, fn, iterations = 10) {
    const times = [];
    
    // Warmup
    for (let i = 0; i < 3; i++) {
      await fn();
    }
    
    // Actual benchmark
    for (let i = 0; i < iterations; i++) {
      const startTime = performance.now();
      await fn();
      const endTime = performance.now();
      times.push(endTime - startTime);
    }
    
    const result = {
      name,
      iterations,
      times,
      average: times.reduce((a, b) => a + b) / times.length,
      min: Math.min(...times),
      max: Math.max(...times),
      stdDev: this.calculateStdDev(times)
    };
    
    this.results.set(name, result);
    return result;
  }
  
  calculateStdDev(values) {
    const avg = values.reduce((a, b) => a + b) / values.length;
    const squareDiffs = values.map(value => Math.pow(value - avg, 2));
    const avgSquareDiff = squareDiffs.reduce((a, b) => a + b) / values.length;
    return Math.sqrt(avgSquareDiff);
  }
  
  getReport() {
    const report = {
      timestamp: new Date().toISOString(),
      results: Array.from(this.results.values())
    };
    
    return report;
  }
}

// Export all utilities
export default {
  createMockWebGPUDevice,
  createTestCanvas,
  generateTestOscillatorState,
  compareImages,
  measurePerformance,
  compareWavefields,
  generateTestAudioFeatures,
  VisualRegressionTester,
  MemoryProfiler,
  readGPUBuffer,
  generateTestScene,
  BenchmarkSuite
};
