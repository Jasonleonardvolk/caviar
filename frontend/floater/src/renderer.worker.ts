/* frontend/floater/src/renderer.worker.ts - REAL COMPONENTS, NO STUBS */
/* eslint-disable no-restricted-globals */

// Import from clean architecture paths (which proxy to real locations)
import { SchrodingerRegistry } from '../../lib/pipeline/schrodinger';
import { OnnxWaveOpRunner, createOnnxWaveRunner } from '../../lib/ai';

// Policy type
export type Policy = {
  donorMode: 'lan'|'off';
  keyframeInterval: number|'auto';
  compressedReturn: boolean;
  roiThreshold: number|'auto';
  maxMsPerFrame: number;
  maxMemMB: number;
};

// Worker state
let device: GPUDevice | null = null;
let registry: typeof SchrodingerRegistry | null = null;
let onnxRunner: OnnxWaveOpRunner | null = null;
let running = false;
let canvas: OffscreenCanvas | null = null;

// Configuration
let dims: [number, number] = [256, 256];
let kernel: 'onnx'|'splitstep'|'biharmonic' = 'biharmonic';
let policy: Policy = {
  donorMode: 'off',
  keyframeInterval: 'auto',
  compressedReturn: false,
  roiThreshold: 'auto',
  maxMsPerFrame: 16,
  maxMemMB: 1024
};

async function initialize() {
  if (!navigator.gpu) {
    throw new Error('WebGPU not available in worker');
  }
  
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No GPU adapter found');
  
  device = await adapter.requestDevice();
  
  // Initialize the kernel registry
  registry = SchrodingerRegistry;
  await registry.initialize(device);
  
  // Initialize ONNX runner if needed
  if (kernel === 'onnx') {
    onnxRunner = await createOnnxWaveRunner({
      modelPath: '/models/waveop_fno_v1.onnx',
      backend: 'webgpu'
    }, device);
  }
  
  self.postMessage({ 
    type: 'ready', 
    features: Array.from(adapter.features || [])
  });
}

async function render() {
  if (!device || !registry) return;
  
  const startTime = performance.now();
  
  // Execute the selected kernel
  const commandEncoder = device.createCommandEncoder();
  
  if (kernel === 'onnx' && onnxRunner) {
    // Use ONNX runner
    await onnxRunner.execute(commandEncoder, { dims });
  } else {
    // Use registry kernels
    const selectedKernel = registry.getKernel(kernel);
    if (selectedKernel) {
      selectedKernel.execute(commandEncoder, { dims });
    }
  }
  
  device.queue.submit([commandEncoder.finish()]);
  
  const elapsed = performance.now() - startTime;
  
  // Send metrics
  self.postMessage({
    type: 'metrics',
    fps: Math.round(1000 / elapsed),
    ms: elapsed,
    kernel,
    dims
  });
  
  // Continue render loop if running
  if (running) {
    requestAnimationFrame(() => render());
  }
}

// Message handler
self.onmessage = async (e) => {
  const { type, ...data } = e.data;
  
  switch (type) {
    case 'init':
      try {
        await initialize();
      } catch (err) {
        self.postMessage({ type: 'error', error: String(err) });
      }
      break;
      
    case 'start':
      running = true;
      if (data.canvas instanceof OffscreenCanvas) {
        canvas = data.canvas;
      }
      if (data.kernel) kernel = data.kernel;
      if (data.dims) dims = data.dims;
      if (data.policy) policy = { ...policy, ...data.policy };
      render();
      break;
      
    case 'stop':
      running = false;
      break;
      
    case 'setKernel':
      kernel = data.kernel;
      if (kernel === 'onnx' && !onnxRunner && device) {
        onnxRunner = await createOnnxWaveRunner({
          modelPath: '/models/waveop_fno_v1.onnx',
          backend: 'webgpu'
        }, device);
      }
      break;
      
    case 'setDims':
      dims = data.dims;
      break;
      
    case 'destroy':
      running = false;
      registry?.destroy();
      onnxRunner?.destroy();
      canvas = null;
      device = null;
      self.postMessage({ type: 'destroyed' });
      break;
  }
};

export {};
