// ${IRIS_ROOT}\frontend\lib\webgpu\enginePerf.ts
/**
 * Performance-optimized WebGPU engine with subgroups, profiling, and feature detection
 * Extends the base engine with iOS 26 / modern browser optimizations
 */

import { WebGPUEngine, type EngineConfig } from './engine';
import { requestDeviceWithCaps, type Caps } from './caps';
import { GpuProfiler } from './profiler';
import { createParallaxComposite } from './pipelines/parallaxComposite';
import { HeadPoseUniform } from './headPoseUniform';
import { drawPacket, makeIndirectBuffer, encodeIndirectDraw } from './indirect';

export interface PerfEngineConfig extends Omit<EngineConfig, 'device'> {
  enableProfiling?: boolean;
  enableSubgroups?: boolean;
  logCapabilities?: boolean;
}

export class WebGPUPerfEngine extends WebGPUEngine {
  private caps!: Caps;
  private adapter!: GPUAdapter;
  private profiler?: GpuProfiler;
  private composite?: Awaited<ReturnType<typeof createParallaxComposite>>;
  private headPose?: HeadPoseUniform;
  private frameCount = 0;
  private lastFrameTime = performance.now();
  private avgFrameTime = 0;
  
  // Performance metrics
  private metrics = {
    fps: 0,
    frameTime: 0,
    gpuTime: 0,
    subgroupsActive: false,
    timestampsAvailable: false,
  };
  
  /**
   * Create a performance-optimized engine with feature detection
   */
  static async create(config: PerfEngineConfig): Promise<WebGPUPerfEngine> {
    // Request device with capabilities detection
    const { adapter, device, caps } = await requestDeviceWithCaps();
    
    if (config.logCapabilities) {
      console.log('[GPU Caps]', {
        subgroups: caps.subgroups,
        subgroupSize: caps.subgroupMinSize ? `${caps.subgroupMinSize}-${caps.subgroupMaxSize}` : 'unknown',
        f16: caps.shaderF16,
        timestamps: caps.timestampQuery,
        indirectFirstInstance: caps.indirectFirstInstance,
      });
    }
    
    // Create engine with detected device
    const engine = new WebGPUPerfEngine({
      ...config,
      device,
    } as EngineConfig);
    
    // Store capabilities
    engine.caps = caps;
    engine.adapter = adapter;
    
    // Initialize performance features
    await engine.initializePerf(config);
    
    return engine;
  }
  
  /**
   * Initialize performance optimizations
   */
  private async initializePerf(config: PerfEngineConfig): Promise<void> {
    // Initialize base engine
    await this.initialize();
    
    // Create GPU profiler if timestamps are available
    if (config.enableProfiling !== false && this.caps.timestampQuery) {
      this.profiler = new GpuProfiler(this.device, true);
      console.log('[PerfEngine] GPU profiling enabled');
      this.metrics.timestampsAvailable = true;
    } else if (config.enableProfiling) {
      // CPU fallback profiler
      this.profiler = new GpuProfiler(this.device, false);
      console.log('[PerfEngine] CPU profiling fallback');
    }
    
    // Create subgroup-optimized pipelines if supported
    if (config.enableSubgroups !== false && this.caps.subgroups) {
      this.composite = await createParallaxComposite(this.device, this.caps);
      console.log('[PerfEngine] Subgroup-optimized pipelines enabled');
      this.metrics.subgroupsActive = true;
    } else {
      // Fallback to basic pipelines
      this.composite = await createParallaxComposite(this.device, { ...this.caps, subgroups: false });
      console.log('[PerfEngine] Basic pipelines (no subgroups)');
    }
    
    // Create head pose uniform buffer for fast updates
    this.headPose = new HeadPoseUniform(this.device);
    console.log('[PerfEngine] Head pose uniform buffer ready');
  }
  
  /**
   * Enhanced render frame with profiling
   */
  renderFramePerf(timestamp: number, headMatrix?: Float32Array): void {
    const startCpu = performance.now();
    
    // Update head pose if provided (fast uniform update, no push constants)
    if (headMatrix && this.headPose) {
      this.headPose.update(headMatrix);
    }
    
    const commandEncoder = this.device.createCommandEncoder({
      label: `Frame ${this.frameCount}`,
    });
    
    // Example: Profile a compute pass with subgroups
    if (this.composite && this.profiler) {
      const { pass } = this.profiler.beginComputePass(commandEncoder, 'parallaxComposite');
      
      // Example dispatch (you'd wire your actual data here)
      const width = this.canvas.width;
      const height = this.canvas.height;
      const layerCount = 4; // Example: 4 parallax layers
      
      // Create parameter buffer
      const paramBuf = this.device.createBuffer({
        size: 12,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Uint32Array(paramBuf.getMappedRange()).set([width, height, layerCount]);
      paramBuf.unmap();
      
      // Output buffer
      const outBuf = this.device.createBuffer({
        size: width * height * 4 * 4, // vec4<f32>
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      
      // Placeholder for actual texture array and sampler
      // In production, you'd use your actual parallax view textures
      
      // pass.setPipeline(this.composite.pipeline);
      // pass.setBindGroup(0, bindGroup);
      // const x = Math.ceil(width / 8), y = Math.ceil(height / 8);
      // pass.dispatchWorkgroups(x, y, 1);
      
      pass.end();
      
      // Resolve timestamps
      this.profiler.resolve(commandEncoder);
    }
    
    // Regular rendering continues
    super.renderFrame(timestamp);
    
    // Submit with profiling
    if (this.profiler) {
      this.device.queue.submit([commandEncoder.finish()]);
      
      // Read profiling results async (non-blocking)
      this.profiler.read().then(samples => {
        if (samples.length > 0) {
          const totalGpu = samples.reduce((sum, s) => sum + s.ns, 0);
          this.metrics.gpuTime = totalGpu / 1e6; // Convert to ms
        }
      });
    }
    
    // Update CPU metrics
    const endCpu = performance.now();
    const frameTime = endCpu - startCpu;
    this.avgFrameTime = this.avgFrameTime * 0.95 + frameTime * 0.05; // Exponential moving average
    this.metrics.frameTime = this.avgFrameTime;
    
    // Calculate FPS
    this.frameCount++;
    const deltaTime = endCpu - this.lastFrameTime;
    if (deltaTime >= 1000) {
      this.metrics.fps = Math.round(this.frameCount * 1000 / deltaTime);
      this.frameCount = 0;
      this.lastFrameTime = endCpu;
    }
  }
  
  /**
   * Example: Safe indirect draw with feature gating
   */
  performIndirectDraw(
    renderPass: GPURenderPassEncoder,
    vertexCount: number,
    instanceCount: number = 1,
    firstVertex: number = 0,
    firstInstance: number = 0
  ): void {
    const packet = drawPacket(vertexCount, instanceCount, firstVertex, firstInstance, this.caps);
    const indirectBuf = makeIndirectBuffer(this.device, packet);
    encodeIndirectDraw(renderPass, this.caps, indirectBuf, 0);
  }
  
  /**
   * Get current performance metrics
   */
  getMetrics() {
    return { ...this.metrics };
  }
  
  /**
   * Get capabilities for diagnostics
   */
  getCapabilities() {
    return { ...this.caps };
  }
  
  /**
   * Enhanced dispose with cleanup
   */
  dispose(): void {
    // Cleanup performance resources
    this.headPose?.buffer.destroy();
    
    // Cleanup base resources
    super.dispose();
  }
}

/**
 * Factory function for quick setup
 */
export async function createPerfEngine(config?: Partial<PerfEngineConfig>): Promise<WebGPUPerfEngine> {
  const canvas = config?.canvas || document.querySelector('canvas') as HTMLCanvasElement;
  if (!canvas) throw new Error('No canvas found');
  
  return WebGPUPerfEngine.create({
    canvas,
    enableProfiling: true,
    enableSubgroups: true,
    logCapabilities: true,
    ...config,
  });
}
