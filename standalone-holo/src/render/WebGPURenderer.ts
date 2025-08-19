// WebGPURenderer.ts - Core WebGPU initialization and management
export type OutputMode = 'quilt' | 'parallax' | 'depth';

export class WebGPURenderer {
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private format!: GPUTextureFormat;
  private canvas: HTMLCanvasElement;
  private outputMode: OutputMode;

  constructor(canvas: HTMLCanvasElement, outputMode: OutputMode = 'parallax') {
    this.canvas = canvas;
    this.outputMode = outputMode;
  }

  static async create(canvas: HTMLCanvasElement, opts?: { output?: OutputMode }): Promise<WebGPURenderer> {
    const renderer = new WebGPURenderer(canvas, opts?.output || 'parallax');
    await renderer.init();
    return renderer;
  }

  private async init() {
    // Check WebGPU support
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported. Please use Chrome/Edge/Safari with WebGPU enabled.');
    }

    // Get adapter and device
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('No WebGPU adapter available');
    }

    this.device = await adapter.requestDevice();
    
    // Setup canvas context
    this.context = this.canvas.getContext('webgpu')!;
    if (!this.context) {
      throw new Error('Failed to get WebGPU context');
    }

    // Get preferred format
    this.format = navigator.gpu.getPreferredCanvasFormat();
    
    // Configure context
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'premultiplied',
    });

    // Set canvas size
    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
  }

  private resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = this.canvas.clientWidth * dpr;
    this.canvas.height = this.canvas.clientHeight * dpr;
  }

  // Getters for other components
  getDevice(): GPUDevice { return this.device; }
  getContext(): GPUCanvasContext { return this.context; }
  getFormat(): GPUTextureFormat { return this.format; }
  getCanvas(): HTMLCanvasElement { return this.canvas; }

  setOutputMode(mode: OutputMode) {
    this.outputMode = mode;
  }

  getOutputMode(): OutputMode {
    return this.outputMode;
  }

  // Frame timing
  private lastFrameTime = 0;
  private frameCount = 0;
  private fps = 0;

  updateFrameStats(timestamp: number): number {
    if (this.lastFrameTime > 0) {
      const delta = timestamp - this.lastFrameTime;
      this.frameCount++;
      if (this.frameCount % 30 === 0) {
        this.fps = Math.round(1000 / delta);
      }
    }
    this.lastFrameTime = timestamp;
    return this.fps;
  }

  getFPS(): number { return this.fps; }
}