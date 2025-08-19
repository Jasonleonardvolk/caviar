// HolographicDisplay.ts - Main orchestration class
import { WebGPURenderer, OutputMode } from './render/WebGPURenderer';
import { QuiltCompose } from './render/QuiltCompose';
import { PoseController } from './parallax/pose_controller';
import { estimateDepth } from './pipelines/depth_estimator';
import { runWaveOp } from './pipelines/waveop_runner';
import { fieldFromDepth } from './pipelines/field_from_depth';
import { PropagationPipeline } from './pipelines/PropagationPipeline';
import { SLMEncoderPipeline, EncodingMode } from './pipelines/SLMEncoderPipeline';

export class HolographicDisplay {
  private renderer!: WebGPURenderer;
  private composer!: QuiltCompose;
  private pose!: PoseController;
  private propagator!: PropagationPipeline;
  private encoder!: SLMEncoderPipeline;
  
  // Cached hologram texture
  private currentHologram: GPUTexture | null = null;
  private hologramSize = { width: 256, height: 256 };
  
  private isRunning = false;
  private parallaxEnabled = true;
  private qualityScale = 1.0;
  private encodingMode: EncodingMode = 'phase_only';

  constructor() {}

  async init(canvas: HTMLCanvasElement) {
    console.log('Initializing HolographicDisplay...');
    
    // Initialize WebGPU
    this.renderer = await WebGPURenderer.create(canvas, { output: 'parallax' });
    
    const device = this.renderer.getDevice();
    const context = this.renderer.getContext();
    const format = this.renderer.getFormat();

    // Initialize render components
    this.composer = new QuiltCompose(device, context, format);
    
    // Initialize pose tracking
    this.pose = new PoseController();
    
    // Initialize holographic pipelines
    this.propagator = new PropagationPipeline(device);
    this.encoder = new SLMEncoderPipeline(device);
    await this.propagator.init();
    await this.encoder.init();
    
    // Initialize with a test pattern
    await this.processTestPattern();
    
    console.log('HolographicDisplay initialized');
  }

  async start() {
    if (this.isRunning) return;
    
    console.log('Starting display...');
    this.isRunning = true;
    
    // Start pose tracking
    this.pose.startDeviceOrientation();
    
    // Start render loop
    this.renderLoop();
  }

  stop() {
    this.isRunning = false;
  }

  toggleParallax() {
    this.parallaxEnabled = !this.parallaxEnabled;
    console.log('Parallax:', this.parallaxEnabled ? 'ON' : 'OFF');
  }

  setQuality(scale: number) {
    this.qualityScale = Math.max(0.25, Math.min(2.0, scale));
    console.log('Quality scale:', this.qualityScale);
  }

  toggleQuality() {
    if (this.qualityScale >= 1.0) {
      this.setQuality(0.5);
    } else {
      this.setQuality(1.0);
    }
  }

  private async renderLoop() {
    if (!this.isRunning) return;

    const timestamp = performance.now();
    const fps = this.renderer.updateFrameStats(timestamp);

    const device = this.renderer.getDevice();
    const canvas = this.renderer.getCanvas();
    
    // Use the current hologram or create a default texture
    let texture: GPUTexture;
    
    if (this.currentHologram) {
      // Convert the phase hologram to RGBA for display
      texture = await this.encoder.phaseToRGBA(this.currentHologram);
    } else {
      // Fallback: create a simple gradient texture
      texture = device.createTexture({
        size: [canvas.width, canvas.height],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
      });

      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginRenderPass({
        colorAttachments: [{
          view: texture.createView(),
          clearValue: { r: 0.1, g: 0.2, b: 0.3, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store'
        }]
      });
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
    }

    // Apply parallax offset if enabled
    const pose = this.parallaxEnabled ? this.pose.getPose() : { tx: 0, ty: 0, tz: 0, rx: 0, ry: 0, rz: 0 };
    
    // Draw to canvas with parallax offset
    this.composer.draw(texture, { x: pose.tx * 0.1, y: pose.ty * 0.1 });

    // Update status
    const status = document.getElementById('status');
    if (status) {
      status.textContent = `FPS: ${fps} | Parallax: ${this.parallaxEnabled ? 'ON' : 'OFF'} | Quality: ${(this.qualityScale * 100).toFixed(0)}%`;
    }

    // Continue loop
    requestAnimationFrame(() => this.renderLoop());
  }

  async processImage(img: ImageBitmap | ImageData) {
    console.log('Processing image...');
    
    try {
      // Estimate depth
      const { depth, width, height } = await estimateDepth(img, {
        inputSize: { w: 256, h: 256 }
      });
      
      console.log(`Depth estimated: ${width}x${height}`);
      
      // Generate field from depth
      let field;
      try {
        field = await runWaveOp(depth, width, height);
        console.log('WaveOp field generated');
      } catch (err) {
        console.log('WaveOp failed, using fallback');
        field = fieldFromDepth(depth, width, height);
      }
      
      // Propagate the field
      console.log('Propagating field...');
      const propagated = await this.propagator.propagate(
        field.amp,
        field.phi,
        width,
        height,
        100.0, // propagation distance in mm
        0.000532 // wavelength in mm (532nm green)
      );
      
      // Convert back to amp/phi
      const propagatedField = this.propagator.complexToAmpPhi(propagated.real, propagated.imag);
      
      // Encode to SLM pattern
      console.log('Encoding to SLM pattern...');
      this.currentHologram = await this.encoder.encodeToTexture(
        propagatedField.amp,
        propagatedField.phi,
        width,
        height,
        this.encodingMode
      );
      
      this.hologramSize = { width, height };
      console.log('Hologram ready for display!');
      
    } catch (err) {
      console.error('Image processing failed:', err);
    }
  }

  // Generate a test pattern for immediate testing
  async generateTestPattern(): Promise<ImageData> {
    const size = 256;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d')!;
    
    // Create gradient background
    const gradient = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
    gradient.addColorStop(0, '#ffffff');
    gradient.addColorStop(0.5, '#808080');
    gradient.addColorStop(1, '#000000');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);
    
    // Add some geometric shapes for depth variation
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(size/4, size/4, size/4, size/4);
    
    ctx.fillStyle = '#666666';
    ctx.beginPath();
    ctx.arc(3*size/4, 3*size/4, size/8, 0, Math.PI * 2);
    ctx.fill();
    
    return ctx.getImageData(0, 0, size, size);
  }

  // Process test pattern on initialization
  async processTestPattern() {
    const testImage = await this.generateTestPattern();
    await this.processImage(testImage);
  }

  // Set encoding mode
  setEncodingMode(mode: EncodingMode) {
    this.encodingMode = mode;
    console.log('Encoding mode:', mode);
  }

  // Get current hologram info
  getHologramInfo() {
    return {
      hasHologram: this.currentHologram !== null,
      size: this.hologramSize,
      encodingMode: this.encodingMode
    };
  }
}