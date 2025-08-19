// ${IRIS_ROOT}\frontend\examples\iris_complete_example.ts
/**
 * Complete working example of IRIS 1.0 integration
 * This shows how to integrate head tracking, phase correction, and wave exclusion
 * into your holographic display application.
 */

import { initializeIris, renderIrisFrame, getIrisSystem } from '../../IRIS_FINAL_INTEGRATION';

// Import your existing holographic components
// These are placeholders - use your actual imports
// import { HolographicRenderer } from '@/lib/webgpu/renderer';
// import { ContentManager } from '@/lib/content/manager';

/**
 * Main application class showing IRIS integration
 */
export class IrisHolographicApp {
  private canvas: HTMLCanvasElement;
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private iris?: ReturnType<typeof getIrisSystem>;
  
  // Your wavefield buffers
  private waveReBuf!: GPUBuffer;
  private waveImBuf!: GPUBuffer;
  private maskBuf?: GPUBuffer;
  
  // Display dimensions
  private width = 1920;
  private height = 1080;
  
  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }
  
  /**
   * Initialize the complete system
   */
  async init(): Promise<void> {
    console.log('🚀 Starting IRIS Holographic App...');
    
    // 1. Initialize WebGPU
    await this.initWebGPU();
    
    // 2. Initialize IRIS system (head tracking + phase correction)
    await this.initIrisSystem();
    
    // 3. Initialize your holographic content
    await this.initHolographicContent();
    
    // 4. Setup event listeners
    this.setupEventListeners();
    
    // 5. Start render loop
    this.startRenderLoop();
    
    console.log('✨ App ready! Try these controls:');
    console.log('  • Move mouse for parallax');
    console.log('  • Alt+↑/↓ to adjust smoothing');
    console.log('  • Alt+Z to cycle correction methods');
    console.log('  • Alt+P for debug overlay');
  }
  
  /**
   * Initialize WebGPU
   */
  private async initWebGPU(): Promise<void> {
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported');
    }
    
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('No GPU adapter found');
    }
    
    this.device = await adapter.requestDevice();
    this.context = this.canvas.getContext('webgpu')!;
    
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: presentationFormat,
      alphaMode: 'premultiplied',
    });
    
    // Create wavefield buffers
    const bufferSize = this.width * this.height * 4; // float32
    
    this.waveReBuf = this.device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    
    this.waveImBuf = this.device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    
    // Optional: Create mask buffer for edge-aware processing
    this.maskBuf = this.device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }
  
  /**
   * Initialize IRIS system
   */
  private async initIrisSystem(): Promise<void> {
    // This single call sets up:
    // - Head tracking with auto sensor selection
    // - Phase correction (TV/Zernike/LUT)
    // - Dynamic tuning (URL params + keyboard)
    // - Debug overlay (Alt+P)
    this.iris = await initializeIris(this.canvas);
    
    // Check current status
    const status = this.iris.getStatus();
    console.log('IRIS Status:', status);
  }
  
  /**
   * Initialize your holographic content
   */
  private async initHolographicContent(): Promise<void> {
    // Your existing holographic initialization
    // This is where you'd load models, textures, etc.
    
    // Example: Load a test hologram
    await this.loadTestHologram();
  }
  
  /**
   * Load a test hologram pattern
   */
  private async loadTestHologram(): Promise<void> {
    // Create a simple test pattern (checkerboard phase)
    const data = new Float32Array(this.width * this.height);
    
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        const idx = y * this.width + x;
        // Checkerboard pattern with phase variations
        const checker = ((x >> 5) ^ (y >> 5)) & 1;
        data[idx] = checker ? 0.5 : -0.5;
        
        // Add some "artifacts" that phase correction will smooth
        if (x % 64 === 0 || y % 64 === 0) {
          data[idx] += Math.random() * 0.2 - 0.1;
        }
      }
    }
    
    // Upload to GPU buffers
    this.device.queue.writeBuffer(this.waveReBuf, 0, data);
    this.device.queue.writeBuffer(this.waveImBuf, 0, data); // Imaginary part
  }
  
  /**
   * Setup event listeners
   */
  private setupEventListeners(): void {
    // Window resize
    window.addEventListener('resize', () => this.handleResize());
    
    // Visibility change (pause when hidden)
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        console.log('🔄 Pausing render loop (tab hidden)');
      } else {
        console.log('▶️ Resuming render loop');
      }
    });
    
    // Custom event from IRIS system
    window.addEventListener('iris:parameterChanged', (e: any) => {
      console.log('IRIS parameter changed:', e.detail);
    });
  }
  
  /**
   * Handle window resize
   */
  private handleResize(): void {
    // Update canvas size
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * window.devicePixelRatio;
    this.canvas.height = rect.height * window.devicePixelRatio;
    
    // Recreate buffers if needed
    // ... your resize logic ...
  }
  
  /**
   * Main render loop
   */
  private startRenderLoop(): void {
    const render = () => {
      // Skip if tab is hidden
      if (document.hidden) {
        requestAnimationFrame(render);
        return;
      }
      
      // Render frame
      this.renderFrame();
      
      // Continue loop
      requestAnimationFrame(render);
    };
    
    requestAnimationFrame(render);
  }
  
  /**
   * Render a single frame
   */
  private renderFrame(): void {
    const commandEncoder = this.device.createCommandEncoder();
    
    // ========================================
    // 1. Generate/update wavefield
    // ========================================
    // Your existing wavefield generation goes here
    // This might include:
    // - Content rendering
    // - Parallax from head tracking (automatic via IRIS)
    // - Initial wavefield computation
    
    // ========================================
    // 2. Apply IRIS corrections (zero overhead!)
    // ========================================
    renderIrisFrame(
      commandEncoder,
      this.waveReBuf,
      this.waveImBuf,
      this.width,
      this.height,
      this.maskBuf
    );
    
    // ========================================
    // 3. Final composition and display
    // ========================================
    const textureView = this.context.getCurrentTexture().createView();
    
    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    };
    
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    
    // Your final rendering pipeline
    // This would typically:
    // - Convert complex wavefield to displayable format
    // - Apply color mapping
    // - Draw to screen
    
    passEncoder.end();
    
    // Submit commands
    this.device.queue.submit([commandEncoder.finish()]);
  }
  
  /**
   * Cleanup resources
   */
  destroy(): void {
    // Cleanup IRIS
    if (this.iris) {
      this.iris.destroy();
    }
    
    // Cleanup buffers
    this.waveReBuf?.destroy();
    this.waveImBuf?.destroy();
    this.maskBuf?.destroy();
    
    // Cleanup WebGPU
    this.device?.destroy();
  }
}

// ============================================
// USAGE EXAMPLE
// ============================================

/**
 * Initialize the app when DOM is ready
 */
export async function initApp() {
  // Wait for DOM
  if (document.readyState !== 'loading') {
    await startApp();
  } else {
    document.addEventListener('DOMContentLoaded', startApp);
  }
}

/**
 * Start the application
 */
async function startApp() {
  try {
    // Get or create canvas
    let canvas = document.querySelector('canvas#iris-display') as HTMLCanvasElement;
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.id = 'iris-display';
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      document.body.appendChild(canvas);
    }
    
    // Create and initialize app
    const app = new IrisHolographicApp(canvas);
    await app.init();
    
    // Store globally for debugging
    (window as any).irisApp = app;
    (window as any).irisSystem = getIrisSystem();
    
    console.log('✅ IRIS app initialized!');
    console.log('Debug commands available:');
    console.log('  window.irisApp - Main app instance');
    console.log('  window.irisSystem - IRIS system');
    console.log('  window.irisSystem.getStatus() - Current status');
    
  } catch (error) {
    console.error('Failed to initialize IRIS app:', error);
    
    // Show user-friendly error
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = `
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: #ff4444;
      color: white;
      padding: 20px;
      border-radius: 10px;
      font-family: system-ui;
      text-align: center;
      z-index: 9999;
    `;
    errorDiv.innerHTML = `
      <h2>⚠️ WebGPU Required</h2>
      <p>This demo requires WebGPU support.</p>
      <p>Try Chrome Canary or Edge Canary with WebGPU enabled.</p>
      <small>${error}</small>
    `;
    document.body.appendChild(errorDiv);
  }
}

// ============================================
// TEST URLS
// ============================================

/*
Test different configurations by adding these to your URL:

1. Basic TV smoothing (default):
   https://yourapp.com/?phase=tv

2. Strong smoothing for heavy artifacts:
   https://yourapp.com/?phase=tv&tv=0.15&maxc=0.35

3. Gentle correction:
   https://yourapp.com/?phase=tv&tv=0.04&maxc=0.15

4. Zernike mode with defocus:
   https://yourapp.com/?phase=zernike&defocus=0.1

5. Static LUT mode:
   https://yourapp.com/?phase=lut&gain=1.2

6. Debug everything:
   https://yourapp.com/?phase=tv&tv=0.08&maxc=0.25&auto=1&metrics=1

7. Disable corrections (parallax only):
   https://yourapp.com/?phase=off
*/

// Auto-start if this is the main entry point
if (import.meta.url === new URL(import.meta.url, window.location.href).href) {
  initApp();
}
