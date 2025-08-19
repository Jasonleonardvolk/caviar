/**
 * quickStart.ts
 * 
 * Quick initialization script to test the integrated holographic system
 * Drop this into your main.ts or run separately for testing
 */

import { HolographicDisplay } from '../../../../standalone-holo/src/HolographicDisplay';
import { HolographicPipelineIntegrator } from './HolographicPipelineIntegrator';
import { UIController } from './UIController';
import { AdaptiveRenderer } from '../adaptiveRenderer';

export async function quickStartHolographic() {
  console.log('🚀 Quick Start: Initializing Holographic Display System...');
  
  try {
    // 1. Get canvas
    const canvas = document.getElementById('canvas') as HTMLCanvasElement;
    if (!canvas) {
      throw new Error('Canvas element with id="canvas" not found');
    }
    
    // 2. Check WebGPU support
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported. Please use Chrome/Edge/Safari with WebGPU enabled.');
    }
    
    // 3. Create adaptive renderer
    const adaptiveRenderer = new AdaptiveRenderer(60); // Target 60 FPS
    
    // 4. Initialize holographic display
    const display = new HolographicDisplay();
    await display.init(canvas);
    
    // 5. Create pipeline integrator
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter available');
    
    const device = await adapter.requestDevice();
    
    const integrator = new HolographicPipelineIntegrator({
      device,
      canvas,
      qualityScale: 1.0,
      enablePrediction: true,
      debugMode: true
    });
    
    // 6. Wire everything together
    await integrator.initialize(
      (display as any).propagator,
      (display as any).encoder,
      (display as any).pose,
      adaptiveRenderer,
      (display as any).composer
    );
    
    // 7. Create UI controller
    const uiContainer = document.body;
    const uiController = new UIController({
      container: uiContainer,
      integrator,
      adaptiveRenderer
    });
    
    // 8. Start the display
    await display.start();
    
    // 9. Success message
    console.log('✅ Holographic Display System Initialized!');
    console.log('   - Press SPACE to toggle parallax');
    console.log('   - Press Q to cycle quality');
    console.log('   - Press M to cycle render modes');
    console.log('   - Press D to toggle debug panel');
    
    // 10. Add test pattern button
    const testBtn = document.createElement('button');
    testBtn.textContent = 'Load Test Pattern';
    testBtn.style.cssText = 'position:fixed;bottom:10px;right:10px;padding:10px;background:#0f0;color:#000;border:none;border-radius:5px;font-family:monospace;cursor:pointer;z-index:1001;';
    testBtn.onclick = () => {
      (display as any).processTestPattern();
    };
    document.body.appendChild(testBtn);
    
    // Return references for external control
    return {
      display,
      integrator,
      uiController,
      adaptiveRenderer
    };
    
  } catch (error) {
    console.error('❌ Failed to initialize holographic display:', error);
    
    // Show error message to user
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#f00;color:#fff;padding:20px;border-radius:10px;font-family:monospace;max-width:400px;text-align:center;z-index:9999;';
    errorDiv.innerHTML = `
      <h2>Initialization Failed</h2>
      <p>${error.message}</p>
      <button onclick="location.reload()" style="margin-top:10px;padding:5px 10px;background:#fff;color:#f00;border:none;border-radius:5px;cursor:pointer;">Reload</button>
    `;
    document.body.appendChild(errorDiv);
    
    throw error;
  }
}

// Auto-start if this is the main entry point
if (typeof window !== 'undefined' && document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', quickStartHolographic);
} else if (typeof window !== 'undefined') {
  quickStartHolographic();
}
