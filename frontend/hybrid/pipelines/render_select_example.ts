// ${IRIS_ROOT}\frontend\hybrid\pipelines\render_select_example.ts
/**
 * Example integration of dynamic phase tuning into render pipeline
 * This shows how to wire PhaseTuner with zero overhead
 */

import { getPhaseTuner } from '@/hybrid/lib/post/PhaseTuner';
import { getPhaseConfig } from '@/hybrid/lib/post/PhaseConfig';

// Import your existing pipeline components
// import { hybridWavefieldBlend } from './wavefield';
// import { compose } from './compose';

export class RenderPipeline {
  private phaseTuner = getPhaseTuner();
  private phaseConfig = getPhaseConfig();
  private deviceKey: string = 'default';
  
  async init(): Promise<void> {
    // Detect device tier (you already have this logic)
    this.deviceKey = await this.detectDeviceTier();
    
    // Initialize phase tuner with device-specific configs
    await this.phaseTuner.init(this.deviceKey);
    
    // Log hotkey instructions once
    if (this.phaseConfig.get('metricsEnabled')) {
      console.log(`
[Phase Tuning] Keyboard shortcuts enabled:
  Alt+↑/↓     : Adjust TV Lambda
  Alt+←/→     : Adjust Max Correction  
  Alt+Z       : Cycle methods (TV→Zernike→LUT→Off)
  Alt+A       : Toggle auto-adaptation
  Alt+D       : Dump current parameters
  Alt+R       : Reset to defaults

Current URL: ${window.location.href}
Test with: ?phase=tv&tv=0.08&maxc=0.25&auto=1&metrics=1
      `);
    }
  }
  
  /**
   * Main render function - called every frame
   */
  render(
    encoder: GPUCommandEncoder,
    width: number,
    height: number,
    timestamp: number
  ): void {
    // ... your existing wavefield accumulation ...
    // const { reBuf, imBuf } = await hybridWavefieldBlend(...);
    
    // Placeholder for example - you'd use your actual buffers
    const reBuf = this.getWaveRealBuffer();
    const imBuf = this.getWaveImagBuffer();
    const maskBuf = this.getSegmentationMask(); // Optional
    
    // Analyze wavefield for artifacts (optional, for auto-adaptation)
    if (this.phaseConfig.get('autoAdaptEnabled')) {
      this.phaseTuner.analyzeWavefield(reBuf, imBuf, width, height);
    }
    
    // Apply phase correction - ZERO OVERHEAD!
    // Config lookups are cached, no allocation, no string parsing
    this.phaseTuner.apply(encoder, reBuf, imBuf, width, height, maskBuf);
    
    // ... continue with your compose/encode pipeline ...
    // compose(encoder, reBuf, imBuf, ...);
  }
  
  /**
   * Detect device tier (placeholder - use your existing logic)
   */
  private async detectDeviceTier(): Promise<string> {
    // You already have this in capabilities.ts / limits_resolver.mjs
    const gpu = navigator.gpu;
    if (!gpu) return 'default';
    
    const adapter = await gpu.requestAdapter();
    if (!adapter) return 'default';
    
    // Simplified detection
    const vendor = adapter.info?.vendor || '';
    const device = adapter.info?.device || '';
    
    if (vendor.includes('Apple')) {
      if (device.includes('M2')) return 'mac_m2';
      if (device.includes('A17')) return 'ios_a17';
    }
    
    return 'default';
  }
  
  // Placeholder methods - replace with your actual buffer getters
  private getWaveRealBuffer(): GPUBuffer {
    throw new Error('Replace with actual implementation');
  }
  
  private getWaveImagBuffer(): GPUBuffer {
    throw new Error('Replace with actual implementation');
  }
  
  private getSegmentationMask(): GPUBuffer | undefined {
    // Optional - return undefined if not using masks
    return undefined;
  }
}

// ============================================
// SIMPLE INTEGRATION (if not using the class)
// ============================================

export async function simpleIntegration(
  encoder: GPUCommandEncoder,
  reBuf: GPUBuffer,
  imBuf: GPUBuffer,
  width: number,
  height: number
): Promise<void> {
  // Get singleton tuner
  const tuner = getPhaseTuner();
  
  // Initialize once (do this at app startup)
  // Using a closure variable to track initialization
  // @ts-ignore - accessing private static-like property
  if (!simpleIntegration.initialized) {
    await tuner.init('default');
    // @ts-ignore
    simpleIntegration.initialized = true;
  }
  
  // Apply correction - that's it!
  tuner.apply(encoder, reBuf, imBuf, width, height);
}

// ============================================
// URL EXAMPLES FOR TESTING
// ============================================

/*
Test URLs - copy and paste into browser:

1. Basic TV polisher with default settings:
   https://yourapp.com/?phase=tv

2. Aggressive smoothing for heavy artifacts:
   https://yourapp.com/?phase=tv&tv=0.12&maxc=0.30

3. Gentle correction for minimal artifacts:
   https://yourapp.com/?phase=tv&tv=0.04&maxc=0.15

4. Zernike with defocus correction:
   https://yourapp.com/?phase=zernike&defocus=0.08&astig0=0.02

5. Auto-adaptation enabled with metrics:
   https://yourapp.com/?phase=auto&auto=1&metrics=1

6. Disable all corrections:
   https://yourapp.com/?phase=off

7. Full debug mode:
   https://yourapp.com/?phase=tv&tv=0.08&maxc=0.25&auto=1&metrics=1

Share settings with team:
1. Press Alt+D to dump current config
2. Copy the generated URL from console
3. Send to teammate - their settings update instantly!
*/
