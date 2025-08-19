// ${IRIS_ROOT}\frontend\hybrid\lib\post\PhaseTuner.ts
/**
 * Zero-overhead integration layer for dynamic phase tuning
 * Connects PhaseConfig to the actual correction components
 */

import { getPhaseConfig, type PhaseParams } from './PhaseConfig';
import { PhasePolisher } from './phasePolisher';
import { ZernikeApply } from './zernikeApply';
import { ApplyPhaseLUT } from './applyPhaseLUT';

export interface PhaseMetrics {
  method: string;
  avgCorrection: number;
  maxCorrection: number;
  frameTime: number;
  seamsDetected: number;
  overSmoothed: number;
}

export class PhaseTuner {
  private config = getPhaseConfig();
  private polisher?: PhasePolisher;
  private zernike?: ZernikeApply;
  private lut?: ApplyPhaseLUT;
  
  private frameCount = 0;
  private metricsInterval = 60;
  private lastMetrics: PhaseMetrics = {
    method: 'off',
    avgCorrection: 0,
    maxCorrection: 0,
    frameTime: 0,
    seamsDetected: 0,
    overSmoothed: 0,
  };
  
  constructor() {
    // Listen for config changes
    this.config.onChange((params) => {
      this.updateComponents(params);
    });
  }
  
  /**
   * Initialize phase correction components
   */
  async init(deviceKey?: string): Promise<void> {
    // Initialize TV Polisher
    this.polisher = new PhasePolisher();
    await this.polisher.init();
    
    // Initialize Zernike
    this.zernike = new ZernikeApply();
    await this.zernike.init();
    
    // Load device-specific Zernike coefficients if available
    if (deviceKey) {
      try {
        await this.zernike.setCoeffsFromUrl(`/corrections/${deviceKey}/zernike.json`);
        console.log('[PhaseTuner] Loaded Zernike coefficients for', deviceKey);
      } catch {
        // Use defaults
      }
    }
    
    // Initialize LUT if available
    if (deviceKey) {
      try {
        this.lut = new ApplyPhaseLUT();
        await this.lut.init();
        await this.lut.loadFromUrls(
          `/corrections/${deviceKey}/phase_offset_f32.bin`,
          `/corrections/${deviceKey}/meta.json`
        );
        console.log('[PhaseTuner] Loaded phase LUT for', deviceKey);
      } catch {
        // LUT not available for this device
      }
    }
    
    // Load remote config if available
    if (deviceKey) {
      await this.config.loadRemote(deviceKey);
    }
    
    console.log('[PhaseTuner] Initialized with method:', this.config.get('activeMethod'));
  }
  
  /**
   * Apply phase correction based on current configuration
   * This is called every frame - zero overhead from config lookups
   */
  apply(
    encoder: GPUCommandEncoder,
    reBuf: GPUBuffer,
    imBuf: GPUBuffer,
    width: number,
    height: number,
    maskBuf?: GPUBuffer
  ): void {
    const startTime = performance.now();
    const method = this.config.get('activeMethod');
    
    switch (method) {
      case 'tv':
        if (this.polisher) {
          this.polisher.run(encoder, reBuf, imBuf, {
            width,
            height,
            tvLambda: this.config.get('tvLambda'),
            maxCorrection: this.config.get('tvMaxCorrection'),
            useMask: this.config.get('tvUseMask') && !!maskBuf,
          }, maskBuf);
        }
        break;
        
      case 'zernike':
        if (this.zernike) {
          this.zernike.run(encoder, reBuf, imBuf, {
            width,
            height,
            maxCorrection: this.config.get('zernikeMaxCorrection'),
            softness: this.config.get('zernikeSoftness'),
            outsideBehavior: 1,
          });
        }
        break;
        
      case 'lut':
        if (this.lut) {
          this.lut.run(encoder, reBuf, imBuf, width, height, {
            gain: this.config.get('lutGain'),
            maxCorrection: this.config.get('lutMaxCorrection'),
          });
        }
        break;
        
      case 'auto':
        // Auto-select based on what's available and metrics
        if (this.lastMetrics.seamsDetected > 0.1 && this.polisher) {
          this.polisher.run(encoder, reBuf, imBuf, {
            width,
            height,
            tvLambda: this.config.get('tvLambda'),
            maxCorrection: this.config.get('tvMaxCorrection'),
            useMask: this.config.get('tvUseMask') && !!maskBuf,
          }, maskBuf);
        } else if (this.lut) {
          this.lut.run(encoder, reBuf, imBuf, width, height, {
            gain: this.config.get('lutGain'),
            maxCorrection: this.config.get('lutMaxCorrection'),
          });
        } else if (this.zernike) {
          this.zernike.run(encoder, reBuf, imBuf, {
            width,
            height,
            maxCorrection: this.config.get('zernikeMaxCorrection'),
            softness: this.config.get('zernikeSoftness'),
            outsideBehavior: 1,
          });
        }
        break;
        
      case 'off':
      default:
        // No correction
        break;
    }
    
    const frameTime = performance.now() - startTime;
    this.updateMetrics(method, frameTime);
  }
  
  /**
   * Update components when config changes
   */
  private updateComponents(params: PhaseParams): void {
    // Update Zernike coefficients if they changed
    if (this.zernike) {
      this.zernike.setCoeffs({
        tipX: params.zernikeTipX,
        tiltY: params.zernikeTiltY,
        defocus: params.zernikeDefocus,
        astig0: params.zernikeAstig0,
        astig45: params.zernikeAstig45,
        comaX: params.zernikeComaX,
        comaY: params.zernikeComaY,
        spherical: params.zernikeSpherical,
      });
    }
  }
  
  /**
   * Update metrics and trigger auto-adaptation
   */
  private updateMetrics(method: string, frameTime: number): void {
    this.frameCount++;
    
    // Update last metrics
    this.lastMetrics.method = method;
    this.lastMetrics.frameTime = frameTime;
    
    // Calculate frame budget remaining (assuming 16.67ms target)
    const frameTimeBudget = Math.max(0, 16.67 - frameTime);
    
    // Update config metrics for auto-adaptation
    this.config.updateMetrics({
      frameTimeBudget,
      seamsDetected: this.lastMetrics.seamsDetected,
      overSmoothed: this.lastMetrics.overSmoothed,
    });
    
    // Log metrics periodically if enabled
    if (this.config.get('metricsEnabled') && this.frameCount % this.metricsInterval === 0) {
      console.log('[PhaseTuner] Metrics:', {
        method,
        frameTime: frameTime.toFixed(2) + 'ms',
        budget: frameTimeBudget.toFixed(2) + 'ms',
        tvLambda: this.config.get('tvLambda'),
        maxCorrection: this.config.get('tvMaxCorrection'),
      });
    }
  }
  
  /**
   * Analyze wavefield for artifacts (call before correction)
   */
  async analyzeWavefield(
    reBuf: GPUBuffer,
    imBuf: GPUBuffer,
    width: number,
    height: number
  ): Promise<void> {
    // This would analyze the wavefield for seams and oversmoothing
    // For now, we'll use placeholder metrics
    // In production, this would run a small compute shader to detect edges
    
    // Placeholder: randomly detect some artifacts for demo
    if (Math.random() < 0.1) {
      this.lastMetrics.seamsDetected = 0.2;
    } else {
      this.lastMetrics.seamsDetected = 0;
    }
    
    // Detect oversmoothing based on TV lambda
    if (this.config.get('tvLambda') > 0.12) {
      this.lastMetrics.overSmoothed = 0.6;
    } else {
      this.lastMetrics.overSmoothed = 0;
    }
  }
  
  /**
   * Get current metrics
   */
  getMetrics(): PhaseMetrics {
    return { ...this.lastMetrics };
  }
  
  /**
   * Clean up resources
   */
  destroy(): void {
    // Components clean up their own GPU resources
  }
}

// Singleton instance
let tunerInstance: PhaseTuner | null = null;

export function getPhaseTuner(): PhaseTuner {
  if (!tunerInstance) {
    tunerInstance = new PhaseTuner();
  }
  return tunerInstance;
}
