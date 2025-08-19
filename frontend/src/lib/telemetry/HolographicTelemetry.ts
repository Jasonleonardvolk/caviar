/**
 * HolographicTelemetry - Specialized telemetry for Carrollian physics A/B testing
 * Tracks frame quality metrics during shadow mode and rollout
 */

import { detectDeviceTier } from '../config/features';

export interface HoloMetrics {
  mode: 'standard' | 'carrollian';
  ssim?: number;                // Structural similarity (optional, computed on downsample)
  speckle?: number;             // Simple variance-based speckle metric
  frameMs: number;              // Frame time in milliseconds
  deviceTier: string;           // Device tier classification
  jitterIdx?: number;           // Jitter index (proxy for thermal/throttle)
  ts: number;                   // Timestamp
}

export class HolographicTelemetry {
  private endpoint: string;
  private frameBuffer: HoloMetrics[] = [];
  private bufferSize = 100;
  private lastFrameTime = 0;
  private frameCount = 0;
  private jitterHistory: number[] = [];
  
  constructor(endpoint = '/api/telemetry/hologram') {
    this.endpoint = endpoint;
  }
  
  /**
   * Log frame quality metrics - fire and forget
   */
  logFrameQuality(metrics: HoloMetrics): void {
    // Add device tier if not present
    if (!metrics.deviceTier) {
      metrics.deviceTier = detectDeviceTier();
    }
    
    // Calculate jitter if we have history
    if (this.lastFrameTime > 0) {
      const expectedFrameTime = 16.67; // 60fps target
      const frameDelta = metrics.frameMs - this.lastFrameTime;
      const jitter = Math.abs(frameDelta - expectedFrameTime);
      this.jitterHistory.push(jitter);
      
      // Keep only last 30 frames for jitter calculation
      if (this.jitterHistory.length > 30) {
        this.jitterHistory.shift();
      }
      
      // Calculate jitter index (0-100 scale)
      const avgJitter = this.jitterHistory.reduce((a, b) => a + b, 0) / this.jitterHistory.length;
      metrics.jitterIdx = Math.min(100, Math.round(avgJitter * 10));
    }
    
    this.lastFrameTime = metrics.frameMs;
    this.frameCount++;
    
    // Buffer metrics
    this.frameBuffer.push(metrics);
    
    // Flush when buffer is full
    if (this.frameBuffer.length >= this.bufferSize) {
      this.flush();
    }
  }
  
  /**
   * Calculate SSIM between two frame buffers (simplified)
   * This would be computed on downsampled versions for performance
   */
  calculateSSIM(reference: Float32Array, test: Float32Array): number {
    // Simplified SSIM calculation
    // In production, this would use a proper SSIM implementation
    const size = Math.min(reference.length, test.length);
    let sum = 0;
    let refMean = 0;
    let testMean = 0;
    
    // Calculate means
    for (let i = 0; i < size; i++) {
      refMean += reference[i];
      testMean += test[i];
    }
    refMean /= size;
    testMean /= size;
    
    // Calculate variance and covariance
    let refVar = 0;
    let testVar = 0;
    let covar = 0;
    
    for (let i = 0; i < size; i++) {
      const refDiff = reference[i] - refMean;
      const testDiff = test[i] - testMean;
      refVar += refDiff * refDiff;
      testVar += testDiff * testDiff;
      covar += refDiff * testDiff;
    }
    
    refVar /= size;
    testVar /= size;
    covar /= size;
    
    // SSIM formula (simplified)
    const c1 = 0.01 * 0.01;
    const c2 = 0.03 * 0.03;
    
    const ssim = ((2 * refMean * testMean + c1) * (2 * covar + c2)) /
                 ((refMean * refMean + testMean * testMean + c1) * (refVar + testVar + c2));
    
    return Math.max(0, Math.min(1, ssim));
  }
  
  /**
   * Calculate speckle metric (variance-based)
   */
  calculateSpeckle(buffer: Float32Array, width: number, height: number): number {
    // Sample a grid of patches and calculate local variance
    const patchSize = 8;
    const stride = 16;
    let totalVariance = 0;
    let patchCount = 0;
    
    for (let y = 0; y < height - patchSize; y += stride) {
      for (let x = 0; x < width - patchSize; x += stride) {
        let mean = 0;
        let values: number[] = [];
        
        // Calculate patch mean
        for (let py = 0; py < patchSize; py++) {
          for (let px = 0; px < patchSize; px++) {
            const idx = (y + py) * width + (x + px);
            const val = buffer[idx] || 0;
            values.push(val);
            mean += val;
          }
        }
        mean /= (patchSize * patchSize);
        
        // Calculate patch variance
        let variance = 0;
        for (const val of values) {
          variance += (val - mean) * (val - mean);
        }
        variance /= values.length;
        
        totalVariance += variance;
        patchCount++;
      }
    }
    
    return patchCount > 0 ? totalVariance / patchCount : 0;
  }
  
  /**
   * Flush buffered metrics to server
   */
  private flush(): void {
    if (this.frameBuffer.length === 0) return;
    
    const payload = {
      metrics: this.frameBuffer,
      summary: {
        deviceTier: detectDeviceTier(),
        frameCount: this.frameCount,
        avgFrameMs: this.frameBuffer.reduce((a, b) => a + b.frameMs, 0) / this.frameBuffer.length,
        timestamp: Date.now()
      }
    };
    
    // Use sendBeacon for reliability
    const blob = new Blob([JSON.stringify(payload) + '\n'], {
      type: 'text/plain'
    });
    
    if (navigator.sendBeacon) {
      navigator.sendBeacon(this.endpoint, blob);
    } else {
      // Fallback to fetch
      fetch(this.endpoint, {
        method: 'POST',
        body: blob,
        headers: {
          'Content-Type': 'text/plain'
        },
        // Don't wait for response
        keepalive: true
      }).catch(() => {
        // Silently fail - this is fire-and-forget telemetry
      });
    }
    
    // Clear buffer
    this.frameBuffer = [];
  }
  
  /**
   * Force flush (e.g., on page unload)
   */
  forceFlush(): void {
    this.flush();
  }
  
  /**
   * Get current telemetry stats
   */
  getStats(): {
    frameCount: number;
    avgJitter: number;
    bufferSize: number;
    deviceTier: string;
  } {
    const avgJitter = this.jitterHistory.length > 0
      ? this.jitterHistory.reduce((a, b) => a + b, 0) / this.jitterHistory.length
      : 0;
    
    return {
      frameCount: this.frameCount,
      avgJitter,
      bufferSize: this.frameBuffer.length,
      deviceTier: detectDeviceTier()
    };
  }
}

// Global instance
export const holographicTelemetry = new HolographicTelemetry();

// Auto-flush on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    holographicTelemetry.forceFlush();
  });
  
  // Also flush on visibility change
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
      holographicTelemetry.forceFlush();
    }
  });
}
