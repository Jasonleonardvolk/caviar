// iosPressure.ts - iOS-specific memory pressure handling
import { getTexturePool } from '../../../lib/webgpu/utils/texturePool';

export interface PressureConfig {
  fpsThreshold?: number;
  frameBudgetMs?: number;
  onDowngrade?: (tier: QualityTier) => void;
}

export type QualityTier = 'motion' | 'medium' | 'full';

export class IOSMemoryManager {
  private device: GPUDevice;
  private config: PressureConfig;
  private currentTier: QualityTier = 'full';
  
  // Performance monitoring
  private frameTimings: number[] = [];
  private lastFrameTime: number = 0;
  private fps: number = 60;
  private frameDrops: number = 0;
  
  // Memory monitoring
  private texturePool: ReturnType<typeof getTexturePool> | null = null;
  
  constructor(device: GPUDevice, config: PressureConfig = {}) {
    this.device = device;
    this.config = {
      fpsThreshold: config.fpsThreshold || 40,
      frameBudgetMs: config.frameBudgetMs || 16.67, // 60 FPS target
      onDowngrade: config.onDowngrade || ((tier) => {
        console.log(`Quality downgraded to: ${tier}`);
      })
    };
    
    this.setupMonitoring();
  }
  
  private setupMonitoring() {
    // Only run on iOS devices
    if (!this.isIOS()) return;
    
    // Monitor FPS
    this.monitorFPS();
    
    // Monitor device lost events
    this.device.lost.then((info) => {
      console.error('WebGPU device lost:', info.reason);
      this.hardDowngrade();
    });
    
    // Monitor texture pool if available
    try {
      this.texturePool = getTexturePool();
      this.texturePool.onBudgetExceeded = () => {
        console.warn('Texture budget exceeded');
        this.downgrade();
      };
    } catch {
      // TexturePool not initialized yet
    }
    
    // Page visibility monitoring
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        // App backgrounded - free resources
        this.releaseResources();
      }
    });
    
    // Low memory warnings (iOS-specific)
    if ('memory' in navigator && 'addEventListener' in (navigator as any).memory) {
      (navigator as any).memory.addEventListener('pressure', (event: any) => {
        if (event.state === 'critical') {
          this.hardDowngrade();
        } else if (event.state === 'serious') {
          this.downgrade();
        }
      });
    }
  }
  
  private monitorFPS() {
    const measureFrame = (timestamp: number) => {
      if (this.lastFrameTime > 0) {
        const frameTime = timestamp - this.lastFrameTime;
        this.frameTimings.push(frameTime);
        
        // Keep last 30 frames
        if (this.frameTimings.length > 30) {
          this.frameTimings.shift();
        }
        
        // Calculate FPS
        const avgFrameTime = this.frameTimings.reduce((a, b) => a + b, 0) / this.frameTimings.length;
        this.fps = 1000 / avgFrameTime;
        
        // Check for frame drops
        if (frameTime > this.config.frameBudgetMs! * 1.5) {
          this.frameDrops++;
          
          // Multiple frame drops = downgrade
          if (this.frameDrops > 5) {
            this.downgrade();
            this.frameDrops = 0;
          }
        } else if (frameTime < this.config.frameBudgetMs!) {
          // Reset counter on good frames
          this.frameDrops = Math.max(0, this.frameDrops - 1);
        }
        
        // Check sustained low FPS
        if (this.fps < this.config.fpsThreshold! && this.frameTimings.length >= 30) {
          this.downgrade();
        }
      }
      
      this.lastFrameTime = timestamp;
      
      // Continue monitoring
      requestAnimationFrame(measureFrame);
    };
    
    requestAnimationFrame(measureFrame);
  }
  
  downgrade() {
    if (this.currentTier === 'full') {
      this.setTier('medium');
    } else if (this.currentTier === 'medium') {
      this.setTier('motion');
    }
    // Already at lowest tier, can't downgrade further
  }
  
  hardDowngrade() {
    // Emergency downgrade straight to motion tier
    this.setTier('motion');
    this.releaseResources();
  }
  
  private setTier(tier: QualityTier) {
    if (tier === this.currentTier) return;
    
    console.log(`Quality tier changed: ${this.currentTier} -> ${tier}`);
    this.currentTier = tier;
    
    // Notify callback
    this.config.onDowngrade?.(tier);
    
    // Reset frame drop counter after tier change
    this.frameDrops = 0;
    this.frameTimings = [];
  }
  
  private releaseResources() {
    // Clear texture pool caches
    if (this.texturePool) {
      // Don't clear the entire pool, just reduce to minimum
      const stats = this.texturePool.getStats();
      if (stats.usage > 0.5) {
        console.log('Releasing texture pool resources');
        // Pool should implement partial clearing
      }
    }
    
    // Force garbage collection if possible
    if ('gc' in globalThis) {
      (globalThis as any).gc();
    }
  }
  
  private isIOS(): boolean {
    return /iPhone|iPad|iPod/.test(navigator.userAgent) ||
           (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1); // iPad Pro
  }
  
  getCurrentTier(): QualityTier {
    return this.currentTier;
  }
  
  getFPS(): number {
    return this.fps;
  }
  
  getStats() {
    return {
      tier: this.currentTier,
      fps: this.fps,
      frameDrops: this.frameDrops,
      avgFrameTime: this.frameTimings.reduce((a, b) => a + b, 0) / (this.frameTimings.length || 1),
      texturePoolUsage: this.texturePool?.getStats().usage || 0
    };
  }
  
  // Manual tier control for testing
  setTierManual(tier: QualityTier) {
    this.setTier(tier);
  }
}

// Quality tier configurations
export const QUALITY_CONFIGS = {
  motion: {
    fftSize: 128,
    slices: 1,
    dispersion: false,
    padding: 1.0,
    edgeKernelRadius: 3
  },
  medium: {
    fftSize: 256,
    slices: 3,
    dispersion: true,
    padding: 1.05,
    edgeKernelRadius: 5
  },
  full: {
    fftSize: 512,
    slices: 5,
    dispersion: true,
    padding: 1.1,
    edgeKernelRadius: 7
  }
} as const;

// Global instance
let globalManager: IOSMemoryManager | null = null;

export function initIOSMemoryManager(device: GPUDevice, config?: PressureConfig): IOSMemoryManager {
  if (!globalManager) {
    globalManager = new IOSMemoryManager(device, config);
  }
  return globalManager;
}

export function getIOSMemoryManager(): IOSMemoryManager | null {
  return globalManager;
}
