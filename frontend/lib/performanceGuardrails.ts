/* frontend/lib/performanceGuardrails.ts
 * Adaptive performance management for mobile and thermal throttling
 * Automatically adjusts quality based on frame timing and device capabilities
 */

export interface PerformanceConfig {
  targetFPS: number;
  minFPS: number;
  maxMemoryMB: number;
  adaptiveQuality: boolean;
  thermalMonitoring: boolean;
  powerPreference: 'high-performance' | 'low-power' | 'balanced';
}

export interface PerformanceMetrics {
  fps: number;
  frameTime: number;
  droppedFrames: number;
  memoryUsageMB: number;
  gpuTime?: number;
  cpuTime?: number;
  thermalState?: 'nominal' | 'fair' | 'serious' | 'critical';
  batteryLevel?: number;
  isCharging?: boolean;
}

export interface QualitySettings {
  resolution: [number, number];
  fftSize: number;
  propagationSteps: number;
  subsampling: number;
  useF16: boolean;
  useSubgroups: boolean;
  tileSize: number;
  temporalSmoothing: number;
}

export type QualityPreset = 'ultra' | 'high' | 'medium' | 'low' | 'potato';

class PerformanceGuardrails {
  private config: PerformanceConfig;
  private metrics: PerformanceMetrics;
  private qualitySettings: QualitySettings;
  private currentPreset: QualityPreset;
  
  // Performance tracking
  private frameTimings: number[] = [];
  private lastFrameTime: number = 0;
  private droppedFrameCount: number = 0;
  private performanceObserver: PerformanceObserver | null = null;
  private memoryInterval: number | null = null;
  private rafHandle: number | null = null;
  
  // Adaptive algorithm state
  private qualityAdjustmentCooldown: number = 0;
  private consecutiveBadFrames: number = 0;
  private consecutiveGoodFrames: number = 0;
  
  // Quality presets
  private readonly presets: Record<QualityPreset, QualitySettings> = {
    ultra: {
      resolution: [2048, 2048],
      fftSize: 512,
      propagationSteps: 100,
      subsampling: 1,
      useF16: true,
      useSubgroups: true,
      tileSize: 256,
      temporalSmoothing: 0.9
    },
    high: {
      resolution: [1024, 1024],
      fftSize: 256,
      propagationSteps: 50,
      subsampling: 1,
      useF16: true,
      useSubgroups: true,
      tileSize: 128,
      temporalSmoothing: 0.85
    },
    medium: {
      resolution: [512, 512],
      fftSize: 256,
      propagationSteps: 30,
      subsampling: 2,
      useF16: false,
      useSubgroups: false,
      tileSize: 64,
      temporalSmoothing: 0.8
    },
    low: {
      resolution: [256, 256],
      fftSize: 128,
      propagationSteps: 20,
      subsampling: 2,
      useF16: false,
      useSubgroups: false,
      tileSize: 32,
      temporalSmoothing: 0.75
    },
    potato: {
      resolution: [128, 128],
      fftSize: 64,
      propagationSteps: 10,
      subsampling: 4,
      useF16: false,
      useSubgroups: false,
      tileSize: 16,
      temporalSmoothing: 0.7
    }
  };
  
  constructor(config: Partial<PerformanceConfig> = {}) {
    this.config = {
      targetFPS: 60,
      minFPS: 30,
      maxMemoryMB: 512,
      adaptiveQuality: true,
      thermalMonitoring: true,
      powerPreference: 'balanced',
      ...config
    };
    
    this.metrics = {
      fps: 0,
      frameTime: 0,
      droppedFrames: 0,
      memoryUsageMB: 0
    };
    
    // Start with medium quality
    this.currentPreset = 'medium';
    this.qualitySettings = { ...this.presets.medium };
    
    // Initialize monitoring
    this.initializeMonitoring();
  }
  
  /**
   * Initialize performance monitoring
   */
  private initializeMonitoring() {
    // Set up frame timing measurement
    if (typeof window !== 'undefined') {
      this.startFrameMonitoring();
      this.startMemoryMonitoring();
      
      // Set up thermal monitoring if available
      if (this.config.thermalMonitoring) {
        this.startThermalMonitoring();
      }
      
      // Monitor battery if available
      this.startBatteryMonitoring();
    }
  }
  
  /**
   * Start monitoring frame timings
   */
  private startFrameMonitoring() {
    const measureFrame = (timestamp: number) => {
      if (this.lastFrameTime > 0) {
        const frameTime = timestamp - this.lastFrameTime;
        
        // Add to rolling buffer (keep last 60 frames)
        this.frameTimings.push(frameTime);
        if (this.frameTimings.length > 60) {
          this.frameTimings.shift();
        }
        
        // Calculate metrics
        const avgFrameTime = this.frameTimings.reduce((a, b) => a + b, 0) / this.frameTimings.length;
        this.metrics.frameTime = avgFrameTime;
        this.metrics.fps = 1000 / avgFrameTime;
        
        // Check for dropped frames (frame took > 2x expected time)
        const expectedFrameTime = 1000 / this.config.targetFPS;
        if (frameTime > expectedFrameTime * 2) {
          this.droppedFrameCount++;
          this.metrics.droppedFrames = this.droppedFrameCount;
        }
        
        // Adaptive quality adjustment
        if (this.config.adaptiveQuality && this.qualityAdjustmentCooldown <= 0) {
          this.adjustQualityBasedOnPerformance();
        } else if (this.qualityAdjustmentCooldown > 0) {
          this.qualityAdjustmentCooldown--;
        }
      }
      
      this.lastFrameTime = timestamp;
      this.rafHandle = requestAnimationFrame(measureFrame);
    };
    
    this.rafHandle = requestAnimationFrame(measureFrame);
  }
  
  /**
   * Start monitoring memory usage
   */
  private startMemoryMonitoring() {
    // Use Performance Memory API if available (Chrome only)
    const measureMemory = () => {
      if ('memory' in performance) {
        const memory = (performance as any).memory;
        this.metrics.memoryUsageMB = memory.usedJSHeapSize / (1024 * 1024);
        
        // Check if we're approaching memory limit
        const memoryRatio = memory.usedJSHeapSize / memory.jsHeapSizeLimit;
        if (memoryRatio > 0.9) {
          console.warn('[Performance] Memory usage critical:', memoryRatio);
          this.downgradeQuality();
        }
      }
    };
    
    // Check memory every second
    this.memoryInterval = window.setInterval(measureMemory, 1000);
    measureMemory();
  }
  
  /**
   * Start thermal state monitoring (Safari only)
   */
  private async startThermalMonitoring() {
    // Check for thermal state API (Safari 16.4+)
    if ('thermalState' in navigator) {
      try {
        const thermal = (navigator as any).thermalState;
        
        // Listen for thermal state changes
        if (thermal && 'addEventListener' in thermal) {
          thermal.addEventListener('change', () => {
            const state = thermal.state;
            this.metrics.thermalState = state;
            
            console.log('[Performance] Thermal state:', state);
            
            // Adjust quality based on thermal state
            if (state === 'critical' || state === 'serious') {
              this.downgradeQuality();
            }
          });
        }
      } catch (err) {
        console.log('[Performance] Thermal monitoring not available');
      }
    }
  }
  
  /**
   * Start battery monitoring
   */
  private async startBatteryMonitoring() {
    if ('getBattery' in navigator) {
      try {
        const battery = await (navigator as any).getBattery();
        
        const updateBattery = () => {
          this.metrics.batteryLevel = battery.level;
          this.metrics.isCharging = battery.charging;
          
          // Switch to low power mode if battery is low and not charging
          if (!battery.charging && battery.level < 0.2) {
            console.log('[Performance] Low battery - switching to power save mode');
            this.setPowerSaveMode(true);
          }
        };
        
        battery.addEventListener('levelchange', updateBattery);
        battery.addEventListener('chargingchange', updateBattery);
        updateBattery();
      } catch (err) {
        console.log('[Performance] Battery monitoring not available');
      }
    }
  }
  
  /**
   * Adjust quality based on performance metrics
   */
  private adjustQualityBasedOnPerformance() {
    const targetFrameTime = 1000 / this.config.targetFPS;
    const minFrameTime = 1000 / this.config.minFPS;
    
    // Check if we're meeting performance targets
    if (this.metrics.frameTime > minFrameTime) {
      // Performance is too low
      this.consecutiveBadFrames++;
      this.consecutiveGoodFrames = 0;
      
      if (this.consecutiveBadFrames >= 10) {
        console.log('[Performance] Downgrading quality due to poor performance');
        this.downgradeQuality();
        this.consecutiveBadFrames = 0;
        this.qualityAdjustmentCooldown = 60; // Wait 60 frames before next adjustment
      }
    } else if (this.metrics.frameTime < targetFrameTime * 0.7) {
      // Performance is good, maybe we can upgrade
      this.consecutiveGoodFrames++;
      this.consecutiveBadFrames = 0;
      
      if (this.consecutiveGoodFrames >= 120) {
        console.log('[Performance] Upgrading quality due to good performance');
        this.upgradeQuality();
        this.consecutiveGoodFrames = 0;
        this.qualityAdjustmentCooldown = 60;
      }
    } else {
      // Performance is acceptable
      this.consecutiveBadFrames = 0;
      this.consecutiveGoodFrames = 0;
    }
  }
  
  /**
   * Downgrade quality preset
   */
  downgradeQuality() {
    const presetOrder: QualityPreset[] = ['ultra', 'high', 'medium', 'low', 'potato'];
    const currentIndex = presetOrder.indexOf(this.currentPreset);
    
    if (currentIndex < presetOrder.length - 1) {
      const newPreset = presetOrder[currentIndex + 1];
      this.setQualityPreset(newPreset);
      
      // Dispatch event for the app to handle
      window.dispatchEvent(new CustomEvent('quality-changed', {
        detail: { preset: newPreset, reason: 'performance' }
      }));
    }
  }
  
  /**
   * Upgrade quality preset
   */
  upgradeQuality() {
    const presetOrder: QualityPreset[] = ['ultra', 'high', 'medium', 'low', 'potato'];
    const currentIndex = presetOrder.indexOf(this.currentPreset);
    
    if (currentIndex > 0) {
      const newPreset = presetOrder[currentIndex - 1];
      this.setQualityPreset(newPreset);
      
      // Dispatch event for the app to handle
      window.dispatchEvent(new CustomEvent('quality-changed', {
        detail: { preset: newPreset, reason: 'performance' }
      }));
    }
  }
  
  /**
   * Set quality preset
   */
  setQualityPreset(preset: QualityPreset) {
    this.currentPreset = preset;
    this.qualitySettings = { ...this.presets[preset] };
    
    console.log('[Performance] Quality preset changed to:', preset, this.qualitySettings);
  }
  
  /**
   * Set power save mode
   */
  setPowerSaveMode(enabled: boolean) {
    if (enabled) {
      // Force low quality for power saving
      this.setQualityPreset('low');
      this.config.adaptiveQuality = false; // Disable auto-adjustment
      this.config.targetFPS = 30; // Lower target FPS
    } else {
      // Restore normal operation
      this.config.adaptiveQuality = true;
      this.config.targetFPS = 60;
      this.setQualityPreset('medium');
    }
    
    console.log('[Performance] Power save mode:', enabled);
  }
  
  /**
   * Get current quality settings
   */
  getQualitySettings(): QualitySettings {
    return { ...this.qualitySettings };
  }
  
  /**
   * Get current performance metrics
   */
  getMetrics(): PerformanceMetrics {
    return { ...this.metrics };
  }
  
  /**
   * Get recommended settings based on device capabilities
   */
  static getRecommendedSettings(caps: {
    tier: 'high' | 'medium' | 'low';
    isMobile: boolean;
    isIOS: boolean;
  }): { preset: QualityPreset; config: Partial<PerformanceConfig> } {
    if (caps.tier === 'high' && !caps.isMobile) {
      return {
        preset: 'high',
        config: {
          targetFPS: 60,
          minFPS: 45,
          maxMemoryMB: 2048,
          powerPreference: 'high-performance'
        }
      };
    } else if (caps.tier === 'medium' || (caps.tier === 'high' && caps.isMobile)) {
      return {
        preset: 'medium',
        config: {
          targetFPS: 60,
          minFPS: 30,
          maxMemoryMB: 512,
          powerPreference: 'balanced'
        }
      };
    } else {
      return {
        preset: 'low',
        config: {
          targetFPS: 30,
          minFPS: 20,
          maxMemoryMB: 256,
          powerPreference: 'low-power'
        }
      };
    }
  }
  
  /**
   * Clean up monitoring
   */
  destroy() {
    if (this.rafHandle) {
      cancelAnimationFrame(this.rafHandle);
      this.rafHandle = null;
    }
    
    if (this.memoryInterval) {
      clearInterval(this.memoryInterval);
      this.memoryInterval = null;
    }
    
    if (this.performanceObserver) {
      this.performanceObserver.disconnect();
      this.performanceObserver = null;
    }
    
    console.log('[Performance] Monitoring stopped');
  }
}

// Export singleton getter
let instance: PerformanceGuardrails | null = null;

export function getPerformanceGuardrails(config?: Partial<PerformanceConfig>): PerformanceGuardrails {
  if (!instance) {
    instance = new PerformanceGuardrails(config);
  }
  return instance;
}

export default PerformanceGuardrails;
