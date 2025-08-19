/**
 * HolographicTelemetry.ts
 * 
 * Lightweight telemetry for v1.0 launch
 * Captures baseline performance for future Carrollian comparison
 * Zero dependencies, non-blocking, privacy-preserving
 */

export interface TelemetryEvent {
  event: string;
  data: any;
  ts: number;
  sessionId: string;
  deviceProfile?: DeviceProfile;
}

export interface DeviceProfile {
  tier: 'low' | 'medium' | 'high' | 'ultra';
  gpu: string;
  memory: number;
  cores: number;
  screen: { width: number; height: number; dpr: number };
  webgpu: boolean;
  browser: string;
  os: string;
}

export interface PerformanceMetrics {
  fps: number;
  frameTime: number;
  propagationTime?: number;
  encodingTime?: number;
  renderTime?: number;
  jitter: number;
  droppedFrames: number;
  memoryUsed?: number;
  gpuTemp?: number;
}

export interface HolographicMetrics {
  mode: 'standard' | 'carrollian';  // For future A/B
  renderMode: 'quilt' | 'stereo' | 'hologram';
  viewCount: number;
  fieldSize: number;
  propagationDistance: number;
  qualityScale: number;
  parallaxEnabled: boolean;
  sensorType: string;
}

export interface UserEngagement {
  sessionDuration: number;
  interactionCount: number;
  qualityChanges: number;
  modeChanges: number;
  headMovementIntensity: number;  // 0-1 scale
}

class HolographicTelemetry {
  private endpoint = '/api/telemetry';
  private sessionId: string;
  private deviceProfile: DeviceProfile | null = null;
  private eventQueue: TelemetryEvent[] = [];
  private flushInterval = 5000; // 5 seconds
  private maxQueueSize = 100;
  private frameCounter = 0;
  private startTime = performance.now();
  private lastFrameTime = performance.now();
  private frameTimes: number[] = [];
  private maxFrameHistory = 60;
  
  // Performance tracking
  private droppedFrames = 0;
  private lastFrameCount = 0;
  private jitterBuffer: number[] = [];
  
  // Engagement tracking
  private interactionCount = 0;
  private qualityChangeCount = 0;
  private modeChangeCount = 0;
  private headMovementSamples: number[] = [];
  
  constructor() {
    this.sessionId = this.generateSessionId();
    this.detectDevice();
    this.startFlushTimer();
    this.attachListeners();
    
    // Log session start
    this.logEvent('session_start', {
      url: window.location.href,
      referrer: document.referrer
    });
  }
  
  /**
   * Generate anonymous session ID
   */
  private generateSessionId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
  
  /**
   * Detect device capabilities
   */
  private detectDevice(): void {
    const nav = navigator as any;
    
    this.deviceProfile = {
      tier: this.detectTier(),
      gpu: this.detectGPU(),
      memory: nav.deviceMemory || 4,
      cores: nav.hardwareConcurrency || 4,
      screen: {
        width: window.screen.width,
        height: window.screen.height,
        dpr: window.devicePixelRatio || 1
      },
      webgpu: !!nav.gpu,
      browser: this.detectBrowser(),
      os: this.detectOS()
    };
    
    // Log device profile
    this.logEvent('device_profile', this.deviceProfile);
  }
  
  private detectTier(): 'low' | 'medium' | 'high' | 'ultra' {
    const nav = navigator as any;
    const memory = nav.deviceMemory || 4;
    const cores = nav.hardwareConcurrency || 4;
    const isMobile = /iPhone|iPad|Android/i.test(navigator.userAgent);
    
    if (!nav.gpu) return 'low';
    if (isMobile) {
      if (memory >= 8 || cores >= 8) return 'high';
      if (memory >= 6 || cores >= 6) return 'medium';
      return 'low';
    }
    if (memory >= 32 || cores >= 16) return 'ultra';
    if (memory >= 16 || cores >= 8) return 'high';
    return 'medium';
  }
  
  private detectGPU(): string {
    // Try to get GPU info from WebGPU
    if ((navigator as any).gpu) {
      // This will be filled when adapter is requested
      return 'WebGPU';
    }
    // Fallback to WebGL for detection
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (gl) {
      const info = (gl as any).getExtension('WEBGL_debug_renderer_info');
      if (info) {
        return (gl as any).getParameter(info.UNMASKED_RENDERER_WEBGL);
      }
    }
    return 'Unknown';
  }
  
  private detectBrowser(): string {
    const ua = navigator.userAgent;
    if (ua.includes('Chrome')) return 'Chrome';
    if (ua.includes('Safari') && !ua.includes('Chrome')) return 'Safari';
    if (ua.includes('Firefox')) return 'Firefox';
    if (ua.includes('Edge')) return 'Edge';
    return 'Unknown';
  }
  
  private detectOS(): string {
    const ua = navigator.userAgent;
    if (/iPhone|iPad/.test(ua)) return 'iOS';
    if (/Android/.test(ua)) return 'Android';
    if (/Windows/.test(ua)) return 'Windows';
    if (/Mac/.test(ua)) return 'macOS';
    if (/Linux/.test(ua)) return 'Linux';
    return 'Unknown';
  }
  
  /**
   * Core logging function
   */
  logEvent(event: string, data: any = {}): void {
    const telemetryEvent: TelemetryEvent = {
      event,
      data,
      ts: Date.now(),
      sessionId: this.sessionId,
      deviceProfile: this.deviceProfile
    };
    
    this.eventQueue.push(telemetryEvent);
    
    // Flush if queue is full
    if (this.eventQueue.length >= this.maxQueueSize) {
      this.flush();
    }
  }
  
  /**
   * Log frame performance
   */
  logFrame(metrics: Partial<PerformanceMetrics>): void {
    const now = performance.now();
    const frameTime = now - this.lastFrameTime;
    this.lastFrameTime = now;
    
    // Track frame times for FPS calculation
    this.frameTimes.push(frameTime);
    if (this.frameTimes.length > this.maxFrameHistory) {
      this.frameTimes.shift();
    }
    
    // Calculate jitter
    if (this.frameTimes.length > 1) {
      const avgFrameTime = this.frameTimes.reduce((a, b) => a + b) / this.frameTimes.length;
      const jitter = Math.abs(frameTime - avgFrameTime);
      this.jitterBuffer.push(jitter);
      if (this.jitterBuffer.length > 10) {
        this.jitterBuffer.shift();
      }
    }
    
    // Check for dropped frames (frame time > 33ms for 30fps, > 16.67ms for 60fps)
    if (frameTime > 33) {
      this.droppedFrames++;
    }
    
    this.frameCounter++;
    
    // Log performance every 60 frames (~1 second at 60fps)
    if (this.frameCounter % 60 === 0) {
      const fps = this.frameTimes.length > 0 
        ? 1000 / (this.frameTimes.reduce((a, b) => a + b) / this.frameTimes.length)
        : 0;
      
      const avgJitter = this.jitterBuffer.length > 0
        ? this.jitterBuffer.reduce((a, b) => a + b) / this.jitterBuffer.length
        : 0;
      
      this.logEvent('performance', {
        fps: Math.round(fps),
        frameTime: Math.round(frameTime * 100) / 100,
        jitter: Math.round(avgJitter * 100) / 100,
        droppedFrames: this.droppedFrames,
        ...metrics
      });
      
      // Reset dropped frames counter
      this.droppedFrames = 0;
    }
  }
  
  /**
   * Log holographic display metrics
   */
  logHolographic(metrics: Partial<HolographicMetrics>): void {
    this.logEvent('holographic', metrics);
  }
  
  /**
   * Log user interaction
   */
  logInteraction(action: string, details: any = {}): void {
    this.interactionCount++;
    
    // Track specific interaction types
    if (action === 'quality_change') {
      this.qualityChangeCount++;
    }
    if (action === 'mode_change') {
      this.modeChangeCount++;
    }
    
    this.logEvent('interaction', {
      action,
      ...details,
      totalInteractions: this.interactionCount
    });
  }
  
  /**
   * Log head tracking data
   */
  logHeadTracking(pose: { x: number; y: number; z: number; rx: number; ry: number }): void {
    // Calculate movement intensity
    const movement = Math.sqrt(pose.x * pose.x + pose.y * pose.y + pose.z * pose.z);
    this.headMovementSamples.push(movement);
    
    if (this.headMovementSamples.length > 100) {
      this.headMovementSamples.shift();
    }
    
    // Log summary every 100 samples
    if (this.headMovementSamples.length === 100) {
      const avgMovement = this.headMovementSamples.reduce((a, b) => a + b) / this.headMovementSamples.length;
      const maxMovement = Math.max(...this.headMovementSamples);
      
      this.logEvent('head_tracking', {
        avgMovement: Math.round(avgMovement * 1000) / 1000,
        maxMovement: Math.round(maxMovement * 1000) / 1000,
        samples: 100
      });
    }
  }
  
  /**
   * Log errors and warnings
   */
  logError(error: string, details: any = {}): void {
    this.logEvent('error', {
      error,
      ...details,
      url: window.location.href,
      userAgent: navigator.userAgent
    });
    
    // Immediately flush errors
    this.flush();
  }
  
  /**
   * Log WebGPU adapter info when available
   */
  logGPUAdapter(adapter: GPUAdapter): void {
    const features = Array.from(adapter.features || []);
    const limits = adapter.limits;
    
    this.logEvent('gpu_adapter', {
      features,
      limits: {
        maxTextureSize: limits?.maxTextureDimension2D,
        maxBufferSize: limits?.maxBufferSize,
        maxComputeWorkgroupSize: limits?.maxComputeWorkgroupSizeX
      }
    });
  }
  
  /**
   * Calculate and log engagement metrics
   */
  logEngagement(): void {
    const sessionDuration = (performance.now() - this.startTime) / 1000; // seconds
    const avgHeadMovement = this.headMovementSamples.length > 0
      ? this.headMovementSamples.reduce((a, b) => a + b) / this.headMovementSamples.length
      : 0;
    
    const engagement: UserEngagement = {
      sessionDuration: Math.round(sessionDuration),
      interactionCount: this.interactionCount,
      qualityChanges: this.qualityChangeCount,
      modeChanges: this.modeChangeCount,
      headMovementIntensity: Math.round(avgHeadMovement * 1000) / 1000
    };
    
    this.logEvent('engagement', engagement);
  }
  
  /**
   * Flush event queue to server
   */
  private flush(): void {
    if (this.eventQueue.length === 0) return;
    
    const events = [...this.eventQueue];
    this.eventQueue = [];
    
    // Use sendBeacon for reliability (survives page unload)
    const blob = new Blob(
      [JSON.stringify({ events, sessionId: this.sessionId })],
      { type: 'application/json' }
    );
    
    // Fallback to fetch if sendBeacon fails
    if (!navigator.sendBeacon(this.endpoint, blob)) {
      fetch(this.endpoint, {
        method: 'POST',
        body: blob,
        headers: { 'Content-Type': 'application/json' },
        keepalive: true
      }).catch(err => {
        // Re-queue events on failure
        this.eventQueue.unshift(...events);
        console.warn('Telemetry flush failed:', err);
      });
    }
  }
  
  /**
   * Start periodic flush timer
   */
  private startFlushTimer(): void {
    setInterval(() => this.flush(), this.flushInterval);
  }
  
  /**
   * Attach global event listeners
   */
  private attachListeners(): void {
    // Flush on page unload
    window.addEventListener('beforeunload', () => {
      this.logEngagement();
      this.flush();
    });
    
    // Track page visibility
    document.addEventListener('visibilitychange', () => {
      this.logEvent('visibility', {
        hidden: document.hidden
      });
      if (document.hidden) {
        this.flush();
      }
    });
    
    // Track errors
    window.addEventListener('error', (e) => {
      this.logError('uncaught_error', {
        message: e.message,
        filename: e.filename,
        line: e.lineno,
        col: e.colno
      });
    });
    
    // Track WebGPU device lost
    if ((navigator as any).gpu) {
      // This will be set when device is created
      (window as any).__telemetry_gpu_device_lost = (reason: string) => {
        this.logError('gpu_device_lost', { reason });
      };
    }
  }
  
  /**
   * Get current session stats
   */
  getSessionStats(): any {
    const sessionDuration = (performance.now() - this.startTime) / 1000;
    const fps = this.frameTimes.length > 0 
      ? 1000 / (this.frameTimes.reduce((a, b) => a + b) / this.frameTimes.length)
      : 0;
    
    return {
      sessionId: this.sessionId,
      duration: Math.round(sessionDuration),
      fps: Math.round(fps),
      interactions: this.interactionCount,
      tier: this.deviceProfile?.tier
    };
  }
}

// Export singleton instance
export const telemetry = new HolographicTelemetry();

// Export for testing
export { HolographicTelemetry };
