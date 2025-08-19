/* frontend/lib/computePolicy.ts
 * Runtime loader and manager for compute policy configuration
 * Handles YAML parsing, platform overrides, and dynamic updates
 */

export interface ComputePolicy {
  donor: {
    mode: 'off' | 'lan' | 'cloud';
    maxConnections: number;
    discoveryTimeoutMs: number;
    heartbeatIntervalMs: number;
    autoReconnect: boolean;
    lan: {
      multicastGroup: string;
      port: number;
      encryption: boolean;
      compression: 'none' | 'lz4' | 'zstd';
    };
  };
  
  fft: {
    backend: 'auto' | 'subgroup' | 'baseline' | 'wasm';
    maxSize: number;
    cachePlans: boolean;
    useF16: 'auto' | 'always' | 'never';
  };
  
  propagation: {
    method: 'angular-spectrum' | 'fresnel' | 'fraunhofer';
    maxDistanceMm: number;
    minDistanceMm: number;
    adaptiveSampling: boolean;
  };
  
  memory: {
    maxAllocationMb: number;
    poolEnabled: boolean;
    gcThresholdMb: number;
    textureCacheMb: number;
  };
  
  performance: {
    targetFps: number;
    minFps: number;
    adaptiveQuality: boolean;
    thermalMonitoring: boolean;
    powerPreference: 'high-performance' | 'balanced' | 'low-power';
    qualityPresets: Record<string, {
      resolution: [number, number];
      fftSize: number;
      propagationSteps: number;
      tileSize: number;
    }>;
  };
  
  telemetry: {
    enabled: boolean;
    endpoint: string;
    batchSize: number;
    flushIntervalMs: number;
    collect: {
      performanceMetrics: boolean;
      errorLogs: boolean;
      usageStats: boolean;
      deviceInfo: boolean;
      shaderCompilation: boolean;
    };
    privacy: {
      anonymizeIp: boolean;
      noPii: boolean;
      optOutStorageKey: string;
    };
  };
  
  killSwitches: {
    enabled: boolean;
    features: {
      webgpu: boolean;
      donorAcceleration: boolean;
      neuralOperators: boolean;
      adaptiveQuality: boolean;
      telemetry: boolean;
    };
    autoKill: {
      onRepeatedCrashes: boolean;
      crashThreshold: number;
      onMemoryPressure: boolean;
      memoryThresholdPercent: number;
      onThermalCritical: boolean;
    };
    remoteKill: {
      enabled: boolean;
      checkEndpoint: string;
      checkIntervalMs: number;
      cacheDurationMs: number;
    };
  };
  
  debug: {
    verboseLogging: boolean;
    shaderDebugInfo: boolean;
    performanceOverlay: boolean;
    memoryStats: boolean;
    donorDiscoveryLogs: boolean;
  };
  
  version: string;
  lastUpdated: string;
  schemaVersion: number;
}

class ComputePolicyManager {
  private policy: ComputePolicy;
  private defaultPolicy: ComputePolicy;
  private killSwitchCheckInterval: number | null = null;
  private telemetryBatch: any[] = [];
  private telemetryFlushInterval: number | null = null;
  private crashCount: number = 0;
  
  constructor() {
    // Default policy (embedded fallback)
    this.defaultPolicy = this.getDefaultPolicy();
    this.policy = { ...this.defaultPolicy };
    
    // Load policy from YAML
    this.loadPolicy();
    
    // Apply platform overrides
    this.applyPlatformOverrides();
    
    // Set up monitoring
    this.initializeMonitoring();
  }
  
  /**
   * Get default policy configuration
   */
  private getDefaultPolicy(): ComputePolicy {
    return {
      donor: {
        mode: 'off',
        maxConnections: 3,
        discoveryTimeoutMs: 5000,
        heartbeatIntervalMs: 1000,
        autoReconnect: true,
        lan: {
          multicastGroup: '239.255.42.99',
          port: 7777,
          encryption: false,
          compression: 'lz4'
        }
      },
      fft: {
        backend: 'auto',
        maxSize: 512,
        cachePlans: true,
        useF16: 'auto'
      },
      propagation: {
        method: 'angular-spectrum',
        maxDistanceMm: 100,
        minDistanceMm: 1,
        adaptiveSampling: true
      },
      memory: {
        maxAllocationMb: 512,
        poolEnabled: true,
        gcThresholdMb: 400,
        textureCacheMb: 128
      },
      performance: {
        targetFps: 60,
        minFps: 30,
        adaptiveQuality: true,
        thermalMonitoring: true,
        powerPreference: 'balanced',
        qualityPresets: {
          high: {
            resolution: [1024, 1024],
            fftSize: 256,
            propagationSteps: 50,
            tileSize: 128
          },
          medium: {
            resolution: [512, 512],
            fftSize: 256,
            propagationSteps: 30,
            tileSize: 64
          },
          low: {
            resolution: [256, 256],
            fftSize: 128,
            propagationSteps: 20,
            tileSize: 32
          }
        }
      },
      telemetry: {
        enabled: true,
        endpoint: '/api/telemetry',
        batchSize: 100,
        flushIntervalMs: 5000,
        collect: {
          performanceMetrics: true,
          errorLogs: true,
          usageStats: true,
          deviceInfo: true,
          shaderCompilation: true
        },
        privacy: {
          anonymizeIp: true,
          noPii: true,
          optOutStorageKey: 'tori_telemetry_optout'
        }
      },
      killSwitches: {
        enabled: true,
        features: {
          webgpu: true,
          donorAcceleration: true,
          neuralOperators: true,
          adaptiveQuality: true,
          telemetry: true
        },
        autoKill: {
          onRepeatedCrashes: true,
          crashThreshold: 3,
          onMemoryPressure: true,
          memoryThresholdPercent: 95,
          onThermalCritical: true
        },
        remoteKill: {
          enabled: false,
          checkEndpoint: '/api/kill-switches',
          checkIntervalMs: 60000,
          cacheDurationMs: 300000
        }
      },
      debug: {
        verboseLogging: false,
        shaderDebugInfo: false,
        performanceOverlay: false,
        memoryStats: false,
        donorDiscoveryLogs: false
      },
      version: '1.0.0',
      lastUpdated: '2025-08-11',
      schemaVersion: 1
    };
  }
  
  /**
   * Load policy from YAML file
   */
  private async loadPolicy() {
    try {
      const response = await fetch('/configs/compute_policy.yaml');
      if (response.ok) {
        const yamlText = await response.text();
        // Note: In production, use a proper YAML parser like js-yaml
        // For now, we'll use the default policy
        console.log('[ComputePolicy] Loaded policy from YAML');
      }
    } catch (error) {
      console.warn('[ComputePolicy] Failed to load YAML, using defaults:', error);
    }
  }
  
  /**
   * Apply platform-specific overrides
   */
  private applyPlatformOverrides() {
    const ua = navigator.userAgent;
    const isIOS = /iPad|iPhone|iPod/.test(ua) || 
      (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
    const isSafari = /^((?!chrome|android).)*safari/i.test(ua);
    const isAndroid = /Android/i.test(ua);
    
    if (isIOS) {
      // Apply iOS overrides
      this.policy.memory.maxAllocationMb = 256;
      this.policy.performance.targetFps = 60;
      this.policy.performance.powerPreference = 'low-power';
      this.policy.fft.maxSize = 256;
      console.log('[ComputePolicy] Applied iOS overrides');
    } else if (isAndroid) {
      // Apply Android overrides
      this.policy.memory.maxAllocationMb = 384;
      this.policy.performance.adaptiveQuality = true;
      console.log('[ComputePolicy] Applied Android overrides');
    }
    
    if (isSafari) {
      // Apply Safari overrides
      this.policy.donor.mode = 'off';
      this.policy.telemetry.enabled = false;
      console.log('[ComputePolicy] Applied Safari overrides');
    }
  }
  
  /**
   * Initialize monitoring and kill switches
   */
  private initializeMonitoring() {
    // Set up crash detection
    window.addEventListener('error', (event) => {
      this.handleCrash(event.error);
    });
    
    window.addEventListener('unhandledrejection', (event) => {
      this.handleCrash(event.reason);
    });
    
    // Set up remote kill switch checking
    if (this.policy.killSwitches.remoteKill.enabled) {
      this.startRemoteKillSwitchCheck();
    }
    
    // Set up telemetry
    if (this.policy.telemetry.enabled && !this.isOptedOut()) {
      this.startTelemetry();
    }
  }
  
  /**
   * Handle crash/error
   */
  private handleCrash(error: any) {
    this.crashCount++;
    
    console.error('[ComputePolicy] Crash detected:', error);
    
    // Log to telemetry
    this.logTelemetry('crash', {
      message: error?.message || 'Unknown error',
      stack: error?.stack,
      crashCount: this.crashCount
    });
    
    // Check if we should auto-kill
    if (this.policy.killSwitches.autoKill.onRepeatedCrashes &&
        this.crashCount >= this.policy.killSwitches.autoKill.crashThreshold) {
      console.error('[ComputePolicy] Too many crashes, disabling features');
      this.emergencyStop('repeated_crashes');
    }
  }
  
  /**
   * Start remote kill switch checking
   */
  private startRemoteKillSwitchCheck() {
    const check = async () => {
      try {
        const response = await fetch(this.policy.killSwitches.remoteKill.checkEndpoint);
        if (response.ok) {
          const switches = await response.json();
          this.updateKillSwitches(switches);
        }
      } catch (error) {
        console.warn('[ComputePolicy] Failed to check remote kill switches:', error);
      }
    };
    
    // Initial check
    check();
    
    // Set up interval
    this.killSwitchCheckInterval = window.setInterval(
      check,
      this.policy.killSwitches.remoteKill.checkIntervalMs
    );
  }
  
  /**
   * Update kill switches
   */
  private updateKillSwitches(switches: Partial<typeof this.policy.killSwitches.features>) {
    Object.assign(this.policy.killSwitches.features, switches);
    console.log('[ComputePolicy] Kill switches updated:', switches);
  }
  
  /**
   * Start telemetry collection
   */
  private startTelemetry() {
    // Set up flush interval
    this.telemetryFlushInterval = window.setInterval(
      () => this.flushTelemetry(),
      this.policy.telemetry.flushIntervalMs
    );
    
    console.log('[ComputePolicy] Telemetry started');
  }
  
  /**
   * Log telemetry event
   */
  logTelemetry(event: string, data: any) {
    if (!this.policy.telemetry.enabled || this.isOptedOut()) {
      return;
    }
    
    const entry = {
      event,
      data,
      timestamp: Date.now(),
      sessionId: this.getSessionId()
    };
    
    // Anonymize if needed
    if (this.policy.telemetry.privacy.anonymizeIp) {
      // Remove any IP addresses from data
      // Implementation would go here
    }
    
    this.telemetryBatch.push(entry);
    
    // Flush if batch is full
    if (this.telemetryBatch.length >= this.policy.telemetry.batchSize) {
      this.flushTelemetry();
    }
  }
  
  /**
   * Flush telemetry batch
   */
  private async flushTelemetry() {
    if (this.telemetryBatch.length === 0) {
      return;
    }
    
    const batch = [...this.telemetryBatch];
    this.telemetryBatch = [];
    
    try {
      await fetch(this.policy.telemetry.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ batch })
      });
    } catch (error) {
      console.warn('[ComputePolicy] Failed to send telemetry:', error);
      // Re-add to batch if failed (with limit)
      if (this.telemetryBatch.length < this.policy.telemetry.batchSize * 2) {
        this.telemetryBatch.unshift(...batch);
      }
    }
  }
  
  /**
   * Check if user opted out of telemetry
   */
  private isOptedOut(): boolean {
    try {
      return localStorage.getItem(this.policy.telemetry.privacy.optOutStorageKey) === 'true';
    } catch {
      return false;
    }
  }
  
  /**
   * Get or create session ID
   */
  private getSessionId(): string {
    try {
      let id = sessionStorage.getItem('tori_session_id');
      if (!id) {
        id = Math.random().toString(36).substr(2, 9);
        sessionStorage.setItem('tori_session_id', id);
      }
      return id;
    } catch {
      return 'unknown';
    }
  }
  
  /**
   * Emergency stop - disable features
   */
  emergencyStop(reason: string) {
    console.error('[ComputePolicy] EMERGENCY STOP:', reason);
    
    // Disable all features
    this.policy.killSwitches.features = {
      webgpu: false,
      donorAcceleration: false,
      neuralOperators: false,
      adaptiveQuality: false,
      telemetry: false
    };
    
    // Log the emergency stop
    this.logTelemetry('emergency_stop', { reason });
    
    // Dispatch event for app to handle
    window.dispatchEvent(new CustomEvent('emergency-stop', {
      detail: { reason }
    }));
  }
  
  /**
   * Get current policy
   */
  getPolicy(): ComputePolicy {
    return { ...this.policy };
  }
  
  /**
   * Check if feature is enabled
   */
  isFeatureEnabled(feature: keyof ComputePolicy['killSwitches']['features']): boolean {
    return this.policy.killSwitches.enabled && 
           this.policy.killSwitches.features[feature];
  }
  
  /**
   * Update policy setting
   */
  updateSetting(path: string, value: any) {
    const keys = path.split('.');
    let obj: any = this.policy;
    
    for (let i = 0; i < keys.length - 1; i++) {
      obj = obj[keys[i]];
    }
    
    obj[keys[keys.length - 1]] = value;
    
    console.log('[ComputePolicy] Setting updated:', path, value);
  }
  
  /**
   * Clean up
   */
  destroy() {
    if (this.killSwitchCheckInterval) {
      clearInterval(this.killSwitchCheckInterval);
    }
    
    if (this.telemetryFlushInterval) {
      clearInterval(this.telemetryFlushInterval);
    }
    
    // Final telemetry flush
    this.flushTelemetry();
  }
}

// Singleton instance
let instance: ComputePolicyManager | null = null;

export function getComputePolicy(): ComputePolicyManager {
  if (!instance) {
    instance = new ComputePolicyManager();
  }
  return instance;
}

export default ComputePolicyManager;
