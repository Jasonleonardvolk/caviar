// ${IRIS_ROOT}\frontend\hybrid\lib\post\PhaseConfig.ts
/**
 * Zero-overhead runtime configuration for phase correction
 * Supports URL params, keyboard shortcuts, auto-adaptation, and remote config
 */

export interface PhaseParams {
  // TV Polisher
  tvLambda: number;
  tvMaxCorrection: number;
  tvUseMask: boolean;
  
  // Zernike
  zernikeTipX: number;
  zernikeTiltY: number;
  zernikeDefocus: number;
  zernikeAstig0: number;
  zernikeAstig45: number;
  zernikeComaX: number;
  zernikeComaY: number;
  zernikeSpherical: number;
  zernikeMaxCorrection: number;
  zernikeSoftness: number;
  
  // LUT
  lutGain: number;
  lutMaxCorrection: number;
  
  // Global
  activeMethod: 'off' | 'tv' | 'zernike' | 'lut' | 'auto';
  autoAdaptEnabled: boolean;
  metricsEnabled: boolean;
}

const DEFAULTS: PhaseParams = {
  // TV Polisher defaults
  tvLambda: 0.08,
  tvMaxCorrection: 0.25,
  tvUseMask: false,
  
  // Zernike defaults
  zernikeTipX: 0.0,
  zernikeTiltY: 0.0,
  zernikeDefocus: 0.06,
  zernikeAstig0: 0.01,
  zernikeAstig45: -0.02,
  zernikeComaX: 0.0,
  zernikeComaY: 0.0,
  zernikeSpherical: 0.0,
  zernikeMaxCorrection: 0.25,
  zernikeSoftness: 0.05,
  
  // LUT defaults
  lutGain: 1.0,
  lutMaxCorrection: 0.30,
  
  // Global defaults
  activeMethod: 'tv',
  autoAdaptEnabled: false,
  metricsEnabled: false,
};

export class PhaseConfig {
  private params: Partial<PhaseParams> = {};
  private listeners: Array<(params: PhaseParams) => void> = [];
  private saveTimeout: number | null = null;
  private frameMetrics = {
    seamsDetected: 0,
    overSmoothed: 0,
    frameTimeBudget: 0,
    frameCount: 0,
  };
  
  constructor() {
    // Parse URL parameters once at startup
    this.parseURL();
    
    // Load saved configuration from localStorage
    this.loadLocalStorage();
    
    // Setup keyboard shortcuts
    this.setupHotkeys();
    
    // Log initial configuration
    console.log('[PhaseConfig] Initialized with:', this.getAll());
  }
  
  /**
   * Parse URL parameters for phase correction
   * Example: ?phase=tv&tv=0.08&maxc=0.25&defocus=0.06
   */
  private parseURL(): void {
    const urlParams = new URLSearchParams(window.location.search);
    
    // Method selection
    const phase = urlParams.get('phase');
    if (phase && ['off', 'tv', 'zernike', 'lut', 'auto'].includes(phase)) {
      this.params.activeMethod = phase as any;
    }
    
    // TV parameters
    const tv = urlParams.get('tv');
    if (tv) this.params.tvLambda = parseFloat(tv);
    
    const maxc = urlParams.get('maxc');
    if (maxc) this.params.tvMaxCorrection = parseFloat(maxc);
    
    const mask = urlParams.get('mask');
    if (mask) this.params.tvUseMask = mask === '1';
    
    // Zernike parameters
    const defocus = urlParams.get('defocus');
    if (defocus) this.params.zernikeDefocus = parseFloat(defocus);
    
    const astig0 = urlParams.get('astig0');
    if (astig0) this.params.zernikeAstig0 = parseFloat(astig0);
    
    const astig45 = urlParams.get('astig45');
    if (astig45) this.params.zernikeAstig45 = parseFloat(astig45);
    
    // Auto-adapt
    const auto = urlParams.get('auto');
    if (auto) this.params.autoAdaptEnabled = auto === '1';
    
    // Metrics
    const metrics = urlParams.get('metrics');
    if (metrics) this.params.metricsEnabled = metrics === '1';
  }
  
  /**
   * Setup keyboard shortcuts for real-time tuning
   */
  private setupHotkeys(): void {
    window.addEventListener('keydown', (e) => {
      // Only process when Alt is held
      if (!e.altKey) return;
      
      let changed = false;
      
      switch(e.key) {
        // TV Lambda adjustment
        case 'ArrowUp':
          e.preventDefault();
          this.params.tvLambda = Math.min(0.20, (this.params.tvLambda ?? DEFAULTS.tvLambda) + 0.01);
          console.log('[PhaseConfig] TV Lambda:', this.params.tvLambda.toFixed(2));
          changed = true;
          break;
          
        case 'ArrowDown':
          e.preventDefault();
          this.params.tvLambda = Math.max(0.01, (this.params.tvLambda ?? DEFAULTS.tvLambda) - 0.01);
          console.log('[PhaseConfig] TV Lambda:', this.params.tvLambda.toFixed(2));
          changed = true;
          break;
          
        // Max correction adjustment
        case 'ArrowLeft':
          e.preventDefault();
          this.params.tvMaxCorrection = Math.max(0.05, (this.params.tvMaxCorrection ?? DEFAULTS.tvMaxCorrection) - 0.02);
          console.log('[PhaseConfig] Max Correction:', this.params.tvMaxCorrection.toFixed(2));
          changed = true;
          break;
          
        case 'ArrowRight':
          e.preventDefault();
          this.params.tvMaxCorrection = Math.min(0.50, (this.params.tvMaxCorrection ?? DEFAULTS.tvMaxCorrection) + 0.02);
          console.log('[PhaseConfig] Max Correction:', this.params.tvMaxCorrection.toFixed(2));
          changed = true;
          break;
          
        // Cycle methods
        case 'z':
        case 'Z':
          e.preventDefault();
          const methods: Array<PhaseParams['activeMethod']> = ['off', 'tv', 'zernike', 'lut', 'auto'];
          const current = this.params.activeMethod ?? DEFAULTS.activeMethod;
          const nextIndex = (methods.indexOf(current) + 1) % methods.length;
          this.params.activeMethod = methods[nextIndex];
          console.log('[PhaseConfig] Active method:', this.params.activeMethod);
          changed = true;
          break;
          
        // Dump current parameters
        case 'd':
        case 'D':
          e.preventDefault();
          this.dumpParams();
          break;
          
        // Reset to defaults
        case 'r':
        case 'R':
          e.preventDefault();
          if (confirm('Reset phase correction to defaults?')) {
            this.reset();
            changed = true;
          }
          break;
          
        // Toggle auto-adaptation
        case 'a':
        case 'A':
          e.preventDefault();
          this.params.autoAdaptEnabled = !(this.params.autoAdaptEnabled ?? DEFAULTS.autoAdaptEnabled);
          console.log('[PhaseConfig] Auto-adapt:', this.params.autoAdaptEnabled ? 'ON' : 'OFF');
          changed = true;
          break;
      }
      
      if (changed) {
        this.notify();
        this.scheduleSave();
      }
    });
  }
  
  /**
   * Get a parameter value with fallback to default
   */
  get<K extends keyof PhaseParams>(key: K): PhaseParams[K] {
    return (this.params[key] ?? DEFAULTS[key]) as PhaseParams[K];
  }
  
  /**
   * Set a parameter value
   */
  set<K extends keyof PhaseParams>(key: K, value: PhaseParams[K]): void {
    this.params[key] = value;
    this.notify();
    this.scheduleSave();
  }
  
  /**
   * Get all parameters with defaults applied
   */
  getAll(): PhaseParams {
    return { ...DEFAULTS, ...this.params };
  }
  
  /**
   * Update frame metrics for auto-adaptation
   */
  updateMetrics(metrics: {
    seamsDetected?: number;
    overSmoothed?: number;
    frameTimeBudget?: number;
  }): void {
    if (metrics.seamsDetected !== undefined) {
      this.frameMetrics.seamsDetected = metrics.seamsDetected;
    }
    if (metrics.overSmoothed !== undefined) {
      this.frameMetrics.overSmoothed = metrics.overSmoothed;
    }
    if (metrics.frameTimeBudget !== undefined) {
      this.frameMetrics.frameTimeBudget = metrics.frameTimeBudget;
    }
    
    this.frameMetrics.frameCount++;
    
    // Auto-adapt every 60 frames if enabled
    if (this.params.autoAdaptEnabled && this.frameMetrics.frameCount % 60 === 0) {
      this.autoAdapt();
    }
  }
  
  /**
   * Auto-adaptation logic based on metrics
   */
  private autoAdapt(): void {
    const current = this.params.tvLambda ?? DEFAULTS.tvLambda;
    let newValue = current;
    
    // Increase smoothing if seams detected and we have frame budget
    if (this.frameMetrics.seamsDetected > 0.1 && this.frameMetrics.frameTimeBudget > 2.0) {
      newValue = Math.min(0.15, current + 0.01);
      if (newValue !== current) {
        console.log('[PhaseConfig] Auto-adapt: Increasing TV lambda to', newValue.toFixed(2));
      }
    }
    // Decrease smoothing if oversmoothed
    else if (this.frameMetrics.overSmoothed > 0.5 && current > 0.04) {
      newValue = Math.max(0.04, current - 0.01);
      if (newValue !== current) {
        console.log('[PhaseConfig] Auto-adapt: Decreasing TV lambda to', newValue.toFixed(2));
      }
    }
    
    if (newValue !== current) {
      this.params.tvLambda = newValue;
      this.notify();
      this.scheduleSave();
    }
  }
  
  /**
   * Subscribe to parameter changes
   */
  onChange(listener: (params: PhaseParams) => void): () => void {
    this.listeners.push(listener);
    return () => {
      const index = this.listeners.indexOf(listener);
      if (index >= 0) this.listeners.splice(index, 1);
    };
  }
  
  /**
   * Notify all listeners of parameter changes
   */
  private notify(): void {
    const params = this.getAll();
    this.listeners.forEach(listener => listener(params));
  }
  
  /**
   * Load configuration from localStorage
   */
  private loadLocalStorage(): void {
    try {
      const saved = localStorage.getItem('phaseConfig');
      if (saved) {
        const parsed = JSON.parse(saved);
        // Only load if not overridden by URL
        Object.keys(parsed).forEach(key => {
          if (!(key in this.params)) {
            (this.params as any)[key] = parsed[key];
          }
        });
      }
    } catch (e) {
      console.warn('[PhaseConfig] Failed to load saved config:', e);
    }
  }
  
  /**
   * Save configuration to localStorage (debounced)
   */
  private scheduleSave(): void {
    if (this.saveTimeout) {
      clearTimeout(this.saveTimeout);
    }
    this.saveTimeout = window.setTimeout(() => {
      try {
        localStorage.setItem('phaseConfig', JSON.stringify(this.params));
        console.log('[PhaseConfig] Saved to localStorage');
      } catch (e) {
        console.warn('[PhaseConfig] Failed to save config:', e);
      }
      this.saveTimeout = null;
    }, 1000);
  }
  
  /**
   * Load remote configuration
   */
  async loadRemote(deviceKey: string): Promise<void> {
    try {
      const response = await fetch(`/api/phase-config/${deviceKey}`);
      if (response.ok) {
        const remote = await response.json();
        // Merge remote config (remote takes precedence over localStorage but not URL)
        Object.keys(remote).forEach(key => {
          if (!(key in this.params)) {
            (this.params as any)[key] = remote[key];
          }
        });
        this.notify();
        console.log('[PhaseConfig] Loaded remote config for', deviceKey);
      }
    } catch (e) {
      console.warn('[PhaseConfig] Failed to load remote config:', e);
    }
  }
  
  /**
   * Dump current parameters to console
   */
  dumpParams(): void {
    const params = this.getAll();
    console.group('[PhaseConfig] Current Parameters');
    console.log('Method:', params.activeMethod);
    console.log('TV Lambda:', params.tvLambda.toFixed(3));
    console.log('TV Max Correction:', params.tvMaxCorrection.toFixed(3));
    console.log('TV Use Mask:', params.tvUseMask);
    console.log('Zernike Defocus:', params.zernikeDefocus.toFixed(3));
    console.log('Auto-adapt:', params.autoAdaptEnabled);
    console.log('Metrics:', params.metricsEnabled);
    console.log('Full config:', params);
    console.groupEnd();
    
    // Also generate URL for sharing
    const url = this.generateURL();
    console.log('[PhaseConfig] Share URL:', url);
  }
  
  /**
   * Generate URL with current parameters
   */
  generateURL(): string {
    const params = this.getAll();
    const url = new URL(window.location.href);
    url.searchParams.set('phase', params.activeMethod);
    url.searchParams.set('tv', params.tvLambda.toFixed(3));
    url.searchParams.set('maxc', params.tvMaxCorrection.toFixed(3));
    url.searchParams.set('mask', params.tvUseMask ? '1' : '0');
    url.searchParams.set('defocus', params.zernikeDefocus.toFixed(3));
    url.searchParams.set('auto', params.autoAdaptEnabled ? '1' : '0');
    url.searchParams.set('metrics', params.metricsEnabled ? '1' : '0');
    return url.toString();
  }
  
  /**
   * Reset to defaults
   */
  reset(): void {
    this.params = {};
    localStorage.removeItem('phaseConfig');
    this.notify();
    console.log('[PhaseConfig] Reset to defaults');
  }
}

// Singleton instance
let instance: PhaseConfig | null = null;

export function getPhaseConfig(): PhaseConfig {
  if (!instance) {
    instance = new PhaseConfig();
  }
  return instance;
}
