/**
 * Feature Flags System
 * Runtime feature toggles for safe rollouts and A/B testing
 */

export interface FeatureFlag {
  key: string;
  enabled: boolean;
  rolloutPercentage?: number;
  targeting?: TargetingRule[];
  metadata?: Record<string, any>;
}

export interface TargetingRule {
  attribute: string;
  operator: 'equals' | 'contains' | 'regex' | 'in' | 'gt' | 'lt';
  value: any;
}

export interface UserContext {
  userId?: string;
  sessionId: string;
  attributes?: Record<string, any>;
  deviceInfo?: {
    gpu?: string;
    browser?: string;
    os?: string;
    memory?: number;
  };
}

class FeatureFlagService {
  private flags: Map<string, FeatureFlag> = new Map();
  private userContext: UserContext;
  private overrides: Map<string, boolean> = new Map();
  private evaluationCache: Map<string, boolean> = new Map();
  private updateInterval: number = 60000; // 1 minute
  private updateTimer?: number;
  
  constructor() {
    this.userContext = this.initUserContext();
    this.loadFlags();
    this.startPolling();
    this.setupDevTools();
  }
  
  /**
   * Initialize user context from browser environment
   */
  private initUserContext(): UserContext {
    const sessionId = this.getOrCreateSessionId();
    
    return {
      sessionId,
      userId: this.getUserId(),
      attributes: {
        language: navigator.language,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        screenResolution: `${screen.width}x${screen.height}`,
        colorDepth: screen.colorDepth,
        touchEnabled: 'ontouchstart' in window,
        cookieEnabled: navigator.cookieEnabled
      },
      deviceInfo: {
        gpu: this.detectGPU(),
        browser: this.detectBrowser(),
        os: this.detectOS(),
        memory: (navigator as any).deviceMemory || undefined
      }
    };
  }
  
  /**
   * Load flags from server or localStorage
   */
  private async loadFlags(): Promise<void> {
    try {
      // Try to load from server
      const response = await fetch('/api/feature-flags', {
        headers: {
          'X-Session-Id': this.userContext.sessionId,
          'X-User-Id': this.userContext.userId || ''
        }
      });
      
      if (response.ok) {
        const flags = await response.json();
        this.updateFlags(flags);
        this.saveToLocalStorage(flags);
      } else {
        // Fallback to localStorage
        this.loadFromLocalStorage();
      }
    } catch (error) {
      console.warn('Failed to load feature flags from server:', error);
      this.loadFromLocalStorage();
    }
  }
  
  /**
   * Update flags and clear evaluation cache
   */
  private updateFlags(flags: FeatureFlag[]): void {
    this.flags.clear();
    this.evaluationCache.clear();
    
    for (const flag of flags) {
      this.flags.set(flag.key, flag);
    }
    
    this.emitUpdate();
  }
  
  /**
   * Check if a feature is enabled for the current user
   */
  public isEnabled(key: string): boolean {
    // Check overrides first (for QA/testing)
    if (this.overrides.has(key)) {
      return this.overrides.get(key)!;
    }
    
    // Check cache
    if (this.evaluationCache.has(key)) {
      return this.evaluationCache.get(key)!;
    }
    
    // Evaluate flag
    const flag = this.flags.get(key);
    if (!flag) {
      console.warn(`Feature flag not found: ${key}`);
      return false;
    }
    
    const result = this.evaluate(flag);
    this.evaluationCache.set(key, result);
    
    // Track evaluation
    this.trackEvaluation(key, result);
    
    return result;
  }
  
  /**
   * Evaluate a flag against the current user context
   */
  private evaluate(flag: FeatureFlag): boolean {
    // Check if globally disabled
    if (!flag.enabled) {
      return false;
    }
    
    // Check targeting rules
    if (flag.targeting && flag.targeting.length > 0) {
      for (const rule of flag.targeting) {
        if (!this.evaluateRule(rule)) {
          return false;
        }
      }
    }
    
    // Check rollout percentage
    if (flag.rolloutPercentage !== undefined && flag.rolloutPercentage < 100) {
      const hash = this.hashString(this.userContext.sessionId + flag.key);
      const bucket = Math.abs(hash % 100);
      return bucket < flag.rolloutPercentage;
    }
    
    return true;
  }
  
  /**
   * Evaluate a single targeting rule
   */
  private evaluateRule(rule: TargetingRule): boolean {
    const value = this.getAttributeValue(rule.attribute);
    
    switch (rule.operator) {
      case 'equals':
        return value === rule.value;
      
      case 'contains':
        return String(value).includes(String(rule.value));
      
      case 'regex':
        return new RegExp(rule.value).test(String(value));
      
      case 'in':
        return Array.isArray(rule.value) && rule.value.includes(value);
      
      case 'gt':
        return Number(value) > Number(rule.value);
      
      case 'lt':
        return Number(value) < Number(rule.value);
      
      default:
        return false;
    }
  }
  
  /**
   * Get attribute value from user context
   */
  private getAttributeValue(path: string): any {
    const parts = path.split('.');
    let value: any = this.userContext;
    
    for (const part of parts) {
      if (value && typeof value === 'object') {
        value = value[part];
      } else {
        return undefined;
      }
    }
    
    return value;
  }
  
  /**
   * Get variation for A/B testing
   */
  public getVariation<T = string>(key: string, variations: T[]): T | null {
    if (!this.isEnabled(key)) {
      return null;
    }
    
    const flag = this.flags.get(key);
    if (!flag || !variations.length) {
      return null;
    }
    
    // Use consistent hashing to assign variation
    const hash = this.hashString(this.userContext.sessionId + key);
    const index = Math.abs(hash % variations.length);
    
    return variations[index];
  }
  
  /**
   * Set override for testing
   */
  public override(key: string, enabled: boolean): void {
    this.overrides.set(key, enabled);
    this.evaluationCache.delete(key);
    this.emitUpdate();
  }
  
  /**
   * Clear all overrides
   */
  public clearOverrides(): void {
    this.overrides.clear();
    this.evaluationCache.clear();
    this.emitUpdate();
  }
  
  /**
   * Get all current flag states
   */
  public getAllFlags(): Record<string, boolean> {
    const result: Record<string, boolean> = {};
    
    for (const [key] of this.flags) {
      result[key] = this.isEnabled(key);
    }
    
    return result;
  }
  
  /**
   * Update user context
   */
  public updateContext(updates: Partial<UserContext>): void {
    Object.assign(this.userContext, updates);
    this.evaluationCache.clear();
    this.emitUpdate();
  }
  
  /**
   * Simple hash function for consistent bucketing
   */
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash;
  }
  
  /**
   * Track flag evaluation for analytics
   */
  private trackEvaluation(key: string, enabled: boolean): void {
    // Send to analytics in batches
    if (typeof window !== 'undefined' && (window as any).analytics) {
      (window as any).analytics.track('feature_flag_evaluated', {
        flag: key,
        enabled,
        sessionId: this.userContext.sessionId,
        userId: this.userContext.userId
      });
    }
  }
  
  /**
   * Start polling for flag updates
   */
  private startPolling(): void {
    if (this.updateTimer) {
      clearInterval(this.updateTimer);
    }
    
    this.updateTimer = window.setInterval(() => {
      this.loadFlags();
    }, this.updateInterval);
  }
  
  /**
   * Stop polling
   */
  public stopPolling(): void {
    if (this.updateTimer) {
      clearInterval(this.updateTimer);
      this.updateTimer = undefined;
    }
  }
  
  /**
   * Emit update event
   */
  private emitUpdate(): void {
    window.dispatchEvent(new CustomEvent('feature-flags-updated', {
      detail: this.getAllFlags()
    }));
  }
  
  /**
   * Save flags to localStorage for offline support
   */
  private saveToLocalStorage(flags: FeatureFlag[]): void {
    try {
      localStorage.setItem('feature-flags', JSON.stringify({
        flags,
        timestamp: Date.now()
      }));
    } catch (e) {
      console.warn('Failed to save feature flags to localStorage:', e);
    }
  }
  
  /**
   * Load flags from localStorage
   */
  private loadFromLocalStorage(): void {
    try {
      const stored = localStorage.getItem('feature-flags');
      if (stored) {
        const { flags, timestamp } = JSON.parse(stored);
        
        // Use cached flags if less than 1 hour old
        if (Date.now() - timestamp < 3600000) {
          this.updateFlags(flags);
        }
      }
    } catch (e) {
      console.warn('Failed to load feature flags from localStorage:', e);
    }
  }
  
  /**
   * Setup dev tools for QA
   */
  private setupDevTools(): void {
    if (typeof window !== 'undefined') {
      (window as any).__featureFlags = {
        isEnabled: (key: string) => this.isEnabled(key),
        override: (key: string, enabled: boolean) => this.override(key, enabled),
        clearOverrides: () => this.clearOverrides(),
        getAllFlags: () => this.getAllFlags(),
        updateContext: (updates: Partial<UserContext>) => this.updateContext(updates),
        reload: () => this.loadFlags()
      };
      
      console.log('Feature flags dev tools available at window.__featureFlags');
    }
  }
  
  /**
   * Get or create session ID
   */
  private getOrCreateSessionId(): string {
    let sessionId = sessionStorage.getItem('session-id');
    
    if (!sessionId) {
      sessionId = this.generateId();
      sessionStorage.setItem('session-id', sessionId);
    }
    
    return sessionId;
  }
  
  /**
   * Get user ID from various sources
   */
  private getUserId(): string | undefined {
    // Try different sources
    return localStorage.getItem('user-id') || 
           (window as any).__userId ||
           undefined;
  }
  
  /**
   * Generate random ID
   */
  private generateId(): string {
    return Math.random().toString(36).substring(2) + Date.now().toString(36);
  }
  
  /**
   * Detect GPU info
   */
  private detectGPU(): string | undefined {
    try {
      const canvas = document.createElement('canvas');
      const gl = (canvas.getContext('webgl2') || canvas.getContext('webgl')) as
        WebGL2RenderingContext | WebGLRenderingContext | null;
      
      if (gl) {
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        if (debugInfo) {
          return gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
        }
      }
    } catch (e) {
      // Ignore errors
    }
    
    return undefined;
  }
  
  /**
   * Detect browser
   */
  private detectBrowser(): string {
    const ua = navigator.userAgent;
    
    if (ua.includes('Chrome')) return 'Chrome';
    if (ua.includes('Firefox')) return 'Firefox';
    if (ua.includes('Safari')) return 'Safari';
    if (ua.includes('Edge')) return 'Edge';
    
    return 'Unknown';
  }
  
  /**
   * Detect OS
   */
  private detectOS(): string {
    const ua = navigator.userAgent;
    
    if (ua.includes('Windows')) return 'Windows';
    if (ua.includes('Mac')) return 'macOS';
    if (ua.includes('Linux')) return 'Linux';
    if (ua.includes('Android')) return 'Android';
    if (ua.includes('iOS')) return 'iOS';
    
    return 'Unknown';
  }
}

// Export singleton instance
export const featureFlags = new FeatureFlagService();

// React hook for feature flags (optional - only if React is available)
// To use this, install React: npm install react
/*
import { useState, useEffect } from 'react';

export function useFeatureFlag(key: string): boolean {
  const [enabled, setEnabled] = useState(() => featureFlags.isEnabled(key));
  
  useEffect(() => {
    const handler = () => {
      setEnabled(featureFlags.isEnabled(key));
    };
    
    window.addEventListener('feature-flags-updated', handler);
    return () => window.removeEventListener('feature-flags-updated', handler);
  }, [key]);
  
  return enabled;
}
*/

// Default feature flags configuration
export const DEFAULT_FLAGS: FeatureFlag[] = [
  {
    key: 'webgpu_renderer',
    enabled: true,
    rolloutPercentage: 100,
    targeting: [
      {
        attribute: 'deviceInfo.browser',
        operator: 'in',
        value: ['Chrome', 'Edge']
      }
    ]
  },
  {
    key: 'holographic_mode',
    enabled: true,
    rolloutPercentage: 100
  },
  {
    key: 'lightfield_mode',
    enabled: true,
    rolloutPercentage: 50,
    metadata: {
      description: 'Advanced lightfield rendering mode'
    }
  },
  {
    key: 'volumetric_fog',
    enabled: false,
    metadata: {
      description: 'Experimental volumetric fog effects'
    }
  },
  {
    key: 'depth_enhanced',
    enabled: true,
    rolloutPercentage: 75
  },
  {
    key: 'streaming_quilts',
    enabled: true,
    targeting: [
      {
        attribute: 'deviceInfo.memory',
        operator: 'gt',
        value: 4
      }
    ]
  },
  {
    key: 'offline_mode',
    enabled: true,
    rolloutPercentage: 100
  },
  {
    key: 'telemetry',
    enabled: true,
    rolloutPercentage: 10,
    metadata: {
      description: 'Client-side telemetry collection'
    }
  },
  {
    key: 'performance_hints',
    enabled: false,
    metadata: {
      description: 'Show performance optimization hints'
    }
  },
  {
    key: 'debug_overlay',
    enabled: false,
    metadata: {
      description: 'Developer debug overlay'
    }
  }
];
