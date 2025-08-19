/**
 * Telemetry Library
 * Privacy-first client-side metrics collection
 */

export interface TelemetryConfig {
  enabled: boolean;
  endpoint: string;
  batchSize: number;
  flushInterval: number;
  sampling: {
    errors: number;      // 1.0 = 100%
    performance: number; // 0.1 = 10%
    interactions: number; // 0.5 = 50%
  };
  privacy: {
    redactPII: boolean;
    anonymizeIP: boolean;
    respectDNT: boolean;
  };
}

export interface TelemetryEvent {
  type: 'error' | 'performance' | 'interaction' | 'custom';
  name: string;
  timestamp: number;
  sessionId: string;
  data: Record<string, any>;
  context?: TelemetryContext;
}

export interface TelemetryContext {
  page?: string;
  component?: string;
  gpu?: string;
  renderMode?: string;
  experimentId?: string;
}

export interface PerformanceMetrics {
  bootTime?: number;
  firstRender?: number;
  shaderCompile?: number;
  frameRate?: number;
  memoryUsage?: number;
  gpuMemory?: number;
}

class TelemetryService {
  private config: TelemetryConfig;
  private queue: TelemetryEvent[] = [];
  private sessionId: string;
  private context: TelemetryContext = {};
  private performanceObserver?: PerformanceObserver;
  private flushTimer?: number;
  private metrics: PerformanceMetrics = {};
  private errorCount = 0;
  private interactionCount = 0;
  
  constructor(config?: Partial<TelemetryConfig>) {
    this.config = {
      enabled: true,
      endpoint: '/api/telemetry',
      batchSize: 50,
      flushInterval: 30000, // 30 seconds
      sampling: {
        errors: 1.0,
        performance: 0.1,
        interactions: 0.5
      },
      privacy: {
        redactPII: true,
        anonymizeIP: true,
        respectDNT: true
      },
      ...config
    };
    
    this.sessionId = this.generateSessionId();
    this.initialize();
  }
  
  /**
   * Initialize telemetry collection
   */
  private initialize(): void {
    // Respect Do Not Track
    if (this.config.privacy.respectDNT && navigator.doNotTrack === '1') {
      this.config.enabled = false;
      console.log('Telemetry disabled: Do Not Track is enabled');
      return;
    }
    
    if (!this.config.enabled) {
      return;
    }
    
    this.setupPerformanceObserver();
    this.setupErrorHandlers();
    this.setupInteractionTracking();
    this.setupVisibilityHandling();
    this.startFlushTimer();
    
    // Track initial page load
    this.trackPageLoad();
  }
  
  /**
   * Setup Performance Observer for Core Web Vitals
   */
  private setupPerformanceObserver(): void {
    if (!('PerformanceObserver' in window)) {
      return;
    }
    
    try {
      // Observe paint timing
      const paintObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.name === 'first-contentful-paint') {
            this.metrics.firstRender = entry.startTime;
            this.track('performance', 'fcp', {
              value: entry.startTime
            });
          }
        }
      });
      paintObserver.observe({ entryTypes: ['paint'] });
      
      // Observe largest contentful paint
      const lcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const lastEntry = entries[entries.length - 1];
        
        this.track('performance', 'lcp', {
          value: lastEntry.startTime,
          element: lastEntry.element?.tagName
        });
      });
      lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });
      
      // Observe layout shifts
      let cumulativeLayoutShift = 0;
      const clsObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (!(entry as any).hadRecentInput) {
            cumulativeLayoutShift += (entry as any).value;
          }
        }
        
        this.track('performance', 'cls', {
          value: cumulativeLayoutShift
        });
      });
      clsObserver.observe({ entryTypes: ['layout-shift'] });
      
      // Observe first input delay
      const fidObserver = new PerformanceObserver((list) => {
        const entry = list.getEntries()[0];
        
        this.track('performance', 'fid', {
          value: (entry as any).processingStart - entry.startTime,
          name: entry.name
        });
      });
      fidObserver.observe({ entryTypes: ['first-input'] });
      
    } catch (e) {
      console.warn('Failed to setup performance observers:', e);
    }
  }
  
  /**
   * Setup error handlers
   */
  private setupErrorHandlers(): void {
    // Global error handler
    window.addEventListener('error', (event) => {
      if (!this.shouldSample('errors')) return;
      
      this.trackError({
        message: event.message,
        source: event.filename,
        line: event.lineno,
        column: event.colno,
        stack: event.error?.stack
      });
    });
    
    // Unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      if (!this.shouldSample('errors')) return;
      
      this.trackError({
        message: 'Unhandled Promise Rejection',
        reason: event.reason?.toString(),
        stack: event.reason?.stack
      });
    });
  }
  
  /**
   * Setup interaction tracking
   */
  private setupInteractionTracking(): void {
    // Track clicks
    document.addEventListener('click', (event) => {
      if (!this.shouldSample('interactions')) return;
      
      const target = event.target as HTMLElement;
      const selector = this.getElementSelector(target);
      
      this.track('interaction', 'click', {
        selector,
        text: target.textContent?.substring(0, 50),
        href: (target as HTMLAnchorElement).href
      });
    }, { passive: true, capture: true });
    
    // Track form submissions
    document.addEventListener('submit', (event) => {
      if (!this.shouldSample('interactions')) return;
      
      const form = event.target as HTMLFormElement;
      
      this.track('interaction', 'form_submit', {
        selector: this.getElementSelector(form),
        method: form.method,
        action: form.action
      });
    }, { passive: true, capture: true });
  }
  
  /**
   * Setup visibility handling for flushing
   */
  private setupVisibilityHandling(): void {
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        this.flush();
      }
    });
    
    // Flush on page unload
    window.addEventListener('beforeunload', () => {
      this.flush(true);
    });
  }
  
  /**
   * Track page load metrics
   */
  private trackPageLoad(): void {
    if (!this.shouldSample('performance')) return;
    
    window.addEventListener('load', () => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      
      if (navigation) {
        this.track('performance', 'page_load', {
          dns: navigation.domainLookupEnd - navigation.domainLookupStart,
          tcp: navigation.connectEnd - navigation.connectStart,
          request: navigation.responseStart - navigation.requestStart,
          response: navigation.responseEnd - navigation.responseStart,
          dom: navigation.domComplete - navigation.domInteractive,
          load: navigation.loadEventEnd - navigation.loadEventStart,
          total: navigation.loadEventEnd - navigation.fetchStart
        });
      }
    });
  }
  
  /**
   * Track custom event
   */
  public track(
    type: TelemetryEvent['type'],
    name: string,
    data: Record<string, any> = {}
  ): void {
    if (!this.config.enabled) return;
    
    const event: TelemetryEvent = {
      type,
      name,
      timestamp: Date.now(),
      sessionId: this.sessionId,
      data: this.sanitizeData(data),
      context: { ...this.context }
    };
    
    this.queue.push(event);
    
    // Flush if queue is full
    if (this.queue.length >= this.config.batchSize) {
      this.flush();
    }
  }
  
  /**
   * Track error
   */
  public trackError(error: any): void {
    this.errorCount++;
    
    this.track('error', 'javascript_error', {
      message: error.message || error.toString(),
      stack: error.stack,
      source: error.source,
      line: error.line,
      column: error.column,
      errorCount: this.errorCount
    });
  }
  
  /**
   * Track performance metric
   */
  public trackPerformance(name: string, value: number, unit = 'ms'): void {
    if (!this.shouldSample('performance')) return;
    
    this.track('performance', name, {
      value,
      unit
    });
  }
  
  /**
   * Track user interaction
   */
  public trackInteraction(name: string, data?: Record<string, any>): void {
    if (!this.shouldSample('interactions')) return;
    
    this.interactionCount++;
    
    this.track('interaction', name, {
      ...data,
      interactionCount: this.interactionCount
    });
  }
  
  /**
   * Track WebGPU metrics
   */
  public trackGPU(metrics: {
    initTime?: number;
    shaderCompileTime?: number;
    frameTime?: number;
    drawCalls?: number;
    triangles?: number;
    textureMemory?: number;
  }): void {
    if (!this.shouldSample('performance')) return;
    
    this.track('performance', 'gpu_metrics', metrics);
  }
  
  /**
   * Update context for future events
   */
  public updateContext(context: Partial<TelemetryContext>): void {
    Object.assign(this.context, context);
  }
  
  /**
   * Set user ID (hashed for privacy)
   */
  public setUserId(userId: string): void {
    // Hash user ID for privacy
    const hashedId = this.hashString(userId);
    this.updateContext({ userId: hashedId } as any);
  }
  
  /**
   * Flush queued events
   */
  private async flush(useBeacon = false): Promise<void> {
    if (this.queue.length === 0) return;
    
    const events = [...this.queue];
    this.queue = [];
    
    const payload = {
      events,
      session: {
        id: this.sessionId,
        timestamp: Date.now(),
        userAgent: navigator.userAgent,
        screen: {
          width: screen.width,
          height: screen.height,
          colorDepth: screen.colorDepth
        },
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight
        },
        connection: (navigator as any).connection ? {
          effectiveType: (navigator as any).connection.effectiveType,
          downlink: (navigator as any).connection.downlink,
          rtt: (navigator as any).connection.rtt
        } : undefined
      }
    };
    
    try {
      if (useBeacon && navigator.sendBeacon) {
        // Use sendBeacon for reliability on page unload
        const blob = new Blob([JSON.stringify(payload)], {
          type: 'application/json'
        });
        navigator.sendBeacon(this.config.endpoint, blob);
      } else {
        // Regular fetch
        await fetch(this.config.endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
        });
      }
    } catch (error) {
      console.warn('Failed to send telemetry:', error);
      
      // Re-queue events for retry
      this.queue.unshift(...events);
    }
  }
  
  /**
   * Start flush timer
   */
  private startFlushTimer(): void {
    this.flushTimer = window.setInterval(() => {
      this.flush();
    }, this.config.flushInterval);
  }
  
  /**
   * Stop telemetry collection
   */
  public stop(): void {
    this.config.enabled = false;
    
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }
    
    if (this.performanceObserver) {
      this.performanceObserver.disconnect();
    }
    
    this.flush();
  }
  
  /**
   * Check if event should be sampled
   */
  private shouldSample(type: keyof TelemetryConfig['sampling']): boolean {
    const rate = this.config.sampling[type];
    return Math.random() < rate;
  }
  
  /**
   * Sanitize data for privacy
   */
  private sanitizeData(data: Record<string, any>): Record<string, any> {
    if (!this.config.privacy.redactPII) {
      return data;
    }
    
    const sanitized = { ...data };
    
    // Redact potential PII
    const piiPatterns = [
      /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi, // Email
      /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/g, // Phone
      /\b\d{3}-\d{2}-\d{4}\b/g, // SSN
      /\b(?:\d{4}[-\s]?){3}\d{4}\b/g // Credit card
    ];
    
    const redact = (value: any): any => {
      if (typeof value === 'string') {
        let redacted = value;
        for (const pattern of piiPatterns) {
          redacted = redacted.replace(pattern, '[REDACTED]');
        }
        return redacted;
      }
      
      if (typeof value === 'object' && value !== null) {
        if (Array.isArray(value)) {
          return value.map(redact);
        }
        
        const result: Record<string, any> = {};
        for (const [key, val] of Object.entries(value)) {
          result[key] = redact(val);
        }
        return result;
      }
      
      return value;
    };
    
    return redact(sanitized);
  }
  
  /**
   * Get element selector for tracking
   */
  private getElementSelector(element: HTMLElement): string {
    const parts: string[] = [];
    let current: HTMLElement | null = element;
    
    while (current && parts.length < 5) {
      let selector = current.tagName.toLowerCase();
      
      if (current.id) {
        selector += `#${current.id}`;
        parts.unshift(selector);
        break;
      }
      
      if (current.className) {
        const classes = current.className.split(' ')
          .filter(c => c && !c.startsWith('svelte-'))
          .slice(0, 2)
          .join('.');
        
        if (classes) {
          selector += `.${classes}`;
        }
      }
      
      parts.unshift(selector);
      current = current.parentElement;
    }
    
    return parts.join(' > ');
  }
  
  /**
   * Generate session ID
   */
  private generateSessionId(): string {
    return `${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
  }
  
  /**
   * Hash string for privacy
   */
  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return hash.toString(36);
  }
  
  /**
   * Get telemetry summary
   */
  public getSummary(): {
    sessionId: string;
    errorCount: number;
    interactionCount: number;
    queueSize: number;
    metrics: PerformanceMetrics;
  } {
    return {
      sessionId: this.sessionId,
      errorCount: this.errorCount,
      interactionCount: this.interactionCount,
      queueSize: this.queue.length,
      metrics: { ...this.metrics }
    };
  }
}

// Export singleton instance
export const telemetry = new TelemetryService({
  enabled: true, // Will be overridden by feature flag
  endpoint: '/api/telemetry',
  sampling: {
    errors: 1.0,     // Capture all errors
    performance: 0.1, // Sample 10% of performance metrics
    interactions: 0.5 // Sample 50% of interactions
  }
});

// Export for debugging
if (typeof window !== 'undefined') {
  (window as any).__telemetry = telemetry;
}
