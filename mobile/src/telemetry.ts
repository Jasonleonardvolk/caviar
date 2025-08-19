/**
 * Mobile Telemetry Service
 * Handles offline buffering and batch upload of metrics
 */

import { openDB, DBSchema, IDBPDatabase } from 'idb';
import { Network } from '@capacitor/network';
import { 
  trace, 
  context, 
  SpanStatusCode,
  SpanKind,
  Tracer
} from '@opentelemetry/api';
import { WebTracerProvider } from '@opentelemetry/sdk-trace-web';
import { 
  BatchSpanProcessor,
  ConsoleSpanExporter
} from '@opentelemetry/sdk-trace-base';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { Resource } from '@opentelemetry/resources';
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions';

interface TelemetryDB extends DBSchema {
  'events': {
    key: number;
    value: TelemetryEvent;
    indexes: { 'by-timestamp': number };
  };
  'spans': {
    key: number;
    value: any; // OpenTelemetry span data
    indexes: { 'by-timestamp': number };
  };
}

interface TelemetryEvent {
  id?: number;
  timestamp: number;
  type: string;
  data: any;
  sessionId: string;
  deviceId: string;
}

export class MobileTelemetryService {
  private db?: IDBPDatabase<TelemetryDB>;
  private tracer?: Tracer;
  private provider?: WebTracerProvider;
  private deviceId: string = '';
  private sessionId: string = '';
  private collectorUrl: string;
  private batchSize: number = 100;
  private flushInterval: number = 30000; // 30 seconds
  private flushTimer?: NodeJS.Timeout;
  private isOnline: boolean = true;

  constructor(collectorUrl: string = '/telemetry') {
    this.collectorUrl = collectorUrl;
    this.initialize();
  }

  private async initialize(): Promise<void> {
    // Initialize IndexedDB
    this.db = await openDB<TelemetryDB>('tori-telemetry', 1, {
      upgrade(db) {
        // Events store
        if (!db.objectStoreNames.contains('events')) {
          const eventStore = db.createObjectStore('events', {
            keyPath: 'id',
            autoIncrement: true
          });
          eventStore.createIndex('by-timestamp', 'timestamp');
        }
        
        // Spans store for OpenTelemetry
        if (!db.objectStoreNames.contains('spans')) {
          const spanStore = db.createObjectStore('spans', {
            keyPath: 'id',
            autoIncrement: true
          });
          spanStore.createIndex('by-timestamp', 'timestamp');
        }
      }
    });

    // Get device ID
    const { Device } = await import('@capacitor/device');
    const info = await Device.getInfo();
    this.deviceId = info.uuid || `mobile-${Date.now()}`;

    // Generate session ID
    this.sessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    // Initialize OpenTelemetry
    this.setupOpenTelemetry();

    // Monitor network status
    this.setupNetworkMonitoring();

    // Start flush timer
    this.startFlushTimer();
  }

  private setupOpenTelemetry(): void {
    // Create resource with device info
    const resource = new Resource({
      [SemanticResourceAttributes.SERVICE_NAME]: 'tori-hologram-mobile',
      [SemanticResourceAttributes.SERVICE_VERSION]: '1.0.0',
      [SemanticResourceAttributes.DEVICE_ID]: this.deviceId,
      'device.session_id': this.sessionId,
      'device.platform': 'mobile'
    });

    // Create provider
    this.provider = new WebTracerProvider({
      resource
    });

    // Configure exporter
    const exporter = new OTLPTraceExporter({
      url: `${this.getCollectorUrl()}/v1/traces`,
      headers: {
        'Authorization': `Bearer ${this.getAuthToken()}`
      }
    });

    // Use batch processor for efficiency
    this.provider.addSpanProcessor(
      new BatchSpanProcessor(exporter, {
        maxQueueSize: 1000,
        maxExportBatchSize: 100,
        scheduledDelayMillis: 5000
      })
    );

    // Register provider
    this.provider.register();

    // Get tracer
    this.tracer = trace.getTracer('tori-hologram-mobile', '1.0.0');
  }

  private async setupNetworkMonitoring(): Promise<void> {
    // Initial status
    const status = await Network.getStatus();
    this.isOnline = status.connected;

    // Listen for changes
    Network.addListener('networkStatusChange', (status) => {
      this.isOnline = status.connected;
      
      if (this.isOnline) {
        // Flush buffered data when coming online
        this.flushAll();
      }
    });
  }

  private startFlushTimer(): void {
    this.flushTimer = setInterval(() => {
      if (this.isOnline) {
        this.flushAll();
      }
    }, this.flushInterval);
  }

  /**
   * Log a telemetry event
   */
  async logEvent(type: string, data: any): Promise<void> {
    const event: TelemetryEvent = {
      timestamp: Date.now(),
      type,
      data,
      sessionId: this.sessionId,
      deviceId: this.deviceId
    };

    // Store in IndexedDB
    await this.db?.add('events', event);

    // Try immediate send if online and critical event
    if (this.isOnline && this.isCriticalEvent(type)) {
      this.flushEvents();
    }
  }

  /**
   * Start a traced operation
   */
  startSpan(name: string, attributes?: any): any {
    if (!this.tracer) return null;

    return this.tracer.startSpan(name, {
      kind: SpanKind.CLIENT,
      attributes: {
        ...attributes,
        'device.session_id': this.sessionId
      }
    });
  }

  /**
   * Record a metric
   */
  async recordMetric(name: string, value: number, tags?: Record<string, string>): Promise<void> {
    await this.logEvent('metric', {
      name,
      value,
      tags: {
        ...tags,
        device_id: this.deviceId,
        session_id: this.sessionId
      },
      timestamp: Date.now()
    });
  }

  /**
   * Record performance timing
   */
  async recordTiming(name: string, duration: number): Promise<void> {
    await this.recordMetric(`timing.${name}`, duration, {
      unit: 'ms'
    });
  }

  /**
   * Record frame metrics
   */
  async recordFrameMetrics(metrics: {
    fps: number;
    frameTime: number;
    quality: string;
    gpuTime?: number;
  }): Promise<void> {
    await this.logEvent('frame_metrics', metrics);
  }

  /**
   * Record quality change
   */
  async recordQualityChange(from: string, to: string, reason: string): Promise<void> {
    await this.logEvent('quality_change', {
      from,
      to,
      reason,
      timestamp: Date.now()
    });
  }

  /**
   * Record error
   */
  async recordError(error: Error, context?: any): Promise<void> {
    await this.logEvent('error', {
      message: error.message,
      stack: error.stack,
      context,
      timestamp: Date.now()
    });
  }

  /**
   * Flush all buffered telemetry
   */
  async flushAll(): Promise<void> {
    await Promise.all([
      this.flushEvents(),
      this.flushSpans()
    ]);
  }

  /**
   * Flush buffered events
   */
  private async flushEvents(): Promise<void> {
    if (!this.isOnline || !this.db) return;

    try {
      // Get buffered events
      const tx = this.db.transaction('events', 'readwrite');
      const store = tx.objectStore('events');
      const events = await store.getAll();

      if (events.length === 0) return;

      // Batch send
      const batches = this.chunk(events, this.batchSize);
      
      for (const batch of batches) {
        const response = await fetch(this.getCollectorUrl(), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.getAuthToken()}`
          },
          body: JSON.stringify({
            events: batch,
            device_id: this.deviceId,
            session_id: this.sessionId
          })
        });

        if (response.ok) {
          // Remove sent events
          const ids = batch.map(e => e.id).filter(id => id !== undefined);
          for (const id of ids) {
            await store.delete(id!);
          }
        }
      }

      await tx.done;
    } catch (error) {
      console.error('Failed to flush events:', error);
    }
  }

  /**
   * Flush OpenTelemetry spans
   */
  private async flushSpans(): Promise<void> {
    if (!this.provider) return;

    try {
      await this.provider.forceFlush();
    } catch (error) {
      console.error('Failed to flush spans:', error);
    }
  }

  /**
   * Check if event type is critical
   */
  private isCriticalEvent(type: string): boolean {
    return ['error', 'crash', 'performance_degradation'].includes(type);
  }

  /**
   * Get collector URL
   */
  private getCollectorUrl(): string {
    // In production, this would be the full URL
    if (this.collectorUrl.startsWith('http')) {
      return this.collectorUrl;
    }
    
    // Relative URL - construct from current location
    const baseUrl = process.env.TELEMETRY_URL || window.location.origin;
    return `${baseUrl}${this.collectorUrl}`;
  }

  /**
   * Get auth token for telemetry
   */
  private getAuthToken(): string {
    // This should be obtained from the pairing process
    return localStorage.getItem('telemetry_token') || '';
  }

  /**
   * Chunk array into batches
   */
  private chunk<T>(array: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  }

  /**
   * Get session statistics
   */
  async getStats(): Promise<any> {
    if (!this.db) return {};

    const eventCount = await this.db.count('events');
    const spanCount = await this.db.count('spans');

    return {
      deviceId: this.deviceId,
      sessionId: this.sessionId,
      bufferedEvents: eventCount,
      bufferedSpans: spanCount,
      isOnline: this.isOnline
    };
  }

  /**
   * Clear all buffered data
   */
  async clear(): Promise<void> {
    if (!this.db) return;

    const tx = this.db.transaction(['events', 'spans'], 'readwrite');
    await Promise.all([
      tx.objectStore('events').clear(),
      tx.objectStore('spans').clear()
    ]);
    await tx.done;
  }

  /**
   * Destroy service
   */
  destroy(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }

    Network.removeAllListeners();

    if (this.provider) {
      this.provider.shutdown();
    }

    if (this.db) {
      this.db.close();
    }
  }
}

// Singleton instance
let telemetryInstance: MobileTelemetryService | null = null;

export function getTelemetryService(): MobileTelemetryService {
  if (!telemetryInstance) {
    telemetryInstance = new MobileTelemetryService();
  }
  return telemetryInstance;
}

// Export convenience functions
export const telemetry = {
  logEvent: (type: string, data: any) => 
    getTelemetryService().logEvent(type, data),
    
  recordMetric: (name: string, value: number, tags?: Record<string, string>) =>
    getTelemetryService().recordMetric(name, value, tags),
    
  recordTiming: (name: string, duration: number) =>
    getTelemetryService().recordTiming(name, duration),
    
  recordError: (error: Error, context?: any) =>
    getTelemetryService().recordError(error, context),
    
  startSpan: (name: string, attributes?: any) =>
    getTelemetryService().startSpan(name, attributes)
};
