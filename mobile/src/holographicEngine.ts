/**
 * Mobile-optimized Holographic Engine
 * Reduced memory footprint and adaptive quality for <100MB app size
 */

import { Device } from '@capacitor/device';
import { Network } from '@capacitor/network';
import SimplePeer from 'simple-peer';
import * as msgpack from '@msgpack/msgpack';
import { openDB, DBSchema } from 'idb';

// Quality presets for mobile
export enum MobileQualityPreset {
  BATTERY_SAVER = 'battery',    // 10x8 views, 30fps, f16
  BALANCED = 'balanced',         // 15x10 views, 45fps, f16
  PERFORMANCE = 'performance',   // 20x12 views, 60fps, f16/f32 mix
  DESKTOP_STREAM = 'stream'      // Full quality via WebRTC
}

interface QualityConfig {
  views: [number, number];
  fps: number;
  precision: 'f16' | 'mixed';
  maxTextureSize: number;
  fftTaps: number;
}

interface CapabilityProbe {
  gpuTier: 'low' | 'medium' | 'high';
  availableMemory: number;
  thermalState: 'nominal' | 'fair' | 'serious' | 'critical';
  networkType: 'wifi' | 'cellular' | 'none';
  batteryLevel: number;
}

interface ShaderDB extends DBSchema {
  'shader-cache': {
    key: string;
    value: {
      code: string;
      compiled: ArrayBuffer;
      timestamp: number;
    };
  };
  'telemetry-buffer': {
    key: number;
    value: {
      event: any;
      timestamp: number;
    };
  };
}

export class MobileHolographicEngine {
  private device?: GPUDevice;
  private canvas?: HTMLCanvasElement;
  private context?: GPUCanvasContext;
  
  private currentQuality: MobileQualityPreset = MobileQualityPreset.BALANCED;
  private qualityConfigs: Record<MobileQualityPreset, QualityConfig> = {
    [MobileQualityPreset.BATTERY_SAVER]: {
      views: [10, 8],
      fps: 30,
      precision: 'f16',
      maxTextureSize: 512,
      fftTaps: 2
    },
    [MobileQualityPreset.BALANCED]: {
      views: [15, 10],
      fps: 45,
      precision: 'f16',
      maxTextureSize: 768,
      fftTaps: 4
    },
    [MobileQualityPreset.PERFORMANCE]: {
      views: [20, 12],
      fps: 60,
      precision: 'mixed',
      maxTextureSize: 1024,
      fftTaps: 4
    },
    [MobileQualityPreset.DESKTOP_STREAM]: {
      views: [45, 1], // Full desktop quilt
      fps: 60,
      precision: 'mixed',
      maxTextureSize: 2048,
      fftTaps: 8
    }
  };

  // Streaming support
  private peer?: SimplePeer.Instance;
  private streamActive: boolean = false;
  private streamBuffer: ArrayBuffer[] = [];
  
  // Performance monitoring
  private frameCount: number = 0;
  private lastFpsCheck: number = 0;
  private autoQualityEnabled: boolean = true;
  
  // Database for caching
  private db?: IDBDatabase;
  
  // WebGPU resources
  private wavefieldPipeline?: GPUComputePipeline;
  private propagationPipeline?: GPUComputePipeline;
  private viewSynthesisPipeline?: GPUComputePipeline;
  private renderPipeline?: GPURenderPipeline;
  
  // Textures
  private wavefieldTexture?: GPUTexture;
  private quiltTexture?: GPUTexture;
  
  // Video decoder for streaming
  private videoDecoder?: VideoDecoder;
  private decodedFrames: VideoFrame[] = [];

  constructor() {
    this.initializeDB();
  }

  /**
   * Initialize IndexedDB for shader caching and telemetry
   */
  private async initializeDB(): Promise<void> {
    this.db = await openDB<ShaderDB>('tori-hologram', 1, {
      upgrade(db) {
        // Shader cache
        if (!db.objectStoreNames.contains('shader-cache')) {
          db.createObjectStore('shader-cache');
        }
        
        // Telemetry buffer for offline metrics
        if (!db.objectStoreNames.contains('telemetry-buffer')) {
          db.createObjectStore('telemetry-buffer', { 
            keyPath: 'timestamp' 
          });
        }
      }
    });
  }

  /**
   * Initialize the mobile engine
   */
  async initialize(canvas: HTMLCanvasElement): Promise<void> {
    this.canvas = canvas;
    
    // Check WebGPU support
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported on this device');
    }

    // Get device capabilities
    const capabilities = await this.probeCapabilities();
    
    // Auto-select quality based on device
    this.currentQuality = this.selectQualityForDevice(capabilities);
    
    // Request GPU device with mobile-friendly limits
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: capabilities.batteryLevel < 20 ? 'low-power' : 'high-performance'
    });

    if (!adapter) {
      throw new Error('Failed to get GPU adapter');
    }

    // Check feature support
    const requiredFeatures: GPUFeatureName[] = [];
    if (adapter.features.has('shader-f16')) {
      requiredFeatures.push('shader-f16');
    }

    this.device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxTextureDimension2D: this.qualityConfigs[this.currentQuality].maxTextureSize,
        maxBufferSize: 64 * 1024 * 1024, // 64MB limit
        maxStorageBufferBindingSize: 32 * 1024 * 1024 // 32MB
      }
    });

    // Configure canvas
    this.context = canvas.getContext('webgpu')!;
    const format = navigator.gpu.getPreferredCanvasFormat();
    
    this.context.configure({
      device: this.device,
      format,
      alphaMode: 'premultiplied',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
    });

    // Initialize video decoder for streaming
    if ('VideoDecoder' in window) {
      this.initializeVideoDecoder();
    }

    // Create GPU resources
    await this.createResources();
    
    // Start performance monitoring
    this.startPerformanceMonitoring();
  }

  /**
   * Probe device capabilities
   */
  private async probeCapabilities(): Promise<CapabilityProbe> {
    const deviceInfo = await Device.getInfo();
    const networkStatus = await Network.getStatus();
    
    // Estimate GPU tier based on device
    let gpuTier: 'low' | 'medium' | 'high' = 'medium';
    
    if (deviceInfo.platform === 'ios') {
      // iOS GPU detection
      const model = deviceInfo.model || '';
      if (model.includes('iPhone 15') || model.includes('iPhone 14 Pro')) {
        gpuTier = 'high';
      } else if (model.includes('iPhone 12') || model.includes('iPhone 13')) {
        gpuTier = 'medium';
      } else {
        gpuTier = 'low';
      }
    } else if (deviceInfo.platform === 'android') {
      // Android - check for high-end chipsets
      // This is simplified - real implementation would check specific GPU models
      gpuTier = 'medium';
    }

    // Get battery info (if available)
    let batteryLevel = 100;
    if ('getBattery' in navigator) {
      try {
        const battery = await (navigator as any).getBattery();
        batteryLevel = battery.level * 100;
      } catch (e) {
        // Battery API not available
      }
    }

    // Check thermal state (iOS only)
    let thermalState: any = 'nominal';
    if (window.webkit?.messageHandlers?.thermalState) {
      try {
        thermalState = await window.webkit.messageHandlers.thermalState.postMessage({});
      } catch (e) {
        // Not available
      }
    }

    return {
      gpuTier,
      availableMemory: (performance as any).memory?.jsHeapSizeLimit || 512 * 1024 * 1024,
      thermalState,
      networkType: networkStatus.connectionType as any || 'none',
      batteryLevel
    };
  }

  /**
   * Auto-select quality based on device capabilities
   */
  private selectQualityForDevice(capabilities: CapabilityProbe): MobileQualityPreset {
    // If on cellular or low battery, use battery saver
    if (capabilities.networkType === 'cellular' || capabilities.batteryLevel < 20) {
      return MobileQualityPreset.BATTERY_SAVER;
    }

    // If thermal throttling, reduce quality
    if (capabilities.thermalState !== 'nominal') {
      return MobileQualityPreset.BATTERY_SAVER;
    }

    // Based on GPU tier
    switch (capabilities.gpuTier) {
      case 'high':
        return MobileQualityPreset.PERFORMANCE;
      case 'medium':
        return MobileQualityPreset.BALANCED;
      case 'low':
      default:
        return MobileQualityPreset.BATTERY_SAVER;
    }
  }

  /**
   * Create GPU resources
   */
  private async createResources(): Promise<void> {
    const config = this.qualityConfigs[this.currentQuality];
    
    // Create textures
    this.wavefieldTexture = this.device!.createTexture({
      size: [config.maxTextureSize, config.maxTextureSize],
      format: config.precision === 'f16' ? 'rg16float' : 'rg32float',
      usage: GPUTextureUsage.STORAGE_BINDING | 
             GPUTextureUsage.TEXTURE_BINDING
    });
    
    this.quiltTexture = this.device!.createTexture({
      size: [
        config.views[0] * (config.maxTextureSize / 4),
        config.views[1] * (config.maxTextureSize / 4)
      ],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | 
             GPUTextureUsage.TEXTURE_BINDING
    });
    
    // Load and create pipelines
    await this.createPipelines();
  }

  /**
   * Create compute and render pipelines
   */
  private async createPipelines(): Promise<void> {
    // Load shader variants for current quality
    const [wavefieldShader, propagationShader, viewShader, renderShader] = await Promise.all([
      this.loadShaderVariant('wavefieldEncoder'),
      this.loadShaderVariant('propagation'),
      this.loadShaderVariant('multiViewSynthesis'),
      this.loadShaderVariant('lenticularRender')
    ]);
    
    // Create compute pipelines
    this.wavefieldPipeline = this.device!.createComputePipeline({
      layout: 'auto',
      compute: {
        module: wavefieldShader,
        entryPoint: 'main'
      }
    });
    
    this.propagationPipeline = this.device!.createComputePipeline({
      layout: 'auto',
      compute: {
        module: propagationShader,
        entryPoint: 'main'
      }
    });
    
    this.viewSynthesisPipeline = this.device!.createComputePipeline({
      layout: 'auto',
      compute: {
        module: viewShader,
        entryPoint: 'main'
      }
    });
    
    // Create render pipeline
    this.renderPipeline = this.device!.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: renderShader,
        entryPoint: 'vs_main'
      },
      fragment: {
        module: renderShader,
        entryPoint: 'fs_main',
        targets: [{
          format: navigator.gpu.getPreferredCanvasFormat()
        }]
      },
      primitive: {
        topology: 'triangle-list'
      }
    });
  }

  /**
   * Load shader variant for current quality
   */
  async loadShaderVariant(name: string): Promise<GPUShaderModule> {
    const quality = this.currentQuality;
    const cacheKey = `${name}_${quality}`;
    
    // Check cache first
    const cached = await this.db?.get('shader-cache', cacheKey);
    if (cached && Date.now() - cached.timestamp < 86400000) { // 24h cache
      return this.device!.createShaderModule({
        code: cached.code
      });
    }

    // Load appropriate shader variant
    const shaderPath = `/shaders/mobile/${quality}/${name}.wgsl`;
    const response = await fetch(shaderPath);
    const code = await response.text();
    
    // Process shader for mobile optimizations
    const optimizedCode = this.optimizeShaderForMobile(code, quality);
    
    // Cache the shader
    await this.db?.put('shader-cache', cacheKey, {
      code: optimizedCode,
      compiled: new ArrayBuffer(0), // Placeholder
      timestamp: Date.now()
    });

    return this.device!.createShaderModule({
      code: optimizedCode
    });
  }

  /**
   * Optimize shader code for mobile
   */
  private optimizeShaderForMobile(code: string, quality: MobileQualityPreset): string {
    const config = this.qualityConfigs[quality];
    
    // Replace precision placeholders
    code = code.replace(/\{\{PRECISION\}\}/g, config.precision === 'f16' ? 'f16' : 'f32');
    
    // Replace view count
    code = code.replace(/\{\{VIEW_COLS\}\}/g, config.views[0].toString());
    code = code.replace(/\{\{VIEW_ROWS\}\}/g, config.views[1].toString());
    
    // Replace texture size
    code = code.replace(/\{\{MAX_TEXTURE_SIZE\}\}/g, config.maxTextureSize.toString());
    
    // Optimize workgroup sizes for mobile GPUs
    code = code.replace(/@workgroup_size\(8, 8\)/g, '@workgroup_size(4, 4)');
    
    return code;
  }

  /**
   * Initialize video decoder for streaming
   */
  private initializeVideoDecoder(): void {
    this.videoDecoder = new VideoDecoder({
      output: (frame) => {
        this.decodedFrames.push(frame);
        // Limit buffer size
        if (this.decodedFrames.length > 3) {
          this.decodedFrames.shift()?.close();
        }
      },
      error: (e) => {
        console.error('Video decode error:', e);
      }
    });

    // Configure for H.265/HEVC
    this.videoDecoder.configure({
      codec: 'hev1.1.6.L93.B0', // H.265 Main profile
      hardwareAcceleration: 'prefer-hardware',
      optimizeForLatency: true
    });
  }

  /**
   * Start performance monitoring and auto-quality adjustment
   */
  private startPerformanceMonitoring(): void {
    const monitor = () => {
      const now = performance.now();
      
      if (now - this.lastFpsCheck > 1000) {
        const fps = this.frameCount;
        this.frameCount = 0;
        this.lastFpsCheck = now;
        
        // Auto-adjust quality if enabled
        if (this.autoQualityEnabled && !this.streamActive) {
          this.adjustQualityBasedOnPerformance(fps);
        }
        
        // Log telemetry
        this.logTelemetry({
          type: 'performance',
          fps,
          quality: this.currentQuality,
          timestamp: Date.now()
        });
      }
      
      requestAnimationFrame(monitor);
    };
    
    requestAnimationFrame(monitor);
  }

  /**
   * Adjust quality based on performance
   */
  private async adjustQualityBasedOnPerformance(fps: number): Promise<void> {
    const targetFps = this.qualityConfigs[this.currentQuality].fps;
    const threshold = 0.8; // 80% of target
    
    if (fps < targetFps * threshold) {
      // Performance is poor, reduce quality
      if (this.currentQuality === MobileQualityPreset.PERFORMANCE) {
        await this.setQuality(MobileQualityPreset.BALANCED);
      } else if (this.currentQuality === MobileQualityPreset.BALANCED) {
        await this.setQuality(MobileQualityPreset.BATTERY_SAVER);
      }
    } else if (fps >= targetFps * 0.95) {
      // Performance is good, can increase quality
      if (this.currentQuality === MobileQualityPreset.BATTERY_SAVER) {
        // Check if we have headroom
        const capabilities = await this.probeCapabilities();
        if (capabilities.gpuTier !== 'low' && capabilities.batteryLevel > 30) {
          await this.setQuality(MobileQualityPreset.BALANCED);
        }
      }
    }
  }

  /**
   * Set quality preset
   */
  async setQuality(preset: MobileQualityPreset): Promise<void> {
    if (preset === this.currentQuality) return;
    
    console.log(`Switching quality from ${this.currentQuality} to ${preset}`);
    this.currentQuality = preset;
    
    // If streaming, notify desktop
    if (this.streamActive && this.peer) {
      this.peer.send(msgpack.encode({
        type: 'quality',
        preset
      }));
    }
    
    // Recreate resources for new quality
    await this.createResources();
    
    // Log quality change
    this.logTelemetry({
      type: 'quality_change',
      from: this.currentQuality,
      to: preset,
      timestamp: Date.now()
    });
  }

  /**
   * Enable desktop streaming mode
   */
  async enableStreaming(jwt: string): Promise<void> {
    if (this.streamActive) return;
    
    // Switch to stream quality
    this.currentQuality = MobileQualityPreset.DESKTOP_STREAM;
    this.streamActive = true;
    
    // Create WebRTC peer
    this.peer = new SimplePeer({
      initiator: true,
      trickle: true
    });

    // Handle peer events
    this.peer.on('signal', async (data) => {
      // Send offer to desktop
      const response = await fetch(`${this.getDesktopUrl()}/webrtc/offer`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${jwt}`
        },
        body: JSON.stringify({
          jwt,
          offer: data,
          profile: 'high',
          capabilities: ['stream', 'metrics']
        })
      });

      const result = await response.json();
      if (result.answer) {
        this.peer!.signal(result.answer);
      }
    });

    this.peer.on('connect', () => {
      console.log('Connected to desktop for streaming');
    });

    this.peer.on('data', (data) => {
      const message = msgpack.decode(data as Uint8Array);
      if (message.type === 'video') {
        this.handleStreamFrame(message);
      }
    });

    this.peer.on('error', (err) => {
      console.error('Streaming error:', err);
      this.disableStreaming();
    });
  }

  /**
   * Disable streaming and fall back to local rendering
   */
  async disableStreaming(): Promise<void> {
    this.streamActive = false;
    
    if (this.peer) {
      this.peer.destroy();
      this.peer = undefined;
    }
    
    // Fall back to appropriate local quality
    const capabilities = await this.probeCapabilities();
    this.currentQuality = this.selectQualityForDevice(capabilities);
    
    await this.createResources();
  }

  /**
   * Handle stream frame from desktop
   */
  private handleStreamFrame(message: any): void {
    if (!this.videoDecoder || this.videoDecoder.state === 'closed') {
      return;
    }

    // Create EncodedVideoChunk from received data
    const chunk = new EncodedVideoChunk({
      type: message.keyFrame ? 'key' : 'delta',
      timestamp: message.timestamp,
      data: message.data
    });

    this.videoDecoder.decode(chunk);
  }

  /**
   * Render hologram frame
   */
  async render(oscillatorState?: any): Promise<void> {
    if (!this.device || !this.context) return;
    
    this.frameCount++;
    
    if (this.streamActive && this.decodedFrames.length > 0) {
      // Render streamed frame
      await this.renderStreamedFrame();
    } else {
      // Local rendering
      await this.renderLocal(oscillatorState);
    }
  }

  /**
   * Render locally computed hologram
   */
  private async renderLocal(oscillatorState: any): Promise<void> {
    const commandEncoder = this.device!.createCommandEncoder();
    
    // Run compute passes based on quality
    // ... (compute pass implementation)
    
    // Render to canvas
    const textureView = this.context!.getCurrentTexture().createView();
    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [{
        view: textureView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store'
      }]
    };
    
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.renderPipeline!);
    // ... bind groups and draw
    passEncoder.draw(6); // Full-screen quad
    passEncoder.end();
    
    this.device!.queue.submit([commandEncoder.finish()]);
  }

  /**
   * Render streamed frame from desktop
   */
  private async renderStreamedFrame(): Promise<void> {
    const frame = this.decodedFrames[0];
    if (!frame) return;
    
    // Copy video frame to canvas
    const bitmap = await createImageBitmap(frame);
    
    // Get 2D context for fast blit (fallback if WebGPU copy is slow)
    const ctx = this.canvas!.getContext('2d');
    if (ctx) {
      ctx.drawImage(bitmap, 0, 0, this.canvas!.width, this.canvas!.height);
    }
    
    bitmap.close();
    frame.close();
    this.decodedFrames.shift();
  }

  /**
   * Get desktop URL for streaming
   */
  private getDesktopUrl(): string {
    // In production, this would be discovered via service discovery
    return process.env.DESKTOP_URL || 'http://localhost:7690';
  }

  /**
   * Log telemetry event
   */
  private async logTelemetry(event: any): Promise<void> {
    // Buffer in IndexedDB
    await this.db?.add('telemetry-buffer', {
      event,
      timestamp: Date.now()
    });
    
    // Flush periodically or when online
    this.flushTelemetry();
  }

  /**
   * Flush telemetry buffer to server
   */
  private async flushTelemetry(): Promise<void> {
    const network = await Network.getStatus();
    if (!network.connected) return;
    
    // Get buffered events
    const tx = this.db!.transaction('telemetry-buffer', 'readwrite');
    const store = tx.objectStore('telemetry-buffer');
    const events = await store.getAll();
    
    if (events.length === 0) return;
    
    try {
      // Send to telemetry endpoint
      await fetch(`${this.getDesktopUrl()}/telemetry`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          events,
          deviceId: await this.getDeviceId()
        })
      });
      
      // Clear buffer on success
      await store.clear();
    } catch (error) {
      console.error('Failed to flush telemetry:', error);
    }
  }

  /**
   * Get unique device ID for telemetry
   */
  private async getDeviceId(): Promise<string> {
    const info = await Device.getInfo();
    return info.uuid || 'unknown';
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    if (this.peer) {
      this.peer.destroy();
    }
    
    if (this.videoDecoder) {
      this.videoDecoder.close();
    }
    
    this.decodedFrames.forEach(frame => frame.close());
    
    if (this.wavefieldTexture) {
      this.wavefieldTexture.destroy();
    }
    
    if (this.quiltTexture) {
      this.quiltTexture.destroy();
    }
    
    if (this.device) {
      this.device.destroy();
    }
  }
}

// Export for use in mobile app
export default MobileHolographicEngine;
