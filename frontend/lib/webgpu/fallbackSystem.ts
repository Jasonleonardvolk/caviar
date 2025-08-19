/**
 * WebGPU Fallback System
 * Provides Canvas2D and WebGL fallbacks for browsers without WebGPU support
 */

import { logger } from '../../utils/logger';

export interface RenderingBackend {
  type: 'webgpu' | 'webgl2' | 'webgl' | 'canvas2d';
  initialize(): Promise<boolean>;
  render(data: any): void;
  destroy(): void;
  isSupported(): boolean;
  getCapabilities(): RenderCapabilities;
}

export interface RenderCapabilities {
  maxTextureSize: number;
  maxComputeWorkgroupSize: number[];
  supportsCompute: boolean;
  supportsFloat32: boolean;
  supportsTimestampQuery: boolean;
}

/**
 * WebGPU backend implementation
 */
export class WebGPUBackend implements RenderingBackend {
  type: 'webgpu' = 'webgpu';
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private canvas: HTMLCanvasElement | null = null;

  async initialize(): Promise<boolean> {
    try {
      if (!navigator.gpu) {
        logger.warn('WebGPU not available in this browser');
        return false;
      }

      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        logger.warn('No WebGPU adapter found');
        return false;
      }

      this.device = await adapter.requestDevice();
      
      this.canvas = document.getElementById('main-canvas') as HTMLCanvasElement;
      if (!this.canvas) {
        throw new Error('Canvas element not found');
      }

      this.context = this.canvas.getContext('webgpu');
      if (!this.context) {
        throw new Error('Could not get WebGPU context');
      }

      const format = navigator.gpu.getPreferredCanvasFormat();
      this.context.configure({
        device: this.device,
        format,
        alphaMode: 'premultiplied',
      });

      logger.info('WebGPU backend initialized successfully');
      return true;
    } catch (error) {
      logger.error('Failed to initialize WebGPU:', error);
      return false;
    }
  }

  render(data: any): void {
    if (!this.device || !this.context) {
      logger.error('WebGPU not initialized');
      return;
    }
    // WebGPU rendering implementation
    // This would use the shaders we've created
  }

  destroy(): void {
    if (this.device) {
      this.device.destroy();
    }
    this.device = null;
    this.context = null;
  }

  isSupported(): boolean {
    return 'gpu' in navigator;
  }

  getCapabilities(): RenderCapabilities {
    return {
      maxTextureSize: 8192,
      maxComputeWorkgroupSize: [256, 256, 64],
      supportsCompute: true,
      supportsFloat32: true,
      supportsTimestampQuery: true,
    };
  }
}

/**
 * WebGL2 fallback backend
 */
export class WebGL2Backend implements RenderingBackend {
  type: 'webgl2' = 'webgl2';
  private gl: WebGL2RenderingContext | null = null;
  private canvas: HTMLCanvasElement | null = null;

  async initialize(): Promise<boolean> {
    try {
      this.canvas = document.getElementById('main-canvas') as HTMLCanvasElement;
      if (!this.canvas) {
        throw new Error('Canvas element not found');
      }

      this.gl = this.canvas.getContext('webgl2', {
        antialias: true,
        alpha: true,
        premultipliedAlpha: true,
      });

      if (!this.gl) {
        logger.warn('WebGL2 not available');
        return false;
      }

      // Enable required extensions
      const requiredExtensions = [
        'EXT_color_buffer_float',
        'OES_texture_float_linear',
      ];

      for (const ext of requiredExtensions) {
        if (!this.gl.getExtension(ext)) {
          logger.warn(`Required WebGL2 extension ${ext} not available`);
        }
      }

      logger.info('WebGL2 backend initialized successfully');
      return true;
    } catch (error) {
      logger.error('Failed to initialize WebGL2:', error);
      return false;
    }
  }

  render(data: any): void {
    if (!this.gl) {
      logger.error('WebGL2 not initialized');
      return;
    }

    const gl = this.gl;
    
    // Clear
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Simplified rendering for fallback
    // Would implement basic visualization without compute shaders
  }

  destroy(): void {
    if (this.gl) {
      const loseContext = this.gl.getExtension('WEBGL_lose_context');
      if (loseContext) {
        loseContext.loseContext();
      }
    }
    this.gl = null;
  }

  isSupported(): boolean {
    const canvas = document.createElement('canvas');
    return !!(canvas.getContext('webgl2'));
  }

  getCapabilities(): RenderCapabilities {
    return {
      maxTextureSize: 4096,
      maxComputeWorkgroupSize: [0, 0, 0], // No compute in WebGL2
      supportsCompute: false,
      supportsFloat32: true,
      supportsTimestampQuery: false,
    };
  }
}

/**
 * Canvas2D ultimate fallback
 */
export class Canvas2DBackend implements RenderingBackend {
  type: 'canvas2d' = 'canvas2d';
  private ctx: CanvasRenderingContext2D | null = null;
  private canvas: HTMLCanvasElement | null = null;

  async initialize(): Promise<boolean> {
    try {
      this.canvas = document.getElementById('main-canvas') as HTMLCanvasElement;
      if (!this.canvas) {
        throw new Error('Canvas element not found');
      }

      this.ctx = this.canvas.getContext('2d');
      if (!this.ctx) {
        logger.error('Canvas2D not available (this should never happen)');
        return false;
      }

      logger.info('Canvas2D fallback backend initialized');
      return true;
    } catch (error) {
      logger.error('Failed to initialize Canvas2D:', error);
      return false;
    }
  }

  render(data: any): void {
    if (!this.ctx || !this.canvas) {
      logger.error('Canvas2D not initialized');
      return;
    }

    const ctx = this.ctx;
    const width = this.canvas.width;
    const height = this.canvas.height;

    // Clear canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    // Draw simple 2D representation
    // This would be a very basic visualization
    ctx.fillStyle = '#00ff00';
    ctx.font = '20px monospace';
    ctx.fillText('TORI System - 2D Fallback Mode', 10, 30);
    
    // Draw some basic shapes to represent the system state
    if (data && data.oscillators) {
      ctx.strokeStyle = '#00ffff';
      ctx.lineWidth = 2;
      
      for (let i = 0; i < Math.min(data.oscillators.length, 100); i++) {
        const osc = data.oscillators[i];
        const x = (osc.position?.[0] || 0) * width;
        const y = (osc.position?.[1] || 0) * height;
        const radius = 5 + Math.abs(osc.charge || 0) * 10;
        
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }

  destroy(): void {
    this.ctx = null;
    this.canvas = null;
  }

  isSupported(): boolean {
    return true; // Canvas2D is always supported
  }

  getCapabilities(): RenderCapabilities {
    return {
      maxTextureSize: 2048,
      maxComputeWorkgroupSize: [0, 0, 0],
      supportsCompute: false,
      supportsFloat32: false,
      supportsTimestampQuery: false,
    };
  }
}

/**
 * Rendering system that automatically selects the best backend
 */
export class RenderingSystem {
  private backend: RenderingBackend | null = null;
  private backends: RenderingBackend[] = [];

  constructor() {
    // Register backends in order of preference
    this.backends = [
      new WebGPUBackend(),
      new WebGL2Backend(),
      new Canvas2DBackend(),
    ];
  }

  async initialize(): Promise<RenderingBackend> {
    // Try each backend in order
    for (const backend of this.backends) {
      if (backend.isSupported()) {
        const success = await backend.initialize();
        if (success) {
          this.backend = backend;
          logger.info(`Rendering system initialized with ${backend.type} backend`);
          this.displayCapabilities(backend);
          return backend;
        }
      }
    }

    throw new Error('No rendering backend could be initialized');
  }

  private displayCapabilities(backend: RenderingBackend): void {
    const caps = backend.getCapabilities();
    logger.info('Rendering capabilities:', {
      backend: backend.type,
      maxTextureSize: caps.maxTextureSize,
      supportsCompute: caps.supportsCompute,
      supportsFloat32: caps.supportsFloat32,
    });

    // Update UI to show current backend
    const statusElement = document.getElementById('render-backend-status');
    if (statusElement) {
      statusElement.textContent = `Renderer: ${backend.type.toUpperCase()}`;
      statusElement.className = backend.type === 'webgpu' ? 'status-optimal' : 'status-fallback';
    }
  }

  render(data: any): void {
    if (!this.backend) {
      logger.error('Rendering system not initialized');
      return;
    }
    this.backend.render(data);
  }

  getBackend(): RenderingBackend | null {
    return this.backend;
  }

  destroy(): void {
    if (this.backend) {
      this.backend.destroy();
      this.backend = null;
    }
  }

  /**
   * Check if WebGPU is available and provide user feedback
   */
  static async checkWebGPUSupport(): Promise<{
    supported: boolean;
    message: string;
    recommendation?: string;
  }> {
    if (!navigator.gpu) {
      return {
        supported: false,
        message: 'WebGPU is not supported in your browser',
        recommendation: 'Please use Chrome 113+, Edge 113+, or Chrome Canary with WebGPU enabled',
      };
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        return {
          supported: false,
          message: 'WebGPU adapter not found',
          recommendation: 'Your GPU may not support WebGPU or drivers need updating',
        };
      }

      const device = await adapter.requestDevice();
      device.destroy(); // Clean up

      return {
        supported: true,
        message: 'WebGPU is fully supported',
      };
    } catch (error) {
      return {
        supported: false,
        message: `WebGPU initialization failed: ${error}`,
        recommendation: 'Check browser console for detailed error information',
      };
    }
  }
}

// Export singleton instance
export const renderingSystem = new RenderingSystem();

// Auto-initialize on module load
if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', async () => {
    try {
      const support = await RenderingSystem.checkWebGPUSupport();
      
      if (!support.supported) {
        logger.warn('WebGPU not supported:', support.message);
        if (support.recommendation) {
          logger.info('Recommendation:', support.recommendation);
        }
      }

      await renderingSystem.initialize();
    } catch (error) {
      logger.error('Failed to initialize rendering system:', error);
    }
  });
}
