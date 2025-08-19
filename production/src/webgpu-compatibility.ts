// production/src/webgpu-compatibility.ts
// WebGPU compatibility layer with fallback support

export interface GPUCompatibilityResult {
  supported: boolean;
  adapter: GPUAdapter | null;
  device: GPUDevice | null;
  fallbackMode: 'none' | 'webgl2' | 'cpu';
  features: Set<GPUFeatureName>;
  limits: GPUSupportedLimits | null;
  warnings: string[];
}

export class WebGPUCompatibility {
  private static instance: WebGPUCompatibility;
  private compatibilityResult?: GPUCompatibilityResult;

  private constructor() {}

  static getInstance(): WebGPUCompatibility {
    if (!WebGPUCompatibility.instance) {
      WebGPUCompatibility.instance = new WebGPUCompatibility();
    }
    return WebGPUCompatibility.instance;
  }

  async checkSupport(): Promise<GPUCompatibilityResult> {
    if (this.compatibilityResult) {
      return this.compatibilityResult;
    }

    const result: GPUCompatibilityResult = {
      supported: false,
      adapter: null,
      device: null,
      fallbackMode: 'none',
      features: new Set(),
      limits: null,
      warnings: []
    };

    // Check for WebGPU support
    if (!navigator.gpu) {
      result.fallbackMode = this.checkWebGL2Support() ? 'webgl2' : 'cpu';
      result.warnings.push('WebGPU not supported, using fallback mode');
      this.compatibilityResult = result;
      return result;
    }

    try {
      // Request adapter with high performance preference
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
        forceFallbackAdapter: false
      });

      if (!adapter) {
        result.fallbackMode = 'webgl2';
        result.warnings.push('No suitable GPU adapter found');
        this.compatibilityResult = result;
        return result;
      }

      result.adapter = adapter;
      result.features = adapter.features;
      result.limits = adapter.limits;

      // Check required features
      const requiredFeatures: GPUFeatureName[] = [];
      const optionalFeatures: GPUFeatureName[] = ['timestamp-query', 'shader-f16'];

      // Filter available features
      const availableFeatures = requiredFeatures.filter(f => adapter.features.has(f));
      const availableOptional = optionalFeatures.filter(f => adapter.features.has(f));

      // Request device
      const device = await adapter.requestDevice({
        requiredFeatures: availableFeatures,
        requiredLimits: {
          maxBufferSize: 512 * 1024 * 1024, // 512MB
          maxStorageBufferBindingSize: 128 * 1024 * 1024, // 128MB
          maxComputeWorkgroupSizeX: 256,
          maxComputeWorkgroupSizeY: 256,
          maxComputeWorkgroupSizeZ: 64
        }
      });

      if (!device) {
        result.fallbackMode = 'webgl2';
        result.warnings.push('Failed to create GPU device');
        this.compatibilityResult = result;
        return result;
      }

      // Set up device lost handler
      device.lost.then((info) => {
        console.error('GPU device lost:', info);
        this.handleDeviceLost(info);
      });

      result.device = device;
      result.supported = true;
      result.fallbackMode = 'none';

      // Add feature warnings
      const missingOptional = optionalFeatures.filter(f => !adapter.features.has(f));
      if (missingOptional.length > 0) {
        result.warnings.push(`Optional features not available: ${missingOptional.join(', ')}`);
      }

      this.compatibilityResult = result;
      return result;

    } catch (error) {
      console.error('WebGPU initialization error:', error);
      result.fallbackMode = 'webgl2';
      result.warnings.push(`WebGPU initialization failed: ${error}`);
      this.compatibilityResult = result;
      return result;
    }
  }

  private checkWebGL2Support(): boolean {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2');
      return gl !== null;
    } catch {
      return false;
    }
  }

  private async handleDeviceLost(info: GPUDeviceLostInfo): Promise<void> {
    console.error('GPU Device lost:', info.reason, info.message);
    
    // Reset compatibility result
    this.compatibilityResult = undefined;

    // Attempt to recover
    if (info.reason === 'destroyed') {
      // Device was intentionally destroyed, no recovery needed
      return;
    }

    // Try to reinitialize after a delay
    setTimeout(async () => {
      console.log('Attempting to recover GPU device...');
      const result = await this.checkSupport();
      if (result.supported) {
        console.log('GPU device recovered successfully');
        // Emit recovery event
        window.dispatchEvent(new CustomEvent('webgpu-recovered', { detail: result }));
      } else {
        console.error('Failed to recover GPU device, using fallback');
        window.dispatchEvent(new CustomEvent('webgpu-fallback', { detail: result }));
      }
    }, 1000);
  }

  async ensureDevice(): Promise<GPUDevice> {
    const result = await this.checkSupport();
    if (!result.supported || !result.device) {
      throw new Error('WebGPU not available. Please use a fallback renderer.');
    }
    return result.device;
  }

  getCapabilities(): { 
    maxBufferSize: number; 
    maxWorkgroupSize: number;
    hasTimestampQuery: boolean;
    hasFloat16: boolean;
  } | null {
    if (!this.compatibilityResult?.device) {
      return null;
    }

    const device = this.compatibilityResult.device;
    const adapter = this.compatibilityResult.adapter!;

    return {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxWorkgroupSize: adapter.limits.maxComputeWorkgroupSizeX,
      hasTimestampQuery: device.features.has('timestamp-query'),
      hasFloat16: device.features.has('shader-f16')
    };
  }

  destroy(): void {
    if (this.compatibilityResult?.device) {
      this.compatibilityResult.device.destroy();
    }
    this.compatibilityResult = undefined;
  }
}

// Export singleton instance
export const webGPUCompat = WebGPUCompatibility.getInstance();
