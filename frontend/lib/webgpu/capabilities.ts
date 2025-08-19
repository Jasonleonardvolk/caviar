export type WebGPULimits = Partial<GPUAdapter['limits']> & {
  maxComputeWorkgroupStorageSize?: number; // alias for older dumps
};

export type WebGPUCapture = {
  limits: WebGPULimits;
  adapterInfo?: {
    vendor?: string;
    architecture?: string;
    device?: string;
    description?: string;
  };
  features?: string[];
};

export async function probeWebGPULimits(): Promise<WebGPULimits | null> {
  if (!('gpu' in navigator)) return null;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  const l = adapter.limits ?? {};
  return {
    maxComputeInvocationsPerWorkgroup: (l as any).maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: (l as any).maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: (l as any).maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: (l as any).maxComputeWorkgroupSizeZ,
    maxComputeWorkgroupStorageSize: (l as any).maxComputeWorkgroupStorageSize ?? (l as any).maxWorkgroupStorageSize,
    maxSampledTexturesPerShaderStage: (l as any).maxSampledTexturesPerShaderStage,
    maxSamplersPerShaderStage: (l as any).maxSamplersPerShaderStage
  };
}

export async function probeWebGPUComplete(): Promise<WebGPUCapture | null> {
  if (!('gpu' in navigator)) return null;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  
  const limits = await probeWebGPULimits();
  if (!limits) return null;
  
  // Get adapter info if available
  const info = ('requestAdapterInfo' in adapter)
    ? await (adapter as any).requestAdapterInfo?.().catch(() => undefined)
    : undefined;
  
  // Get features
  const features = adapter.features ? Array.from(adapter.features) : [];
  
  return {
    limits,
    adapterInfo: info ? {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description
    } : undefined,
    features
  };
}

export async function pushLimitsToDevServer(name: string, limits: WebGPULimits | WebGPUCapture) {
  try {
    // Check if it's a simple limits object or full capture
    const isFullCapture = 'adapterInfo' in limits || 'features' in limits;
    
    const payload = {
      name,
      limits: isFullCapture ? (limits as WebGPUCapture).limits : limits,
      metadata: {
        userAgent: navigator.userAgent,
        platform: detectPlatform(),
        adapterInfo: isFullCapture ? (limits as WebGPUCapture).adapterInfo : undefined,
        features: isFullCapture ? (limits as WebGPUCapture).features : undefined
      }
    };
    
    const response = await fetch("/api/dev/save-gpu-limits", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    
    if (response.ok) {
      const result = await response.json();
      console.log(`✅ GPU limits saved: ${result.checksum || 'success'}`);
      return result;
    } else {
      const error = await response.json();
      console.error(`❌ Failed to save limits: ${error.error}`);
    }
  } catch (e) {
    // best-effort; no crash if dev server not running
    console.log('📡 Dev server not available (this is normal in production)');
  }
}

export function getHolographicCapability(device?: GPUDevice): HolographicCapability {
  const hasDevice = !!device;
  const features = device?.features ? Array.from(device.features) : [];
  
  return {
    supported: hasDevice && features.includes('texture-compression-bc'),
    maxViews: 64,
    maxResolution: [8192, 8192],
    features: features
  };
}

export interface HolographicCapability {
  supported: boolean;
  maxViews: number;
  maxResolution: [number, number];
  features: string[];
}

function detectPlatform(): string {
  const ua = navigator.userAgent.toLowerCase();
  
  if (ua.includes('iphone') || ua.includes('ipad')) return 'ios';
  if (ua.includes('android')) return 'android';
  if (ua.includes('mac')) return 'mac';
  if (ua.includes('windows')) return 'windows';
  if (ua.includes('linux')) return 'linux';
  
  return 'unknown';
}