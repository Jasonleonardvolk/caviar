// WebGPU device limits validation and variant selection

export interface DeviceVariant {
  name: string;
  minTextureSize: number;
  minBufferSize: number;
  requiredFeatures: GPUFeatureName[];
  optionalFeatures?: GPUFeatureName[];
}

const variants: Record<string, DeviceVariant> = {
  'high-end': {
    name: 'High End',
    minTextureSize: 8192,
    minBufferSize: 256 * 1024 * 1024, // 256MB
    requiredFeatures: ['timestamp-query', 'depth-clip-control'],
    optionalFeatures: ['shader-f16', 'float32-filterable']
  },
  'standard': {
    name: 'Standard',
    minTextureSize: 4096,
    minBufferSize: 128 * 1024 * 1024, // 128MB
    requiredFeatures: ['depth-clip-control'],
    optionalFeatures: ['timestamp-query']
  },
  'low-end': {
    name: 'Low End',
    minTextureSize: 2048,
    minBufferSize: 64 * 1024 * 1024, // 64MB
    requiredFeatures: [],
    optionalFeatures: ['depth-clip-control']
  }
};

export function chooseVariant(device: GPUDevice): DeviceVariant {
  const limits = device.limits;
  const features = device.features;
  
  // Check high-end first
  if (limits.maxTextureDimension2D >= 8192 &&
      limits.maxBufferSize >= 256 * 1024 * 1024 &&
      features.has('timestamp-query')) {
    return variants['high-end'];
  }
  
  // Check standard
  if (limits.maxTextureDimension2D >= 4096 &&
      limits.maxBufferSize >= 128 * 1024 * 1024) {
    return variants['standard'];
  }
  
  // Default to low-end
  return variants['low-end'];
}

export function validateDeviceLimits(device: GPUDevice, required?: Partial<Record<string, number>>): boolean {
  if (!required) return true;
  
  const limits = device.limits;
  
  for (const [key, value] of Object.entries(required)) {
    const deviceValue = (limits as any)[key];
    if ((deviceValue as number) < (value as number)) {
      console.warn(`Device limit ${key} (${deviceValue}) is less than required (${value})`);
      return false;
    }
  }
  
  return true;
}

export function getOptimalLimits(device: GPUDevice): GPUSupportedLimits {
  return device.limits;
}
