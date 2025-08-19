// GPU Helper utilities for WebGPU compatibility

/**
 * Get texture dimensions in a compatible way
 * GPUTexture doesn't have a direct 'size' property
 */
export function getTextureSize(texture: GPUTexture): [number, number, number] {
    // GPUTexture has width, height, depthOrArrayLayers properties
    return [
        texture.width,
        texture.height,
        texture.depthOrArrayLayers
    ];
}

/**
 * Get 2D texture dimensions
 */
export function getTextureSize2D(texture: GPUTexture): [number, number] {
    return [texture.width, texture.height];
}

/**
 * Create texture descriptor with proper size format
 */
export function createTextureDescriptor(
    width: number,
    height: number,
    options: Partial<GPUTextureDescriptor> = {}
): GPUTextureDescriptor {
    return {
        size: { width, height, depthOrArrayLayers: options.dimension === '3d' ? ((options as any).depthOrArrayLayers || 1) : 1 },
        format: options.format || 'rgba8unorm',
        usage: options.usage || (GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST),
        ...options
    };
}

/**
 * Helper for timestamp queries (handles deprecation)
 */
export function writeTimestamp(
    encoder: GPUCommandEncoder | GPUComputePassEncoder | GPURenderPassEncoder,
    querySet: GPUQuerySet | undefined,
    index: number
): void {
    if (!querySet) return;
    
    // Timestamp queries are only available on command encoder now
    // Pass encoders no longer have writeTimestamp
    if ('beginComputePass' in encoder) {
        // This is a command encoder
        // Note: Direct timestamp writes on command encoder are limited
        // You need to use timestamp writes in pass descriptors instead
        console.warn('Timestamp queries should be specified in pass descriptors');
    }
}

/**
 * Create compute pass with timestamp queries
 */
export function createComputePassWithTimestamps(
    encoder: GPUCommandEncoder,
    querySet?: GPUQuerySet,
    startIndex?: number
): GPUComputePassEncoder {
    const descriptor: GPUComputePassDescriptor = {};
    
    if (querySet && startIndex !== undefined) {
        descriptor.timestampWrites = {
            querySet,
            beginningOfPassWriteIndex: startIndex,
            endOfPassWriteIndex: startIndex + 1
        };
    }
    
    return encoder.beginComputePass(descriptor);
}

/**
 * Check if a feature is supported
 */
export function hasFeature(device: GPUDevice, feature: GPUFeatureName): boolean {
    return device.features.has(feature);
}

/**
 * Safe feature check for non-standard features
 */
export function hasNonStandardFeature(device: GPUDevice, feature: string): boolean {
    try {
        // Some features might not be in the GPUFeatureName enum yet
        return device.features.has(feature as GPUFeatureName);
    } catch {
        return false;
    }
}

/**
 * Get workgroup memory limit safely
 */
export function getWorkgroupMemoryLimit(device: GPUDevice): number {
    // Handle different property names across versions
    const limit = device.limits.maxComputeWorkgroupStorageSize;
    return limit || 16384; // Default to 16KB if not available
}

/**
 * Create a buffer with size validation
 */
export function createBuffer(
    device: GPUDevice,
    size: number,
    usage: GPUBufferUsageFlags,
    label?: string
): GPUBuffer {
    // Ensure size is a multiple of 4 (WebGPU requirement)
    const alignedSize = Math.ceil(size / 4) * 4;
    
    return device.createBuffer({
        label,
        size: alignedSize,
        usage
    });
}

/**
 * Helper to handle render pipeline fragment targets
 */
export function createFragmentState(
    module: GPUShaderModule,
    format: GPUTextureFormat = 'bgra8unorm'
): GPUFragmentState {
    return {
        module,
        entryPoint: 'fs_main',
        targets: [{
            format
        }]
    };
}

export default {
    getTextureSize,
    getTextureSize2D,
    createTextureDescriptor,
    writeTimestamp,
    createComputePassWithTimestamps,
    hasFeature,
    hasNonStandardFeature,
    getWorkgroupMemoryLimit,
    createBuffer,
    createFragmentState
};
