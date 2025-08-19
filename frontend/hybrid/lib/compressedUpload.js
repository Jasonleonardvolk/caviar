/**
 * WebGPU compressed texture upload helpers
 * Handles both RGBA and block-compressed texture formats
 */
// Block compression format information
const BLOCK_FORMATS = {
    'astc-4x4-unorm': { blockWidth: 4, blockHeight: 4, bytesPerBlock: 16 },
    'astc-4x4-unorm-srgb': { blockWidth: 4, blockHeight: 4, bytesPerBlock: 16 },
    'bc7-rgba-unorm': { blockWidth: 4, blockHeight: 4, bytesPerBlock: 16 },
    'bc7-rgba-unorm-srgb': { blockWidth: 4, blockHeight: 4, bytesPerBlock: 16 },
    'etc2-rgb8unorm': { blockWidth: 4, blockHeight: 4, bytesPerBlock: 8 },
    'etc2-rgb8unorm-srgb': { blockWidth: 4, blockHeight: 4, bytesPerBlock: 8 },
    'etc2-rgb8a1unorm': { blockWidth: 4, blockHeight: 4, bytesPerBlock: 8 },
    'etc2-rgb8a1unorm-srgb': { blockWidth: 4, blockHeight: 4, bytesPerBlock: 8 },
    'etc2-rgba8unorm': { blockWidth: 4, blockHeight: 4, bytesPerBlock: 16 },
    'etc2-rgba8unorm-srgb': { blockWidth: 4, blockHeight: 4, bytesPerBlock: 16 },
};
/**
 * Upload RGBA texture data to WebGPU
 */
export function uploadRGBA(device, levels, srgb = false) {
    if (levels.length === 0) {
        throw new Error('No texture levels provided');
    }
    const format = srgb ? 'rgba8unorm-srgb' : 'rgba8unorm';
    // Create texture with mipmaps
    const texture = device.createTexture({
        size: {
            width: levels[0].width,
            height: levels[0].height,
            depthOrArrayLayers: 1
        },
        format,
        mipLevelCount: levels.length,
        usage: GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
        label: 'RGBA Texture'
    });
    // Upload each mip level
    for (let i = 0; i < levels.length; i++) {
        const level = levels[i];
        const data = new Uint8Array(level.data);
        // Calculate bytes per row with alignment
        const bytesPerPixel = 4; // RGBA
        const bytesPerRow = Math.max(256, level.width * bytesPerPixel);
        // Write texture data
        device.queue.writeTexture({
            texture,
            mipLevel: i,
            origin: { x: 0, y: 0, z: 0 }
        }, data, {
            offset: 0,
            bytesPerRow,
            rowsPerImage: level.height
        }, {
            width: level.width,
            height: level.height,
            depthOrArrayLayers: 1
        });
    }
    return texture;
}
/**
 * Upload compressed texture data to WebGPU
 */
export function uploadCompressed(device, levels, format, gpuFormat) {
    if (levels.length === 0) {
        throw new Error('No texture levels provided');
    }
    const blockInfo = BLOCK_FORMATS[format];
    if (!blockInfo) {
        throw new Error(`Unknown compressed format: ${format}`);
    }
    // Create texture with compressed format
    const texture = device.createTexture({
        size: {
            width: levels[0].width,
            height: levels[0].height,
            depthOrArrayLayers: 1
        },
        format: gpuFormat,
        mipLevelCount: levels.length,
        usage: GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
        label: `Compressed Texture (${format})`
    });
    // Upload each mip level
    for (let i = 0; i < levels.length; i++) {
        const level = levels[i];
        const data = new Uint8Array(level.data);
        // Calculate block dimensions
        const blocksWide = Math.ceil(level.width / blockInfo.blockWidth);
        const blocksHigh = Math.ceil(level.height / blockInfo.blockHeight);
        // Calculate bytes per row with alignment
        const bytesPerRow = Math.max(256, blocksWide * blockInfo.bytesPerBlock);
        // Write compressed texture data
        device.queue.writeTexture({
            texture,
            mipLevel: i,
            origin: { x: 0, y: 0, z: 0 }
        }, data, {
            offset: 0,
            bytesPerRow,
            rowsPerImage: blocksHigh
        }, {
            width: level.width,
            height: level.height,
            depthOrArrayLayers: 1
        });
    }
    return texture;
}
/**
 * Map transcoder format to WebGPU format string
 */
export function getGPUFormat(format, srgb = false) {
    switch (format) {
        case 'astc':
            return {
                format: srgb ? 'astc-4x4-unorm-srgb' : 'astc-4x4-unorm',
                blockFormat: srgb ? 'astc-4x4-unorm-srgb' : 'astc-4x4-unorm'
            };
        case 'bc7':
            return {
                format: srgb ? 'bc7-rgba-unorm-srgb' : 'bc7-rgba-unorm',
                blockFormat: srgb ? 'bc7-rgba-unorm-srgb' : 'bc7-rgba-unorm'
            };
        case 'etc2':
            return {
                format: srgb ? 'etc2-rgb8unorm-srgb' : 'etc2-rgb8unorm',
                blockFormat: srgb ? 'etc2-rgb8unorm-srgb' : 'etc2-rgb8unorm'
            };
        case 'rgba':
        default:
            return {
                format: srgb ? 'rgba8unorm-srgb' : 'rgba8unorm'
            };
    }
}
/**
 * Upload texture based on format
 */
export function uploadTexture(device, levels, format, srgb = false) {
    const { format: gpuFormat, blockFormat } = getGPUFormat(format, srgb);
    if (format === 'rgba') {
        return uploadRGBA(device, levels, srgb);
    }
    else if (blockFormat) {
        return uploadCompressed(device, levels, blockFormat, gpuFormat);
    }
    else {
        throw new Error(`Unsupported format: ${format}`);
    }
}
/**
 * Create a sampler for compressed textures
 */
export function createTextureSampler(device, options = {}) {
    return device.createSampler({
        minFilter: options.minFilter || 'linear',
        magFilter: options.magFilter || 'linear',
        mipmapFilter: options.mipmapFilter || 'linear',
        addressModeU: options.addressModeU || 'repeat',
        addressModeV: options.addressModeV || 'repeat',
        addressModeW: options.addressModeW || 'repeat',
        lodMinClamp: options.lodMinClamp || 0,
        lodMaxClamp: options.lodMaxClamp || 32,
        maxAnisotropy: options.maxAnisotropy || 16,
        label: 'Texture Sampler'
    });
}
/**
 * Check if a compressed format is supported by the device
 */
export function isFormatSupported(adapter, format) {
    const features = adapter.features;
    switch (format) {
        case 'astc':
            return features.has('texture-compression-astc');
        case 'bc7':
            return features.has('texture-compression-bc');
        case 'etc2':
            return features.has('texture-compression-etc2');
        default:
            return false;
    }
}
/**
 * Get all supported compressed formats for a device
 */
export function getSupportedFormats(adapter) {
    const supported = new Set();
    if (isFormatSupported(adapter, 'astc')) {
        supported.add('astc');
    }
    if (isFormatSupported(adapter, 'bc7')) {
        supported.add('bc7');
    }
    if (isFormatSupported(adapter, 'etc2')) {
        supported.add('etc2');
    }
    return supported;
}
