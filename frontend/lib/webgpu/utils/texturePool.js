export class TexturePool {
    constructor(device, config = {}) {
        this.allocatedTextures = new Map(); // texture -> size in bytes
        this.totalAllocated = 0;
        this.allocationFailures = 0;
        this.device = device;
        this.maxBudget = (config.maxBudgetMB || 256) * 1024 * 1024; // Default 256MB
        this.onBudgetExceeded = config.onBudgetExceeded || (() => {
            console.warn('TexturePool: Budget exceeded');
        });
        this.onAllocationFailed = config.onAllocationFailed || ((e) => {
            console.error('TexturePool: Allocation failed', e);
        });
    }
    create(desc) {
        // Calculate texture size
        const size = this.calculateTextureSize(desc);
        // Check budget before allocation
        if (this.totalAllocated + size > this.maxBudget) {
            this.onBudgetExceeded();
            // Try to free some memory
            this.gc();
            // If still over budget, throw
            if (this.totalAllocated + size > this.maxBudget) {
                throw new Error(`TexturePool: Would exceed budget (${this.totalAllocated + size} > ${this.maxBudget})`);
            }
        }
        try {
            const texture = this.device.createTexture(desc);
            this.allocatedTextures.set(texture, size);
            this.totalAllocated += size;
            // Reset failure counter on success
            this.allocationFailures = 0;
            return texture;
        }
        catch (e) {
            this.allocationFailures++;
            this.onAllocationFailed(e);
            // If we've had multiple failures, trigger emergency downgrade
            if (this.allocationFailures > 2) {
                this.onBudgetExceeded();
            }
            throw e;
        }
    }
    destroy(texture) {
        const size = this.allocatedTextures.get(texture);
        if (size) {
            this.allocatedTextures.delete(texture);
            this.totalAllocated -= size;
            texture.destroy();
        }
    }
    calculateTextureSize(desc) {
        const width = desc.size.width || 1;
        const height = desc.size.height || 1;
        const depth = desc.size.depthOrArrayLayers || 1;
        const mipLevels = desc.mipLevelCount || 1;
        // Get bytes per pixel based on format
        const bytesPerPixel = this.getBytesPerPixel(desc.format);
        // Calculate total size including mip levels
        let totalSize = 0;
        for (let mip = 0; mip < mipLevels; mip++) {
            const mipWidth = Math.max(1, width >> mip);
            const mipHeight = Math.max(1, height >> mip);
            const mipDepth = Math.max(1, depth >> mip);
            totalSize += mipWidth * mipHeight * mipDepth * bytesPerPixel;
        }
        return totalSize;
    }
    getBytesPerPixel(format) {
        const formatSizes = {
            'r8unorm': 1,
            'r8snorm': 1,
            'r8uint': 1,
            'r8sint': 1,
            'r16uint': 2,
            'r16sint': 2,
            'r16float': 2,
            'rg8unorm': 2,
            'rg8snorm': 2,
            'rg8uint': 2,
            'rg8sint': 2,
            'r32uint': 4,
            'r32sint': 4,
            'r32float': 4,
            'rg16uint': 4,
            'rg16sint': 4,
            'rg16float': 4,
            'rgba8unorm': 4,
            'rgba8unorm-srgb': 4,
            'rgba8snorm': 4,
            'rgba8uint': 4,
            'rgba8sint': 4,
            'bgra8unorm': 4,
            'bgra8unorm-srgb': 4,
            'rgb10a2unorm': 4,
            'rg11b10ufloat': 4,
            'rgb9e5ufloat': 4,
            'rg32uint': 8,
            'rg32sint': 8,
            'rg32float': 8,
            'rgba16uint': 8,
            'rgba16sint': 8,
            'rgba16float': 8,
            'rgba32uint': 16,
            'rgba32sint': 16,
            'rgba32float': 16,
            // Depth/stencil formats
            'depth16unorm': 2,
            'depth24plus': 4,
            'depth24plus-stencil8': 4,
            'depth32float': 4,
            'depth32float-stencil8': 5,
        };
        return formatSizes[format] || 4; // Default to 4 bytes
    }
    gc() {
        // Simple GC: remove destroyed textures from tracking
        const toRemove = [];
        for (const [texture, size] of this.allocatedTextures) {
            // Check if texture is still valid (this is a simplification)
            // In practice, you'd need a way to detect destroyed textures
            try {
                // Try to access a property - will throw if destroyed
                const _ = texture.width;
            }
            catch {
                toRemove.push(texture);
            }
        }
        for (const texture of toRemove) {
            const size = this.allocatedTextures.get(texture);
            this.allocatedTextures.delete(texture);
            this.totalAllocated -= size;
        }
    }
    getStats() {
        return {
            totalAllocated: this.totalAllocated,
            maxBudget: this.maxBudget,
            textureCount: this.allocatedTextures.size,
            allocationFailures: this.allocationFailures,
            usage: this.totalAllocated / this.maxBudget
        };
    }
    clear() {
        for (const [texture] of this.allocatedTextures) {
            texture.destroy();
        }
        this.allocatedTextures.clear();
        this.totalAllocated = 0;
        this.allocationFailures = 0;
    }
}
// Singleton instance for global usage
let globalPool = null;
export function initTexturePool(device, config) {
    if (!globalPool) {
        globalPool = new TexturePool(device, config);
    }
    return globalPool;
}
export function getTexturePool() {
    if (!globalPool) {
        throw new Error('TexturePool not initialized. Call initTexturePool first.');
    }
    return globalPool;
}
