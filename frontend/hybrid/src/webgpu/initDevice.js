/* frontend/hybrid/src/webgpu/initDevice.ts
 * Enhanced WebGPU initialization with comprehensive feature detection
 * iOS 26 Safari optimized with graceful fallbacks
 */
// Detect platform
function detectPlatform() {
    const ua = navigator.userAgent;
    const isIOS = /iPad|iPhone|iPod/.test(ua) ||
        (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
    const isSafari = /^((?!chrome|android).)*safari/i.test(ua);
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(ua);
    return { isIOS, isSafari, isMobile };
}
// Determine performance tier based on limits
function determinePerformanceTier(limits) {
    const maxTextureSize = limits.maxTextureDimension2D || 0;
    const maxComputeWorkgroupSizeX = limits.maxComputeWorkgroupSizeX || 0;
    const maxBufferSize = limits.maxBufferSize || 0;
    // High tier: Desktop GPUs, M-series chips
    if (maxTextureSize >= 16384 &&
        maxComputeWorkgroupSizeX >= 512 &&
        maxBufferSize >= 2147483648) { // 2GB
        return 'high';
    }
    // Medium tier: Good mobile GPUs, older desktops
    if (maxTextureSize >= 8192 &&
        maxComputeWorkgroupSizeX >= 256 &&
        maxBufferSize >= 268435456) { // 256MB
        return 'medium';
    }
    // Low tier: Basic mobile GPUs
    return 'low';
}
// Get safe limits based on platform and tier
function getSafeLimits(adapter, platform, options) {
    const limits = adapter.limits;
    const requested = {};
    // iOS Safari specific limits
    if (platform.isIOS && platform.isSafari) {
        // Conservative limits for iOS Safari
        requested.maxTextureDimension2D = Math.min(limits.maxTextureDimension2D || 8192, 8192);
        requested.maxBufferSize = Math.min(limits.maxBufferSize || 268435456, 268435456 // 256MB max for iOS
        );
        requested.maxComputeWorkgroupSizeX = Math.min(limits.maxComputeWorkgroupSizeX || 256, 256);
        requested.maxComputeWorkgroupSizeY = Math.min(limits.maxComputeWorkgroupSizeY || 256, 256);
        requested.maxComputeInvocationsPerWorkgroup = Math.min(limits.maxComputeInvocationsPerWorkgroup || 256, 256);
    }
    else if (platform.isMobile) {
        // General mobile limits
        requested.maxTextureDimension2D = Math.min(limits.maxTextureDimension2D || 4096, 4096);
        requested.maxBufferSize = Math.min(limits.maxBufferSize || 134217728, 134217728 // 128MB for mobile
        );
    }
    // Apply user-requested limits if provided
    if (options?.requiredLimits) {
        Object.entries(options.requiredLimits).forEach(([key, value]) => {
            const adapterLimit = limits[key];
            if (adapterLimit !== undefined) {
                requested[key] = Math.min(adapterLimit, value);
            }
        });
    }
    return requested;
}
// Main initialization function
export async function initWebGPU(options) {
    // Check WebGPU availability
    if (!('gpu' in navigator)) {
        throw new Error('WebGPU not available in this browser. Please use Chrome, Edge, or Safari 18+');
    }
    const platform = detectPlatform();
    // Request adapter with appropriate power preference
    const powerPreference = options?.powerPreference ||
        (platform.isMobile ? 'low-power' : 'high-performance');
    const adapter = await navigator.gpu.requestAdapter({
        powerPreference,
    });
    if (!adapter) {
        throw new Error('No suitable GPU adapter found. WebGPU may be disabled in browser settings.');
    }
    // Feature detection - check what's available
    const availableFeatures = adapter.features;
    const hasSubgroups = availableFeatures.has('subgroups');
    const hasF16 = availableFeatures.has('shader-f16');
    const hasDepthClipControl = availableFeatures.has('depth-clip-control');
    const hasTimestamp = availableFeatures.has('timestamp-query');
    const hasIndirectFirstInstance = availableFeatures.has('indirect-first-instance');
    const hasDepth32FloatStencil8 = availableFeatures.has('depth32float-stencil8');
    const hasTextureCompressionBC = availableFeatures.has('texture-compression-bc');
    const hasTextureCompressionETC2 = availableFeatures.has('texture-compression-etc2');
    const hasTextureCompressionASTC = availableFeatures.has('texture-compression-astc');
    // Build feature list - only request features that are available
    const requiredFeatures = [];
    let fallbackMode = false;
    // Add features if available and not forcing fallback
    if (!options?.forceFallback) {
        if (hasSubgroups) {
            requiredFeatures.push('subgroups');
        }
        else {
            console.warn('[WebGPU] Subgroups not available - using fallback compute paths');
            fallbackMode = true;
        }
        if (hasF16) {
            requiredFeatures.push('shader-f16');
        }
        else {
            console.warn('[WebGPU] Float16 not available - using float32 fallback');
        }
        // Add other beneficial features if available
        if (hasDepthClipControl)
            requiredFeatures.push('depth-clip-control');
        if (hasTimestamp && !platform.isIOS) { // iOS doesn't like timestamp queries
            requiredFeatures.push('timestamp-query');
        }
        if (hasIndirectFirstInstance)
            requiredFeatures.push('indirect-first-instance');
        if (hasDepth32FloatStencil8)
            requiredFeatures.push('depth32float-stencil8');
        // Texture compression (useful for holographic data)
        if (hasTextureCompressionBC && !platform.isMobile) {
            requiredFeatures.push('texture-compression-bc');
        }
        if (hasTextureCompressionETC2 && platform.isMobile) {
            requiredFeatures.push('texture-compression-etc2');
        }
        if (hasTextureCompressionASTC && platform.isIOS) {
            requiredFeatures.push('texture-compression-astc');
        }
    }
    // Add user-requested features if available
    if (options?.requiredFeatures) {
        for (const feature of options.requiredFeatures) {
            if (availableFeatures.has(feature) && !requiredFeatures.includes(feature)) {
                requiredFeatures.push(feature);
            }
            else if (!availableFeatures.has(feature)) {
                console.warn(`[WebGPU] Requested feature '${feature}' not available`);
                fallbackMode = true;
            }
        }
    }
    // Get safe limits
    const limits = getSafeLimits(adapter, platform, options);
    // Request device with detected features and limits
    let device;
    try {
        device = await adapter.requestDevice({
            requiredFeatures,
            requiredLimits: Object.keys(limits).length > 0 ? limits : undefined,
        });
    }
    catch (error) {
        console.error('[WebGPU] Failed to create device with requested features:', error);
        // Fallback: try with no features
        console.warn('[WebGPU] Attempting fallback mode with minimal features');
        device = await adapter.requestDevice({
            requiredFeatures: [],
            requiredLimits: undefined,
        });
        fallbackMode = true;
    }
    // Get adapter info if available
    let adapterInfo;
    try {
        // Use the standard API if available
        if ('requestAdapterInfo' in adapter) {
            adapterInfo = await adapter.requestAdapterInfo();
        }
        else if ('info' in adapter) {
            adapterInfo = adapter.info;
        }
    }
    catch (err) {
        console.log('[WebGPU] Adapter info not available');
    }
    // Determine performance tier
    const tier = determinePerformanceTier(adapter.limits);
    // Log device capabilities
    console.log('[WebGPU] Device initialized:', {
        tier,
        features: Array.from(device.features),
        limits: Object.fromEntries(Object.entries(limits).slice(0, 5) // Just log first few limits
        ),
        platform,
        fallbackMode,
        adapterInfo
    });
    // Build capabilities object
    const caps = {
        // Core features
        subgroups: device.features.has('subgroups'),
        f16: device.features.has('shader-f16'),
        // Additional features
        depthClipControl: device.features.has('depth-clip-control'),
        timestamp: device.features.has('timestamp-query'),
        indirectFirstInstance: device.features.has('indirect-first-instance'),
        depth32FloatStencil8: device.features.has('depth32float-stencil8'),
        textureCompressionBC: device.features.has('texture-compression-bc'),
        textureCompressionETC2: device.features.has('texture-compression-etc2'),
        textureCompressionASTC: device.features.has('texture-compression-astc'),
        // Limits and info
        limits: device.limits,
        adapterInfo,
        // Platform
        isMobile: platform.isMobile,
        isIOS: platform.isIOS,
        isSafari: platform.isSafari,
        // Performance
        tier
    };
    // Set up device lost handler
    device.lost.then((info) => {
        console.error('[WebGPU] Device lost:', info);
        // Dispatch event for app to handle
        window.dispatchEvent(new CustomEvent('webgpu-device-lost', {
            detail: {
                reason: info.reason,
                message: info.message || 'Unknown error'
            }
        }));
    });
    return { device, caps, fallbackMode };
}
// Helper function to check if WebGPU is supported
export function isWebGPUSupported() {
    return 'gpu' in navigator;
}
// Helper function to get a fallback message for unsupported browsers
export function getWebGPUErrorMessage() {
    const platform = detectPlatform();
    if (!('gpu' in navigator)) {
        if (platform.isIOS) {
            return 'WebGPU requires iOS 17+ with Safari 18+. Please update your device.';
        }
        else if (platform.isSafari) {
            return 'WebGPU requires Safari 18+. Please update Safari or use Chrome/Edge.';
        }
        else {
            return 'WebGPU is not supported in this browser. Please use Chrome 113+, Edge 113+, or Safari 18+.';
        }
    }
    return 'WebGPU is available but failed to initialize. Check browser settings.';
}
// Export platform detection for use elsewhere
export { detectPlatform };
