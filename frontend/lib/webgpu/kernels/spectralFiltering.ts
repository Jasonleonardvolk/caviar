/**
 * spectralFiltering.ts
 * PLATINUM Edition: Advanced spectral filtering in k-space
 * 
 * Implements various filter types for k-space manipulation:
 * - Ideal (brick-wall) filter
 * - Raised-cosine filter  
 * - Gaussian filter
 * - Butterworth filter
 * - Custom filter from texture
 * 
 * Features:
 * - Anisotropic filtering support
 * - Adaptive filter parameters
 * - Anti-aliasing with smooth rolloff
 * - Direction-dependent filtering
 */

export interface SpectralFilterConfig {
    width: number;
    height: number;
    dx: number;
    dy: number;
    filterType: FilterType;
    cutoffFrequency: number;      // Normalized 0-1 (1 = Nyquist)
    rolloff?: number;              // Transition width for smooth filters
    order?: number;                // Filter order (sharpness)
    anisotropic?: boolean;         // Different cutoffs for x/y
    cutoffX?: number;
    cutoffY?: number;
    adaptiveParams?: AdaptiveParams;
}

export enum FilterType {
    Ideal = 'ideal',
    RaisedCosine = 'raised-cosine',
    Gaussian = 'gaussian',
    Butterworth = 'butterworth',
    SuperGaussian = 'super-gaussian',
    Custom = 'custom',
}

interface AdaptiveParams {
    energyThreshold?: number;      // Adapt based on energy content
    noiseLevel?: number;           // Adapt based on noise estimate
    preserveFeatures?: boolean;    // Preserve high-energy features
}

/**
 * Generate spectral filter shader code
 */
export function generateSpectralFilterShader(config: SpectralFilterConfig): string {
    const filterFunction = getFilterFunction(config);
    
    return `
// spectral_filter_${config.filterType}.wgsl
// Auto-generated spectral filter shader

struct FilterParams {
    width: u32,
    height: u32,
    dx: f32,
    dy: f32,
    cutoff: f32,          // Primary cutoff frequency
    cutoff_x: f32,        // Anisotropic x cutoff
    cutoff_y: f32,        // Anisotropic y cutoff
    rolloff: f32,         // Transition width
    order: f32,           // Filter order/sharpness
    anisotropic: u32,     // 0/1 flag
    adaptive: u32,        // 0/1 flag
    noise_level: f32,     // Estimated noise level
}

@group(0) @binding(0) var<uniform> params: FilterParams;
@group(0) @binding(1) var<storage, read_write> field_k: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> energy_density: array<f32>;  // Optional
@group(0) @binding(3) var filter_texture: texture_2d<f32>;  // For custom filters
@group(0) @binding(4) var filter_sampler: sampler;

const PI = 3.14159265359;
const TWO_PI = 6.28318530718;

// Convert array index to k-space frequency
fn get_k_coords(idx: u32) -> vec2<f32> {
    let x = idx % params.width;
    let y = idx / params.width;
    
    // Handle FFT indexing (DC at origin, then positive, then negative freqs)
    let kx = select(
        f32(x) * TWO_PI / (f32(params.width) * params.dx),
        (f32(x) - f32(params.width)) * TWO_PI / (f32(params.width) * params.dx),
        x > params.width / 2u
    );
    
    let ky = select(
        f32(y) * TWO_PI / (f32(params.height) * params.dy),
        (f32(y) - f32(params.height)) * TWO_PI / (f32(params.height) * params.dy),
        y > params.height / 2u
    );
    
    return vec2<f32>(kx, ky);
}

// Normalize k-vector to Nyquist units
fn normalize_k(k: vec2<f32>) -> vec2<f32> {
    let nyquist_x = PI / params.dx;
    let nyquist_y = PI / params.dy;
    return vec2<f32>(k.x / nyquist_x, k.y / nyquist_y);
}

${filterFunction}

// Adaptive filtering based on local energy
fn adaptive_filter(base_response: f32, idx: u32) -> f32 {
    if (params.adaptive == 0u) {
        return base_response;
    }
    
    // Get local energy density
    let energy = energy_density[idx];
    let avg_energy = energy_density[0];  // Assume first element stores average
    
    // Adapt filter based on energy
    if (energy > avg_energy * 2.0) {
        // High energy region - preserve more
        return mix(base_response, 1.0, 0.5);
    } else if (energy < params.noise_level) {
        // Likely noise - filter more aggressively
        return base_response * base_response;
    }
    
    return base_response;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.width * params.height) { return; }
    
    // Get k-space coordinates
    let k = get_k_coords(idx);
    let k_norm = normalize_k(k);
    
    // Calculate filter response
    var response: f32;
    
    if (params.anisotropic == 1u) {
        // Anisotropic filtering
        response = filter_anisotropic(k_norm);
    } else {
        // Isotropic filtering
        let k_mag = length(k_norm);
        response = filter_response(k_mag);
    }
    
    // Apply adaptive filtering if enabled
    response = adaptive_filter(response, idx);
    
    // Apply filter to field
    field_k[idx] *= response;
}

// Specialized entry point for custom texture-based filters
@compute @workgroup_size(8, 8, 1)
fn apply_custom_filter(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    
    let idx = gid.y * params.width + gid.x;
    
    // Sample filter texture
    let uv = vec2<f32>(
        f32(gid.x) / f32(params.width),
        f32(gid.y) / f32(params.height)
    );
    
    let filter_value = textureSampleLevel(filter_texture, filter_sampler, uv, 0.0).r;
    
    // Apply to field
    field_k[idx] *= filter_value;
}
    `;
}

/**
 * Get filter function based on type
 */
function getFilterFunction(config: SpectralFilterConfig): string {
    switch (config.filterType) {
        case FilterType.Ideal:
            return `
// Ideal (brick-wall) filter
fn filter_response(k_mag: f32) -> f32 {
    return select(0.0, 1.0, k_mag <= params.cutoff);
}

fn filter_anisotropic(k_norm: vec2<f32>) -> f32 {
    let in_x = abs(k_norm.x) <= params.cutoff_x;
    let in_y = abs(k_norm.y) <= params.cutoff_y;
    return select(0.0, 1.0, in_x && in_y);
}
            `;
            
        case FilterType.RaisedCosine:
            return `
// Raised-cosine filter with smooth rolloff
fn filter_response(k_mag: f32) -> f32 {
    if (k_mag <= params.cutoff - params.rolloff) {
        return 1.0;
    } else if (k_mag >= params.cutoff + params.rolloff) {
        return 0.0;
    }
    
    // Cosine rolloff region
    let t = (k_mag - params.cutoff + params.rolloff) / (2.0 * params.rolloff);
    return 0.5 * (1.0 + cos(PI * t));
}

fn filter_anisotropic(k_norm: vec2<f32>) -> f32 {
    let response_x = filter_response_1d(abs(k_norm.x), params.cutoff_x);
    let response_y = filter_response_1d(abs(k_norm.y), params.cutoff_y);
    return response_x * response_y;
}

fn filter_response_1d(k: f32, cutoff: f32) -> f32 {
    if (k <= cutoff - params.rolloff) {
        return 1.0;
    } else if (k >= cutoff + params.rolloff) {
        return 0.0;
    }
    
    let t = (k - cutoff + params.rolloff) / (2.0 * params.rolloff);
    return 0.5 * (1.0 + cos(PI * t));
}
            `;
            
        case FilterType.Gaussian:
            return `
// Gaussian filter
fn filter_response(k_mag: f32) -> f32 {
    // Standard deviation related to cutoff frequency
    let sigma = params.cutoff / (2.0 * sqrt(2.0 * log(2.0)));  // FWHM = cutoff
    return exp(-0.5 * pow(k_mag / sigma, 2.0));
}

fn filter_anisotropic(k_norm: vec2<f32>) -> f32 {
    let sigma_x = params.cutoff_x / (2.0 * sqrt(2.0 * log(2.0)));
    let sigma_y = params.cutoff_y / (2.0 * sqrt(2.0 * log(2.0)));
    
    let gauss_x = exp(-0.5 * pow(k_norm.x / sigma_x, 2.0));
    let gauss_y = exp(-0.5 * pow(k_norm.y / sigma_y, 2.0));
    
    return gauss_x * gauss_y;
}
            `;
            
        case FilterType.Butterworth:
            return `
// Butterworth filter
fn filter_response(k_mag: f32) -> f32 {
    let ratio = k_mag / params.cutoff;
    return 1.0 / sqrt(1.0 + pow(ratio, 2.0 * params.order));
}

fn filter_anisotropic(k_norm: vec2<f32>) -> f32 {
    let ratio_x = abs(k_norm.x) / params.cutoff_x;
    let ratio_y = abs(k_norm.y) / params.cutoff_y;
    
    let butter_x = 1.0 / sqrt(1.0 + pow(ratio_x, 2.0 * params.order));
    let butter_y = 1.0 / sqrt(1.0 + pow(ratio_y, 2.0 * params.order));
    
    return butter_x * butter_y;
}
            `;
            
        case FilterType.SuperGaussian:
            return `
// Super-Gaussian filter (tunable sharpness)
fn filter_response(k_mag: f32) -> f32 {
    let sigma = params.cutoff / (2.0 * pow(log(2.0), 1.0 / params.order));
    return exp(-pow(k_mag / sigma, params.order));
}

fn filter_anisotropic(k_norm: vec2<f32>) -> f32 {
    let sigma_x = params.cutoff_x / (2.0 * pow(log(2.0), 1.0 / params.order));
    let sigma_y = params.cutoff_y / (2.0 * pow(log(2.0), 1.0 / params.order));
    
    let super_x = exp(-pow(abs(k_norm.x) / sigma_x, params.order));
    let super_y = exp(-pow(abs(k_norm.y) / sigma_y, params.order));
    
    return super_x * super_y;
}
            `;
            
        case FilterType.Custom:
            return `
// Custom filter (uses texture)
fn filter_response(k_mag: f32) -> f32 {
    // This is overridden by apply_custom_filter entry point
    return 1.0;
}

fn filter_anisotropic(k_norm: vec2<f32>) -> f32 {
    // This is overridden by apply_custom_filter entry point
    return 1.0;
}
            `;
            
        default:
            return `
// Default pass-through
fn filter_response(k_mag: f32) -> f32 {
    return 1.0;
}

fn filter_anisotropic(k_norm: vec2<f32>) -> f32 {
    return 1.0;
}
            `;
    }
}

/**
 * Create filter texture for custom filtering
 */
export function createFilterTexture(
    device: GPUDevice,
    width: number,
    height: number,
    filterFunction: (kx: number, ky: number) => number
): GPUTexture {
    // Generate filter data
    const data = new Float32Array(width * height * 4);  // RGBA
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            // Convert to k-space coordinates (centered)
            const kx = (x - width / 2) / (width / 2);
            const ky = (y - height / 2) / (height / 2);
            
            const value = filterFunction(kx, ky);
            const idx = (y * width + x) * 4;
            
            data[idx] = value;      // R
            data[idx + 1] = value;  // G
            data[idx + 2] = value;  // B
            data[idx + 3] = 1.0;    // A
        }
    }
    
    // Create texture
    const texture = device.createTexture({
        size: { width, height },
        format: 'rgba32float',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        label: 'Custom Filter Texture',
    });
    
    // Upload data
    device.queue.writeTexture(
        { texture },
        data,
        { bytesPerRow: width * 16, rowsPerImage: height },
        { width, height }
    );
    
    return texture;
}

/**
 * Common filter presets
 */
export const FilterPresets = {
    antiAliasing: (): Partial<SpectralFilterConfig> => ({
        filterType: FilterType.RaisedCosine,
        cutoffFrequency: 0.9,  // 90% of Nyquist
        rolloff: 0.1,
    }),
    
    lowPass: (cutoff: number = 0.5): Partial<SpectralFilterConfig> => ({
        filterType: FilterType.Butterworth,
        cutoffFrequency: cutoff,
        order: 4,
    }),
    
    highPass: (cutoff: number = 0.1): Partial<SpectralFilterConfig> => ({
        filterType: FilterType.Custom,
        // Requires custom implementation (1 - lowpass)
    }),
    
    bandPass: (low: number = 0.2, high: number = 0.8): Partial<SpectralFilterConfig> => ({
        filterType: FilterType.Custom,
        // Requires custom implementation
    }),
    
    denoise: (noiseLevel: number = 0.01): Partial<SpectralFilterConfig> => ({
        filterType: FilterType.Gaussian,
        cutoffFrequency: 0.7,
        adaptiveParams: {
            noiseLevel,
            preserveFeatures: true,
        },
    }),
    
    sharpening: (): Partial<SpectralFilterConfig> => ({
        filterType: FilterType.Custom,
        // High-frequency emphasis filter
    }),
};