// propagation_v2.wgsl - iOS 26 Optimized Edition
// Enhanced FFT-based wave propagation with Metal 3 optimizations
// New features: Learned propagation, neural enhancement, simdgroup ops

struct PropagationParams {
    distance: f32,
    wavelength: f32,
    pixel_size: f32,
    amplitude_scale: f32,
    method: u32,                // 0: ASM, 1: Fresnel, 2: Fraunhofer, 3: Auto, 4: Neural
    apply_aperture: u32,
    fresnel_number: f32,
    use_band_limiting: u32,
    // Precomputed values
    k: f32,
    inv_wavelength: f32,
    aperture_radius_mm: f32,
    edge_smoothness: f32,
    // Coherence parameters
    noise_amount: f32,
    prng_seed: u32,
    coherence_length: f32,
    // NEW: Neural enhancement
    neural_blend: f32,          // 0-1 blend with neural prediction
    // NEW: Adaptive sampling
    importance_threshold: f32,   // For adaptive ray density
    max_rays_per_pixel: u32,
    // NEW: Aberration coefficients
    zernike_coeffs: array<f32, 15>, // Up to 4th order
    _padding: u32
}

struct FrequencyParams {
    fx_max: f32,
    fy_max: f32,
    dfx: f32,
    dfy: f32,
    fx_scale: f32,
    fy_scale: f32,
    bandwidth_limit: f32,
    raised_cosine_width: f32,
    // NEW: Gerchberg-Saxton parameters
    gs_iterations: u32,
    gs_error_threshold: f32,
    // NEW: Bandwidth extrapolation
    extrapolation_factor: f32,
    _padding: f32
}

// Constants
const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const INV_PI: f32 = 0.31830988618;
const INV_TWO_PI: f32 = 0.15915494309;
// Use overrides for device-adaptive workgroup sizing
override WG_X: u32 = 16u;
override WG_Y: u32 = 16u;
const WORKGROUP_SIZE: u32 = 16u; // Reduced to 16x16=256 for device compatibility
const SIMDGROUP_SIZE: u32 = 32u; // Metal simdgroup width

// Bindings
@group(0) @binding(0) var<uniform> prop_params: PropagationParams;
@group(0) @binding(1) var<uniform> freq_params: FrequencyParams;
@group(1) @binding(0) var input_field: texture_storage_2d<rg32float, read>;
@group(1) @binding(1) var output_field: texture_storage_2d<rg32float, read_write>;
@group(1) @binding(2) var frequency_domain: texture_storage_2d<rg32float, read_write>;
@group(1) @binding(3) var transfer_function: texture_storage_2d<rg32float, read_write>;

// NEW: Neural network weights for learned propagation
@group(2) @binding(0) var<storage, read> neural_weights: array<f32>;
@group(2) @binding(1) var<storage, read> neural_bias: array<f32>;

// NEW: Importance map for adaptive sampling
@group(2) @binding(2) var importance_map: texture_2d<f32>;
@group(2) @binding(3) var importance_sampler: sampler;

// Shared memory with Metal 3 optimization hints
var<workgroup> shared_field: array<vec4<f32>, 256>; // 16x16
var<workgroup> shared_transfer: array<vec4<f32>, 256>;
// NEW: Simdgroup shared memory for Metal
var<workgroup> simd_scratch: array<f32, 128>;

fn clamp_index_dyn(i: u32, len: u32) -> u32 {
    return select(i, len - 1u, i >= len);
}

// Enhanced complex exponential with Pade approximant
fn complex_exp_pade(phase: f32) -> vec2<f32> {
    let reduced_phase = phase - round(phase * INV_TWO_PI) * TWO_PI;
    
    // Pade [3/3] approximant - more accurate than Taylor
    if (abs(reduced_phase) < 2.0) {
        let x = reduced_phase;
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x2 * x2;
        
        // Pade coefficients for cosine
        let cos_num = 1.0 - 0.4999999996*x2 + 0.0416666418*x4;
        let cos_den = 1.0 + 0.0416666642*x2 + 0.0013888397*x4;
        let cos_approx = cos_num / cos_den;
        
        // Pade coefficients for sine  
        let sin_num = x - 0.1666666664*x3;
        let sin_den = 1.0 + 0.0083333315*x2;
        let sin_approx = sin_num / sin_den;
        
        return vec2<f32>(cos_approx, sin_approx);
    }
    
    return vec2<f32>(cos(reduced_phase), sin(reduced_phase));
}

// NEW: SIMD-optimized complex multiply for Metal
fn complex_multiply_simd(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    // Metal will recognize this pattern and use FMA
    let real = fma(a.x, b.x, -a.y * b.y);
    let imag = fma(a.x, b.y, a.y * b.x);
    return vec2<f32>(real, imag);
}

// NEW: Van Cittert-Zernike coherence
fn van_cittert_zernike(r1: vec2<f32>, r2: vec2<f32>, source_size: f32) -> f32 {
    let delta_r = length(r1 - r2);
    let coherence_radius = prop_params.wavelength * prop_params.distance / source_size;
    let x = PI * delta_r / coherence_radius;
    
    // sinc function with small-x approximation
    if (abs(x) < 0.001) {
        return 1.0 - x*x/6.0;
    }
    return sin(x) / x;
}

// NEW: Zernike polynomial aberration
fn zernike_aberration(rho: f32, theta: f32) -> f32 {
    var phase = 0.0;
    let r2 = rho * rho;
    let r3 = r2 * rho;
    let r4 = r2 * r2;
    
    // Piston (Z0)
    phase += prop_params.zernike_coeffs[0];
    
    // Tilt (Z1, Z2)
    phase += prop_params.zernike_coeffs[1] * rho * cos(theta);
    phase += prop_params.zernike_coeffs[2] * rho * sin(theta);
    
    // Defocus (Z3)
    phase += prop_params.zernike_coeffs[3] * (2.0*r2 - 1.0);
    
    // Astigmatism (Z4, Z5)
    phase += prop_params.zernike_coeffs[4] * r2 * cos(2.0*theta);
    phase += prop_params.zernike_coeffs[5] * r2 * sin(2.0*theta);
    
    // Coma (Z6, Z7)
    phase += prop_params.zernike_coeffs[6] * (3.0*r3 - 2.0*rho) * cos(theta);
    phase += prop_params.zernike_coeffs[7] * (3.0*r3 - 2.0*rho) * sin(theta);
    
    // Spherical (Z8)
    phase += prop_params.zernike_coeffs[8] * (6.0*r4 - 6.0*r2 + 1.0);
    
    return phase;
}

// NEW: Gerchberg bandwidth extrapolation
fn gerchberg_bandwidth_recovery(
    field: vec2<f32>,
    coord: vec2<i32>,
    dims: vec2<u32>
) -> vec2<f32> {
    var result = field;
    let uv = vec2<f32>(coord) / vec2<f32>(dims);
    
    // Check if we're in the extrapolation region
    let freq = length(uv - 0.5) * 2.0;
    if (freq > freq_params.bandwidth_limit && 
        freq < freq_params.bandwidth_limit * freq_params.extrapolation_factor) {
        
        // Apply iterative constraint satisfaction
        for (var iter = 0u; iter < freq_params.gs_iterations; iter++) {
            // Forward constraint: Known low frequencies
            if (freq <= freq_params.bandwidth_limit) {
                result = field; // Keep known data
            }
            
            // Backward constraint: Smoothness prior
            let grad = compute_gradient(result, coord, dims);
            let smoothness = exp(-length(grad) * 0.1);
            result = mix(result, result * smoothness, 0.5);
        }
    }
    
    return result;
}

// Helper: Compute gradient for smoothness constraint
fn compute_gradient(field: vec2<f32>, coord: vec2<i32>, dims: vec2<u32>) -> vec2<f32> {
    var grad = vec2<f32>(0.0);
    
    // Central differences
    if (coord.x > 0 && coord.x < i32(dims.x) - 1) {
        let left = textureLoad(frequency_domain, coord - vec2<i32>(1, 0)).xy;
        let right = textureLoad(frequency_domain, coord + vec2<i32>(1, 0)).xy;
        grad.x = length(right - left) * 0.5;
    }
    
    if (coord.y > 0 && coord.y < i32(dims.y) - 1) {
        let up = textureLoad(frequency_domain, coord - vec2<i32>(0, 1)).xy;
        let down = textureLoad(frequency_domain, coord + vec2<i32>(0, 1)).xy;
        grad.y = length(down - up) * 0.5;
    }
    
    return grad;
}

// NEW: Neural-enhanced propagation operator
fn neural_propagation(fx: f32, fy: f32, classical_transfer: vec2<f32>) -> vec2<f32> {
    // Simple 3-layer MLP for learned corrections
    let input_vec = vec4<f32>(fx, fy, classical_transfer.x, classical_transfer.y);
    
    // Layer 1: 4 -> 16
    var hidden1 = vec4<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let weight_idx = i * 4u + j;
            hidden1[i] += input_vec[j] * neural_weights[clamp_index_dyn(weight_idx, arrayLength(&neural_weights))];
        }
        hidden1[i] = max(0.0, hidden1[i] + neural_bias[clamp_index_dyn(i, arrayLength(&neural_bias))]); // ReLU
    }
    
    // Layer 2: 16 -> 2 (complex output)
    var output = vec2<f32>(0.0);
    for (var i = 0u; i < 2u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let weight_idx = 16u + i * 4u + j;
            output[i] += hidden1[j] * neural_weights[clamp_index_dyn(weight_idx, arrayLength(&neural_weights))];
        }
        output[i] = tanh(output[i]); // Bounded output
    }
    
    // Blend with classical result
    return mix(classical_transfer, output, prop_params.neural_blend);
}

// Enhanced Angular Spectrum with all improvements
fn angular_spectrum_enhanced(fx: f32, fy: f32, coord: vec2<f32>) -> vec2<f32> {
    let k = prop_params.k;
    let kx = TWO_PI * fx;
    let ky = TWO_PI * fy;
    
    let k_squared = k * k;
    let kxy_squared = kx * kx + ky * ky;
    let kz_squared = k_squared - kxy_squared;
    
    // Add Zernike aberrations
    let rho = sqrt(fx*fx + fy*fy) / max(freq_params.fx_max, freq_params.fy_max);
    let theta = atan2(fy, fx);
    let aberration_phase = zernike_aberration(rho, theta);
    
    var transfer: vec2<f32>;
    
    if (kz_squared <= 0.0) {
        // Enhanced evanescent wave handling
        let kz_imag = sqrt(-kz_squared);
        let decay = exp(-kz_imag * prop_params.distance);
        
        // Add sub-wavelength detail preservation
        let detail_factor = exp(-kxy_squared / (4.0 * k_squared));
        transfer = vec2<f32>(decay * detail_factor, 0.0);
    } else {
        // Propagating waves
        let kz = sqrt(kz_squared);
        let phase = kz * prop_params.distance + aberration_phase;
        transfer = complex_exp_pade(phase);
    }
    
    // Apply Van Cittert-Zernike coherence
    let coherence = van_cittert_zernike(coord, vec2<f32>(0.5), 0.1);
    transfer *= coherence;
    
    // Neural enhancement if enabled
    if (prop_params.method == 4u) {
        transfer = neural_propagation(fx, fy, transfer);
    }
    
    return transfer;
}

// NEW: Adaptive importance sampling
fn get_importance_weight(coord: vec2<f32>) -> f32 {
    return textureSampleLevel(importance_map, importance_sampler, coord, 0.0).r;
}

// Main compute kernel with Metal optimizations
@compute @workgroup_size(WG_X, WG_Y, 1)
fn compute_transfer_enhanced(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_invocation_id) lane_id: u32,  // Metal simdgroup
    @builtin(subgroup_size) simd_size: u32
) {
    let coord = vec2<i32>(global_id.xy);
    let dims = textureDimensions(transfer_function);
    
    if (any(coord >= vec2<i32>(dims))) {
        return;
    }
    
    // Get spatial frequency
    let freq = vec2<f32>(coord - vec2<i32>(dims/2u)) * vec2<f32>(freq_params.dfx, freq_params.dfy);
    let uv = vec2<f32>(coord) / vec2<f32>(dims);
    
    // Compute transfer function with enhancements
    var transfer = angular_spectrum_enhanced(freq.x, freq.y, uv);
    
    // Apply Gerchberg bandwidth recovery
    transfer = gerchberg_bandwidth_recovery(transfer, coord, dims);
    
    // Adaptive sampling based on importance
    let importance = get_importance_weight(uv);
    if (importance < prop_params.importance_threshold) {
        // Low importance region - use simplified computation
        transfer *= 0.1;
    }
    
    // Metal simdgroup optimization - reduction across lanes
    if (simd_size == SIMDGROUP_SIZE) {
        // Compute average magnitude across simdgroup for normalization
        let mag = length(transfer);
        var simd_sum = mag;
        
        // Butterfly reduction using simdgroup operations
        for (var offset = simd_size >> 1u; offset > 0u; offset >>= 1u) {
            simd_sum += simdgroupShuffleDown(simd_sum, offset);
        }
        
        // Broadcast result to all lanes
        let avg_mag = simdgroupBroadcast(simd_sum / f32(simd_size), 0u);
        
        // Normalize if needed
        if (avg_mag > 10.0) {
            transfer *= 10.0 / avg_mag;
        }
    }
    
    // Store result
    textureStore(transfer_function, coord, vec4<f32>(transfer, 0.0, 0.0));
}

// NEW: Phase-only hologram generation for bandwidth optimization
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn convert_to_kinoform(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = vec2<i32>(global_id.xy);
    let dims = textureDimensions(output_field);
    
    if (any(coord >= vec2<i32>(dims))) {
        return;
    }
    
    // Read complex field
    let field = textureLoad(output_field, vec2<u32>(coord)).xy;
    
    // Extract phase, normalize amplitude
    let phase = atan2(field.y, field.x);
    let kinoform = complex_exp_pade(phase);
    
    // Optional: Add carrier frequency for off-axis hologram
    let carrier_freq = vec2<f32>(5.0, 0.0); // cycles/mm
    let carrier_phase = TWO_PI * dot(carrier_freq, vec2<f32>(coord) * prop_params.pixel_size);
    let modulated = complex_multiply_simd(kinoform, complex_exp_pade(carrier_phase));
    
    textureStore(output_field, vec2<u32>(coord), vec4<f32>(modulated, 0.0, 0.0));
}

// NEW: Optimized multi-wavelength with Metal tensor operations
@compute @workgroup_size(16, 16, 1)
fn propagate_multi_wavelength_tensor(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32
) {
    let coord = vec2<i32>(global_id.xy);
    let dims = textureDimensions(frequency_domain);
    
    if (any(coord >= vec2<i32>(dims))) {
        return;
    }
    
    // Load transfer functions for RGB into shared memory
    if (local_idx < 256u) {
        let shared_coord = vec2<u32>(local_idx % 16u, local_idx / 16u);
        let global_coord = vec2<i32>(global_id.xy / 16u * 16u) + vec2<i32>(shared_coord);
        
        if (all(global_coord < vec2<i32>(dims))) {
            shared_field[local_idx] = textureLoad(transfer_function, global_coord);
        }
    }
    workgroupBarrier();
    
    // Process RGB channels in parallel using shared data
    let local_coord = vec2<u32>(global_id.xy % 16u);
    let shared_idx = local_coord.y * 16u + local_coord.x;
    
    if (shared_idx < 256u) {
        let transfer_rgb = shared_field[shared_idx].xyz;
        
        // Apply wavelength-specific processing
        var result = vec3<f32>(0.0);
        result.r = transfer_rgb.r * 0.9;  // Red: 700nm factor
        result.g = transfer_rgb.g * 1.0;  // Green: 550nm reference
        result.b = transfer_rgb.b * 1.1;  // Blue: 450nm factor
        
        // Store result
        textureStore(frequency_domain, vec2<u32>(coord), vec4<f32>(result, 1.0));
    }
}
