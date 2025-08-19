// ${IRIS_ROOT}\frontend\shaders\propagation.wgsl
// Enhanced FFT-based wave propagation with complete method implementations
// Addresses all feedback for production-ready holographic rendering

struct PropagationParams {
    distance: f32,              // Propagation distance in mm
    wavelength: f32,            // Wavelength in mm
    pixel_size: f32,            // Physical pixel size in mm
    amplitude_scale: f32,       // Global amplitude scaling
    method: u32,                // 0: Angular Spectrum, 1: Fresnel, 2: Fraunhofer, 3: Auto
    apply_aperture: u32,        // Whether to apply circular aperture
    fresnel_number: f32,        // Pre-computed Fresnel number
    use_band_limiting: u32,     // Enable band-limiting for anti-aliasing
    // Precomputed values for performance
    k: f32,                     // Wave number (2pi/lambda)
    inv_wavelength: f32,        // 1/lambda
    aperture_radius_mm: f32,    // Physical aperture radius in mm
    edge_smoothness: f32,       // Edge smoothing factor (0.01-0.1 typical)
    // Noise and coherence parameters
    noise_amount: f32,          // Phase noise amplitude (0-1)
    prng_seed: u32,             // Random seed for phase noise
    coherence_length: f32,      // Spatial coherence length in mm
    _padding: u32
}

struct FrequencyParams {
    fx_max: f32,                // Maximum spatial frequency in x (1/mm)
    fy_max: f32,                // Maximum spatial frequency in y (1/mm)
    dfx: f32,                   // Frequency spacing in x (1/mm)
    dfy: f32,                   // Frequency spacing in y (1/mm)
    fx_scale: f32,              // 2pi * dfx - precomputed
    fy_scale: f32,              // 2pi * dfy - precomputed
    bandwidth_limit: f32,       // Bandwidth limit factor (0.8 typical)
    raised_cosine_width: f32    // Width of raised cosine filter transition
}

struct NormalizationParams {
    num_wavelengths: u32,       // Number of wavelengths for multi-spectral
    spectral_weight_sum: f32,   // Sum of spectral weights for normalization
    fft_normalization: f32,     // FFT normalization factor (1/N or 1/sqrt(N))
    energy_conservation: u32    // Enable energy conservation normalization
}

// Uniform buffers - organized by update frequency
@group(0) @binding(0) var<uniform> prop_params: PropagationParams;
@group(0) @binding(1) var<uniform> freq_params: FrequencyParams;
@group(0) @binding(2) var<uniform> norm_params: NormalizationParams;

// Textures - using appropriate formats
@group(1) @binding(0) var input_field: texture_storage_2d<rg32float, read>;      // Complex field input
@group(1) @binding(1) var output_field: texture_storage_2d<rg32float, read_write>;    // Complex field output
@group(1) @binding(2) var frequency_domain: texture_storage_2d<rg32float, read_write>; // FFT workspace
@group(1) @binding(3) var transfer_function: texture_storage_2d<rg32float, read_write>; // Precomputed H(fx,fy)

// Optional debug textures
@group(2) @binding(0) var debug_magnitude: texture_storage_2d<r32float, write>;  // Magnitude visualization
@group(2) @binding(1) var debug_phase: texture_storage_2d<r32float, write>;      // Phase visualization

// Constants
const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const INV_PI: f32 = 0.31830988618;
const INV_TWO_PI: f32 = 0.15915494309;
const SQRT_TWO: f32 = 1.41421356237;
const WORKGROUP_SIZE: u32 = 16u;
const SHARED_SIZE: u32 = 256u; // 16x16 tile

// Shared memory for cooperative operations
var<workgroup> shared_field: array<vec4<f32>, SHARED_SIZE>;
var<workgroup> shared_transfer: array<vec4<f32>, SHARED_SIZE>;

// Hash function for PRNG (PCG variant)

fn clamp_index_dyn(i: u32, len: u32) -> u32 {
    return select(i, len - 1u, i >= len);
}

fn pcg_hash(seed: u32) -> u32 {
    var state = seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Generate uniform random float [0, 1)
fn random_float(seed: u32, coord: vec2<u32>) -> f32 {
    let hash = pcg_hash(seed ^ (coord.x * 1973u + coord.y * 9277u));
    return f32(hash) / 4294967296.0;
}

// Box-Muller transform for Gaussian noise
fn gaussian_noise(seed: u32, coord: vec2<u32>) -> vec2<f32> {
    let u1 = max(1e-6, random_float(seed, coord));
    let u2 = random_float(seed ^ 0x5A5A5A5Au, coord);
    let r = sqrt(-2.0 * log(u1));
    let theta = TWO_PI * u2;
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

// Accurate complex exponential with range reduction
fn complex_exp_accurate(phase: f32) -> vec2<f32> {
    // Range reduction to [-pi, pi]
    let reduced_phase = phase - round(phase * INV_TWO_PI) * TWO_PI;
    
    // Use Taylor series for small angles, built-in for large
    if (abs(reduced_phase) < 0.5) {
        let p2 = reduced_phase * reduced_phase;
        let p3 = p2 * reduced_phase;
        let p4 = p2 * p2;
        let p5 = p4 * reduced_phase;
        
        // Higher order Taylor series for better accuracy
        let cos_approx = 1.0 - p2 * 0.5 + p4 * 0.041666667 - p4 * p2 * 0.001388889;
        let sin_approx = reduced_phase - p3 * 0.166666667 + p5 * 0.008333333;
        
        return vec2<f32>(cos_approx, sin_approx);
    }
    
    return vec2<f32>(cos(reduced_phase), sin(reduced_phase));
}

// Complex operations with FMA
fn complex_multiply_fma(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        fma(a.x, b.x, -a.y * b.y),
        fma(a.x, b.y, a.y * b.x)
    );
}

// Get spatial frequency with proper physical units
fn get_spatial_frequency_physical(coord: vec2<i32>, size: vec2<u32>) -> vec2<f32> {
    // FFT-shifted coordinates to physical frequencies
    let half_size = vec2<f32>(size) * 0.5;
    let shifted_coord = vec2<f32>(coord) - half_size;
    
    // Return frequencies in cycles/mm
    return shifted_coord * vec2<f32>(freq_params.dfx, freq_params.dfy);
}

// Physical aperture function with proper dimensions
fn apply_physical_aperture(coord: vec2<f32>, dims: vec2<f32>) -> f32 {
    // Convert from pixels to physical coordinates
    let physical_pos = (coord - dims * 0.5) * prop_params.pixel_size;
    let r = length(physical_pos);
    
    // Use physical aperture radius
    let radius = prop_params.aperture_radius_mm;
    let edge_width = radius * prop_params.edge_smoothness;
    
    if (r < radius - edge_width) {
        return 1.0;
    } else if (r > radius + edge_width) {
        return 0.0;
    }
    
    // Super-Gaussian edge for smooth transition
    let t = (radius - r) / (2.0 * edge_width) + 0.5;
    let t_clamped = clamp(t, 0.0, 1.0);
    
    // Raised cosine edge
    return 0.5 * (1.0 + cos(PI * (1.0 - t_clamped)));
}

// Raised cosine filter for band limiting
fn raised_cosine_filter(freq: vec2<f32>) -> f32 {
    let freq_mag = length(freq);
    let cutoff = min(freq_params.fx_max, freq_params.fy_max) * freq_params.bandwidth_limit;
    let transition_width = cutoff * freq_params.raised_cosine_width;
    
    if (freq_mag < cutoff - transition_width) {
        return 1.0;
    } else if (freq_mag > cutoff + transition_width) {
        return 0.0;
    }
    
    // Raised cosine transition
    let t = (freq_mag - cutoff + transition_width) / (2.0 * transition_width);
    return 0.5 * (1.0 + cos(PI * t));
}

// Angular Spectrum transfer function with evanescent wave handling
fn angular_spectrum_transfer(fx: f32, fy: f32) -> vec2<f32> {
    let k = prop_params.k;
    let kx = TWO_PI * fx;  // Convert to rad/mm
    let ky = TWO_PI * fy;
    
    let k_squared = k * k;
    let kxy_squared = kx * kx + ky * ky;
    let kz_squared = k_squared - kxy_squared;
    
    if (kz_squared <= 0.0) {
        // Evanescent waves - exponential decay
        let kz_imag = sqrt(-kz_squared);
        let decay = exp(-kz_imag * prop_params.distance);
        
        // Cutoff for numerical stability
        return vec2<f32>(select(0.0, decay, decay > 1e-8), 0.0);
    }
    
    // Propagating waves
    let kz = sqrt(kz_squared);
    let phase = kz * prop_params.distance;
    return complex_exp_accurate(phase);
}

// Fresnel transfer function (paraxial approximation)
fn fresnel_transfer(fx: f32, fy: f32) -> vec2<f32> {
    // H(fx,fy) = exp(j * pi * lambda * z * (fx^2 + fy^2))
    let factor = PI * prop_params.wavelength * prop_params.distance;
    let freq_squared = fx * fx + fy * fy;
    let phase = factor * freq_squared;
    
    return complex_exp_accurate(phase);
}

// Fraunhofer transfer function (far-field)
fn fraunhofer_transfer(fx: f32, fy: f32) -> vec2<f32> {
    // Additional quadratic phase factor for far-field
    let factor = -PI * prop_params.distance * prop_params.inv_wavelength;
    let freq_squared = fx * fx + fy * fy;
    let phase = factor * freq_squared;
    
    // Include Fresnel integral normalization
    let fresnel_factor = complex_exp_accurate(PI * 0.25); // exp(jpi/4)
    return complex_multiply_fma(complex_exp_accurate(phase), fresnel_factor);
}

// Compute transfer function with method selection
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn compute_transfer_function(@builtin(global_invocation_id) global_id: vec3<u32>,
                           @builtin(local_invocation_id) local_id: vec3<u32>) {
    let coord = vec2<i32>(global_id.xy);
    let dims = textureDimensions(transfer_function);
    
    if (any(coord >= vec2<i32>(dims))) {
        return;
    }
    
    // Get physical spatial frequency
    let freq = get_spatial_frequency_physical(coord, dims);
    
    // Determine method (auto-select if method == 3)
    var selected_method = prop_params.method;
    if (selected_method == 3u) {
        // Auto-select based on Fresnel number
        if (prop_params.fresnel_number > 100.0) {
            selected_method = 0u; // Angular Spectrum for near-field
        } else if (prop_params.fresnel_number > 1.0) {
            selected_method = 1u; // Fresnel for medium distances
        } else {
            selected_method = 2u; // Fraunhofer for far-field
        }
    }
    
    // Compute transfer function based on selected method
    var transfer: vec2<f32>;
    switch (selected_method) {
        case 0u: {
            transfer = angular_spectrum_transfer(freq.x, freq.y);
        }
        case 1u: {
            transfer = fresnel_transfer(freq.x, freq.y);
        }
        case 2u: {
            transfer = fraunhofer_transfer(freq.x, freq.y);
        }
        default: {
            // Fallback to identity
            transfer = vec2<f32>(1.0, 0.0);
        }
    }
    
    // Apply band limiting with raised cosine filter
    if (prop_params.use_band_limiting != 0u) {
        let band_filter = raised_cosine_filter(freq);
        transfer *= band_filter;
    }
    
    // Add phase noise for partial coherence simulation
    if (prop_params.noise_amount > 0.0) {
        let noise = gaussian_noise(prop_params.prng_seed, global_id.xy);
        let phase_noise = noise.x * prop_params.noise_amount * PI;
        let noise_phasor = complex_exp_accurate(phase_noise);
        transfer = complex_multiply_fma(transfer, noise_phasor);
    }
    
    // Store in shared memory for potential reuse with bounds checking
    let local_idx = local_id.y * WORKGROUP_SIZE + local_id.x;
    if (local_idx < SHARED_SIZE) {
        shared_transfer[local_idx] = vec4<f32>(transfer, 0.0, 1.0);
    }
    
    // Write to global memory
    textureStore(transfer_function, coord, vec4<f32>(transfer, 0.0, 0.0));
}

// Main propagation kernel with proper normalization
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn propagate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = vec2<i32>(global_id.xy);
    let dims = textureDimensions(frequency_domain);
    
    if (any(coord >= vec2<i32>(dims))) {
        return;
    }
    
    // Read frequency domain field
    let field_freq = textureLoad(frequency_domain, vec2<u32>(coord)).xy;
    
    // Read precomputed transfer function
    let transfer = textureLoad(transfer_function, vec2<u32>(coord)).xy;
    
    // Apply transfer function
    var propagated = complex_multiply_fma(field_freq, transfer);
    
    // Apply normalization
    if (norm_params.energy_conservation != 0u) {
        // Energy conservation: scale by sqrt of Jacobian for coordinate transform
        let jacobian = prop_params.distance * prop_params.wavelength;
        propagated *= sqrt(jacobian);
    }
    
    // FFT normalization (forward transform uses 1/N, inverse uses 1)
    propagated *= norm_params.fft_normalization;
    
    // Amplitude scaling
    propagated *= prop_params.amplitude_scale;
    
    // Store result
    textureStore(frequency_domain, vec2<u32>(coord), vec4<f32>(propagated, 0.0, 0.0));
}

// Post-processing with aperture and coherence effects
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn post_process(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = vec2<i32>(global_id.xy);
    let dims = textureDimensions(output_field);
    
    if (any(coord >= vec2<i32>(dims))) {
        return;
    }
    
    // Read propagated field
    var field = textureLoad(output_field, vec2<u32>(coord)).xy;
    
    // Apply physical aperture if requested
    if (prop_params.apply_aperture != 0u) {
        let aperture = apply_physical_aperture(vec2<f32>(coord), vec2<f32>(dims));
        field *= aperture;
    }
    
    // Simulate partial coherence with correlation function
    if (prop_params.coherence_length > 0.0 && prop_params.coherence_length < 1000.0) {
        let correlation_distance = prop_params.coherence_length / prop_params.pixel_size;
        let coherence_kernel = exp(-length(vec2<f32>(coord) - vec2<f32>(dims) * 0.5) / correlation_distance);
        field *= coherence_kernel;
    }
    
    // Clamp for numerical stability
    let mag = length(field);
    if (mag > 100.0) {
        field = field * (100.0 / mag);
    }
    
    textureStore(output_field, vec2<u32>(coord), vec4<f32>(field, 0.0, 0.0));
}

// Multi-wavelength propagation with proper spectral weighting
@group(2) @binding(2) var<storage, read> wavelengths: array<f32>;
@group(2) @binding(3) var<storage, read> spectral_weights: array<f32>;
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn propagate_multi_wavelength(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = vec2<i32>(global_id.xy);
    let dims = textureDimensions(frequency_domain);
    
    if (any(coord >= vec2<i32>(dims))) {
        return;
    }
    
    let num_wavelengths = norm_params.num_wavelengths;
    var accumulated_field = vec2<f32>(0.0);
    
    // Get spatial frequency once
    let freq = get_spatial_frequency_physical(coord, dims);
    
    // Unroll loop for small wavelength counts
    if (num_wavelengths <= 4u) {
        // Unrolled version for RGB or RGBA
        for (var i = 0u; i < num_wavelengths; i++) {
            let wavelength = wavelengths[clamp_index_dyn(i, arrayLength(&wavelengths))];
            let weight = spectral_weights[clamp_index_dyn(i, arrayLength(&spectral_weights))];
            let k = TWO_PI / wavelength;
            
            // Read frequency domain field for this wavelength channel
            let field_freq = textureLoad(frequency_domain, vec2<u32>(coord)).xy;
            
            // Compute wavelength-specific transfer function
            var transfer: vec2<f32>;
            switch (prop_params.method) {
                case 0u: {
                    // Angular spectrum with wavelength-specific k
                    let kx = TWO_PI * freq.x;
                    let ky = TWO_PI * freq.y;
                    let kz_squared = k * k - kx * kx - ky * ky;
                    
                    if (kz_squared > 0.0) {
                        let kz = sqrt(kz_squared);
                        let phase = kz * prop_params.distance;
                        transfer = complex_exp_accurate(phase);
                    } else {
                        transfer = vec2<f32>(0.0);
                    }
                }
                case 1u: {
                    // Fresnel with wavelength
                    let phase = PI * wavelength * prop_params.distance * dot(freq, freq);
                    transfer = complex_exp_accurate(phase);
                }
                default: {
                    transfer = vec2<f32>(1.0, 0.0);
                }
            }
            
            // Accumulate weighted contribution
            accumulated_field += complex_multiply_fma(field_freq, transfer) * weight;
        }
    } else {
        // General loop for many wavelengths
        for (var i = 0u; i < num_wavelengths; i++) {
            let wavelength = wavelengths[clamp_index_dyn(i, arrayLength(&wavelengths))];
            let weight = spectral_weights[clamp_index_dyn(i, arrayLength(&spectral_weights))];
            let k = TWO_PI / wavelength;
            
            // Read frequency domain field for this wavelength channel
            let field_freq = textureLoad(frequency_domain, vec2<u32>(coord)).xy;
            
            // Compute wavelength-specific transfer function
            var transfer: vec2<f32>;
            switch (prop_params.method) {
                case 0u: {
                    // Angular spectrum with wavelength-specific k
                    let kx = TWO_PI * freq.x;
                    let ky = TWO_PI * freq.y;
                    let kz_squared = k * k - kx * kx - ky * ky;
                    
                    if (kz_squared > 0.0) {
                        let kz = sqrt(kz_squared);
                        let phase = kz * prop_params.distance;
                        transfer = complex_exp_accurate(phase);
                    } else {
                        transfer = vec2<f32>(0.0);
                    }
                }
                case 1u: {
                    // Fresnel with wavelength
                    let phase = PI * wavelength * prop_params.distance * dot(freq, freq);
                    transfer = complex_exp_accurate(phase);
                }
                default: {
                    transfer = vec2<f32>(1.0, 0.0);
                }
            }
            
            // Accumulate weighted contribution
            accumulated_field += complex_multiply_fma(field_freq, transfer) * weight;
        }
    }
    
    // Normalize by spectral weight sum
    accumulated_field /= norm_params.spectral_weight_sum;
    
    textureStore(frequency_domain, vec2<u32>(coord), vec4<f32>(accumulated_field, 0.0, 0.0));
}

// Debug visualization kernels
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn visualize_magnitude(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = vec2<i32>(global_id.xy);
    let dims = textureDimensions(output_field);
    
    if (any(coord >= vec2<i32>(dims))) {
        return;
    }
    
    let field = textureLoad(output_field, vec2<u32>(coord)).xy;
    let magnitude = length(field);
    
    // Log scale for better dynamic range
    let log_mag = log(1.0 + magnitude) / log(10.0);
    
    textureStore(debug_magnitude, vec2<u32>(coord), vec4<f32>(log_mag, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn visualize_phase(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = vec2<i32>(global_id.xy);
    let dims = textureDimensions(output_field);
    
    if (any(coord >= vec2<i32>(dims))) {
        return;
    }
    
    let field = textureLoad(output_field, vec2<u32>(coord)).xy;
    let phase = atan2(field.y, field.x);
    
    // Normalize to [0, 1] for visualization
    let normalized_phase = (phase + PI) * INV_TWO_PI;
    
    textureStore(debug_phase, vec2<u32>(coord), vec4<f32>(normalized_phase, 0.0, 0.0, 0.0));
}

// Integration point for multi-view synthesis
// This kernel prepares the propagated field for view generation
@group(2) @binding(4) var multiview_buffer: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn prepare_for_multiview(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = vec2<i32>(global_id.xy);
    let dims = textureDimensions(output_field);
    
    if (any(coord >= vec2<i32>(dims))) {
        return;
    }
    
    // Read propagated field
    let field = textureLoad(output_field, vec2<u32>(coord)).xy;
    
    // Apply any final transformations needed for multi-view
    // This could include coordinate remapping, additional phase factors, etc.
    var transformed_field = field;
    
    // Example: Add linear phase ramp for off-axis viewing
    // const view_angle = 0.1; // radians
    // let kx = prop_params.k * sin(view_angle);
    // let x_pos = f32(coord.x) * prop_params.pixel_size;
    // let phase_ramp = kx * x_pos;
    // transformed_field = complex_multiply_fma(field, complex_exp_accurate(phase_ramp));
    
    // Write to multi-view buffer
    textureStore(multiview_buffer, vec2<u32>(coord), vec4<f32>(transformed_field, 0.0, 0.0));
}
