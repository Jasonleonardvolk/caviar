// ${IRIS_ROOT}\frontend\shaders\wavefieldEncoder.wgsl
// Enhanced wavefield encoder with unified array sizes and optimizations

// Unified constant for oscillator count
const MAX_OSCILLATORS: u32 = 32u;
const WORKGROUP_SIZE: u32 = 8u;
const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

// Override for runtime configuration
override HOLOGRAM_SIZE: u32 = 1024u;

struct WavefieldParams {
    phase_modulation: f32,
    coherence: f32,
    time: f32,
    scale: f32,
    phases: array<f32, MAX_OSCILLATORS>,
    spatial_freqs: array<vec2<f32>, MAX_OSCILLATORS>,
    amplitudes: array<f32, MAX_OSCILLATORS>  // Pre-computed on CPU
}

struct OscillatorData {
    psi_phase: f32,
    phase_coherence: f32,
    coupling_strength: f32,
    dominant_frequency: f32,
    phases: array<f32, MAX_OSCILLATORS>,
    spatial_freqs: array<vec2<f32>, MAX_OSCILLATORS>,
    amplitudes: array<f32, MAX_OSCILLATORS>  // Pre-computed amplitudes
}

struct PropagationParams {
    wavelength: f32,
    z_offset: f32,
    amplitude_scale: f32,
    phase_noise: f32
}

struct QualitySettings {
    resolution_scale: f32,
    sample_count: u32,
    enable_chromatic: u32,
    enable_volumetric: u32
}

@group(0) @binding(0) var<storage, read> wavefield_params: WavefieldParams;
@group(0) @binding(1) var depth_tex: texture_storage_2d<r32float, read>;
@group(0) @binding(2) var wavefield_out: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var noise_tex: texture_2d<f32>;  // Pre-computed noise texture

@group(1) @binding(0) var<storage, read> osc_data: OscillatorData;
@group(1) @binding(1) var<uniform> prop_params: PropagationParams;
@group(1) @binding(2) var<uniform> quality: QualitySettings;
@group(1) @binding(3) var color_tex: texture_2d<f32>;
@group(1) @binding(4) var color_sampler: sampler;

// Shared memory for oscillator data (loaded once per workgroup)
var<workgroup> shared_spatial_freqs: array<vec2<f32>, MAX_OSCILLATORS>;
var<workgroup> shared_phases: array<f32, MAX_OSCILLATORS>;
var<workgroup> shared_amplitudes: array<f32, MAX_OSCILLATORS>;

// Complex number operations

fn clamp_index_dyn(i: u32, len: u32) -> u32 {
    return select(i, len - 1u, i >= len);
}

fn complex_multiply(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

fn complex_exp(phase: f32) -> vec2<f32> {
    return vec2<f32>(cos(phase), sin(phase));
}

// Hash function for position-based effects
fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.13);
    p3 += dot(p3, p3.yzx + 3.333);
    return fract((p3.x + p3.y) * p3.z);
}

// Convert depth to phase delay
fn depth_to_phase(depth: f32, wavelength: f32, dispersion: f32) -> f32 {
    let k = TWO_PI / wavelength;
    let actual_depth = depth * 500.0;  // Map 0-1 to 0-500mm
    let dispersed_k = k * (1.0 + dispersion * 0.1);
    return dispersed_k * actual_depth;
}

// Load shared data once per workgroup
fn load_shared_data(local_id: vec3<u32>) {
    let thread_idx = local_id.y * WORKGROUP_SIZE + local_id.x;
    let stride = WORKGROUP_SIZE * WORKGROUP_SIZE;
    
    // Each thread loads a portion of the data
    for (var i = thread_idx; i < MAX_OSCILLATORS; i += stride) {
        let clamped_i = clamp_index_dyn(i, MAX_OSCILLATORS);
        shared_spatial_freqs[clamped_i] = wavefield_params.spatial_freqs[clamped_i];
        shared_phases[clamped_i] = wavefield_params.phases[clamped_i];
        shared_amplitudes[clamped_i] = wavefield_params.amplitudes[clamped_i];
    }
    
    workgroupBarrier();
}

// Optimized interference computation using shared memory
fn compute_interference_fast(pos: vec2<f32>, time: f32) -> vec2<f32> {
    var field = vec2<f32>(0.0, 0.0);
    
    // Unrolled loop for better performance (process 4 at a time)
    for (var i = 0u; i < MAX_OSCILLATORS; i += 4u) {
        // Make sure we don't go out of bounds
        let idx0 = clamp_index_dyn(i, MAX_OSCILLATORS);
        let idx1 = clamp_index_dyn(i + 1u, MAX_OSCILLATORS);
        let idx2 = clamp_index_dyn(i + 2u, MAX_OSCILLATORS);
        let idx3 = clamp_index_dyn(i + 3u, MAX_OSCILLATORS);
        
        // Process 4 oscillators in parallel
        let freq0 = shared_spatial_freqs[idx0];
        let freq1 = shared_spatial_freqs[idx1];
        let freq2 = shared_spatial_freqs[idx2];
        let freq3 = shared_spatial_freqs[idx3];
        
        let phase0 = shared_phases[idx0];
        let phase1 = shared_phases[idx1];
        let phase2 = shared_phases[idx2];
        let phase3 = shared_phases[idx3];
        
        let amp0 = shared_amplitudes[idx0];
        let amp1 = shared_amplitudes[idx1];
        let amp2 = shared_amplitudes[idx2];
        let amp3 = shared_amplitudes[idx3];
        
        // Compute all dot products
        let k_dot_r0 = dot(freq0, pos);
        let k_dot_r1 = dot(freq1, pos);
        let k_dot_r2 = dot(freq2, pos);
        let k_dot_r3 = dot(freq3, pos);
        
        // Add time evolution
        let t_scale = time * 0.1;
        
        // Accumulate contributions (branchless)
        field += amp0 * complex_exp(k_dot_r0 + phase0 + t_scale * length(freq0));
        field += amp1 * complex_exp(k_dot_r1 + phase1 + t_scale * length(freq1));
        field += amp2 * complex_exp(k_dot_r2 + phase2 + t_scale * length(freq2));
        field += amp3 * complex_exp(k_dot_r3 + phase3 + t_scale * length(freq3));
    }
    
    return field;
}

// Simplified coherence modulation using pre-computed noise
fn apply_coherence_modulation_fast(phase: f32, coherence: f32, noise_value: f32) -> f32 {
    let noise_strength = (1.0 - coherence) * prop_params.phase_noise;
    let phase_noise = (noise_value - 0.5) * TWO_PI * noise_strength;
    let sharpness = mix(0.7, 1.3, coherence);
    return phase * sharpness + phase_noise;
}

// Super-Gaussian aperture
fn super_gaussian_aperture(pos: vec2<f32>, sigma: f32, order: f32) -> f32 {
    let center = vec2<f32>(0.5, 0.5);
    let r = length(pos - center) / sigma;
    return exp(-pow(r, order));
}

// Optimized chromatic dispersion
fn apply_chromatic_dispersion_fast(field: vec2<f32>, channel: u32) -> vec2<f32> {
    if (quality.enable_chromatic == 0u) {
        return field;
    }
    
    // Pre-computed dispersion factors
    const dispersion_factors = array<f32, 3>(
        0.9,   // Red: 700nm
        1.0,   // Green: 550nm  
        1.1    // Blue: 450nm
    );
    
    let dispersion = dispersion_factors[clamp_index_dyn(channel % 3u, 3u)];
    let phase_idx = clamp_index_dyn((channel + 4u) * 2u, MAX_OSCILLATORS);
    let phase_shift = osc_data.phases[phase_idx] * 0.05 * dispersion;
    
    return complex_multiply(field, complex_exp(phase_shift));
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    // Load shared data once per workgroup
    load_shared_data(local_id);
    
    let coord = global_id.xy;
    let dims = vec2<u32>(HOLOGRAM_SIZE, HOLOGRAM_SIZE);
    
    if (coord.x >= dims.x || coord.y >= dims.y) {
        return;
    }
    
    let uv = vec2<f32>(coord) / vec2<f32>(dims);
    
    // Sample inputs
    let depth = textureLoad(depth_tex, coord).r;
    let noise_value = textureLoad(noise_tex, coord, 0).r;
    let color = textureSampleLevel(color_tex, color_sampler, uv, 0.0);
    let luminance = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    
    // === Core computation ===
    
    // 1. Base amplitude
    let base_amplitude = prop_params.amplitude_scale * 
                        mix(luminance, 1.0, 0.5) * 
                        mix(0.6, 1.0, wavefield_params.coherence);
    
    // 2. Phase from depth
    var phase = depth_to_phase(depth, prop_params.wavelength, 
                              f32(coord.x % 3u) - 1.0);
    phase += wavefield_params.phase_modulation;
    
    // 3. Fast interference computation
    let interference = compute_interference_fast(uv, wavefield_params.time);
    
    // 4. Apply modulations
    let interference_strength = osc_data.coupling_strength * 0.5;
    phase += atan2(interference.y, interference.x) * interference_strength;
    phase = apply_coherence_modulation_fast(phase, wavefield_params.coherence, noise_value);
    
    // 5. Aperture
    let aperture = super_gaussian_aperture(uv, 0.45, 4.0);
    let amplitude = base_amplitude * aperture * wavefield_params.scale;
    
    // 6. Generate field
    phase = phase - floor(phase / TWO_PI) * TWO_PI;
    var field = amplitude * complex_exp(phase) + interference * interference_strength;
    
    // 7. Reference beam
    let ref_angle = PI / 6.0;
    let ref_phase = TWO_PI * (uv.x * sin(ref_angle) + uv.y * cos(ref_angle)) / prop_params.wavelength;
    field += 0.5 * complex_exp(ref_phase);
    
    // 8. Time effects
    let pulse = 1.0 + 0.1 * sin(osc_data.psi_phase * 4.0 + wavefield_params.time) * 
                osc_data.phase_coherence;
    field *= pulse;
    
    // 9. Chromatic dispersion
    field = apply_chromatic_dispersion_fast(field, coord.x);
    
    // Store result
    textureStore(wavefield_out, coord, vec4<f32>(field, 0.0, 0.0));
}

// Simplified test pattern generator
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn generate_test_pattern(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = global_id.xy;
    let dims = vec2<u32>(HOLOGRAM_SIZE, HOLOGRAM_SIZE);
    
    if (coord.x >= dims.x || coord.y >= dims.y) {
        return;
    }
    
    let uv = vec2<f32>(coord) / vec2<f32>(dims);
    let center = vec2<f32>(0.5, 0.5);
    
    // Animated point
    let time = wavefield_params.time;
    let point_pos = center + 0.2 * vec2<f32>(
        sin(time + osc_data.psi_phase),
        cos(time * 0.7)
    );
    
    let r = distance(uv, point_pos);
    let k = TWO_PI / prop_params.wavelength;
    let phase = k * r * 1000.0 + osc_data.psi_phase * 2.0;
    let amplitude = exp(-r * 5.0) * wavefield_params.coherence;
    
    let field = amplitude * complex_exp(phase);
    let ref_phase = TWO_PI * uv.x * 5.0;
    let reference = 0.3 * complex_exp(ref_phase);
    
    textureStore(wavefield_out, coord, vec4<f32>(field + reference, 0.0, 0.0));
}
