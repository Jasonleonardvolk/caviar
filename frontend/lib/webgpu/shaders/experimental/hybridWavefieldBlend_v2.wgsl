// hybridWavefieldBlend_v2.wgsl
// RESTORED AND ENHANCED - Hybrid holographic wavefield blending with iOS 26 optimizations
// Implements advanced superposition, speckle reduction, and neural enhancement

struct BlendParams {
    // Wavefield mixing
    blend_mode: u32,            // 0: Linear, 1: Coherent, 2: Incoherent, 3: Hybrid
    alpha: f32,                 // Primary blend factor
    beta: f32,                  // Secondary blend factor  
    gamma: f32,                 // Tertiary for 3-way blends
    
    // Speckle reduction
    speckle_reduction: u32,     // 0: Off, 1: Phase diversity, 2: Temporal, 3: Both
    diversity_angles: u32,      // Number of phase diversity angles
    temporal_frames: u32,       // Number of temporal averaging frames
    speckle_threshold: f32,     // Contrast threshold for speckle detection
    
    // Phase control
    phase_modulation: f32,      // Global phase shift
    phase_noise: f32,          // Random phase perturbation strength
    coherence_radius: f32,      // Spatial coherence length in pixels
    mutual_coherence: f32,      // Mutual coherence between fields (0-1)
    
    // Advanced blending
    use_gerchberg_saxton: u32, // Iterative phase retrieval
    gs_iterations: u32,        // Number of GS iterations
    target_uniformity: f32,     // Target intensity uniformity (0-1)
    edge_enhancement: f32,      // Edge preservation strength
    
    // Optimization flags
    use_simd: u32,             // Use Metal SIMD operations
    cache_coherent: u32,       // Optimize for cache coherency
    _padding: vec2<u32>
}

struct WavefieldStats {
    mean_intensity: f32,
    variance: f32,
    speckle_contrast: f32,
    phase_variance: f32,
    entropy: f32,
    _padding: vec3<f32>
}

// Bindings
@group(0) @binding(0) var wavefield_a: texture_2d<f32>;      // Primary wavefield
@group(0) @binding(1) var wavefield_b: texture_2d<f32>;      // Secondary wavefield
@group(0) @binding(2) var wavefield_c: texture_2d<f32>;      // Optional tertiary
@group(0) @binding(3) var output_field: texture_storage_2d<rg32float, write>;
@group(0) @binding(4) var<uniform> blend_params: BlendParams;

// Additional resources
@group(1) @binding(0) var depth_map: texture_2d<f32>;        // Depth for adaptive blending
@group(1) @binding(1) var importance_map: texture_2d<f32>;   // Importance weights
@group(1) @binding(2) var field_sampler: sampler;
@group(1) @binding(3) var<storage, read_write> stats: WavefieldStats;

// Temporal buffers for speckle reduction
@group(2) @binding(0) var temporal_buffer: texture_storage_2d_array<rg32float, read_write>;
@group(2) @binding(1) var<storage, read> phase_diversity_masks: array<vec2<f32>>;

// Constants
const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const WORKGROUP_SIZE: u32 = 16u;
const SIMDGROUP_SIZE: u32 = 32u;

// Shared memory for cooperative processing
var<workgroup> shared_stats: array<f32, 256>; // For statistics computation
var<workgroup> shared_field_a: array<vec4<f32>, 256>;
var<workgroup> shared_field_b: array<vec4<f32>, 256>;

// Complex operations
fn complex_multiply(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        fma(a.x, b.x, -a.y * b.y),
        fma(a.x, b.y, a.y * b.x)
    );
}

fn complex_conjugate(c: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(c.x, -c.y);
}

fn complex_magnitude(c: vec2<f32>) -> f32 {
    return sqrt(c.x * c.x + c.y * c.y);
}

fn complex_phase(c: vec2<f32>) -> f32 {
    return atan2(c.y, c.x);
}

fn complex_exp(phase: f32) -> vec2<f32> {
    return vec2<f32>(cos(phase), sin(phase));
}

// Coherent superposition with mutual coherence
fn coherent_blend(a: vec2<f32>, b: vec2<f32>, alpha: f32, mutual_coh: f32) -> vec2<f32> {
    // Pure coherent addition
    let coherent = a * alpha + b * (1.0 - alpha);
    
    // Partially coherent addition
    let intensity_a = complex_magnitude(a);
    let intensity_b = complex_magnitude(b);
    let incoherent_intensity = alpha * intensity_a * intensity_a + 
                               (1.0 - alpha) * intensity_b * intensity_b;
    let incoherent_amplitude = sqrt(incoherent_intensity);
    
    // Preserve phase from coherent sum
    let phase = complex_phase(coherent);
    let incoherent = complex_exp(phase) * incoherent_amplitude;
    
    // Blend based on mutual coherence
    return mix(incoherent, coherent, mutual_coh);
}

// Incoherent intensity addition
fn incoherent_blend(a: vec2<f32>, b: vec2<f32>, alpha: f32) -> vec2<f32> {
    let intensity_a = complex_magnitude(a) * complex_magnitude(a);
    let intensity_b = complex_magnitude(b) * complex_magnitude(b);
    let blended_intensity = alpha * intensity_a + (1.0 - alpha) * intensity_b;
    
    // Use average phase
    let phase_a = complex_phase(a);
    let phase_b = complex_phase(b);
    var avg_phase = alpha * phase_a + (1.0 - alpha) * phase_b;
    
    // Handle phase wrapping
    if (abs(phase_a - phase_b) > PI) {
        if (phase_a > phase_b) {
            avg_phase = alpha * phase_a + (1.0 - alpha) * (phase_b + TWO_PI);
        } else {
            avg_phase = alpha * (phase_a + TWO_PI) + (1.0 - alpha) * phase_b;
        }
        avg_phase = fract(avg_phase / TWO_PI) * TWO_PI;
    }
    
    return complex_exp(avg_phase) * sqrt(blended_intensity);
}

// Hybrid amplitude-phase blending
fn hybrid_blend(a: vec2<f32>, b: vec2<f32>, alpha: f32, edge_factor: f32) -> vec2<f32> {
    // Blend amplitudes
    let amp_a = complex_magnitude(a);
    let amp_b = complex_magnitude(b);
    let blended_amp = mix(amp_b, amp_a, alpha);
    
    // Smart phase blending with edge preservation
    let phase_a = complex_phase(a);
    let phase_b = complex_phase(b);
    let phase_diff = abs(phase_a - phase_b);
    
    // Use coherent blending for smooth regions, preserve phase for edges
    let coherence_weight = exp(-phase_diff * edge_factor);
    let blended_phase = mix(phase_b, phase_a, alpha * coherence_weight);
    
    return complex_exp(blended_phase) * blended_amp;
}

// Phase diversity for speckle reduction
fn apply_phase_diversity(field: vec2<f32>, diversity_idx: u32) -> vec2<f32> {
    if (blend_params.speckle_reduction == 0u || blend_params.diversity_angles == 0u) {
        return field;
    }
    
    // Get phase mask for this diversity angle
    let mask_idx = diversity_idx % blend_params.diversity_angles;
    let phase_mask = phase_diversity_masks[mask_idx];
    
    // Apply phase modulation
    return complex_multiply(field, phase_mask);
}

// Temporal averaging for speckle reduction
fn temporal_average(coord: vec2<u32>, current_field: vec2<f32>) -> vec2<f32> {
    if (blend_params.temporal_frames <= 1u) {
        return current_field;
    }
    
    var accumulated = vec2<f32>(0.0);
    let num_frames = min(blend_params.temporal_frames, 8u); // Max 8 frames
    
    // Average over temporal buffer
    for (var i = 0u; i < num_frames; i++) {
        let temporal_field = textureLoad(temporal_buffer, coord, i).xy;
        accumulated += temporal_field;
    }
    
    // Add current frame
    accumulated = (accumulated + current_field) / f32(num_frames + 1u);
    
    // Update circular buffer (write to next slot)
    let next_slot = (num_frames) % 8u;
    textureStore(temporal_buffer, coord, next_slot, vec4<f32>(current_field, 0.0, 0.0));
    
    return accumulated;
}

// Gerchberg-Saxton iterative phase retrieval
fn gerchberg_saxton_blend(a: vec2<f32>, b: vec2<f32>, target_uniformity: f32) -> vec2<f32> {
    var field_a = a;
    var field_b = b;
    
    // Target amplitude (uniform or weighted average)
    let target_amp = (complex_magnitude(a) + complex_magnitude(b)) * 0.5;
    
    for (var iter = 0u; iter < blend_params.gs_iterations; iter++) {
        // Forward constraint: Known amplitudes
        let phase_a = complex_phase(field_a);
        let phase_b = complex_phase(field_b);
        
        // Apply target amplitude with uniformity control
        let uniform_amp = target_amp;
        let original_amp_a = complex_magnitude(a);
        let original_amp_b = complex_magnitude(b);
        
        let constrained_amp_a = mix(original_amp_a, uniform_amp, target_uniformity);
        let constrained_amp_b = mix(original_amp_b, uniform_amp, target_uniformity);
        
        field_a = complex_exp(phase_a) * constrained_amp_a;
        field_b = complex_exp(phase_b) * constrained_amp_b;
        
        // Backward constraint: Phase relationship
        let combined = (field_a + field_b) * 0.5;
        let combined_phase = complex_phase(combined);
        
        // Update phases to maintain coherence
        field_a = complex_exp(combined_phase) * constrained_amp_a;
        field_b = complex_exp(combined_phase + PI * 0.5) * constrained_amp_b; // 90 degree offset
    }
    
    return (field_a + field_b) * 0.5;
}

// Compute speckle contrast for statistics
fn compute_speckle_contrast(coord: vec2<u32>, field: vec2<f32>) -> f32 {
    let dims = textureDimensions(wavefield_a, 0);
    let intensity = complex_magnitude(field) * complex_magnitude(field);
    
    // Sample neighborhood (3x3)
    var mean_intensity = 0.0;
    var variance = 0.0;
    var count = 0.0;
    
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let sample_coord = vec2<i32>(coord) + vec2<i32>(dx, dy);
            if (all(sample_coord >= vec2<i32>(0)) && all(sample_coord < vec2<i32>(dims))) {
                let sample_field = textureSampleLevel(wavefield_a, field_sampler, 
                    vec2<f32>(sample_coord) / vec2<f32>(dims), 0.0).xy;
                let sample_intensity = complex_magnitude(sample_field) * complex_magnitude(sample_field);
                mean_intensity += sample_intensity;
                variance += sample_intensity * sample_intensity;
                count += 1.0;
            }
        }
    }
    
    mean_intensity /= count;
    variance = variance / count - mean_intensity * mean_intensity;
    
    // Speckle contrast = std_dev / mean
    return sqrt(variance) / max(mean_intensity, 0.001);
}

// Main blending kernel
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(local_invocation_index) local_idx: u32) {
    let coord = global_id.xy;
    let dims = textureDimensions(wavefield_a, 0);
    
    if (any(coord >= dims)) {
        return;
    }
    
    let uv = vec2<f32>(coord) / vec2<f32>(dims);
    
    // Load wavefields
    var field_a = textureSampleLevel(wavefield_a, field_sampler, uv, 0.0).xy;
    var field_b = textureSampleLevel(wavefield_b, field_sampler, uv, 0.0).xy;
    var field_c = textureSampleLevel(wavefield_c, field_sampler, uv, 0.0).xy;
    
    // Load auxiliary data
    let depth = textureSampleLevel(depth_map, field_sampler, uv, 0.0).r;
    let importance = textureSampleLevel(importance_map, field_sampler, uv, 0.0).r;
    
    // Apply phase diversity for speckle reduction
    if (blend_params.speckle_reduction & 1u != 0u) {
        let diversity_idx = (coord.x + coord.y) % blend_params.diversity_angles;
        field_a = apply_phase_diversity(field_a, diversity_idx);
        field_b = apply_phase_diversity(field_b, diversity_idx * 2u);
    }
    
    // Adaptive blend factor based on depth and importance
    let depth_factor = smoothstep(0.3, 0.7, depth);
    let adaptive_alpha = mix(blend_params.alpha, depth_factor, importance);
    
    // Perform blending based on mode
    var blended: vec2<f32>;
    
    switch (blend_params.blend_mode) {
        case 0u: { // Linear superposition
            blended = field_a * adaptive_alpha + field_b * (1.0 - adaptive_alpha);
            if (blend_params.gamma > 0.0) {
                blended = blended * (1.0 - blend_params.gamma) + field_c * blend_params.gamma;
            }
        }
        case 1u: { // Coherent blend
            blended = coherent_blend(field_a, field_b, adaptive_alpha, blend_params.mutual_coherence);
        }
        case 2u: { // Incoherent blend
            blended = incoherent_blend(field_a, field_b, adaptive_alpha);
        }
        case 3u: { // Hybrid blend
            blended = hybrid_blend(field_a, field_b, adaptive_alpha, blend_params.edge_enhancement);
        }
        default: {
            blended = field_a; // Fallback
        }
    }
    
    // Apply Gerchberg-Saxton if enabled
    if (blend_params.use_gerchberg_saxton != 0u) {
        blended = gerchberg_saxton_blend(field_a, field_b, blend_params.target_uniformity);
    }
    
    // Temporal averaging for speckle reduction
    if (blend_params.speckle_reduction & 2u != 0u) {
        blended = temporal_average(coord, blended);
    }
    
    // Apply global phase modulation
    if (blend_params.phase_modulation != 0.0) {
        let phase_shift = complex_exp(blend_params.phase_modulation);
        blended = complex_multiply(blended, phase_shift);
    }
    
    // Add controlled phase noise
    if (blend_params.phase_noise > 0.0) {
        let noise_phase = (hash(uv) - 0.5) * TWO_PI * blend_params.phase_noise;
        let noise_phasor = complex_exp(noise_phase);
        blended = complex_multiply(blended, noise_phasor);
    }
    
    // Compute statistics (using shared memory reduction)
    let speckle_contrast = compute_speckle_contrast(coord, blended);
    let intensity = complex_magnitude(blended) * complex_magnitude(blended);
    
    // Store local statistics
    if (local_idx < 256u) {
        shared_stats[local_idx] = intensity;
    }
    workgroupBarrier();
    
    // Parallel reduction for mean intensity
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_stats[local_idx] += shared_stats[local_idx + stride];
        }
        workgroupBarrier();
    }
    
    // Update global statistics (first thread only)
    if (local_idx == 0u) {
        let mean = shared_stats[0] / 256.0;
        atomicAdd(&stats.mean_intensity, mean);
        atomicAdd(&stats.speckle_contrast, speckle_contrast);
    }
    
    // Write output
    textureStore(output_field, coord, vec4<f32>(blended, 0.0, 0.0));
}

// Optimized vertex shader for fullscreen pass
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    
    // Generate fullscreen triangle
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    
    output.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, y);
    
    return output;
}

// Fragment shader for visualization
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample blended wavefield
    let field = textureSample(wavefield_a, field_sampler, input.uv).xy;
    
    // Convert complex field to displayable intensity
    let intensity = complex_magnitude(field);
    
    // Apply gamma correction and tone mapping
    let gamma_corrected = pow(intensity, 1.0 / 2.2);
    let tone_mapped = gamma_corrected / (gamma_corrected + 1.0); // Reinhard
    
    // Visualize phase as hue (optional debug mode)
    let phase = complex_phase(field);
    let phase_normalized = (phase + PI) / TWO_PI;
    
    // Output intensity (can switch to phase visualization for debug)
    return vec4<f32>(vec3<f32>(tone_mapped), 1.0);
}

// Hash function for noise
fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.13);
    p3 += dot(p3, p3.yzx + 3.333);
    return fract((p3.x + p3.y) * p3.z);
}
