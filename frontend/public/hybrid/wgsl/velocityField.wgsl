// ${IRIS_ROOT}\frontend\shaders\velocityField.wgsl
// Enhanced compute shader for phase velocity field visualization with optimizations

struct WavefieldParams {
    phase_modulation: f32,
    coherence: f32,
    time: f32,
    scale: f32,
    phases: array<vec4<f32>, 4>,  // 16 floats packed as 4 vec4s
    spatial_freqs: array<vec4<f32>, 16>  // 32 vec2s packed as 16 vec4s
}

struct VelocityFieldParams {
    scale_factor: f32,
    time_step: f32,
    viscosity: f32,
    vorticity_strength: f32,
    damping_factor: f32,
    max_speed: f32,
    coherence_blend: f32,
    vortex_falloff: f32
}

@group(0) @binding(0) var<uniform> wavefield_params: WavefieldParams;
@group(0) @binding(1) var<uniform> velocity_params: VelocityFieldParams;
@group(0) @binding(2) var wavefield_texture: texture_2d<f32>;
@group(0) @binding(3) var velocity_out: texture_storage_2d<rg32float, read_write>;
@group(0) @binding(4) var sampler_linear: sampler;
@group(0) @binding(5) var<storage, read_write> particles: array<vec4<f32>>;
@group(0) @binding(6) var flow_vis_out: texture_storage_2d<rgba8unorm, write>;

// Pre-computed constants
const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const INV_PI: f32 = 0.31830988618;
const INV_TWO_PI: f32 = 0.15915494309;
const HALF_PI: f32 = 1.57079632679;
const SHARED_SIZE: u32 = 100u; // 10x10 tile

// Workgroup shared memory for caching texture reads
var<workgroup> shared_wavefield: array<vec4<f32>, SHARED_SIZE>;

// Fast phase unwrapping using bit manipulation

fn clamp_index_dyn(i: u32, len: u32) -> u32 {
    return select(i, len - 1u, i >= len);
}

fn fast_unwrap_phase(phase_diff: f32) -> f32 {
    // Optimized phase unwrapping without branches
    let wrapped = phase_diff + PI;
    let cycles = floor(wrapped * INV_TWO_PI);
    return phase_diff - cycles * TWO_PI;
}

// Optimized atan2 approximation for performance
fn fast_atan2(y: f32, x: f32) -> f32 {
    let abs_y = abs(y);
    let abs_x = abs(x);
    let a = min(abs_x, abs_y) / max(abs_x, abs_y);
    let s = a * a;
    
    // Polynomial approximation
    var r = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;
    
    if (abs_y > abs_x) {
        r = HALF_PI - r;
    }
    
    if (x < 0.0) {
        r = PI - r;
    }
    
    return select(-r, r, y >= 0.0);
}

// Load wavefield data into shared memory
fn load_shared_memory(local_id: vec2<u32>, workgroup_id: vec2<u32>, dims: vec2<u32>) {
    let tile_start = workgroup_id.xy * 8u;
    let shared_idx = local_id.y * 10u + local_id.x;
    
    // Each thread loads multiple values to fill the 10x10 tile
    if (local_id.x < 10u && local_id.y < 10u && shared_idx < SHARED_SIZE) {
        let global_coord = vec2<i32>(tile_start) + vec2<i32>(local_id) - vec2<i32>(1);
        
        // Clamp to texture boundaries
        let clamped_coord = clamp(global_coord, vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
        shared_wavefield[shared_idx] = textureLoad(wavefield_texture, clamped_coord, 0);
    }
    
    workgroupBarrier();
}

// Compute phase gradient using shared memory
fn compute_phase_gradient_shared(local_id: vec2<u32>) -> vec2<f32> {
    let idx = (local_id.y + 1u) * 10u + (local_id.x + 1u);
    
    // Bounds check for shared memory
    if (idx >= SHARED_SIZE || idx < 1u || idx >= SHARED_SIZE - 1u) {
        return vec2<f32>(0.0);
    }
    
    // Check bounds for all needed indices
    let idx_left = idx - 1u;
    let idx_right = idx + 1u;
    let idx_down = idx - 10u;
    let idx_up = idx + 10u;
    
    if (idx_up >= SHARED_SIZE) {
        return vec2<f32>(0.0);
    }
    
    // Get values from shared memory with bounds checking
    let center = shared_wavefield[clamp_index_dyn(idx, SHARED_SIZE)].xy;
    let left = shared_wavefield[clamp_index_dyn(idx_left, SHARED_SIZE)].xy;
    let right = shared_wavefield[clamp_index_dyn(idx_right, SHARED_SIZE)].xy;
    let down = shared_wavefield[clamp_index_dyn(idx_down, SHARED_SIZE)].xy;
    let up = shared_wavefield[clamp_index_dyn(idx_up, SHARED_SIZE)].xy;
    
    // Use fast atan2 for phase computation
    let phase_center = fast_atan2(center.y, center.x);
    let phase_left = fast_atan2(left.y, left.x);
    let phase_right = fast_atan2(right.y, right.x);
    let phase_down = fast_atan2(down.y, down.x);
    let phase_up = fast_atan2(up.y, up.x);
    
    // Optimized phase unwrapping
    let dx = fast_unwrap_phase(phase_right - phase_left);
    let dy = fast_unwrap_phase(phase_up - phase_down);
    
    return vec2<f32>(dx, dy) * 0.5;
}

// Vectorized theoretical velocity computation
fn compute_theoretical_velocity_optimized(pos: vec2<f32>) -> vec2<f32> {
    var velocity = vec2<f32>(0.0);
    var total_weight = 0.0;
    
    // Process spatial frequencies in groups of 4 for better vectorization
    for (var i = 0u; i < 32u; i += 4u) {
        // Load 4 spatial frequencies at once from packed vec4 array
        // Each vec4 contains 2 vec2s (xy=first vec2, zw=second vec2)
        let packed_idx0 = clamp_index_dyn(i / 2u, 16u);
        let packed_idx1 = clamp_index_dyn((i + 2u) / 2u, 16u);
        let packed0 = wavefield_params.spatial_freqs[packed_idx0];
        let packed1 = wavefield_params.spatial_freqs[packed_idx1];
        
        let k0 = vec2<f32>(packed0.xy);
        let k1 = vec2<f32>(packed0.zw);
        let k2 = vec2<f32>(packed1.xy);
        let k3 = vec2<f32>(packed1.zw);
        
        // Compute magnitudes
        let k_mag0 = length(k0);
        let k_mag1 = length(k1);
        let k_mag2 = length(k2);
        let k_mag3 = length(k3);
        
        // Early termination check
        if (k_mag0 < 0.001 && k_mag1 < 0.001 && k_mag2 < 0.001 && k_mag3 < 0.001) {
            continue;
        }
        
        // Vectorized computation
        let scale = velocity_params.scale_factor;
        let coherence = wavefield_params.coherence;
        
        // Process each valid frequency
        if (k_mag0 >= 0.001) {
            let omega0 = k_mag0 * scale;
            let phase_vel0 = omega0 / (k_mag0 * k_mag0);
            let weight0 = exp(-f32(i) * 0.1) * coherence;
            velocity += k0 * phase_vel0 * weight0;
            total_weight += weight0;
        }
        
        if (k_mag1 >= 0.001) {
            let omega1 = k_mag1 * scale;
            let phase_vel1 = omega1 / (k_mag1 * k_mag1);
            let weight1 = exp(-f32(i + 1u) * 0.1) * coherence;
            velocity += k1 * phase_vel1 * weight1;
            total_weight += weight1;
        }
        
        if (k_mag2 >= 0.001) {
            let omega2 = k_mag2 * scale;
            let phase_vel2 = omega2 / (k_mag2 * k_mag2);
            let weight2 = exp(-f32(i + 2u) * 0.1) * coherence;
            velocity += k2 * phase_vel2 * weight2;
            total_weight += weight2;
        }
        
        if (k_mag3 >= 0.001) {
            let omega3 = k_mag3 * scale;
            let phase_vel3 = omega3 / (k_mag3 * k_mag3);
            let weight3 = exp(-f32(i + 3u) * 0.1) * coherence;
            velocity += k3 * phase_vel3 * weight3;
            total_weight += weight3;
        }
    }
    
    return select(vec2<f32>(0.0), velocity / total_weight, total_weight > 0.0);
}

// Optimized vorticity with precomputed values
fn add_vorticity_optimized(pos: vec2<f32>, base_velocity: vec2<f32>) -> vec2<f32> {
    var vorticity = vec2<f32>(0.0);
    let falloff = velocity_params.vortex_falloff;
    let strength = velocity_params.vorticity_strength;
    
    // Unroll loop for better performance
    for (var i = 0u; i < 4u; i++) {
        let phase_idx = clamp_index_dyn(i / 4u, 4u);
        let phase_component = i % 4u;
        let phase = wavefield_params.phases[phase_idx][phase_component];
        let angle = phase + f32(i) * HALF_PI;
        
        // Precompute trigonometric values
        let cos_angle = cos(angle);
        let sin_angle = sin(angle);
        let center = vec2<f32>(0.5 + 0.3 * cos_angle, 0.5 + 0.3 * sin_angle);
        
        let r = pos - center;
        let r_squared = dot(r, r);
        
        // Use squared distance to avoid sqrt
        if (r_squared > 0.000001 && r_squared < 0.09) { // 0.09 = 0.3^2
            let r_mag = sqrt(r_squared);
            let inv_r_mag = 1.0 / r_mag;
            
            // Tangential velocity for vortex
            let tangent = vec2<f32>(-r.y, r.x) * inv_r_mag;
            let vortex_strength = exp(-r_mag * falloff) * strength;
            vorticity += tangent * vortex_strength;
        }
    }
    
    return base_velocity + vorticity;
}

// Optimized viscosity using separable filter
fn apply_viscosity_separable(velocity: vec2<f32>, coord: vec2<i32>, local_id: vec2<u32>) -> vec2<f32> {
    if (velocity_params.viscosity < 0.001) {
        return velocity;
    }
    
    let idx = (local_id.y + 1u) * 10u + (local_id.x + 1u);
    
    // Bounds check
    if (idx >= SHARED_SIZE || idx < 1u || idx >= SHARED_SIZE - 1u || idx < 10u || idx >= SHARED_SIZE - 10u) {
        return velocity;
    }
    
    // Horizontal pass with bounds checking
    var h_sum = vec2<f32>(0.0);
    h_sum += shared_wavefield[clamp_index_dyn(idx - 1u, SHARED_SIZE)].xy * 0.25;
    h_sum += shared_wavefield[clamp_index_dyn(idx, SHARED_SIZE)].xy * 0.5;
    h_sum += shared_wavefield[clamp_index_dyn(idx + 1u, SHARED_SIZE)].xy * 0.25;
    
    // Vertical pass with bounds checking
    var v_sum = vec2<f32>(0.0);
    v_sum += shared_wavefield[clamp_index_dyn(idx - 10u, SHARED_SIZE)].xy * 0.25;
    v_sum += shared_wavefield[clamp_index_dyn(idx, SHARED_SIZE)].xy * 0.5;
    v_sum += shared_wavefield[clamp_index_dyn(idx + 10u, SHARED_SIZE)].xy * 0.25;
    
    // Combine and apply viscosity
    let smoothed = (h_sum + v_sum) * 0.5;
    return mix(velocity, smoothed, velocity_params.viscosity);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let coord = vec2<i32>(global_id.xy);
    let dims = textureDimensions(velocity_out);
    
    // Early exit for out-of-bounds threads
    if (any(coord >= vec2<i32>(dims))) {
        return;
    }
    
    // Load shared memory
    load_shared_memory(local_id.xy, workgroup_id.xy, dims);
    
    let uv = vec2<f32>(coord) / vec2<f32>(dims);
    
    // Compute velocities using optimized functions
    let phase_grad = compute_phase_gradient_shared(local_id.xy);
    let measured_velocity = -phase_grad * velocity_params.scale_factor;
    let theoretical_velocity = compute_theoretical_velocity_optimized(uv);
    
    // Optimized blending
    let blend_factor = velocity_params.coherence_blend * wavefield_params.coherence;
    var velocity = mix(theoretical_velocity, measured_velocity, blend_factor);
    
    // Add vorticity
    velocity = add_vorticity_optimized(uv, velocity);
    
    // Apply viscosity using shared memory
    velocity = apply_viscosity_separable(velocity, coord, local_id.xy);
    
    // Apply damping
    velocity *= exp(-velocity_params.time_step * velocity_params.damping_factor);
    
    // Clamp velocity magnitude
    let speed_squared = dot(velocity, velocity);
    let max_speed_squared = velocity_params.max_speed * velocity_params.max_speed;
    if (speed_squared > max_speed_squared) {
        velocity *= sqrt(max_speed_squared / speed_squared);
    }
    
    // Store result
    textureStore(velocity_out, coord, vec4<f32>(velocity, 0.0, 0.0));
}

// Enhanced particle advection with better integration
@compute @workgroup_size(128, 1, 1)
fn advect_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_particles = arrayLength(&particles);
    
    if (idx >= num_particles) {
        return;
    }
    
    // Load particle data with bounds checking
    let clamped_idx = clamp_index_dyn(idx, num_particles);
    var particle = particles[clamped_idx];
    let pos = particle.xy;
    let old_vel = particle.zw;
    
    // Bicubic interpolation for smoother velocity field sampling
    let dims = vec2<f32>(textureDimensions(velocity_out));
    let tex_coord = pos * dims;
    let base_coord = floor(tex_coord);
    let frac = tex_coord - base_coord;
    
    // Sample 4x4 grid for bicubic interpolation
    var velocity_sum = vec2<f32>(0.0);
    var weight_sum = 0.0;
    
    for (var dy = -1; dy <= 2; dy++) {
        for (var dx = -1; dx <= 2; dx++) {
            let sample_coord = vec2<i32>(base_coord) + vec2<i32>(dx, dy);
            let clamped = clamp(sample_coord, vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
            
            let sample_vel = textureLoad(velocity_out, clamped).xy;
            
            // Bicubic weight
            let dist_x = abs(f32(dx) - frac.x);
            let dist_y = abs(f32(dy) - frac.y);
            let weight_x = max(0.0, 1.0 - dist_x);
            let weight_y = max(0.0, 1.0 - dist_y);
            let weight = weight_x * weight_y;
            
            velocity_sum += sample_vel * weight;
            weight_sum += weight;
        }
    }
    
    let field_velocity = velocity_sum / max(weight_sum, 0.001);
    
    // Improved velocity blending with momentum
    const momentum = 0.85;
    let new_vel = old_vel * momentum + field_velocity * (1.0 - momentum);
    
    // RK4 integration for better accuracy
    let dt = velocity_params.time_step;
    
    // k1
    let k1_vel = new_vel;
    let k1_pos = pos + k1_vel * dt * 0.5;
    
    // k2
    let k2_coord = clamp(k1_pos * dims, vec2<f32>(0.0), dims - vec2<f32>(1.0));
    let k2_vel = textureLoad(velocity_out, vec2<i32>(k2_coord)).xy;
    let k2_pos = pos + k2_vel * dt * 0.5;
    
    // k3
    let k3_coord = clamp(k2_pos * dims, vec2<f32>(0.0), dims - vec2<f32>(1.0));
    let k3_vel = textureLoad(velocity_out, vec2<i32>(k3_coord)).xy;
    let k3_pos = pos + k3_vel * dt;
    
    // k4
    let k4_coord = clamp(k3_pos * dims, vec2<f32>(0.0), dims - vec2<f32>(1.0));
    let k4_vel = textureLoad(velocity_out, vec2<i32>(k4_coord)).xy;
    
    // Combine
    let final_vel = (k1_vel + 2.0 * k2_vel + 2.0 * k3_vel + k4_vel) / 6.0;
    var new_pos = pos + final_vel * dt;
    
    // Toroidal wrapping for continuous flow
    new_pos = fract(new_pos + vec2<f32>(1.0));
    
    // Store updated particle with bounds checking
    particles[clamped_idx] = vec4<f32>(new_pos, new_vel);
}

// Enhanced visualization with gradient magnitude
@compute @workgroup_size(8, 8, 1)
fn visualize_flow(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = vec2<i32>(global_id.xy);
    let dims = textureDimensions(flow_vis_out);
    
    if (any(coord >= vec2<i32>(dims))) {
        return;
    }
    
    // Load velocity
    let velocity = textureLoad(velocity_out, coord).xy;
    let magnitude = length(velocity);
    
    // Compute divergence and curl for additional visualization
    let left = textureLoad(velocity_out, coord + vec2<i32>(-1, 0)).xy;
    let right = textureLoad(velocity_out, coord + vec2<i32>(1, 0)).xy;
    let down = textureLoad(velocity_out, coord + vec2<i32>(0, -1)).xy;
    let up = textureLoad(velocity_out, coord + vec2<i32>(0, 1)).xy;
    
    let divergence = (right.x - left.x + up.y - down.y) * 0.5;
    let curl = (right.y - left.y - up.x + down.x) * 0.5;
    
    // HSV to RGB conversion for better visualization
    let hue = atan2(velocity.y, velocity.x) * INV_TWO_PI + 0.5;
    let saturation = clamp(magnitude / velocity_params.max_speed, 0.0, 1.0);
    const value = 1.0;
    
    // Convert HSV to RGB
    let c = value * saturation;
    let x = c * (1.0 - abs((hue * 6.0) % 2.0 - 1.0));
    let m = value - c;
    
    var rgb: vec3<f32>;
    let h_segment = i32(hue * 6.0);
    switch (h_segment) {
        case 0: { rgb = vec3<f32>(c, x, 0.0); }
        case 1: { rgb = vec3<f32>(x, c, 0.0); }
        case 2: { rgb = vec3<f32>(0.0, c, x); }
        case 3: { rgb = vec3<f32>(0.0, x, c); }
        case 4: { rgb = vec3<f32>(x, 0.0, c); }
        default: { rgb = vec3<f32>(c, 0.0, x); }
    }
    
    rgb = rgb + vec3<f32>(m);
    
    // Add divergence/curl visualization in alpha channel
    let div_curl_vis = clamp(abs(divergence) + abs(curl) * 0.5, 0.0, 1.0);
    
    textureStore(flow_vis_out, coord, vec4<f32>(rgb, div_curl_vis));
}
