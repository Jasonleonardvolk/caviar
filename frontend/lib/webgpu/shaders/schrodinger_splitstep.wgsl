// schrodinger_splitstep.wgsl
// Split-Step Fourier Method: MAXIMUM STABILITY!
// Can use dt = 1.0 or even larger!
// Requires FFT passes between steps

struct Params {
    width: u32,
    height: u32,
    is_forward_fft: u32,   // 1 for FFT, 0 for IFFT
    pad0: u32,
    dt: f32,
    alpha: f32,            // Kinetic coefficient
    beta: f32,             // Dispersion coefficient  
    k_max: f32,            // Maximum k-vector magnitude
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> inField: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> outField: array<vec2<f32>>;
@group(0) @binding(3) var potential: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;

// Complex multiplication
fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Complex exponential
fn c_exp(phase: f32) -> vec2<f32> {
    return vec2<f32>(cos(phase), sin(phase));
}

// Step 1: Apply potential in position space (half step)
@compute @workgroup_size(8, 8, 1)
fn apply_potential_half(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.width || gid.y >= P.height) { return; }
    let id = gid.y * P.width + gid.x;
    
    // Sample potential
    let uv = vec2<f32>((f32(gid.x) + 0.5) / f32(P.width),
                       (f32(gid.y) + 0.5) / f32(P.height));
    let V = textureSampleLevel(potential, samp, uv, 0.0).r;
    
    // Apply exp(-i*V*dt/2)
    let phase = -V * P.dt * 0.5;
    let propagator = c_exp(phase);
    
    outField[id] = c_mul(inField[id], propagator);
}

// Step 2: Apply kinetic + dispersion in k-space (full step)
@compute @workgroup_size(8, 8, 1)
fn apply_kinetic_full(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.width || gid.y >= P.height) { return; }
    let id = gid.y * P.width + gid.x;
    
    // Get k-space coordinates (assumes FFT shift)
    let kx_idx = select(f32(gid.x), f32(gid.x) - f32(P.width), gid.x >= P.width / 2u);
    let ky_idx = select(f32(gid.y), f32(gid.y) - f32(P.height), gid.y >= P.height / 2u);
    
    // Normalize to [-π, π]
    let kx = kx_idx * 2.0 * 3.14159265 / f32(P.width);
    let ky = ky_idx * 2.0 * 3.14159265 / f32(P.height);
    let k2 = kx * kx + ky * ky;
    let k4 = k2 * k2;
    
    // Dispersion relation: E(k) = α*k² + β*k⁴
    let energy = P.alpha * k2 + P.beta * k4;
    
    // Apply exp(-i*E*dt) in k-space
    let phase = -energy * P.dt;
    let propagator = c_exp(phase);
    
    // Band limiting: suppress high frequencies
    let k_mag = sqrt(k2);
    let suppress = select(1.0, 0.0, k_mag > P.k_max);
    
    outField[id] = c_mul(inField[id], propagator) * suppress;
}

// Step 3: Apply potential again (half step) - same as step 1
@compute @workgroup_size(8, 8, 1)
fn apply_potential_half_final(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.width || gid.y >= P.height) { return; }
    let id = gid.y * P.width + gid.x;
    
    let uv = vec2<f32>((f32(gid.x) + 0.5) / f32(P.width),
                       (f32(gid.y) + 0.5) / f32(P.height));
    let V = textureSampleLevel(potential, samp, uv, 0.0).r;
    
    let phase = -V * P.dt * 0.5;
    let propagator = c_exp(phase);
    
    outField[id] = c_mul(inField[id], propagator);
}

// Helper: Apply absorbing boundary conditions (sponge layer)
@compute @workgroup_size(8, 8, 1)
fn apply_absorbing_boundary(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.width || gid.y >= P.height) { return; }
    let id = gid.y * P.width + gid.x;
    
    // Distance from edges
    let margin = 16.0;  // Width of absorbing layer
    let dx = min(f32(gid.x), f32(P.width - 1u - gid.x));
    let dy = min(f32(gid.y), f32(P.height - 1u - gid.y));
    let d = min(dx, dy);
    
    // Smooth absorption using tanh profile
    let absorption = select(1.0, 0.5 * (1.0 + tanh((d - margin) / 4.0)), d < margin);
    
    outField[id] = inField[id] * absorption;
}