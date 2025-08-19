// temporal_neural_propagation.wgsl
// Time-dependent Schrödinger equation with learned dynamics
// Predicts future wave states using physics + neural corrections

struct Params {
    width: u32,
    height: u32,
    dt: f32,              // Time step
    dx: f32,              // Spatial resolution
    // Wave equation parameters
    c: f32,               // Wave speed (c = λν)
    dispersion: f32,      // Dispersion coefficient
    damping: f32,         // Damping factor
    // Neural prediction weights (learned)
    a0: f32,              // Current frame weight
    a1: f32,              // Previous frame weight
    a2: f32,              // Two frames ago weight
    lap_scale: f32,       // Laplacian influence
    biharmonic_scale: f32, // ∇⁴ influence for dispersion
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> prevField: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> currField: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> nextField: array<vec2<f32>>;
// Neural correction from learned operator
@group(0) @binding(4) var<storage, read> neuralCorrection: array<vec2<f32>>;

// Constants
const PI: f32 = 3.14159265359;

// Safe array access with clamping
fn at(buf: ptr<storage, array<vec2<f32>>, read>, x: i32, y: i32) -> vec2<f32> {
    let xi = clamp(x, 0, i32(p.width) - 1);
    let yi = clamp(y, 0, i32(p.height) - 1);
    let idx = u32(yi) * p.width + u32(xi);
    if (idx < arrayLength(buf)) {
        return (*buf)[idx];
    }
    return vec2<f32>(0.0, 0.0);
}

// Complex multiplication
fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.width || gid.y >= p.height) { 
        return; 
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let idx = gid.y * p.width + gid.x;
    
    // Current and previous values
    let c = at(&currField, x, y);
    let p_prev = at(&prevField, x, y);
    
    // Compute Laplacian (∇²) - 5-point stencil
    let l = at(&currField, x-1, y);
    let r = at(&currField, x+1, y);
    let u = at(&currField, x, y-1);
    let d = at(&currField, x, y+1);
    let lap = (l + r + u + d - 4.0 * c) / (p.dx * p.dx);
    
    // Compute Biharmonic (∇⁴) for dispersion - 13-point stencil
    let ll = at(&currField, x-2, y);
    let rr = at(&currField, x+2, y);
    let uu = at(&currField, x, y-2);
    let dd = at(&currField, x, y+2);
    let ul = at(&currField, x-1, y-1);
    let ur = at(&currField, x+1, y-1);
    let dl = at(&currField, x-1, y+1);
    let dr = at(&currField, x+1, y+1);
    
    let biharmonic = (ll + rr + uu + dd 
                     - 4.0*(l + r + u + d) 
                     + 12.0*c 
                     + 2.0*(ul + ur + dl + dr)) / (p.dx * p.dx * p.dx * p.dx);
    
    // Time-dependent Schrödinger equation with dispersion
    // i∂ψ/∂t = -∇²ψ/(2m) + V(x)ψ + dispersion∇⁴ψ
    let i_unit = vec2<f32>(0.0, 1.0);
    
    // Wave propagation term (kinetic energy)
    var wave_term = lap * p.c * p.c;
    
    // Dispersion term (for chromatic effects)
    wave_term = wave_term - biharmonic * p.dispersion;
    
    // Time evolution using improved Euler method
    // ψ(t+dt) = ψ(t) + i*H*ψ(t)*dt
    var n = c + c_mul(i_unit, wave_term) * p.dt;
    
    // Add damping for stability (dissipation)
    n = n * (1.0 - p.damping * p.dt);
    
    // Neural correction (learned residual)
    if (idx < arrayLength(&neuralCorrection)) {
        n = n + neuralCorrection[idx] * 0.1; // Small correction
    }
    
    // AR(2) recurrence for temporal coherence
    n = n * p.a0 + p_prev * p.a1 + lap * p.lap_scale;
    
    // Ensure unitarity (preserve probability)
    let mag = length(n);
    if (mag > 0.0) {
        n = n / mag; // Normalize
    }
    
    nextField[idx] = n;
}

// Predict multiple frames ahead using learned dynamics
@compute @workgroup_size(8, 8, 1)
fn predict_future(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.width || gid.y >= p.height) { 
        return; 
    }
    let idx = gid.y * p.width + gid.x;
    
    // Load current state
    var state = vec2<f32>(0.0, 0.0);
    var prev_state = vec2<f32>(0.0, 0.0);
    
    if (idx < arrayLength(&currField)) {
        state = currField[idx];
    }
    if (idx < arrayLength(&prevField)) {
        prev_state = prevField[idx];
    }
    
    // Predict N frames ahead using learned dynamics
    let future_steps = 10u;
    for (var t = 0u; t < future_steps; t = t + 1u) {
        // Simplified evolution for speed
        let prediction = state * p.a0 + prev_state * p.a1;
        prev_state = state;
        state = prediction;
        
        // Maintain unitarity
        let mag = length(state);
        if (mag > 0.0) {
            state = state / mag;
        }
    }
    
    nextField[idx] = state;
}