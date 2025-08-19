// schrodinger_kspace_multiply.wgsl
// PLATINUM Edition: K-space evolution with anisotropic dispersion
// - Support for direction-dependent dispersion
// - Band limiting for anti-aliasing
// - Nonlinear corrections

struct Params {
    width: u32;
    height: u32;
    pad0: u32;
    pad1: u32;
    dt: f32;
    alpha: f32;          // Isotropic kinetic coefficient
    beta: f32;           // Isotropic biharmonic coefficient
    dx: f32;             // Spatial resolution x
    dy: f32;             // Spatial resolution y
    // ENHANCED: Anisotropic dispersion
    alpha_x: f32;        // Direction-dependent kinetic
    alpha_y: f32;
    beta_x: f32;         // Direction-dependent biharmonic
    beta_y: f32;
    // Nonlinear corrections
    nonlinear_strength: f32;  // For |ψ|²ψ term in k-space
    band_limit: f32;          // Cutoff frequency (0.5 = Nyquist)
    filter_order: f32;        // Sharpness of band limiting
    use_anisotropic: u32;     // 0: isotropic, 1: anisotropic
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> fieldK: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> density: array<f32>; // Optional: |ψ|² for nonlinear term

const PI = 3.14159265359;
const TWO_PI = 6.28318530718;

// Convert array index to k-space coordinate
fn kcoord(i: u32, n: u32, d: f32) -> f32 {
    let ii = i32(i);
    let nn = i32(n);
    // Standard FFT indexing: positive frequencies first, then negative
    let iw = select(ii, ii - nn, ii > nn / 2);
    return TWO_PI * (f32(iw) / (f32(n) * d));
}

// Enhanced k-coordinate with Nyquist handling
fn kcoord_enhanced(i: u32, n: u32, d: f32) -> f32 {
    let ii = f32(i);
    let nn = f32(n);
    
    // Handle Nyquist frequency specially for even n
    if (i == n / 2u && (n & 1u) == 0u) {
        return PI / d;  // Nyquist frequency
    }
    
    // Standard mapping
    let k_index = select(ii, ii - nn, ii > nn * 0.5);
    return TWO_PI * k_index / (nn * d);
}

// Band limiting filter (smoother than hard cutoff)
fn band_limit_filter(k_mag: f32, cutoff: f32, order: f32) -> f32 {
    let k_normalized = k_mag / (PI / min(P.dx, P.dy));  // Normalize to Nyquist
    
    if (k_normalized <= cutoff) {
        return 1.0;
    }
    
    if (order < 0.5) {
        // Hard cutoff
        return 0.0;
    }
    
    // Smooth cutoff using super-Gaussian
    let excess = (k_normalized - cutoff) / (1.0 - cutoff);
    return exp(-pow(excess * 2.0, order));
}

// Complex operations
fn complex_multiply(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

fn complex_exp_stable(phase: f32) -> vec2<f32> {
    // Stability for large phase values
    let reduced = phase - floor(phase / TWO_PI) * TWO_PI;
    
    // Use Taylor series for small angles (better accuracy)
    if (abs(reduced) < 0.1) {
        let p2 = reduced * reduced;
        let p3 = p2 * reduced;
        let p4 = p2 * p2;
        let p5 = p4 * reduced;
        
        let cos_approx = 1.0 - p2 * 0.5 + p4 * 0.041666667;
        let sin_approx = reduced - p3 * 0.166666667 + p5 * 0.008333333;
        
        return vec2<f32>(cos_approx, sin_approx);
    }
    
    return vec2<f32>(cos(reduced), sin(reduced));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.width || gid.y >= P.height) { return; }
    
    let id = gid.y * P.width + gid.x;
    
    // Get k-space coordinates
    let kx = kcoord_enhanced(gid.x, P.width, P.dx);
    let ky = kcoord_enhanced(gid.y, P.height, P.dy);
    
    // Calculate dispersion based on mode
    var phase: f32;
    
    if (P.use_anisotropic == 1u) {
        // Anisotropic dispersion
        let kx2 = kx * kx;
        let ky2 = ky * ky;
        let kx4 = kx2 * kx2;
        let ky4 = ky2 * ky2;
        
        // Direction-dependent coefficients
        let kinetic_term = P.alpha_x * kx2 + P.alpha_y * ky2;
        let biharmonic_term = P.beta_x * kx4 + P.beta_y * ky4;
        
        // Cross terms for full anisotropy
        let cross_term = 2.0 * sqrt(P.beta_x * P.beta_y) * kx2 * ky2;
        
        phase = P.dt * (kinetic_term - biharmonic_term - cross_term * 0.5);
    } else {
        // Isotropic dispersion (standard)
        let k2 = kx * kx + ky * ky;
        let k4 = k2 * k2;
        phase = P.dt * (P.alpha * k2 - P.beta * k4);
    }
    
    // Add nonlinear correction if available
    if (P.nonlinear_strength > 0.0 && id < arrayLength(&density)) {
        let rho = density[id];
        phase += P.dt * P.nonlinear_strength * rho;
    }
    
    // Apply band limiting
    let k_mag = sqrt(kx * kx + ky * ky);
    let filter = band_limit_filter(k_mag, P.band_limit, P.filter_order);
    
    // Load field value
    let psi_k = fieldK[id];
    
    // Apply evolution operator
    let evolution = complex_exp_stable(-phase);  // Note: negative for correct sign
    var result = complex_multiply(psi_k, evolution);
    
    // Apply band limit filter
    result *= filter;
    
    // Store result
    fieldK[id] = result;
}

// ENHANCED: Specialized version for radially symmetric dispersion
@compute @workgroup_size(8, 8, 1)
fn apply_radial_dispersion(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= P.width || gid.y >= P.height) { return; }
    
    let id = gid.y * P.width + gid.x;
    
    // Get k-space coordinates
    let kx = kcoord_enhanced(gid.x, P.width, P.dx);
    let ky = kcoord_enhanced(gid.y, P.height, P.dy);
    let k2 = kx * kx + ky * ky;
    let k = sqrt(k2);
    
    // Radial dispersion relation (can be more complex)
    // Example: ω(k) = α*k² - β*k⁴ + γ*k³ (odd power for asymmetry)
    let omega = P.alpha * k2 - P.beta * k2 * k2;
    
    // Phase velocity correction for radial symmetry
    let phase = -P.dt * omega;
    
    // Apply with band limiting
    let filter = band_limit_filter(k, P.band_limit, P.filter_order);
    let evolution = complex_exp_stable(phase);
    
    let psi_k = fieldK[id];
    fieldK[id] = complex_multiply(psi_k, evolution) * filter;
}