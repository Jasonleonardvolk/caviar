// schrodinger_phase_multiply.wgsl
// PLATINUM Edition: Phase multiplication with absorbing boundaries and APO
// - Support for complex potentials (absorption)
// - Airy function absorbing boundaries
// - Perfectly matched layer (PML) option

struct Params {
    width: u32;
    height: u32;
    pad0: u32;
    pad1: u32;
    dt_half: f32;
    vscale: f32;
    useMask: u32;         // 0/1 for absorbing mask
    maskStrength: f32;    // 0..1 damping strength
    // ENHANCED: Additional boundary options
    boundary_type: u32;   // 0: none, 1: mask, 2: PML, 3: Airy
    pml_width: f32;       // Width of PML region in pixels
    pml_strength: f32;    // PML absorption coefficient
    airy_scale: f32;      // Airy function scaling
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> field: array<vec2<f32>>;
@group(0) @binding(2) var potentialTex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

// Complex multiplication
fn complex_multiply(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

// Complex exponential with accuracy improvements
fn complex_exp_accurate(theta: f32) -> vec2<f32> {
    // Range reduction for better accuracy
    let reduced_theta = theta - floor(theta / (2.0 * 3.14159265359)) * 2.0 * 3.14159265359;
    return vec2<f32>(cos(reduced_theta), sin(reduced_theta));
}

// Airy function absorbing boundary (approximate)
fn airy_absorbing(dist_from_edge: f32, scale: f32) -> f32 {
    // Approximation of Ai(x) decay for x > 0
    let x = max(0.0, -dist_from_edge / scale);
    if (x < 0.1) { return 1.0; }
    
    // Asymptotic approximation: Ai(x) ~ exp(-2/3 * x^(3/2)) / (2 * sqrt(pi) * x^(1/4))
    let x_pow = pow(x, 1.5);
    return exp(-0.666667 * x_pow) / (2.0 * sqrt(3.14159265359 * pow(x, 0.25)));
}

// Perfectly Matched Layer (PML) absorption
fn pml_sigma(dist_from_edge: f32, width: f32, strength: f32) -> f32 {
    if (dist_from_edge >= width) { return 0.0; }
    
    let x = (width - dist_from_edge) / width;  // 0 at edge, 1 at inner boundary
    // Quadratic profile for smooth absorption
    return strength * x * x;
}

// Super-Gaussian window for smooth edges
fn super_gaussian(r: f32, order: f32) -> f32 {
    return exp(-pow(r, order));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.width || gid.y >= P.height) { return; }
    
    let id = gid.y * P.width + gid.x;
    let uv = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(P.width),
        (f32(gid.y) + 0.5) / f32(P.height)
    );
    
    // Sample potential (can be complex for absorption)
    let pot_sample = textureSampleLevel(potentialTex, samp, uv, 0.0);
    let V_real = pot_sample.r * P.vscale;
    let V_imag = pot_sample.g * P.vscale;  // Imaginary part for absorption
    
    // Load field value
    var psi = field[id];
    
    // Apply potential evolution: exp(-i * (V_r - i*V_i) * dt/2)
    // = exp(-i * V_r * dt/2) * exp(-V_i * dt/2)
    let phase = -V_real * P.dt_half;
    let damping = exp(-V_imag * P.dt_half);
    
    let rotation = complex_exp_accurate(phase);
    psi = complex_multiply(psi, rotation) * damping;
    
    // Apply boundary conditions based on type
    var boundary_factor = 1.0;
    
    if (P.boundary_type > 0u) {
        // Calculate distance from edges
        let fx = f32(gid.x);
        let fy = f32(gid.y);
        let dist_left = fx;
        let dist_right = f32(P.width - 1u) - fx;
        let dist_top = fy;
        let dist_bottom = f32(P.height - 1u) - fy;
        let min_dist = min(min(dist_left, dist_right), min(dist_top, dist_bottom));
        
        switch (P.boundary_type) {
            case 1u: { // Simple mask from texture alpha
                let mask = clamp(pot_sample.a, 0.0, 1.0);
                boundary_factor = mix(1.0, mask, P.maskStrength);
            }
            case 2u: { // PML
                let sigma = pml_sigma(min_dist, P.pml_width, P.pml_strength);
                boundary_factor = exp(-sigma * P.dt_half);
            }
            case 3u: { // Airy function
                boundary_factor = airy_absorbing(min_dist, P.airy_scale);
            }
            default: {}
        }
    }
    
    // Apply boundary absorption
    psi *= boundary_factor;
    
    // Store result
    field[id] = psi;
}

// ENHANCED: Variant for split potential (real and imaginary parts separate)
@compute @workgroup_size(8, 8, 1)
fn apply_split_potential(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (gid.x >= P.width || gid.y >= P.height) { return; }
    
    let id = gid.y * P.width + gid.x;
    
    // For cases where V_real and V_imag are in separate textures
    // This allows for higher precision or different resolutions
    let uv = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(P.width),
        (f32(gid.y) + 0.5) / f32(P.height)
    );
    
    // Red channel: real potential, Green channel: imaginary potential
    let V_complex = textureSampleLevel(potentialTex, samp, uv, 0.0).rg * P.vscale;
    
    // Apply split-operator method more accurately
    let psi = field[id];
    
    // First apply real part rotation
    let phase_real = -V_complex.x * P.dt_half;
    let rot_real = complex_exp_accurate(phase_real);
    var result = complex_multiply(psi, rot_real);
    
    // Then apply imaginary part damping
    let damping = exp(-V_complex.y * P.dt_half);
    result *= damping;
    
    field[id] = result;
}