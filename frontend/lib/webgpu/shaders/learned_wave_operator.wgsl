// learned_wave_operator.wgsl
// Neural operator stub - will be replaced with ONNX model later
// For now, implements sophisticated convolution-based wave propagation

struct Params { 
    width: u32,
    height: u32,
    kernel_size: u32,      // 3, 5, or 7
    num_layers: u32,        // Number of conv layers to apply
    nonlinearity: u32,      // 0: None, 1: ReLU, 2: Tanh, 3: GELU
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> inField: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> kernel: array<vec2<f32>>;    // Complex kernels, flattened
@group(0) @binding(3) var<storage, read_write> outField: array<vec2<f32>>;
// Additional kernels for multi-layer network
@group(0) @binding(4) var<storage, read> kernel2: array<vec2<f32>>;   // Second layer kernel
@group(0) @binding(5) var<storage, read> bias: array<vec2<f32>>;      // Complex bias terms

// Safe array access
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

// GELU activation for complex numbers (applied to magnitude)
fn c_gelu(z: vec2<f32>) -> vec2<f32> {
    let mag = length(z);
    let phase = atan2(z.y, z.x);
    
    // GELU(x) = x * Φ(x) where Φ is CDF of standard normal
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let c = sqrt(2.0 / 3.14159265359);
    let gelu_mag = 0.5 * mag * (1.0 + tanh(c * (mag + 0.044715 * mag * mag * mag)));
    
    return vec2<f32>(gelu_mag * cos(phase), gelu_mag * sin(phase));
}

// Apply nonlinearity
fn apply_activation(z: vec2<f32>, activation_type: u32) -> vec2<f32> {
    switch (activation_type) {
        case 0u: { return z; }                    // Linear
        case 1u: {                                 // ReLU (on magnitude)
            let mag = max(0.0, length(z));
            let phase = atan2(z.y, z.x);
            return vec2<f32>(mag * cos(phase), mag * sin(phase));
        }
        case 2u: {                                 // Tanh (on real and imag separately)
            return vec2<f32>(tanh(z.x), tanh(z.y));
        }
        case 3u: { return c_gelu(z); }             // GELU
        default: { return z; }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.width || gid.y >= p.height) { 
        return; 
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let idx = gid.y * p.width + gid.x;
    
    var acc = vec2<f32>(0.0, 0.0);
    let half_kernel = i32(p.kernel_size) / 2;
    
    // First convolution layer
    var k_idx = 0u;
    for (var j = -half_kernel; j <= half_kernel; j = j + 1) {
        for (var i = -half_kernel; i <= half_kernel; i = i + 1) {
            if (k_idx < arrayLength(&kernel)) {
                let field_val = at(&inField, x + i, y + j);
                let kernel_val = kernel[k_idx];
                acc = acc + c_mul(field_val, kernel_val);
            }
            k_idx = k_idx + 1u;
        }
    }
    
    // Add bias if available
    if (idx < arrayLength(&bias)) {
        acc = acc + bias[idx];
    }
    
    // Apply activation
    acc = apply_activation(acc, p.nonlinearity);
    
    // Second layer if requested (residual connection)
    if (p.num_layers > 1u && arrayLength(&kernel2) > 0u) {
        var acc2 = vec2<f32>(0.0, 0.0);
        k_idx = 0u;
        
        for (var j = -half_kernel; j <= half_kernel; j = j + 1) {
            for (var i = -half_kernel; i <= half_kernel; i = i + 1) {
                if (k_idx < arrayLength(&kernel2)) {
                    // Use acc as input to second layer
                    let neighbor_idx = clamp(
                        u32(clamp(y + j, 0, i32(p.height) - 1)) * p.width + 
                        u32(clamp(x + i, 0, i32(p.width) - 1)),
                        0u,
                        p.width * p.height - 1u
                    );
                    // This is simplified - in reality we'd need intermediate storage
                    let field_val = at(&inField, x + i, y + j); // Using original for now
                    let kernel_val = kernel2[k_idx];
                    acc2 = acc2 + c_mul(field_val, kernel_val);
                }
                k_idx = k_idx + 1u;
            }
        }
        
        // Residual connection
        acc = acc + acc2 * 0.5;
    }
    
    outField[idx] = acc;
}

// Fourier Neural Operator (FNO) inspired kernel
@compute @workgroup_size(8, 8, 1)
fn fno_layer(@builtin(global_invocation_id) gid: vec3<u32>) {
    // This would implement spectral convolution in Fourier space
    // For now, it's a placeholder for the full FNO implementation
    // The idea: 
    // 1. FFT the input
    // 2. Multiply by learned spectral weights
    // 3. IFFT back
    // This learns the Green's function of the PDE!
    
    if (gid.x >= p.width || gid.y >= p.height) { 
        return; 
    }
    let idx = gid.y * p.width + gid.x;
    
    // Placeholder: just copy through with phase shift
    if (idx < arrayLength(&inField)) {
        let phase_shift = vec2<f32>(0.9659, 0.2588); // cos(15°), sin(15°)
        outField[idx] = c_mul(inField[idx], phase_shift);
    }
}