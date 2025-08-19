// normalize.wgsl
// Apply normalization factor to FFT output based on configured normalization mode

struct FFTUniforms {
    size: u32,
    log_size: u32,
    batch_size: u32,
    normalization: f32,    // Normalization factor calculated on CPU
    dimensions: u32,
    direction: u32,
    stage: u32,
    twiddle_offset: u32
}

@group(0) @binding(0) var<uniform> uniforms: FFTUniforms;
@group(0) @binding(1) var<storage, read> dummy: array<vec2<f32>>;  // Unused but needed for consistent layout
@group(0) @binding(2) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec2<f32>>;

// Specialization constant for workgroup size
@id(0) override workgroup_size_x: u32 = 256u;


fn clamp_index_dyn(i: u32, len: u32) -> u32 {
    return select(i, len - 1u, i >= len);
}

@compute @workgroup_size(workgroup_size_x, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    let total_size = uniforms.size * uniforms.batch_size;
    
    // Bounds check
    if (idx >= total_size) {
        return;
    }
    
    // Apply normalization factor
    let value = input[clamp_index_dyn(idx, arrayLength(&input))];
    output[clamp_index_dyn(idx, arrayLength(&output))] = value * uniforms.normalization;
}
