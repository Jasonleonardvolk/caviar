// fftShift.wgsl ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â Fixed for bundling
struct FFTUniforms {
    size: u32,
    log_size: u32,
    batch_size: u32,
    normalization: f32,
    dimensions: u32,
    direction: u32,
    stage: u32,
    _padding: u32
}

@group(0) @binding(0) var<uniform> uniforms: FFTUniforms;
@group(0) @binding(2) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec2<f32>>;

const workgroup_size_x: u32 = 256u;


fn clamp_index_dyn(i: u32, len: u32) -> u32 {
    return select(i, len - 1u, i >= len);
}

fn fft_shift_2d(idx: u32) {
    let N = uniforms.size;
    let N2 = N * N;
    let b = uniforms.batch_size;
    if (idx >= N2 * b) { return; }

    let batch = idx / N2;
    let i = idx % N2;

    let x = i % N;
    let y = i / N;
    let hx = x ^ (N >> 1u);
    let hy = y ^ (N >> 1u);
    let j = hy * N + hx;

    output[clamp_index_dyn(batch * N2 + j, arrayLength(&output))] = input[clamp_index_dyn(batch * N2 + i, arrayLength(&input))];
}

@compute @workgroup_size(workgroup_size_x, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;

    switch uniforms.dimensions {
        case 2u { fft_shift_2d(i); }
        default { return; }
    }
}
