// File: frontend/shaders/bitReversal.wgsl
// Bit-reversal permutation for FFT with precomputed lookup table

// -------------------------------
// Struct Definitions
// -------------------------------
struct FFTUniforms {
    size: u32,             // FFT size (e.g., 1024)
    log_size: u32,         // log2(size)
    batch_size: u32,       // Number of parallel batches
    normalization: f32,    // Normalization factor
    dimensions: u32,       // 1D or 2D FFT
    direction: u32,        // Forward or inverse
    stage: u32,            // FFT stage
    _padding: u32          // Padding for 16-byte alignment
}

// -------------------------------
// Bindings
// -------------------------------
@group(0) @binding(0) var<uniform> uniforms: FFTUniforms;
@group(0) @binding(1) var<storage, read> bit_reversal: array<u32>;
@group(0) @binding(2) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec2<f32>>;

// -------------------------------
// Constants
// -------------------------------
@id(0) override workgroup_size_x: u32 = 256u;
@id(1) override normalization_mode: u32 = 0u;  // Add the missing constant

// -------------------------------
// Entry Point
// -------------------------------

fn clamp_index_dyn(i: u32, len: u32) -> u32 {
    return select(i, len - 1u, i >= len);
}

@compute @workgroup_size(workgroup_size_x, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let idx = global_id.x;
    let fft_size = uniforms.size;
    let total = fft_size * uniforms.batch_size;

    // Defensive bound check
    if (idx >= total) {
        return;
    }

    // Batch and element index
    let batch = idx / fft_size;
    let i = idx % fft_size;
    let offset = batch * fft_size;

    // Bit-reversal lookup with bounds checking
    let j = bit_reversal[clamp_index_dyn(i, arrayLength(&bit_reversal))];

    // Perform the copy with bounds checking
    output[clamp_index_dyn(offset + j, arrayLength(&output))] = input[clamp_index_dyn(offset + i, arrayLength(&input))];
}
