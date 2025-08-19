// butterflyStage.wgsl
// Radix-2 butterfly computation for FFT stages with optimizations
// Processes one stage of the FFT using Cooley-Tukey algorithm

struct FFTUniforms {
    size: u32,
    log_size: u32,
    batch_size: u32,
    normalization: f32,
    dimensions: u32,
    direction: u32,
    stage: u32,
    twiddle_offset: u32  // Precomputed offset for this stage
}

@group(0) @binding(0) var<uniform> uniforms: FFTUniforms;
@group(0) @binding(1) var<storage, read> twiddles: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec2<f32>>;

// Specialization constant for workgroup size
const workgroup_size_x: u32 = 256u;

// Optimized complex multiplication using FMA (fused multiply-add)

fn clamp_index_dyn(i: u32, len: u32) -> u32 {
    return select(i, len - 1u, i >= len);
}

fn complex_multiply_fma(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        fma(a.x, b.x, -a.y * b.y),  // a.x * b.x - a.y * b.y
        fma(a.x, b.y,  a.y * b.x)   // a.x * b.y + a.y * b.x
    );
}

// Alternative complex multiply for when FMA is not available
fn complex_multiply(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

@compute @workgroup_size(workgroup_size_x, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread_id = global_id.x;
    let stage = uniforms.stage;
    let half_size = uniforms.size >> 1u;
    let total_butterflies = half_size * uniforms.batch_size;
    
    // Bounds check
    if (thread_id >= total_butterflies) { 
        return; 
    }
    
    // Calculate batch and butterfly indices
    let batch_idx = thread_id / half_size;
    let butterfly_idx = thread_id % half_size;
    let batch_offset = batch_idx * uniforms.size;
    
    // Calculate indices for butterfly operation
    let stage_size = 1u << (stage + 1u);
    let half_stage = stage_size >> 1u;
    let group = butterfly_idx / half_stage;
    let pair = butterfly_idx % half_stage;
    
    let idx_a = batch_offset + group * stage_size + pair;
    let idx_b = idx_a + half_stage;
    
    // Get twiddle factor using precomputed offset with bounds checking
    let twiddle_idx = uniforms.twiddle_offset + pair;
    let twiddle = twiddles[clamp_index_dyn(twiddle_idx, arrayLength(&twiddles))];
    
    // Load values with bounds checking
    let a = input[clamp_index_dyn(idx_a, arrayLength(&input))];
    let b = input[clamp_index_dyn(idx_b, arrayLength(&input))];
    
    // Butterfly operation with FMA optimization
    let b_twiddle = complex_multiply_fma(b, twiddle);
    
    // Store results with bounds checking
    output[clamp_index_dyn(idx_a, arrayLength(&output))] = a + b_twiddle;
    output[clamp_index_dyn(idx_b, arrayLength(&output))] = a - b_twiddle;
}

// Shared memory variant for better cache utilization
// Use when butterfly pairs fit in shared memory
const BUTTERFLIES_PER_WORKGROUP: u32 = 128u;
var<workgroup> shared_data: array<vec2<f32>, BUTTERFLIES_PER_WORKGROUP * 2u>;

@compute @workgroup_size(BUTTERFLIES_PER_WORKGROUP, 1, 1)
fn main_shared(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let stage = uniforms.stage;
    let stage_size = 1u << (stage + 1u);
    let half_stage = stage_size >> 1u;
    
    // Calculate global butterfly index
    let butterfly_idx = workgroup_id.x * BUTTERFLIES_PER_WORKGROUP + local_id.x;
    let batch_idx = butterfly_idx / (uniforms.size >> 1u);
    
    if (batch_idx >= uniforms.batch_size) {
        return;
    }
    
    // Load data into shared memory
    let local_idx = local_id.x;
    let group = (butterfly_idx % (uniforms.size >> 1u)) / half_stage;
    let pair = (butterfly_idx % (uniforms.size >> 1u)) % half_stage;
    
    let batch_offset = batch_idx * uniforms.size;
    let idx_a = batch_offset + group * stage_size + pair;
    let idx_b = idx_a + half_stage;
    
    // Shared memory accesses with bounds checking
    let shared_idx_a = local_idx * 2u;
    let shared_idx_b = local_idx * 2u + 1u;
    
    if (shared_idx_a < BUTTERFLIES_PER_WORKGROUP * 2u) {
        shared_data[shared_idx_a] = input[clamp_index_dyn(idx_a, arrayLength(&input))];
    }
    if (shared_idx_b < BUTTERFLIES_PER_WORKGROUP * 2u) {
        shared_data[shared_idx_b] = input[clamp_index_dyn(idx_b, arrayLength(&input))];
    }
    
    workgroupBarrier();
    
    // Perform butterfly operation with bounds checking
    var a = vec2<f32>(0.0, 0.0);
    var b = vec2<f32>(0.0, 0.0);
    
    if (shared_idx_a < BUTTERFLIES_PER_WORKGROUP * 2u) {
        a = shared_data[shared_idx_a];
    }
    if (shared_idx_b < BUTTERFLIES_PER_WORKGROUP * 2u) {
        b = shared_data[shared_idx_b];
    }
    
    let twiddle_idx = uniforms.twiddle_offset + pair;
    let twiddle = twiddles[clamp_index_dyn(twiddle_idx, arrayLength(&twiddles))];
    let b_twiddle = complex_multiply_fma(b, twiddle);
    
    // Write results with bounds checking
    output[clamp_index_dyn(idx_a, arrayLength(&output))] = a + b_twiddle;
    output[clamp_index_dyn(idx_b, arrayLength(&output))] = a - b_twiddle;
}

// Documentation:
// - Stage 0: butterflies span 2 elements (groups of 2)
// - Stage 1: butterflies span 4 elements (groups of 4)
// - Stage n: butterflies span 2^(n+1) elements
// Twiddle factors are precomputed and stored consecutively per stage
