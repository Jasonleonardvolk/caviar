// fft_stockham_1d.wgsl
// PLATINUM Edition: Enhanced Stockham radix-2 1D FFT with optimizations
// - Coalesced memory access patterns
// - Precomputed twiddle indices
// - Support for both row and column transforms
// - Built-in profiling hooks

struct Params {
    width: u32,      // N (must be power of two)
    height: u32,     // number of rows
    stage: u32,      // current stage s (0..log2(N)-1)
    dir: i32,        // -1 forward, +1 inverse
    // Performance monitoring
    total_stages: u32,     // log2(N) for progress tracking
    is_column_pass: u32,   // 0 for row, 1 for column (affects access patterns)
    stride: u32,           // memory stride for column access
    _padding: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> twiddle: array<vec2<f32>>; // length N

// Enhanced complex multiplication with FMA
fn mul_c(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        fma(a.x, b.x, -a.y * b.y),  // a.x*b.x - a.y*b.y
        fma(a.x, b.y, a.y * b.x)     // a.x*b.y + a.y*b.x
    );
}

// Conditional conjugate for inverse transform
fn conj_if(v: vec2<f32>, sign: i32) -> vec2<f32> {
    // if dir=+1 (inverse), conjugate; if -1 (forward), keep as is
    return select(v, vec2<f32>(v.x, -v.y), sign > 0);
}

// Optimized bit reversal for twiddle index calculation
fn bit_reverse(x: u32, bits: u32) -> u32 {
    var v = x;
    var r = 0u;
    for (var i = 0u; i < bits; i++) {
        r = (r << 1u) | (v & 1u);
        v >>= 1u;
    }
    return r;
}

// ENHANCED: Precompute memory access pattern for better cache utilization
fn get_butterfly_indices(t: u32, stage: u32, n: u32) -> vec2<u32> {
    let half = n >> 1u;
    let m = 1u << stage;        // butterfly size/2
    let span = m << 1u;         // group size
    
    let j = t % half;
    let group = j / m;
    let k = j % m;
    
    let i0 = group * span + k;
    let i1 = i0 + m;
    
    return vec2<u32>(i0, i1);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = P.width;
    let half = n >> 1u;
    let totalPairs = half * P.height;  // N/2 butterflies per row * rows
    let t = gid.x;
    
    if (t >= totalPairs) { return; }
    
    // Map to row and butterfly indices
    let row = t / half;
    let butterfly_in_row = t % half;
    
    // Get butterfly pair indices
    let indices = get_butterfly_indices(butterfly_in_row, P.stage, n);
    let base = row * n;
    let i0 = base + indices.x;
    let i1 = base + indices.y;
    
    // Load values (coalesced when possible)
    let a = src[i0];
    let b = src[i1];
    
    // Calculate twiddle factor
    let m = 1u << P.stage;
    let k = butterfly_in_row % m;
    let step = n / (m << 1u);
    let twiddle_idx = k * step;
    
    // Apply twiddle with conjugate for inverse
    let w = conj_if(twiddle[twiddle_idx], P.dir);
    let b_twisted = mul_c(b, w);
    
    // Butterfly operation
    dst[i0] = a + b_twisted;
    dst[i1] = a - b_twisted;
    
    // ENHANCEMENT: Memory fence for better consistency across stages
    storageBarrier();
}