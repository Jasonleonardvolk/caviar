// transpose.wgsl
// Matrix transpose for 2D FFT using shared memory tiles with bank conflict avoidance
// Optimized for coalesced memory access patterns

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

// Tile configuration
const TILE_SIZE: u32 = 16u;
// Add padding to avoid bank conflicts on shared memory
const TILE_DIM: u32 = TILE_SIZE + 1u;
const TILE_ARRAY_SIZE: u32 = TILE_DIM * TILE_SIZE;

// Shared memory with padding to avoid bank conflicts
var<workgroup> tile: array<vec2<f32>, TILE_ARRAY_SIZE>;


fn clamp_index_dyn(i: u32, len: u32) -> u32 {
    return select(i, len - 1u, i >= len);
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let size = uniforms.size;
    let batch_idx = workgroup_id.z;
    let batch_offset = batch_idx * size * size;
    
    // Calculate tile coordinates
    let tile_x = workgroup_id.x * TILE_SIZE;
    let tile_y = workgroup_id.y * TILE_SIZE;
    
    // Global coordinates for this thread
    let global_x = tile_x + local_id.x;
    let global_y = tile_y + local_id.y;
    
    // Load tile into shared memory with bounds checking
    // Use padded indexing to avoid bank conflicts
    let local_idx = local_id.y * TILE_DIM + local_id.x;
    
    if (global_x < size && global_y < size && local_idx < TILE_ARRAY_SIZE) {
        let idx = batch_offset + global_y * size + global_x;
        tile[local_idx] = input[clamp_index_dyn(idx, arrayLength(&input))];
    } else if (local_idx < TILE_ARRAY_SIZE) {
        // Initialize out-of-bounds elements to zero
        tile[local_idx] = vec2<f32>(0.0, 0.0);
    }
    
    workgroupBarrier();
    
    // Calculate transposed coordinates
    let transposed_x = tile_y + local_id.x;
    let transposed_y = tile_x + local_id.y;
    
    // Write transposed tile with bounds checking
    if (transposed_x < size && transposed_y < size) {
        let transposed_idx = batch_offset + transposed_y * size + transposed_x;
        // Read from shared memory with transposed local indices
        // Note the swapped indices for transpose
        let transposed_local = local_id.x * TILE_DIM + local_id.y;
        if (transposed_local < TILE_ARRAY_SIZE) {
            output[clamp_index_dyn(transposed_idx, arrayLength(&output))] = tile[transposed_local];
        }
    }
}

// Alternative implementation for non-square matrices or different tile sizes
@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn transpose_rect(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    // For rectangular matrices, adjust tile loading strategy
    let width = uniforms.size;  // Assume width
    let height = uniforms.size; // Could be different for rect matrices
    let batch_idx = workgroup_id.z;
    let batch_offset = batch_idx * width * height;
    
    // Load phase - coalesced reads
    let tile_x = workgroup_id.x * TILE_SIZE;
    let tile_y = workgroup_id.y * TILE_SIZE;
    
    for (var i = 0u; i < TILE_SIZE; i += 1u) {
        let global_x = tile_x + local_id.x;
        let global_y = tile_y + i;
        
        if (global_x < width && global_y < height) {
            let idx = batch_offset + global_y * width + global_x;
            let shared_idx = i * TILE_DIM + local_id.x;
            if (shared_idx < TILE_ARRAY_SIZE) {
                tile[shared_idx] = input[clamp_index_dyn(idx, arrayLength(&input))];
            }
        }
    }
    
    workgroupBarrier();
    
    // Store phase - coalesced writes
    for (var i = 0u; i < TILE_SIZE; i += 1u) {
        let transposed_x = tile_y + i;
        let transposed_y = tile_x + local_id.x;
        
        if (transposed_x < height && transposed_y < width) {
            let transposed_idx = batch_offset + transposed_y * height + transposed_x;
            let shared_idx = local_id.x * TILE_DIM + i;
            if (shared_idx < TILE_ARRAY_SIZE) {
                output[clamp_index_dyn(transposed_idx, arrayLength(&output))] = tile[shared_idx];
            }
        }
    }
}

// In-place transpose for square matrices (requires careful synchronization)
@compute @workgroup_size(TILE_SIZE, 1, 1)
fn transpose_inplace(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let size = uniforms.size;
    let batch_idx = global_id.z;
    let batch_offset = batch_idx * size * size;
    
    // Process upper triangle only to avoid double swapping
    let row = global_id.x;
    if (row >= size) { return; }
    
    for (var col = row + 1u; col < size; col++) {
        let idx1 = batch_offset + row * size + col;
        let idx2 = batch_offset + col * size + row;
        
        // Swap elements with bounds checking
        let clamped_idx1 = clamp_index_dyn(idx1, arrayLength(&input));
        let clamped_idx2 = clamp_index_dyn(idx2, arrayLength(&input));
        let temp = input[clamped_idx1];
        output[clamp_index_dyn(idx1, arrayLength(&output))] = input[clamped_idx2];
        output[clamp_index_dyn(idx2, arrayLength(&output))] = temp;
    }
}

// Documentation:
// Bank conflicts occur when multiple threads in a warp access the same
// memory bank simultaneously. Adding padding (TILE_DIM = TILE_SIZE + 1)
// ensures that consecutive threads access different banks.
//
// Performance tips:
// - TILE_SIZE should be a multiple of warp size (typically 32)
// - Smaller tiles (16x16) often perform better than larger ones
// - Padding eliminates bank conflicts at the cost of slightly more shared memory
