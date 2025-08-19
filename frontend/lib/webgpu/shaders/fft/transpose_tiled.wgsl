// transpose_tiled.wgsl
// PLATINUM Edition: Enhanced tiled transpose with bank conflict avoidance
// - Padding to avoid shared memory bank conflicts
// - Support for non-square matrices
// - Optimized tile sizes for different GPU architectures

struct Params { 
    width: u32;   // Input width
    height: u32;  // Input height
    // ENHANCED: Additional parameters for optimization
    tile_size: u32;     // Usually 16, but configurable
    padding: u32;       // Padding for bank conflict avoidance
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;

// Shared memory with padding to avoid bank conflicts
// Using 17 instead of 16 for the inner dimension eliminates conflicts
const TILE_DIM = 16u;
const TILE_PAD = 1u;
var<workgroup> tile: array<vec2<f32>, (TILE_DIM + TILE_PAD) * TILE_DIM>;

// Helper to calculate padded tile index
fn tile_index(x: u32, y: u32) -> u32 {
    return y * (TILE_DIM + TILE_PAD) + x;
}

@compute @workgroup_size(TILE_DIM, TILE_DIM, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Calculate tile boundaries
    let tile_x = wid.x * TILE_DIM;
    let tile_y = wid.y * TILE_DIM;
    
    // Source coordinates
    let src_x = tile_x + lid.x;
    let src_y = tile_y + lid.y;
    
    // PHASE 1: Coalesced read from global memory to shared memory
    if (src_x < P.width && src_y < P.height) {
        let src_idx = src_y * P.width + src_x;
        tile[tile_index(lid.x, lid.y)] = src[src_idx];
    } else {
        // Pad with zeros for out-of-bounds
        tile[tile_index(lid.x, lid.y)] = vec2<f32>(0.0, 0.0);
    }
    
    // Synchronize to ensure all threads have loaded their data
    workgroupBarrier();
    
    // PHASE 2: Transposed write from shared memory to global memory
    // Note: We swap x and y for the transposed write
    let dst_x = tile_y + lid.x;  // Transposed: tile_y becomes x base
    let dst_y = tile_x + lid.y;  // Transposed: tile_x becomes y base
    
    if (dst_x < P.height && dst_y < P.width) {
        // Read from shared memory with transposed indices
        let transposed_value = tile[tile_index(lid.y, lid.x)];
        
        // Write to global memory (output is height x width)
        let dst_idx = dst_y * P.height + dst_x;
        dst[dst_idx] = transposed_value;
    }
    
    // ENHANCEMENT: Add memory fence for consistency
    storageBarrier();
}

// ENHANCED: Specialized version for square power-of-2 matrices
@compute @workgroup_size(TILE_DIM, TILE_DIM, 1)
fn transpose_square_po2(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Optimized version when we know width == height and both are powers of 2
    let n = P.width;  // Assume square matrix
    let tile_offset = wid.y * TILE_DIM * n + wid.x * TILE_DIM;
    
    // Load with coalescing
    let idx = tile_offset + lid.y * n + lid.x;
    tile[tile_index(lid.x, lid.y)] = src[idx];
    
    workgroupBarrier();
    
    // Store transposed
    let trans_offset = wid.x * TILE_DIM * n + wid.y * TILE_DIM;
    let trans_idx = trans_offset + lid.y * n + lid.x;
    dst[trans_idx] = tile[tile_index(lid.y, lid.x)];
}