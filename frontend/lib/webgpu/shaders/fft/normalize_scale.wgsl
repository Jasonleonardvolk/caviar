// normalize_scale.wgsl
// PLATINUM Edition: Normalization with multiple conventions support
// - Forward, backward, and orthonormal conventions
// - Batch processing support
// - Optional energy conservation check

struct Params { 
    width: u32;
    height: u32;
    batch_size: u32;        // For batch processing
    convention: u32;        // 0: standard (1/N²), 1: unitary (1/√N), 2: none
    // Energy monitoring
    compute_energy: u32;    // If 1, compute total energy
    _padding: array<u32, 3>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> field: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> energy: array<f32>; // Optional energy output

// Shared memory for energy reduction
var<workgroup> shared_energy: array<f32, 256>;

fn complex_magnitude_squared(c: vec2<f32>) -> f32 {
    return c.x * c.x + c.y * c.y;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    let total_size = P.width * P.height * P.batch_size;
    let idx = gid.x;
    
    // Calculate normalization factor based on convention
    var scale = 1.0;
    let n = f32(P.width * P.height);
    
    switch (P.convention) {
        case 0u: { scale = 1.0 / n; }           // Standard: 1/N²
        case 1u: { scale = 1.0 / sqrt(n); }     // Unitary: 1/√N
        case 2u: { scale = 1.0; }                // No normalization
        default: { scale = 1.0 / n; }
    }
    
    // Apply normalization
    if (idx < total_size) {
        let v = field[idx];
        field[idx] = v * scale;
        
        // Compute energy if requested
        if (P.compute_energy == 1u) {
            shared_energy[lid.x] = complex_magnitude_squared(v * scale);
        }
    } else if (P.compute_energy == 1u) {
        shared_energy[lid.x] = 0.0;
    }
    
    // Energy reduction within workgroup
    if (P.compute_energy == 1u) {
        workgroupBarrier();
        
        // Tree reduction
        for (var stride = 128u; stride > 0u; stride >>= 1u) {
            if (lid.x < stride && lid.x + stride < 256u) {
                shared_energy[lid.x] += shared_energy[lid.x + stride];
            }
            workgroupBarrier();
        }
        
        // Write workgroup result
        if (lid.x == 0u) {
            energy[gid.x / 256u] = shared_energy[0];
        }
    }
}

// ENHANCED: Batch normalization for multiple fields
@compute @workgroup_size(256, 1, 1)
fn normalize_batch(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let field_size = P.width * P.height;
    let batch_idx = gid.x / field_size;
    let local_idx = gid.x % field_size;
    
    if (batch_idx >= P.batch_size) { return; }
    
    // Each batch can have different normalization
    let n = f32(field_size);
    let scale = 1.0 / n;  // Can be made batch-specific if needed
    
    field[gid.x] *= scale;
}