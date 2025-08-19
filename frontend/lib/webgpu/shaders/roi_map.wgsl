// roi_map.wgsl
// Generates Region of Interest map for selective tile offloading
// Combines gradient (Sobel), temporal delta, and phase variance metrics

struct ROIParams {
    width: u32,
    height: u32,
    tile_size: u32,        // 2, 4, or 8 for reduction
    temporal_weight: f32,  // Weight for temporal change
    gradient_weight: f32,  // Weight for spatial gradient
    phase_weight: f32,     // Weight for phase variance
    threshold: f32,        // Energy threshold for "interesting"
}

@group(0) @binding(0) var<uniform> params: ROIParams;
@group(0) @binding(1) var<storage, read> currentField: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> previousField: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> roiMap: array<f32>; // Energy per tile

// Sobel kernels for gradient
const SOBEL_X = array<f32, 9>(-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0);
const SOBEL_Y = array<f32, 9>(-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0);

fn sample_field(field: ptr<storage, array<vec2<f32>>, read>, x: i32, y: i32) -> vec2<f32> {
    let xi = clamp(x, 0, i32(params.width) - 1);
    let yi = clamp(y, 0, i32(params.height) - 1);
    let idx = u32(yi) * params.width + u32(xi);
    if (idx < arrayLength(field)) {
        return (*field)[idx];
    }
    return vec2<f32>(0.0, 0.0);
}

fn compute_gradient_energy(x: i32, y: i32) -> f32 {
    var gx = 0.0;
    var gy = 0.0;
    var kernel_idx = 0;
    
    // Apply Sobel operator
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let field = sample_field(&currentField, x + dx, y + dy);
            let mag = length(field);
            gx += mag * SOBEL_X[kernel_idx];
            gy += mag * SOBEL_Y[kernel_idx];
            kernel_idx++;
        }
    }
    
    return sqrt(gx * gx + gy * gy);
}

fn compute_temporal_delta(idx: u32) -> f32 {
    if (idx >= arrayLength(&currentField) || idx >= arrayLength(&previousField)) {
        return 0.0;
    }
    let curr = currentField[idx];
    let prev = previousField[idx];
    return length(curr - prev);
}

fn compute_phase_variance(x: i32, y: i32, tile_size: i32) -> f32 {
    var mean_phase = 0.0;
    var count = 0.0;
    
    // First pass: compute mean phase
    for (var dy = 0; dy < tile_size; dy++) {
        for (var dx = 0; dx < tile_size; dx++) {
            let field = sample_field(&currentField, x + dx, y + dy);
            mean_phase += atan2(field.y, field.x);
            count += 1.0;
        }
    }
    mean_phase /= count;
    
    // Second pass: compute variance
    var variance = 0.0;
    for (var dy = 0; dy < tile_size; dy++) {
        for (var dx = 0; dx < tile_size; dx++) {
            let field = sample_field(&currentField, x + dx, y + dy);
            let phase = atan2(field.y, field.x);
            let diff = phase - mean_phase;
            variance += diff * diff;
        }
    }
    
    return sqrt(variance / count);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tiles_x = params.width / params.tile_size;
    let tiles_y = params.height / params.tile_size;
    
    if (gid.x >= tiles_x || gid.y >= tiles_y) {
        return;
    }
    
    let tile_x = i32(gid.x * params.tile_size);
    let tile_y = i32(gid.y * params.tile_size);
    
    var total_energy = 0.0;
    
    // Accumulate energy metrics over the tile
    for (var dy = 0; dy < i32(params.tile_size); dy++) {
        for (var dx = 0; dx < i32(params.tile_size); dx++) {
            let x = tile_x + dx;
            let y = tile_y + dy;
            let idx = u32(y) * params.width + u32(x);
            
            // Gradient energy (Sobel)
            let gradient = compute_gradient_energy(x, y);
            
            // Temporal delta
            let temporal = compute_temporal_delta(idx);
            
            // Combine metrics
            let pixel_energy = gradient * params.gradient_weight + 
                              temporal * params.temporal_weight;
            total_energy += pixel_energy;
        }
    }
    
    // Add phase variance for the whole tile
    let phase_var = compute_phase_variance(tile_x, tile_y, i32(params.tile_size));
    total_energy += phase_var * params.phase_weight;
    
    // Normalize by tile area
    total_energy /= f32(params.tile_size * params.tile_size);
    
    //