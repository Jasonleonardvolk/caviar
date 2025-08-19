// roi_interest_map_enhanced.wgsl
// ENHANCED: Proper circular variance + shared memory for Sobel

struct Params {
    width: u32,
    height: u32,
    pad0: u32,
    pad1: u32,
    w_grad: f32,
    w_temp: f32,
    w_phase: f32,
    k_norm: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> currField: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> prevField: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> interest: array<f32>;

// Shared memory for 10x10 tile (8x8 + 1-pixel border)
var<workgroup> tile_curr: array<vec2<f32>, 100>;
var<workgroup> tile_prev: array<vec2<f32>, 100>;

fn idx(x: i32, y: i32) -> u32 {
    let xi = clamp(x, 0, i32(P.width) - 1);
    let yi = clamp(y, 0, i32(P.height) - 1);
    return u32(yi) * P.width + u32(xi);
}

fn loadTile(lx: u32, ly: u32, gx: i32, gy: i32) {
    let id = idx(gx, gy);
    let tile_idx = ly * 10u + lx;
    tile_curr[tile_idx] = currField[id];
    tile_prev[tile_idx] = prevField[id];
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    if (gid.x >= P.width || gid.y >= P.height) { return; }
    
    let gx = i32(gid.x);
    let gy = i32(gid.y);
    
    // Cooperative loading with 1-pixel halo
    let lx = lid.x + 1u;
    let ly = lid.y + 1u;
    
    // Load main pixel
    loadTile(lx, ly, gx, gy);
    
    // Load borders
    if (lid.x == 0u) {
        loadTile(0u, ly, gx - 1, gy);      // Left border
    }
    if (lid.x == 7u) {
        loadTile(9u, ly, gx + 1, gy);      // Right border
    }
    if (lid.y == 0u) {
        loadTile(lx, 0u, gx, gy - 1);      // Top border
    }
    if (lid.y == 7u) {
        loadTile(lx, 9u, gx, gy + 1);      // Bottom border
    }
    
    // Load corners
    if (lid.x == 0u && lid.y == 0u) loadTile(0u, 0u, gx - 1, gy - 1);
    if (lid.x == 7u && lid.y == 0u) loadTile(9u, 0u, gx + 1, gy - 1);
    if (lid.x == 0u && lid.y == 7u) loadTile(0u, 9u, gx - 1, gy + 1);
    if (lid.x == 7u && lid.y == 7u) loadTile(9u, 9u, gx + 1, gy + 1);
    
    workgroupBarrier();
    
    // Now compute using shared memory
    let c = tile_curr[ly * 10u + lx];
    let p = tile_prev[ly * 10u + lx];
    
    // Temporal delta (magnitude change)
    let temporal = abs(length(c) - length(p));
    
    // Sobel gradient on magnitude using shared memory
    var gx_sobel = 0.0;
    var gy_sobel = 0.0;
    
    for (var j = -1; j <= 1; j = j + 1) {
        for (var i = -1; i <= 1; i = i + 1) {
            let tile_idx = u32(i32(ly) + j) * 10u + u32(i32(lx) + i);
            let mag = length(tile_curr[tile_idx]);
            
            // Sobel X kernel: [-1 0 1; -2 0 2; -1 0 1]
            let wx = f32(i) * (1.0 + f32(abs(j) == 0));
            gx_sobel += mag * wx;
            
            // Sobel Y kernel: [-1 -2 -1; 0 0 0; 1 2 1]
            let wy = f32(j) * (1.0 + f32(abs(i) == 0));
            gy_sobel += mag * wy;
        }
    }
    
    let gradient = sqrt(gx_sobel * gx_sobel + gy_sobel * gy_sobel);
    
    // ENHANCED: Circular variance using von Mises distribution
    // Compute mean direction and concentration parameter
    var sum_cos = 0.0;
    var sum_sin = 0.0;
    var sum_mag = 0.0;
    
    for (var j = -1; j <= 1; j = j + 1) {
        for (var i = -1; i <= 1; i = i + 1) {
            let tile_idx = u32(i32(ly) + j) * 10u + u32(i32(lx) + i);
            let z = tile_curr[tile_idx];
            let mag = length(z);
            if (mag > 1e-6) {
                let phase = atan2(z.y, z.x);
                sum_cos += cos(phase) * mag;
                sum_sin += sin(phase) * mag;
                sum_mag += mag;
            }
        }
    }
    
    // Mean resultant length (0 = high variance, 1 = low variance)
    let R = sqrt(sum_cos * sum_cos + sum_sin * sum_sin) / max(sum_mag, 1e-6);
    
    // Circular variance: V = 1 - R
    // But we want high values for high variance, so use 1 - R
    let phasevar = 1.0 - R;
    
    // Angular deviation bonus: penalize rapid phase changes
    let center_phase = atan2(c.y, c.x);
    var phase_dev = 0.0;
    for (var j = -1; j <= 1; j = j + 1) {
        for (var i = -1; i <= 1; i = i + 1) {
            if (i == 0 && j == 0) { continue; }
            let tile_idx = u32(i32(ly) + j) * 10u + u32(i32(lx) + i);
            let z = tile_curr[tile_idx];
            if (length(z) > 1e-6) {
                let neighbor_phase = atan2(z.y, z.x);
                var diff = neighbor_phase - center_phase;
                // Wrap to [-π, π]
                diff = diff - round(diff / (2.0 * 3.14159265)) * 2.0 * 3.14159265;
                phase_dev += abs(diff);
            }
        }
    }
    phase_dev /= 8.0;  // Average over 8 neighbors
    
    // Combine phase variance and deviation
    let phase_metric = phasevar * 0.7 + (phase_dev / 3.14159265) * 0.3;
    
    // Soft normalization
    let gN = gradient / (gradient + P.k_norm);
    let tN = temporal / (temporal + P.k_norm);
    
    // Final interest score
    let id = gid.y * P.width + gid.x;
    interest[id] = P.w_grad * gN + P.w_temp * tN + P.w_phase * phase_metric;
}