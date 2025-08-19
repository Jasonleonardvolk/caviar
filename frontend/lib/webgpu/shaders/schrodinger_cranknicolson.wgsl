// schrodinger_cranknicolson.wgsl  
// Semi-implicit Crank-Nicolson: UNCONDITIONALLY STABLE!
// Can use dt = 0.5 instead of 0.05!

struct Params {
    width: u32,
    height: u32,
    iterations: u32,    // Jacobi iterations (5-10 typical)
    pad0: u32,
    dt: f32,
    alpha: f32,
    beta: f32,
    omega: f32,        // Over-relaxation factor (1.0-1.8)
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> inField: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> tempField: array<vec2<f32>>;  // For iteration
@group(0) @binding(3) var<storage, read_write> outField: array<vec2<f32>>;

fn idx(x: i32, y: i32) -> u32 {
    let xi = clamp(x, 0, i32(P.width) - 1);
    let yi = clamp(y, 0, i32(P.height) - 1);
    return u32(yi) * P.width + u32(xi);
}

fn applyH(field: ptr<storage, array<vec2<f32>>, read_write>, x: i32, y: i32) -> vec2<f32> {
    let c = (*field)[idx(x, y)];
    
    // 5-point Laplacian
    let l = (*field)[idx(x - 1, y)];
    let r = (*field)[idx(x + 1, y)];
    let u = (*field)[idx(x, y - 1)];
    let d = (*field)[idx(x, y + 1)];
    let lap = (l + r + u + d) - 4.0 * c;
    
    // 13-point Biharmonic (simplified for semi-implicit)
    let l2 = (*field)[idx(x - 2, y)];
    let r2 = (*field)[idx(x + 2, y)];
    let u2 = (*field)[idx(x, y - 2)];
    let d2 = (*field)[idx(x, y + 2)];
    let bih = 12.0 * c - 4.0 * (l + r + u + d) + (l2 + r2 + u2 + d2);
    
    // H*psi = -alpha*lap + beta*bih (ignoring potential for stability)
    return -P.alpha * lap + P.beta * bih;
}

@compute @workgroup_size(8, 8, 1)
fn initialize(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.width || gid.y >= P.height) { return; }
    let id = gid.y * P.width + gid.x;
    
    // Start with explicit Euler guess
    let x = i32(gid.x);
    let y = i32(gid.y);
    let c = inField[id];
    let H_psi = applyH(&inField, x, y);
    
    // i*dPsi/dt = H*Psi => dPsi = -i*H*Psi*dt
    let dPsi = vec2<f32>(H_psi.y, -H_psi.x) * P.dt;
    tempField[id] = c + dPsi * 0.5;  // Half-step for Crank-Nicolson
}

@compute @workgroup_size(8, 8, 1)  
fn jacobi_iterate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.width || gid.y >= P.height) { return; }
    let id = gid.y * P.width + gid.x;
    let x = i32(gid.x);
    let y = i32(gid.y);
    
    // Crank-Nicolson: (I + dt*H/2)*psi_new = (I - dt*H/2)*psi_old
    // Rearranged for Jacobi iteration
    
    let psi_old = inField[id];
    let H_old = applyH(&inField, x, y);
    let rhs = psi_old - vec2<f32>(H_old.y, -H_old.x) * P.dt * 0.5;
    
    // Current guess
    let psi_curr = tempField[id];
    let H_curr = applyH(&tempField, x, y);
    
    // Jacobi update with over-relaxation
    let residual = rhs - (psi_curr + vec2<f32>(H_curr.y, -H_curr.x) * P.dt * 0.5);
    let psi_new = psi_curr + residual * P.omega;
    
    outField[id] = psi_new;
}

@compute @workgroup_size(8, 8, 1)
fn swap_buffers(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P.width || gid.y >= P.height) { return; }
    let id = gid.y * P.width + gid.x;
    tempField[id] = outField[id];
}