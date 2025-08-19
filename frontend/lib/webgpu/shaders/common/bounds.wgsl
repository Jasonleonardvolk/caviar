// bounds.wgsl - Helper functions for safe array indexing
// Reduces dynamic array access warnings and prevents out-of-bounds errors

// Clamp index to valid range [0, n-1] without branching
fn clamp_index(i: u32, n: u32) -> u32 {
    return select(n - 1u, i, i < n); // min(i, n-1) without branching
}

// Clamp signed index to valid range
fn clamp_index_i32(i: i32, n: u32) -> u32 {
    let ui = max(0, i);
    return clamp_index(u32(ui), n);
}

// Safe 2D indexing
fn idx_2d_safe(x: u32, y: u32, width: u32, height: u32) -> u32 {
    let cx = clamp_index(x, width);
    let cy = clamp_index(y, height);
    return cy * width + cx;
}

// Check if index is in bounds
fn in_bounds(i: u32, n: u32) -> bool {
    return i < n;
}

// Check if 2D coordinate is in bounds
fn in_bounds_2d(x: u32, y: u32, width: u32, height: u32) -> bool {
    return x < width && y < height;
}
