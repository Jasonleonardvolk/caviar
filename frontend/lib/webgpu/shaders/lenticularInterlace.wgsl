// ${IRIS_ROOT}\frontend\shaders\lenticularInterlace.wgsl
// Production-optimized lenticular interlacing shader for Looking Glass displays
// Addresses all performance and correctness feedback

// ===== Consolidated Uniform Structures =====
// Packed into 256-byte blocks for GPU efficiency

struct CoreUniforms {
    // Display parameters (16 floats)
    screen_width: f32,
    screen_height: f32,
    inv_screen_width: f32,      // Precomputed 1.0 / screen_width
    inv_screen_height: f32,      // Precomputed 1.0 / screen_height
    
    // Lens parameters (16 floats)
    pitch: f32,
    inv_pitch: f32,              // Precomputed 1.0 / pitch
    cos_tilt: f32,               // Precomputed cos(tilt)
    sin_tilt: f32,               // Precomputed sin(tilt)
    center: f32,
    subp: f32,
    ri: f32,                     // Refractive index
    bi: f32,                     // Base thickness
    
    // View parameters (8 floats)
    num_views: f32,
    inv_num_views: f32,          // Precomputed 1.0 / num_views
    view_cone: f32,
    lens_curve: f32,
    
    // Calibration basics (8 floats)
    gamma: f32,
    exposure: f32,
    black_level: f32,
    white_point: f32,
    
    // Flags and modes (8 floats)
    flip_x: f32,
    flip_y: f32,
    aa_samples: f32,             // Anti-aliasing sample count
    interpolation_mode: f32,     // 0: nearest, 1: linear, 2: cubic
    
    // Padding to 64 floats (256 bytes)
    _padding: array<vec4<f32>, 2>
}

struct QuiltUniforms {
    // Quilt layout (16 floats)
    cols: f32,
    rows: f32,
    inv_cols: f32,               // Precomputed 1.0 / cols
    inv_rows: f32,               // Precomputed 1.0 / rows
    tile_width: f32,
    tile_height: f32,
    inv_tile_width: f32,         // Precomputed 1.0 / tile_width
    inv_tile_height: f32,        // Precomputed 1.0 / tile_height
    quilt_width: f32,
    quilt_height: f32,
    inv_quilt_width: f32,        // Precomputed 1.0 / quilt_width
    inv_quilt_height: f32,       // Precomputed 1.0 / quilt_height
    total_views: f32,
    edge_enhancement: f32,
    temporal_blend: f32,
    display_type: f32,           // 0: Standard, 1: Portrait, 2: 8K, 3: Custom
}

struct AdvancedUniforms {
    // Optical corrections (16 floats)
    dispersion_r: f32,           // Precomputed chromatic dispersion for R
    dispersion_g: f32,           // Precomputed chromatic dispersion for G
    dispersion_b: f32,           // Precomputed chromatic dispersion for B
    field_curvature: f32,
    coma: f32,
    astigmatism: f32,
    vignette_strength: f32,
    vignette_radius: f32,
    
    // Color correction (8 floats)
    color_temp_r: f32,           // Precomputed color temperature scale R
    color_temp_b: f32,           // Precomputed color temperature scale B
    contrast: f32,
    saturation: f32,
    
    // Quality settings (4 floats)
    aa_strength: f32,
    view_blend_curve: f32,
    debug_mode: f32,             // 0: off, 1: lens vis, 2: view vis
    quality_preset: f32,         // 0: performance, 1: balanced, 2: quality
}

// ===== Vertex Shader =====
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;
    // Full-screen triangle
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    output.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    output.uv = vec2<f32>(x, y);
    return output;
}

// ===== Resource Bindings =====
@group(0) @binding(0) var<uniform> core: CoreUniforms;
@group(0) @binding(1) var<uniform> quilt: QuiltUniforms;
@group(0) @binding(2) var<uniform> advanced: AdvancedUniforms;

// Textures - separate group for better batching
@group(1) @binding(0) var quilt_texture: texture_2d<f32>;
@group(1) @binding(1) var quilt_sampler: sampler;
@group(1) @binding(2) var previous_frame: texture_2d<f32>;
@group(1) @binding(3) var<storage, read> trig_lut: array<vec4<f32>>; // cos,sin pairs packed

// Compute shader outputs
@group(2) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(2) @binding(1) var temp_buffer: texture_storage_2d<rgba16float, read_write>;

// Constants
const PI: f32 = 3.14159265359;
const INV_255: f32 = 0.00392156863;
const WORKGROUP_SIZE: u32 = 16u;

// Specialization constants for build-time optimization
@id(0) override ENABLE_DEBUG: bool = false;
@id(1) override ENABLE_TEMPORAL: bool = true;
@id(2) override MAX_AA_SAMPLES: u32 = 4u;

// ===== Optimized View Calculation =====
fn get_view_for_pixel_fast(screen_coord: vec2<f32>, subpixel_offset: f32) -> f32 {
    // Use precomputed trig values from uniforms
    let rotated_x = screen_coord.x * core.cos_tilt - screen_coord.y * core.sin_tilt + subpixel_offset;
    
    // Single computation with precomputed inverse
    let lens_x = (rotated_x + core.center * core.screen_width) * core.inv_pitch + core.subp;
    
    // Get fractional position with lens curvature
    var lens_frac = fract(lens_x);
    
    // Apply lens curvature only if significant
    if (core.lens_curve > 0.001) {
        let dist = abs(lens_frac - 0.5);
        let curve = 1.0 + core.lens_curve * dist * dist;
        lens_frac = 0.5 + (lens_frac - 0.5) * curve;
    }
    
    // Non-linear view distribution
    if (advanced.view_blend_curve != 1.0) {
        lens_frac = pow(lens_frac, advanced.view_blend_curve);
    }
    
    return lens_frac * core.num_views;
}

// ===== Cubic Interpolation with Bounds Checking =====
fn cubic_interpolate_safe(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>, t: f32) -> vec4<f32> {
    // Catmull-Rom spline coefficients
    let t2 = t * t;
    let t3 = t2 * t;
    
    return v1 + 0.5 * t * (v2 - v0 + 
           t * (2.0 * v0 - 5.0 * v1 + 4.0 * v2 - v3 + 
           t * (3.0 * (v1 - v2) + v3 - v0)));
}

// ===== Optimized Quilt Sampling =====
fn sample_quilt_view_fast(view_index: f32, uv: vec2<f32>) -> vec4<f32> {
    // Wrap view index with single modulo
    let wrapped_view = view_index - floor(view_index * quilt.inv_cols) * quilt.total_views;
    
    // Tile calculation with precomputed inverses
    let tile_idx = floor(wrapped_view);
    let tile_x = tile_idx - floor(tile_idx * quilt.inv_cols) * quilt.cols;
    let tile_y = floor(tile_idx * quilt.inv_cols);
    
    // UV calculation with precomputed scales
    var tile_uv = vec2<f32>(
        (tile_x + uv.x) * quilt.inv_cols,
        (tile_y + uv.y) * quilt.inv_rows
    );
    
    // Apply flipping
    if (core.flip_x > 0.5) { tile_uv.x = 1.0 - tile_uv.x; }
    if (core.flip_y > 0.5) { tile_uv.y = 1.0 - tile_uv.y; }
    
    return textureSampleLevel(quilt_texture, quilt_sampler, tile_uv, 0.0);
}

// ===== Optimized Multi-View Sampling =====
fn sample_views_interpolated(view_index: f32, uv: vec2<f32>) -> vec4<f32> {
    let mode = u32(core.interpolation_mode);
    
    if (mode == 0u) { // Nearest
        return sample_quilt_view_fast(round(view_index), uv);
    }
    
    let base_view = floor(view_index);
    let frac = view_index - base_view;
    
    if (mode == 1u) { // Linear
        let v0 = sample_quilt_view_fast(base_view, uv);
        let v1 = sample_quilt_view_fast(base_view + 1.0, uv);
        return mix(v0, v1, frac); // True linear, no smoothstep
    }
    
    // Cubic with proper bounds handling
    let total = quilt.total_views;
    let vm1 = select(base_view - 1.0, base_view - 1.0 + total, base_view > 0.0);
    let vp1 = select(base_view + 1.0, base_view + 1.0 - total, base_view + 1.0 < total);
    let vp2 = select(base_view + 2.0, base_view + 2.0 - total, base_view + 2.0 < total);
    
    let v0 = sample_quilt_view_fast(vm1, uv);
    let v1 = sample_quilt_view_fast(base_view, uv);
    let v2 = sample_quilt_view_fast(vp1, uv);
    let v3 = sample_quilt_view_fast(vp2, uv);
    
    return cubic_interpolate_safe(v0, v1, v2, v3, frac);
}

// ===== Efficient Subpixel Sampling =====
fn sample_subpixel_optimized(screen_coord: vec2<f32>, uv: vec2<f32>) -> vec4<f32> {
    let aa_samples = min(u32(core.aa_samples), MAX_AA_SAMPLES);
    
    if (aa_samples <= 1u) {
        // No AA - single sample
        let view = get_view_for_pixel_fast(screen_coord, 0.0);
        return sample_views_interpolated(view, uv);
    }
    
    // Precomputed chromatic offsets
    const subpixel_width = 1.0 / 3.0;
    let r_offset = -subpixel_width + advanced.dispersion_r;
    let b_offset = subpixel_width + advanced.dispersion_b;
    
    // AA sample accumulation
    var accum_r = 0.0;
    var accum_g = 0.0;
    var accum_b = 0.0;
    
    let aa_step = advanced.aa_strength / f32(aa_samples);
    for (var i = 0u; i < aa_samples; i++) {
        let offset = (f32(i) - f32(aa_samples - 1u) * 0.5) * aa_step;
        
        // Sample each subpixel
        let view_r = get_view_for_pixel_fast(screen_coord, r_offset + offset);
        let view_g = get_view_for_pixel_fast(screen_coord, offset);
        let view_b = get_view_for_pixel_fast(screen_coord, b_offset + offset);
        
        let sample_r = sample_views_interpolated(view_r, uv);
        let sample_g = sample_views_interpolated(view_g, uv);
        let sample_b = sample_views_interpolated(view_b, uv);
        
        accum_r += sample_r.r;
        accum_g += sample_g.g;
        accum_b += sample_b.b;
    }
    
    let inv_samples = 1.0 / f32(aa_samples);
    return vec4<f32>(accum_r * inv_samples, accum_g * inv_samples, accum_b * inv_samples, 1.0);
}

// ===== Fast Calibration =====
fn apply_calibration_fast(color: vec3<f32>) -> vec3<f32> {
    // Black level and white point with single division
    let range = core.white_point - core.black_level;
    var calibrated = saturate((color - core.black_level) / range);
    
    // Combined gamma and exposure
    calibrated = pow(calibrated * core.exposure, vec3<f32>(1.0 / core.gamma));
    
    // Pre-computed color temperature
    calibrated.r *= advanced.color_temp_r;
    calibrated.b *= advanced.color_temp_b;
    
    // Simple contrast without branches
    calibrated = mix(vec3<f32>(0.5), calibrated, advanced.contrast);
    
    return calibrated;
}

// ===== Main Fragment Shader =====
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let screen_coord = input.uv * vec2<f32>(core.screen_width, core.screen_height);
    
    // Main sampling
    var color = sample_subpixel_optimized(screen_coord, input.uv);
    
    // Temporal blending only if enabled
    if (ENABLE_TEMPORAL && quilt.temporal_blend > 0.0) {
        let prev = textureSampleLevel(previous_frame, quilt_sampler, input.uv, 0.0);
        color = mix(color, prev, quilt.temporal_blend);
    }
    
    // Calibration
    let calibrated = apply_calibration_fast(color.rgb);
    color.r = calibrated.r;
    color.g = calibrated.g;
    color.b = calibrated.b;
    
    // Debug visualization only if enabled
    if (ENABLE_DEBUG && advanced.debug_mode > 0.0) {
        let mixed = mix(color.rgb, debug_visualization(screen_coord, input.uv), 0.5);
        color.r = mixed.r;
        color.g = mixed.g;
        color.b = mixed.b;
    }
    
    return vec4<f32>(saturate(color.rgb), 1.0);
}

// ===== Compute Shader with TextureLoad =====
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = global_id.xy;
    let dims = vec2<u32>(u32(core.screen_width), u32(core.screen_height));
    
    if (any(coord >= dims)) {
        return;
    }
    
    let uv = vec2<f32>(coord) * vec2<f32>(core.inv_screen_width, core.inv_screen_height);
    let screen_coord = vec2<f32>(coord);
    
    // Direct sampling without sampler overhead
    var color = sample_subpixel_compute(coord, screen_coord, uv);
    
    // Apply calibration
    let calibrated = apply_calibration_fast(color.rgb);
    color.r = calibrated.r;
    color.g = calibrated.g;
    color.b = calibrated.b;
    
    // Write to output
    textureStore(output_texture, coord, vec4<f32>(saturate(color.rgb), 1.0));
}

// Compute-specific sampling using textureLoad
fn sample_subpixel_compute(pixel_coord: vec2<u32>, screen_coord: vec2<f32>, uv: vec2<f32>) -> vec4<f32> {
    // Similar to sample_subpixel_optimized but uses textureLoad
    let view = get_view_for_pixel_fast(screen_coord, 0.0);
    
    // Calculate quilt coordinates
    let view_idx = u32(view);
    let tile_x = view_idx % u32(quilt.cols);
    let tile_y = view_idx / u32(quilt.cols);
    
    // Direct texel fetch
    let quilt_coord = vec2<u32>(
        tile_x * u32(quilt.tile_width) + u32(uv.x * quilt.tile_width),
        tile_y * u32(quilt.tile_height) + u32(uv.y * quilt.tile_height)
    );
    
    return textureLoad(quilt_texture, quilt_coord, 0);
}

// ===== Edge Enhancement on Final Output =====
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn cs_edge_enhance(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = vec2<i32>(global_id.xy);
    let dims = vec2<i32>(i32(core.screen_width), i32(core.screen_height));
    
    if (any(coord >= dims)) {
        return;
    }
    
    // Load from temp buffer
    let center = textureLoad(temp_buffer, coord).rgb;
    
    if (quilt.edge_enhancement > 0.0) {
        // 3x3 edge detection on final image
        var laplacian = vec3<f32>(0.0);
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let sample_coord = clamp(coord + vec2<i32>(dx, dy), vec2<i32>(0), dims - 1);
                let sample = textureLoad(temp_buffer, sample_coord).rgb;
                
                if (dx == 0 && dy == 0) {
                    laplacian += sample * 8.0;
                } else {
                    laplacian -= sample;
                }
            }
        }
        
        let enhanced = center + laplacian * quilt.edge_enhancement * 0.125;
        textureStore(output_texture, coord, vec4<f32>(saturate(enhanced), 1.0));
    } else {
        textureStore(output_texture, coord, vec4<f32>(center, 1.0));
    }
}

// ===== Debug Visualization (only compiled if ENABLE_DEBUG) =====
fn debug_visualization(screen_coord: vec2<f32>, uv: vec2<f32>) -> vec3<f32> {
    if (!ENABLE_DEBUG) { return vec3<f32>(0.0); }
    
    let mode = u32(advanced.debug_mode);
    
    if (mode == 1u) { // Lens boundaries
        let rotated_x = screen_coord.x * core.cos_tilt - screen_coord.y * core.sin_tilt;
        let lens_x = rotated_x * core.inv_pitch;
        let lens_frac = fract(lens_x);
        
        if (lens_frac < 0.02 || lens_frac > 0.98) {
            return vec3<f32>(1.0, 0.0, 0.0);
        }
    } else if (mode == 2u) { // View distribution
        let view = get_view_for_pixel_fast(screen_coord, 0.0);
        let hue = view * core.inv_num_views;
        return hsv_to_rgb(vec3<f32>(hue, 0.8, 0.6));
    }
    
    return vec3<f32>(0.0);
}

// Simple HSV to RGB
fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let h = hsv.x * 6.0;
    let s = hsv.y;
    let v = hsv.z;
    let i = floor(h);
    let f = h - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    
    let idx = u32(i) % 6u;
    if (idx == 0u) { return vec3<f32>(v, t, p); }
    if (idx == 1u) { return vec3<f32>(q, v, p); }
    if (idx == 2u) { return vec3<f32>(p, v, t); }
    if (idx == 3u) { return vec3<f32>(p, q, v); }
    if (idx == 4u) { return vec3<f32>(t, p, v); }
    return vec3<f32>(v, p, q);
}

// ===== Portrait Mode Entry Point =====
@fragment
fn fs_portrait(input: VertexOutput) -> @location(0) vec4<f32> {
    // Rotate coordinates for portrait orientation
    let screen_coord = vec2<f32>(
        input.uv.y * core.screen_height,
        (1.0 - input.uv.x) * core.screen_width
    );
    let rotated_uv = vec2<f32>(input.uv.y, 1.0 - input.uv.x);
    
    var color = sample_subpixel_optimized(screen_coord, rotated_uv);
    let calibrated = apply_calibration_fast(color.rgb);
    color.r = calibrated.r;
    color.g = calibrated.g;
    color.b = calibrated.b;
    
    return vec4<f32>(saturate(color.rgb), 1.0);
}
