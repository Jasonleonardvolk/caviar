/**
 * shaderSources.ts
 * 
 * Temporary shader source exports for SLMEncoderPipeline
 * This bridges the gap until the shader bundler runs
 */

// Phase-only encoding shader
export const phase_only_encode_wgsl = `
struct EncodeParams {
  width: u32,
  height: u32,
  amplitude_scale: f32,
  phase_scale: f32,
}

@group(0) @binding(0) var<storage, read> amplitude: array<f32>;
@group(0) @binding(1) var<storage, read> phase: array<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params: EncodeParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let x = global_id.x;
  let y = global_id.y;
  
  if (x >= params.width || y >= params.height) {
    return;
  }
  
  let idx = y * params.width + x;
  let amp = amplitude[idx] * params.amplitude_scale;
  let phi = phase[idx] * params.phase_scale;
  
  // Encode phase as grayscale (0-1 range)
  let gray = (phi + 3.14159265359) / (2.0 * 3.14159265359);
  
  // Output as RGBA (use R channel for phase, A for amplitude)
  let color = vec4<f32>(gray, gray, gray, amp);
  textureStore(output_texture, vec2<i32>(i32(x), i32(y)), color);
}
`;

// Lee hologram off-axis encoding shader
export const lee_offaxis_encode_wgsl = `
struct EncodeParams {
  width: u32,
  height: u32,
  carrier_frequency_x: f32,
  carrier_frequency_y: f32,
  amplitude_scale: f32,
  dc_term: f32,
}

@group(0) @binding(0) var<storage, read> amplitude: array<f32>;
@group(0) @binding(1) var<storage, read> phase: array<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params: EncodeParams;

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let x = global_id.x;
  let y = global_id.y;
  
  if (x >= params.width || y >= params.height) {
    return;
  }
  
  let idx = y * params.width + x;
  let amp = amplitude[idx] * params.amplitude_scale;
  let phi = phase[idx];
  
  // Calculate carrier wave phase
  let carrier_phase = TWO_PI * (
    params.carrier_frequency_x * f32(x) / f32(params.width) +
    params.carrier_frequency_y * f32(y) / f32(params.height)
  );
  
  // Lee hologram encoding: I = dc + amp * cos(phi + carrier_phase)
  let intensity = params.dc_term + amp * cos(phi + carrier_phase);
  
  // Normalize to 0-1 range
  let normalized = clamp(intensity, 0.0, 1.0);
  
  // Output as grayscale
  let color = vec4<f32>(normalized, normalized, normalized, 1.0);
  textureStore(output_texture, vec2<i32>(i32(x), i32(y)), color);
}
`;

// Propagation shader (simplified version)
export const propagation_wgsl = `
struct PropagationParams {
  distance: f32,
  wavelength: f32,
  pixel_size: f32,
  width: u32,
  height: u32,
}

@group(0) @binding(0) var<uniform> params: PropagationParams;
@group(0) @binding(1) var input_field: texture_2d<f32>;
@group(0) @binding(2) var output_field: texture_storage_2d<rg32float, write>;

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

@compute @workgroup_size(8, 8)
fn angular_spectrum_propagate(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let x = global_id.x;
  let y = global_id.y;
  
  if (x >= params.width || y >= params.height) {
    return;
  }
  
  // Load complex field (real, imaginary)
  let field = textureLoad(input_field, vec2<i32>(i32(x), i32(y)), 0);
  
  // Calculate spatial frequencies
  let fx = f32(x) / f32(params.width) - 0.5;
  let fy = f32(y) / f32(params.height) - 0.5;
  
  // Wave number
  let k = TWO_PI / params.wavelength;
  
  // Transfer function for angular spectrum propagation
  let H_arg = params.distance * sqrt(k * k - fx * fx - fy * fy);
  let H_real = cos(H_arg);
  let H_imag = sin(H_arg);
  
  // Complex multiplication: field * H
  let out_real = field.r * H_real - field.g * H_imag;
  let out_imag = field.r * H_imag + field.g * H_real;
  
  // Store propagated field
  textureStore(output_field, vec2<i32>(i32(x), i32(y)), vec4<f32>(out_real, out_imag, 0.0, 1.0));
}
`;

// Multi-view synthesis shader
export const multiViewSynthesis_wgsl = `
struct ViewParams {
  num_views: u32,
  current_view: u32,
  tile_width: u32,
  tile_height: u32,
  view_cone: f32,
  convergence_distance: f32,
}

@group(0) @binding(0) var<uniform> params: ViewParams;
@group(0) @binding(1) var wavefield: texture_2d<f32>;
@group(0) @binding(2) var quilt_output: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn synthesize_view(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let x = global_id.x;
  let y = global_id.y;
  
  if (x >= params.tile_width || y >= params.tile_height) {
    return;
  }
  
  // Calculate view angle
  let view_angle = (f32(params.current_view) / f32(params.num_views - 1u) - 0.5) * params.view_cone;
  
  // Apply parallax shift based on view angle
  let shift_x = tan(view_angle) * params.convergence_distance;
  let sample_x = i32(f32(x) + shift_x);
  
  // Sample the wavefield with shift
  let field = textureLoad(wavefield, vec2<i32>(sample_x, i32(y)), 0);
  
  // Convert complex field to intensity
  let intensity = sqrt(field.r * field.r + field.g * field.g);
  
  // Calculate tile position in quilt
  let col = params.current_view % 8u;
  let row = params.current_view / 8u;
  let quilt_x = col * params.tile_width + x;
  let quilt_y = row * params.tile_height + y;
  
  // Write to quilt
  let color = vec4<f32>(intensity, intensity, intensity, 1.0);
  textureStore(quilt_output, vec2<i32>(i32(quilt_x), i32(quilt_y)), color);
}
`;

// Export all shaders
export const shaderSources = {
  phase_only_encode_wgsl,
  lee_offaxis_encode_wgsl,
  propagation_wgsl,
  multiViewSynthesis_wgsl
};

export default shaderSources;
