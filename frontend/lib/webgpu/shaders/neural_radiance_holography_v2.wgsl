// neural_radiance_holography_v2.wgsl (lite, production-safe)
// Wave-aware volumetric march with proper physics
// Works on iOS WebGPU at modest sizes; scales up automatically on desktop donor

struct Params {
    width: u32,
    height: u32,
    depth: u32,
    samples_per_ray: u32,      // 64-256 (not 10,000)
    dz: f32,                    // step along z
    phase_scale: f32,           // 2π/λ scaling
    wavelength: f32,            // in mm
    absorption: f32,           // Beer-Lambert coefficient
    // Interference parameters
    coherence_length: f32,
    mutual_intensity: f32,
    // Adaptive sampling
    importance_threshold: f32,
    use_russian_roulette: u32,
}

@group(0) @binding(0) var volumeTex: texture_3d<f32>;
@group(0) @binding(1) var volumeSamp: sampler;
@group(0) @binding(2) var<uniform> params: Params;
// Complex field: real, imag
@group(0) @binding(3) var<storage, read_write> outField: array<vec2<f32>>;
// Previous frame for temporal coherence
@group(0) @binding(4) var<storage, read> prevField: array<vec2<f32>>;

// Constants
const PI: f32 = 3.14159265359;

// Complex math helpers
fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

fn c_from_phase(phi: f32) -> vec2<f32> {
    return vec2<f32>(cos(phi), sin(phi));
}

fn c_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { 
    return a + b; 
}

// Fresnel-Kirchhoff integral for proper wave propagation
fn fresnel_kirchhoff_kernel(r: f32, z: f32) -> vec2<f32> {
    let k = 2.0 * PI / params.wavelength;
    let dist = sqrt(r*r + z*z);
    let phase = k * dist;
    
    // Inclination factor (1 + cos(θ))/2
    let cos_theta = z / dist;
    let inclination = (1.0 + cos_theta) * 0.5;
    
    // Complex amplitude with 1/r falloff
    let amp = inclination / dist;
    return c_from_phase(phase) * amp;
}

// Van Cittert-Zernike for partial coherence
fn coherence_factor(r: f32) -> f32 {
    if (params.coherence_length <= 0.0) { 
        return 1.0; 
    }
    let normalized_r = r / params.coherence_length;
    return exp(-normalized_r * normalized_r);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { 
        return; 
    }
    let wh = params.width * params.height;
    let idx = gid.y * params.width + gid.x;
    
    // Project pixel (x,y) into volume [0,1]×[0,1]×[0,1]
    let uv = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(params.width),
        (f32(gid.y) + 0.5) / f32(params.height)
    );
    
    // Adaptive sampling based on previous frame
    var samples = params.samples_per_ray;
    if (params.use_russian_roulette > 0u && idx < arrayLength(&prevField)) {
        let prev_intensity = length(prevField[idx]);
        if (prev_intensity < params.importance_threshold) {
            samples = samples / 2u; // Fewer samples in low-importance regions
        }
    }
    
    // Enhanced volumetric integration with proper wave physics
    var acc = vec2<f32>(0.0, 0.0);
    var transmittance = 1.0;
    var phase_accumulated = 0.0;
    
    // March through z with stratified sampling
    for (var s = 0u; s < samples; s = s + 1u) {
        let z = (f32(s) + 0.5) * params.dz;  // in [0,1]
        let sigma = textureSampleLevel(volumeTex, volumeSamp, vec3<f32>(uv, z), 0.0).r;
        
        // Beer-Lambert absorption
        let absorption = exp(-sigma * params.absorption * params.dz);
        transmittance *= absorption;
        
        // Russian roulette termination
        if (params.use_russian_roulette > 0u && transmittance < 0.01) {
            break;
        }
        
        // Wave propagation with Fresnel-Kirchhoff
        let r = length(uv - vec2<f32>(0.5));
        let wave_kernel = fresnel_kirchhoff_kernel(r, z);
        
        // Accumulate with coherence
        let coherence = coherence_factor(r);
        let contribution = wave_kernel * sigma * transmittance * coherence;
        acc = c_add(acc, contribution * params.dz);
        
        // Phase accumulation for interference
        phase_accumulated += sigma * params.phase_scale * params.dz;
    }
    
    // Apply accumulated phase
    acc = c_mul(acc, c_from_phase(phase_accumulated));
    
    // Temporal smoothing with previous frame
    if (idx < arrayLength(&prevField) && length(prevField[idx]) > 0.0) {
        acc = mix(prevField[idx], acc, 0.7); // 30% temporal coherence
    }
    
    outField[idx] = acc;
}