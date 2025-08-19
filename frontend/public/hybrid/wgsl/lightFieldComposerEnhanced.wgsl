/* 
 LightFieldComposerEnhanced.wgsl
 
 Enhanced version integrating tensor/phase backend for holographic display.
 Adds phase-weighted blending, tensor field distortion, and soliton wave modulation.
 
 New Features:
    - Phase-based view interpolation for smooth transitions
    - Tensor field integration for gravitational lensing effects (ALBERT system)
    - Soliton wave modulation for dynamic holographic patterns
    - Foveated rendering support with non-uniform view distributions
    - Full RGBA occlusion for complex masking effects
 
 Modes:
    0 = Quilt with phase blending
    1 = Depth Layers with tensor distortion
    2 = Stereo with soliton modulation
    3 = Tensor Field Visualization (NEW)
    4 = Phase-Coherent Hologram (NEW)
*/

struct Params {
    width: u32,              // Width of each single view image
    height: u32,             // Height of each single view image
    viewCount: u32,          // Total number of views
    tileCountX: u32,         // Number of tiles horizontally
    tileCountY: u32,         // Number of tiles vertically
    mode: u32,               // Output mode (0-4)
    time: f32,               // Animation time for soliton dynamics
    phaseShift: f32,         // Global phase shift for holographic effects
    tensorStrength: f32,     // Strength of tensor field distortion
    coherenceRadius: f32,    // Radius for phase coherence calculations
}

struct TensorField {
    // Kerr metric components for gravitational effects
    mass: f32,               // Black hole mass parameter
    spin: f32,               // Angular momentum parameter
    rho_squared: f32,        // ρ² = r² + a²cos²θ
    delta: f32,              // Δ = r² - 2Mr + a²
}

struct SolitonParams {
    amplitude: f32,          // Soliton wave amplitude
    wavelength: f32,         // Characteristic wavelength
    velocity: f32,           // Propagation velocity
    nonlinearity: f32,       // γ coefficient for |ψ|²ψ term
}

@group(0) @binding(0) var baseTex: texture_2d_array<f32>;
@group(0) @binding(1) var occTex: texture_2d_array<f32>;
@group(0) @binding(2) var outputTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params: Params;

// Phase field from multiDepthWaveSynth output
@group(0) @binding(4) var phaseTex: texture_2d_array<f32>;

// Tensor field data from ALBERT system
@group(0) @binding(5) var<uniform> tensorField: TensorField;

// Soliton wave parameters
@group(0) @binding(6) var<uniform> soliton: SolitonParams;

// Constants for phase calculations
const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const GOLDEN_RATIO: f32 = 1.61803398875;

// Complex number operations for phase calculations
fn complexMultiply(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

fn complexExp(phase: f32) -> vec2<f32> {
    return vec2<f32>(cos(phase), sin(phase));
}

// Calculate tensor field distortion at given position
fn applyTensorDistortion(pos: vec2<f32>, strength: f32) -> vec2<f32> {
    let r = length(pos);
    let theta = atan2(pos.y, pos.x);
    
    // Boyer-Lindquist coordinates transformation
    let a = tensorField.spin;
    let M = tensorField.mass;
    let rho2 = r * r + a * a * cos(theta) * cos(theta);
    let delta = r * r - 2.0 * M * r + a * a;
    
    // Frame-dragging effect (Lense-Thirring precession)
    let omega = (2.0 * M * a * r) / (rho2 * rho2 + a * a * delta);
    
    // Apply gravitational lensing distortion
    let deflection = strength * omega / (1.0 + r * 0.1);
    let distorted = vec2<f32>(
        pos.x * cos(deflection) - pos.y * sin(deflection),
        pos.x * sin(deflection) + pos.y * cos(deflection)
    );
    
    return distorted;
}

// Calculate soliton wave amplitude at position and time
fn solitonWave(pos: vec2<f32>, t: f32) -> f32 {
    let k = TWO_PI / soliton.wavelength;
    let omega = soliton.velocity * k;
    
    // Nonlinear Schrödinger equation solution (bright soliton)
    let xi = pos.x - soliton.velocity * t;
    let envelope = soliton.amplitude * (1.0 / cosh(xi / soliton.wavelength));
    let phase = k * pos.x - omega * t;
    
    // Add nonlinear phase modulation
    let nonlinearPhase = soliton.nonlinearity * envelope * envelope;
    
    return envelope * cos(phase + nonlinearPhase);
}

// Hyperbolic cosine for soliton envelope
fn cosh(x: f32) -> f32 {
    return (exp(x) + exp(-x)) * 0.5;
}

// Phase-weighted interpolation between views
fn phaseInterpolate(viewIndex: f32, localCoord: vec2<u32>) -> vec4<f32> {
    let baseViewIdx = u32(floor(viewIndex));
    let nextViewIdx = min(baseViewIdx + 1u, params.viewCount - 1u);
    let blend = fract(viewIndex);
    
    // Load base and next view colors
    let baseColor1 = textureLoad(baseTex, vec2<i32>(localCoord), i32(baseViewIdx), 0);
    let baseColor2 = textureLoad(baseTex, vec2<i32>(localCoord), i32(nextViewIdx), 0);
    
    // Load phase information
    let phase1 = textureLoad(phaseTex, vec2<i32>(localCoord), i32(baseViewIdx), 0).r;
    let phase2 = textureLoad(phaseTex, vec2<i32>(localCoord), i32(nextViewIdx), 0).r;
    
    // Complex phase-weighted blending
    let complexWeight1 = complexExp(phase1 * TWO_PI);
    let complexWeight2 = complexExp(phase2 * TWO_PI);
    
    // Blend in complex domain then convert back
    let blendedComplex = complexWeight1 * (1.0 - blend) + complexWeight2 * blend;
    let blendedPhase = atan2(blendedComplex.y, blendedComplex.x);
    
    // Apply phase to color interpolation
    let phaseFactor = (1.0 + cos(blendedPhase)) * 0.5;
    let blendedColor = mix(baseColor1, baseColor2, blend * phaseFactor);
    
    return blendedColor;
}

// Foveated view distribution for optimal sampling
fn getFoveatedViewIndex(normalizedPos: vec2<f32>, viewCount: u32) -> f32 {
    let center = vec2<f32>(0.5, 0.5);
    let dist = length(normalizedPos - center);
    
    // More views in the center, fewer at edges
    let foveationStrength = 2.0;
    let adjustedDist = pow(dist * 2.0, foveationStrength) * 0.5;
    
    return adjustedDist * f32(viewCount - 1u);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) GlobalId: vec3<u32>) {
    let out_x = GlobalId.x;
    let out_y = GlobalId.y;
    
    let totalWidth = params.width * params.tileCountX;
    let totalHeight = params.height * params.tileCountY;
    
    if (out_x >= totalWidth || out_y >= totalHeight) {
        return;
    }
    
    var outColor: vec4<f32>;
    let normalizedPos = vec2<f32>(f32(out_x) / f32(totalWidth), f32(out_y) / f32(totalHeight));
    
    if (params.mode == 0u) {
        // Enhanced Quilt mode with phase blending
        let tile_x = out_x / params.width;
        let tile_y = out_y / params.height;
        let local_x = out_x - tile_x * params.width;
        let local_y = out_y - tile_y * params.height;
        
        // Apply tensor field distortion to tile coordinates
        let tilePos = vec2<f32>(f32(tile_x), f32(tile_y));
        let distortedPos = applyTensorDistortion(tilePos, params.tensorStrength);
        
        // Calculate view index with phase-based interpolation
        let viewIndexFloat = f32(tile_y * params.tileCountX + tile_x) + 
                            solitonWave(normalizedPos * 10.0, params.time) * 0.5;
        
        outColor = phaseInterpolate(viewIndexFloat, vec2<u32>(local_x, local_y));
        
        // Apply full RGBA occlusion
        let occVal = textureLoad(occTex, vec2<i32>(i32(local_x), i32(local_y)), i32(viewIndexFloat), 0);
        outColor *= occVal;
        
    } else if (params.mode == 1u) {
        // Depth layers with tensor field visualization
        let tile_x = out_x / params.width;
        let tile_y = out_y / params.height;
        let viewIndex = tile_y * params.tileCountX + tile_x;
        let local_x = out_x - tile_x * params.width;
        let local_y = out_y - tile_y * params.height;
        
        if (viewIndex < params.viewCount) {
            let baseColor = textureLoad(baseTex, vec2<i32>(i32(local_x), i32(local_y)), i32(viewIndex), 0);
            let phase = textureLoad(phaseTex, vec2<i32>(i32(local_x), i32(local_y)), i32(viewIndex), 0).r;
            
            // Modulate color by phase
            let phaseColor = vec3<f32>(
                0.5 + 0.5 * cos(phase * TWO_PI + 0.0),
                0.5 + 0.5 * cos(phase * TWO_PI + 2.094),
                0.5 + 0.5 * cos(phase * TWO_PI + 4.189)
            );
            
            outColor = vec4<f32>(baseColor.rgb * phaseColor, 1.0);
        } else {
            outColor = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        
    } else if (params.mode == 2u) {
        // Stereo with soliton modulation
        var viewIndex: u32;
        var local_x: u32;
        
        if (out_x < params.width) {
            viewIndex = 0u;
            local_x = out_x;
        } else {
            viewIndex = max(params.viewCount, 1u) - 1u;
            local_x = out_x - params.width;
        }
        
        let local_y = out_y;
        
        if (viewIndex < params.viewCount) {
            let baseColor = textureLoad(baseTex, vec2<i32>(i32(local_x), i32(local_y)), i32(viewIndex), 0);
            let phase = textureLoad(phaseTex, vec2<i32>(i32(local_x), i32(local_y)), i32(viewIndex), 0).r;
            
            // Apply soliton wave modulation
            let solitonMod = solitonWave(normalizedPos * 20.0, params.time);
            let modPhase = phase + solitonMod * 0.1;
            
            let phaseFactor = 0.5 + 0.5 * cos(modPhase * TWO_PI + params.phaseShift);
            outColor = vec4<f32>(baseColor.rgb * phaseFactor, 1.0);
        } else {
            outColor = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        
    } else if (params.mode == 3u) {
        // Tensor field visualization mode
        let distortedPos = applyTensorDistortion(
            (normalizedPos - 0.5) * 2.0, 
            params.tensorStrength
        );
        
        // Visualize curvature as color
        let curvature = length(distortedPos - (normalizedPos - 0.5) * 2.0);
        let hue = curvature * 3.0;
        
        outColor = vec4<f32>(
            0.5 + 0.5 * cos(hue + 0.0),
            0.5 + 0.5 * cos(hue + 2.094),
            0.5 + 0.5 * cos(hue + 4.189),
            1.0
        );
        
    } else if (params.mode == 4u) {
        // Phase-coherent hologram mode
        let foveatedView = getFoveatedViewIndex(normalizedPos, params.viewCount);
        let viewIdx = u32(foveatedView);
        
        // Calculate local coordinates with sub-pixel precision
        let pixelPos = vec2<f32>(f32(out_x), f32(out_y));
        let localCoord = vec2<u32>(
            u32(pixelPos.x) % params.width,
            u32(pixelPos.y) % params.height
        );
        
        // Get phase-interpolated color
        outColor = phaseInterpolate(foveatedView, localCoord);
        
        // Apply coherence radius for holographic effect
        let coherenceMask = exp(-length(normalizedPos - 0.5) / params.coherenceRadius);
        outColor.r *= coherenceMask;
        outColor.g *= coherenceMask;
        outColor.b *= coherenceMask;
        
        // Add holographic interference pattern
        let interference = 0.5 + 0.5 * cos(
            length(pixelPos) * 0.1 + 
            params.phaseShift + 
            solitonWave(normalizedPos * 50.0, params.time) * PI
        );
        outColor.r *= interference;
        outColor.g *= interference;
        outColor.b *= interference;
    }
    
    // Ensure alpha is always 1.0 for proper display
    outColor.a = 1.0;
    
    // Write the composed color to the output texture
    textureStore(outputTex, vec2<i32>(i32(out_x), i32(out_y)), outColor);
}