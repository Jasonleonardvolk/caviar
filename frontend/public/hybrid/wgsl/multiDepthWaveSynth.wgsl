// multiDepthWaveSynth.wgsl - Compute shader for multi-depth wavefield synthesis
const MAX_LAYERS: u32 = 8;  // maximum layers supported
@group(0) @binding(0) var<storage, read> inputWave: array<vec2<f32>>;    // Input complex wave (after occlusion)
@group(0) @binding(1) var<storage, read_write> outputWave: array<vec2<f32>>; // Output combined wave
struct ParamsData {
    width: u32,
    height: u32,
    numLayers: u32,
    _pad: u32,                       // padding (unused)
    depths: array<f32, MAX_LAYERS>,  // Depth values for each layer (e.g., in meters or normalized units)
    emotion: f32,    // User emotional intensity (0 = calm, 1 = excited)
    proximity: f32,  // User proximity factor (0 = far, 1 = very close) - can be used to adjust focus dynamically
    gazeX: f32,      // Horizontal gaze direction factor (relative to center)
    gazeY: f32,      // Vertical gaze direction factor 
    personaPhaseSeed: f32, // Seed derived from persona embedding for random phase offsets
    // (padding to 16-byte alignment, if needed)
}

@group(0) @binding(2) var<storage, read> params: ParamsData;


fn clamp_index_dyn(i: u32, len: u32) -> u32 {
    return select(i, len - 1u, i >= len);
}

fn rand01(x: f32) -> f32 {
    // Simple deterministic hash to get a pseudorandom 0-1 value from input x
    return fract(sin(x) * 43758.5453);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    let x = gid.x;
    let y = gid.y;
    if (x >= w || y >= h) {
        return;
    }
    let idx = y * w + x;
    // Center coordinates for phase calculations
    let cx = f32(w) * 0.5;
    let cy = f32(h) * 0.5;
    let xf = f32(x) - cx;
    let yf = f32(y) - cy;
    // Read input wave at (x,y) with bounds checking
    let clamped_idx = clamp_index_dyn(idx, arrayLength(&inputWave));
    let inRe = inputWave[clamped_idx].x;
    let inIm = inputWave[clamped_idx].y;
    // Initialize output accumulation
    var accRe: f32 = 0.0;
    var accIm: f32 = 0.0;
    // Calculate viewing angle phase tilt based on gaze
    let gx = params.gazeX;
    let gy = params.gazeY;
    // Introduce a linear phase gradient for parallax: one full 2pi phase shift across the entire width for gaze factor = 1
    let tiltPhase = (gx * xf + gy * yf) * (2.0 * 3.141592653589793 / f32(w));
    // Coherence factor based on emotion
    let coherence = max(0.0, 1.0 - params.emotion);
    // Loop over each depth layer
    for (var i: u32 = 0u; i < params.numLayers; i = i + 1u) {
        if (i >= params.numLayers) { break; }
        // Access depths array with bounds checking (static array, so clamp to MAX_LAYERS)
        let depth_idx = clamp_index_dyn(i, MAX_LAYERS);
        let z = params.depths[depth_idx];
        // Compute quadratic phase for focusing at depth z:
        // Using Fresnel approximation: phase_curvature = (pi/(lambda*z)) * (x^2 + y^2). Here we pick lambda ~ 0.000633 (633nm) in simulation.
        const lambda = 0.000633;
        let invZ = 1.0 / (z + 1e-6);
        let phaseCurv = 3.1415927 * invZ / lambda * (xf*xf + yf*yf);
        // Compute a random phase offset for this layer (for decoherence) based on persona seed and layer index
        let randPhase = rand01(params.personaPhaseSeed + f32(i) * 13.37) * 2.0 * 3.1415927;
        let offsetPhase = (1.0 - coherence) * randPhase;
        // Total phase for this layer = focus phase + viewpoint tilt + offset
        let totalPhase = phaseCurv + tiltPhase + offsetPhase;
        let cosP = cos(totalPhase);
        let sinP = sin(totalPhase);
        // Rotate input wave by totalPhase and accumulate
        let contribRe = inRe * cosP - inIm * sinP;
        let contribIm = inRe * sinP + inIm * cosP;
        accRe = accRe + contribRe;
        accIm = accIm + contribIm;
    }
    // Write accumulated wave to output with bounds checking
    outputWave[clamp_index_dyn(idx, arrayLength(&outputWave))] = vec2<f32>(accRe, accIm);
}
