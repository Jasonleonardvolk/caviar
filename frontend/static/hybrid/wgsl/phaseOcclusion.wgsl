// phaseOcclusion.wgsl - Compute shader for phase-aware spectral occlusion
@group(0) @binding(0) var<storage, read> inputWave: array<vec2<f32>>;    // Input complex wavefield (real, imag)
@group(0) @binding(1) var<storage, read> occlusion: array<f32>;         // Occlusion map (0 = opaque, 1 = transparent)
@group(0) @binding(2) var<storage, read_write> outputWave: array<vec2<f32>>; // Output wavefield after occlusion
struct ParamsData {
    width: u32,
    height: u32,
    cognitiveFactor: f32,   // Cognitive transparency override factor (0 = purely physical, 1 = fully see-through)
    phaseShiftMax: f32      // Maximum phase delay (radians) through a fully opaque occluder
}

@group(0) @binding(3) var<uniform> params: ParamsData;;


fn clamp_index_dyn(i: u32, len: u32) -> u32 {
    return select(i, len - 1u, i >= len);
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
    
    // Fetch input wave at this pixel with bounds checking
    let clamped_idx = clamp_index_dyn(idx, arrayLength(&inputWave));
    let inRe = inputWave[clamped_idx].x;
    let inIm = inputWave[clamped_idx].y;
    
    // Fetch occlusion value (between 0 and 1) with bounds checking
    let occ = occlusion[clamp_index_dyn(idx, arrayLength(&occlusion))];
    
    // Soft-blend occlusion edges by averaging with immediate neighbors (cross pattern)
    var occSmooth = occ;
    if (x + 1u < w) { 
        occSmooth += occlusion[clamp_index_dyn(idx + 1u, arrayLength(&occlusion))]; 
    }
    if (x > 0u) { 
        occSmooth += occlusion[clamp_index_dyn(idx - 1u, arrayLength(&occlusion))]; 
    }
    if (y + 1u < h) { 
        occSmooth += occlusion[clamp_index_dyn(idx + w, arrayLength(&occlusion))]; 
    }
    if (y > 0u) { 
        occSmooth += occlusion[clamp_index_dyn(idx - w, arrayLength(&occlusion))]; 
    }
    
    // Divide by the number of samples (itself + neighbors)
    // Count neighbors included:
    var count: f32 = 1.0;
    if (x + 1u < w)  { count += 1.0; }
    if (x > 0u)      { count += 1.0; }
    if (y + 1u < h)  { count += 1.0; }
    if (y > 0u)      { count += 1.0; }
    occSmooth = occSmooth / count;
    
    // Determine effective transparency with cognitive override:
    // If cognitiveFactor > 0, reduce occlusion (increase transparency)
    let cf = params.cognitiveFactor;
    let effectiveT = occSmooth + cf * (1.0 - occSmooth);
    
    let output_idx = clamp_index_dyn(idx, arrayLength(&outputWave));
    if (effectiveT <= 0.0) {
        // Completely opaque after cognitive adjustment: block wave entirely
        outputWave[output_idx] = vec2<f32>(0.0, 0.0);
    } else {
        // Apply phase-aware attenuation
        // Compute phase delay proportional to opacity (1 - effectiveT)
        let phaseShift = (1.0 - effectiveT) * params.phaseShiftMax;
        let cosP = cos(phaseShift);
        let sinP = sin(phaseShift);
        // Modulate the input wave amplitude by effective transparency and rotate by phaseShift
        let outRe = effectiveT * (inRe * cosP - inIm * sinP);
        let outIm = effectiveT * (inRe * sinP + inIm * cosP);
        outputWave[output_idx] = vec2<f32>(outRe, outIm);
    }
}
