// quantum_superposition.wgsl
// Linear superposition with unitary mixing - pure math, no quantum hardware needed
// Creates quantum-inspired holographic states

struct Params { 
    width: u32,
    height: u32,
    states: u32,        // Number of states to superpose
    coherence: f32,     // Coherence between states (0-1)
    phase_offset: f32,  // Global phase offset
    entanglement: f32,  // Entanglement strength (0-1)
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> inStates: array<vec2<f32>>;   // [states * W*H]
@group(0) @binding(2) var<storage, read> weights: array<vec2<f32>>;    // [states] complex amplitudes
@group(0) @binding(3) var<storage, read_write> outField: array<vec2<f32>>;

// Complex operations
fn c_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { 
    return a + b; 
}

fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

fn c_conj(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}

fn c_from_phase(phi: f32) -> vec2<f32> {
    return vec2<f32>(cos(phi), sin(phi));
}

// Quantum-inspired entanglement operator
fn entangle_states(state1: vec2<f32>, state2: vec2<f32>, strength: f32) -> vec2<f32> {
    // Create Bell-like state: |ψ⟩ = α|00⟩ + β|11⟩
    let bell_component = c_mul(state1, state2);
    return mix(state1, bell_component, strength);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.width || gid.y >= p.height) { 
        return; 
    }
    let wh = p.width * p.height;
    let idx = gid.y * p.width + gid.x;
    
    var acc = vec2<f32>(0.0, 0.0);
    var norm = 0.0;
    
    // Superpose all states with proper quantum amplitudes
    for (var s = 0u; s < p.states; s = s + 1u) {
        // Get complex weight for this state
        var w = vec2<f32>(1.0, 0.0);
        if (s < arrayLength(&weights)) {
            w = weights[s];
        }
        
        // Apply coherence decay
        let coherence_factor = pow(p.coherence, f32(s));
        w = w * coherence_factor;
        
        // Get the field for this state
        let state_idx = s * wh + idx;
        var f = vec2<f32>(0.0, 0.0);
        if (state_idx < arrayLength(&inStates)) {
            f = inStates[state_idx];
        }
        
        // Apply phase evolution
        let phase = p.phase_offset * f32(s);
        f = c_mul(f, c_from_phase(phase));
        
        // Entangle with previous state if requested
        if (s > 0u && p.entanglement > 0.0) {
            let prev_idx = (s - 1u) * wh + idx;
            if (prev_idx < arrayLength(&inStates)) {
                let prev_state = inStates[prev_idx];
                f = entangle_states(f, prev_state, p.entanglement);
            }
        }
        
        // Accumulate with weight
        acc = c_add(acc, c_mul(w, f));
        norm += length(w) * length(w);
    }
    
    // Normalize to preserve unitarity
    if (norm > 0.0) {
        acc = acc / sqrt(norm);
    }
    
    outField[idx] = acc;
}

// Create GHZ state (generalized entangled state)
@compute @workgroup_size(8, 8, 1)
fn create_ghz_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.width || gid.y >= p.height) { 
        return; 
    }
    let idx = gid.y * p.width + gid.x;
    
    // GHZ state: |ψ⟩ = (|000...⟩ + |111...⟩)/√2
    var ghz = vec2<f32>(0.0, 0.0);
    let wh = p.width * p.height;
    
    // All zeros component
    if (idx < arrayLength(&inStates)) {
        ghz = inStates[idx];  // First state
    }
    
    // All ones component
    let last_state_idx = (p.states - 1u) * wh + idx;
    if (last_state_idx < arrayLength(&inStates)) {
        ghz = c_add(ghz, inStates[last_state_idx]);
    }
    
    // Normalize
    ghz = ghz * 0.70710678118; // 1/√2
    
    outField[idx] = ghz;
}