/**
 * wavefrontReconstructor.ts
 *
 * Reconstructs a 3D holographic wavefront from cognitive & symbolic inputs.
 * - Integrates persona embeddings (ENOLA or similar) and semantic concept meshes into a complex wavefield.
 * - Supports full-field propagation in depth via FFT (WebGPU compute) or CPU fallback.
 * - Outputs a complex wavefront encoded as phase and amplitude (intensity) maps.
 *
 * Optimized for software simulation: uses WebGPU for parallel FFT and vector math when available,
 * and falls back to CPU (with potential SIMD optimizations) if GPU is not supported.
 */
export class WavefrontReconstructor {
    constructor(resolution = 512) {
        this.device = null;
        this.useWebGPU = false;
        this.phaseOcclusionPipeline = null;
        this.multiDepthPipeline = null;
        this.bindGroupOcclusion = null;
        this.bindGroupMultiDepth = null;
        // GPU buffers for wavefield and occlusion data
        this.waveBufferIn = null;
        this.waveBufferMid = null;
        this.waveBufferOut = null;
        this.occlusionBuffer = null;
        // GPU uniform buffers for shader parameters
        this.occlusionParamsBuffer = null;
        this.multiDepthParamsBuffer = null;
        this.dim = resolution;
        // Attempt to initialize WebGPU
        if (navigator.gpu) {
            navigator.gpu.requestAdapter().then(adapter => {
                if (!adapter) {
                    console.warn("WebGPU adapter not found, using CPU fallback.");
                    return;
                }
                adapter.requestDevice().then(device => {
                    this.device = device;
                    this.useWebGPU = true;
                    this.initGPUPipeline().catch(err => {
                        console.error("WebGPU pipeline initialization failed, falling back to CPU:", err);
                        this.useWebGPU = false;
                    });
                }).catch(err => {
                    console.warn("WebGPU device request failed:", err);
                });
            }).catch(err => {
                console.warn("WebGPU adapter request failed:", err);
            });
        }
        else {
            console.warn("WebGPU not supported, using CPU simulation.");
        }
    }
    /**
     * Initialize GPU resources and compile compute shaders for occlusion and multi-depth synthesis.
     */
    async initGPUPipeline() {
        if (!this.device)
            return;
        const device = this.device;
        // Allocate storage buffers for wavefield and occlusion data
        const N = this.dim * this.dim;
        const waveBufferSize = N * 2 * 4; // 2 floats (real, imag) per pixel
        const occBufferSize = N * 4; // 1 float per pixel for occlusion map
        this.waveBufferIn = device.createBuffer({ size: waveBufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.waveBufferMid = device.createBuffer({ size: waveBufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        this.waveBufferOut = device.createBuffer({ size: waveBufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.occlusionBuffer = device.createBuffer({ size: occBufferSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        // Create uniform buffers for parameters (16-byte aligned)
        const occParamSize = 4 * 4; // OcclusionParams: 4 floats (width, height, cognitiveFactor, phaseShiftMax)
        this.occlusionParamsBuffer = device.createBuffer({ size: occParamSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const maxLayers = 8;
        // MultiDepthParams struct size:
        // {width(u32), height(u32), numLayers(u32), pad(u32)} = 16 bytes, 
        // depths[8] = 8*4 = 32 bytes, 
        // {emotion, proximity, gazeX, gazeY, personaPhaseSeed, coherencePadding(=0 to align)} = 6*4 = 24 bytes.
        // Total = 16+32+24 = 72 bytes, round up to 80 for alignment safety.
        const mdParamSize = 80;
        this.multiDepthParamsBuffer = device.createBuffer({ size: mdParamSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        // Load and compile WGSL shaders for phase occlusion and multi-depth synthesis
        const phaseOcclusionCode = /* wgsl */ `
            @group(0) @binding(0) var<storage, read> inputWave: array<vec2<f32>>;
            @group(0) @binding(1) var<storage, read> occlusion: array<f32>;
            @group(0) @binding(2) var<storage, write> outputWave: array<vec2<f32>>;
            @group(0) @binding(3) var<uniform> params: struct {
                width: u32,
                height: u32,
                cognitiveFactor: f32,
                phaseShiftMax: f32
            };

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
                // Read input wave (real and imaginary components)
                let inRe = inputWave[idx].x;
                let inIm = inputWave[idx].y;
                // Read occlusion value (0 = fully blocked, 1 = no occlusion)
                let oc = occlusion[idx];
                // Simple edge-aware smoothing of occlusion (average with cross neighbors)
                var ocSmooth = oc;
                if (x + 1u < w) { ocSmooth += occlusion[idx + 1u]; }
                if (x > 0u)    { ocSmooth += occlusion[idx - 1u]; }
                if (y + 1u < h) { ocSmooth += occlusion[idx + w]; }
                if (y > 0u)    { ocSmooth += occlusion[idx - w]; }
                // Divide by count of samples (itself + up to 4 neighbors)
                ocSmooth = ocSmooth / (1.0 + 
                    f32(if (x + 1u < w) { 1 } else { 0 } + if (x > 0u) { 1 } else { 0 } + 
                        if (y + 1u < h) { 1 } else { 0 } + if (y > 0u) { 1 } else { 0 }));
                // Compute effective transparency considering cognitive override
                let cogFactor = params.cognitiveFactor;
                // cognitiveFactor drives "seeing through" occluders: higher factor -> occluder becomes more transparent
                let effectiveT = ocSmooth + cogFactor * (1.0 - ocSmooth);
                if (effectiveT < 0.0001) {
                    // Nearly fully opaque case: block wave (output ~ 0)
                    outputWave[idx] = vec2<f32>(0.0, 0.0);
                } else {
                    // Phase-aware transparency: occluder may impart a phase delay
                    let phaseShift = (1.0 - effectiveT) * params.phaseShiftMax;
                    let cosP = cos(phaseShift);
                    let sinP = sin(phaseShift);
                    // Apply attenuation and phase shift to the input wave
                    let outRe = effectiveT * (inRe * cosP - inIm * sinP);
                    let outIm = effectiveT * (inRe * sinP + inIm * cosP);
                    outputWave[idx] = vec2<f32>(outRe, outIm);
                }
            }
        `;
        const multiDepthCode = /* wgsl */ `
            const MAX_LAYERS: u32 = 8;
            @group(0) @binding(0) var<storage, read> inputWave: array<vec2<f32>>;
            @group(0) @binding(1) var<storage, write> outputWave: array<vec2<f32>>;
            @group(0) @binding(2) var<uniform> params: struct {
                width: u32,
                height: u32,
                numLayers: u32,
                _pad: u32,                 // padding
                depths: array<f32, ${8 /* MAX_LAYERS */}>,
                emotion: f32,
                proximity: f32,
                gazeX: f32,
                gazeY: f32,
                personaPhaseSeed: f32,
                _pad2: vec3<f32>          // padding to 16-byte align (if needed)
            };

            // Pseudorandom number generator for layer phase offsets (based on seed and layer index)
            fn rand01(seed: f32) -> f32 {
                // Simple hash: sine chaos
                return fract(sin(seed) * 43758.5453);
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
                // Determine normalized coordinates (center at 0)
                let cx = f32(w) * 0.5;
                let cy = f32(h) * 0.5;
                let xf = f32(x) - cx;
                let yf = f32(y) - cy;
                // Pre-read the input wave at this pixel
                let inRe = inputWave[idx].x;
                let inIm = inputWave[idx].y;
                // Initialize output accumulator
                var sumRe: f32 = 0.0;
                var sumIm: f32 = 0.0;
                // Determine user state factors
                let gazeX = params.gazeX;
                let gazeY = params.gazeY;
                // Linear phase tilt for viewpoint parallax (approximate)
                // The gazeX/Y are assumed to be small fractions indicating viewing angle offset.
                let tiltFactor = 2.0 * 3.1415926;  // 2π phase across full width for gaze=1
                let phaseTilt = (gazeX * xf + gazeY * yf) * tiltFactor / f32(w);
                // Coherence control based on emotion: high emotion -> low coherence between depth layers
                let coherence = max(0.0, 1.0 - params.emotion);
                // Loop over each depth layer to accumulate its wave contribution
                for (var layer: u32 = 0u; layer < params.numLayers; layer = layer + 1u) {
                    if (layer >= params.numLayers) { break; } // safety check (though loop bound is numLayers)
                    let z = params.depths[layer];
                    // Compute layer-specific phase curvature for depth z (Fresnel lens approximation)
                    // phase_curv ≈ (π / (λ * z)) * (x^2 + y^2), here we use a scaled factor for simulation
                    let invDepth = 1.0 / (z + 1e-6);
                    // Choose a scale for curvature: using a constant factor relative to image size and an assumed wavelength
                    let wavelength = 0.000633; // 633 nm (~red laser) as a reference
                    let lensPhase = 3.1415926 * invDepth / wavelength * ((xf * xf) + (yf * yf));
                    // Global phase offset per layer (randomized to reduce inter-layer coherence if coherence < 1)
                    // Use personaPhaseSeed and layer index to generate a pseudo-random phase
                    let baseSeed = params.personaPhaseSeed + f32(layer) * 17.654; 
                    let randPhase = rand01(baseSeed) * 2.0 * 3.1415926;
                    let offsetPhase = (1.0 - coherence) * randPhase;
                    // Total phase = lens focusing phase + viewpoint tilt phase + layer offset phase
                    let totalPhase = lensPhase + phaseTilt + offsetPhase;
                    let cosP = cos(totalPhase);
                    let sinP = sin(totalPhase);
                    // Rotate the input wave by this phase and accumulate
                    // (This effectively adds the wave contribution focused at depth z)
                    let partRe = inRe * cosP - inIm * sinP;
                    let partIm = inRe * sinP + inIm * cosP;
                    sumRe = sumRe + partRe;
                    sumIm = sumIm + partIm;
                }
                // Write the accumulated multi-depth wavefield to output
                outputWave[idx] = vec2<f32>(sumRe, sumIm);
            }
        `;
        // Compile shader modules and pipelines
        const phaseModule = device.createShaderModule({ code: phaseOcclusionCode });
        const mdModule = device.createShaderModule({ code: multiDepthCode });
        this.phaseOcclusionPipeline = device.createComputePipeline({
            layout: "auto",
            compute: { module: phaseModule, entryPoint: "main" }
        });
        this.multiDepthPipeline = device.createComputePipeline({
            layout: "auto",
            compute: { module: mdModule, entryPoint: "main" }
        });
        // Create bind groups for the compute shaders
        this.bindGroupOcclusion = device.createBindGroup({
            layout: this.phaseOcclusionPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.waveBufferIn } },
                { binding: 1, resource: { buffer: this.occlusionBuffer } },
                { binding: 2, resource: { buffer: this.waveBufferMid } },
                { binding: 3, resource: { buffer: this.occlusionParamsBuffer } }
            ]
        });
        this.bindGroupMultiDepth = device.createBindGroup({
            layout: this.multiDepthPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.waveBufferMid } },
                { binding: 1, resource: { buffer: this.waveBufferOut } },
                { binding: 2, resource: { buffer: this.multiDepthParamsBuffer } }
            ]
        });
    }
    /**
     * Reconstruct the holographic wavefront using the provided inputs.
     * @param personaEmbedding - Array of floats representing cognitive persona embedding (ENOLA, etc.)
     * @param conceptMesh - Semantic concept mesh or graph (can influence coherence or patterns)
     * @param audioData - Optional audio data (e.g., waveform or frequency) to modulate the wavefront
     * @param userState - Optional user state parameters: { emotion, proximity, gazeX, gazeY, cognitiveFactor }
     * @param occlusionMap - Optional occlusion map (2D array of size [dim x dim], values 0.0-1.0)
     * @returns A structure containing the final complex wavefield (and possibly intensity map for visualization).
     */
    async reconstructWavefront(personaEmbedding, conceptMesh, audioData, userState, occlusionMap) {
        const width = this.dim, height = this.dim;
        // 1. Generate initial wavefront (complex field) from cognitive overlays on CPU
        // Prepare data structures
        const N = width * height;
        const waveField = new Float32Array(N * 2); // interleaved real, imag
        waveField.fill(0);
        // Parameters for oscillator-based wave synthesis
        const maxOscillators = 32;
        const personaLength = personaEmbedding ? personaEmbedding.length : 0;
        // Derive number of oscillators from persona and concept data
        let numOsc = Math.min(maxOscillators, personaLength > 0 ? personaLength : maxOscillators);
        if (conceptMesh && conceptMesh.nodes) {
            // If concept mesh has many nodes, we could allocate additional oscillators for it
            numOsc = Math.min(maxOscillators, Math.max(numOsc, conceptMesh.nodes.length));
        }
        // Define spatial frequency and phase for each oscillator
        // We will distribute oscillators in frequency space and modulate by persona embedding values
        const centerX = width / 2;
        const centerY = height / 2;
        // Base frequency (cycles across image) and range
        const baseFreq = 5.0;
        const freqRange = 10.0;
        // To ensure deep coherence modulated by concept complexity:
        // If the concept mesh is very complex (many nodes/relationships), we introduce more phase noise (lower coherence)
        let conceptComplexity = conceptMesh && conceptMesh.nodeCount ? conceptMesh.nodeCount : (conceptMesh?.nodes?.length || 0);
        // Normalize concept complexity to [0,1] (assuming >=0, cap at 100 for normalization)
        if (conceptComplexity > 100)
            conceptComplexity = 100;
        const conceptComplexityNorm = conceptComplexity / 100;
        // We'll add a small random noise component to the wave (speckle) proportional to concept complexity
        const noiseAmplitude = conceptComplexityNorm * 0.2; // up to 20% amplitude as noise
        // Compute oscillators contributions
        for (let o = 0; o < numOsc; o++) {
            // Determine oscillator properties
            const personaVal = personaEmbedding && o < personaLength ? personaEmbedding[o] : Math.random();
            const normVal = Math.min(Math.max(personaVal, 0.0), 1.0); // normalize value to [0,1]
            // Spatial frequency magnitude and direction
            const freq = baseFreq + normVal * freqRange; // cycles across field
            const angle = (2 * Math.PI * o) / numOsc; // distribute evenly by angle
            const kx = freq * Math.cos(angle) * (2 * Math.PI / width); // convert to rad/pixel
            const ky = freq * Math.sin(angle) * (2 * Math.PI / height);
            // Initial phase for this oscillator (could be influenced by personaVal and audio)
            let phase0 = normVal * 2 * Math.PI;
            if (audioData && audioData.length > 0 && o === 0) {
                // Example: modulate first oscillator phase by audio (e.g., use first audio sample or frequency)
                phase0 += audioData[0] * 2 * Math.PI; // simplistic mapping of audio amplitude to phase
            }
            // Loop over the grid and accumulate this oscillator's wave
            for (let yi = 0; yi < height; yi++) {
                // Compute phase increment per x and base phase at start of row
                const dyPhase = ky * (yi - centerY);
                const basePhase = dyPhase + phase0;
                // Use incremental trig to avoid repetitive cos/sin calls per pixel (for performance)
                let cosPhase = Math.cos(basePhase);
                let sinPhase = Math.sin(basePhase);
                // Precompute cosine and sine of the per-pixel increment (kx)
                const cosStep = Math.cos(kx);
                const sinStep = Math.sin(kx);
                let index = yi * width;
                for (let xi = 0; xi < width; xi++) {
                    // Compute oscillator contribution at (xi, yi)
                    // real = cos(total phase), imag = sin(total phase) for this oscillator
                    // (Using planar wave e^(i(kx*x + ky*y + phase0)))
                    // Current cosPhase, sinPhase represent cos/sin of (kx*xi + ky*yi + phase0)
                    // Accumulate into waveField
                    const waveIndex = index * 2;
                    waveField[waveIndex] += cosPhase;
                    waveField[waveIndex + 1] += sinPhase;
                    // Advance phase incrementally for next pixel: rotate (cosPhase, sinPhase) by step (cosStep, sinStep)
                    let tempCos = cosPhase * cosStep - sinPhase * sinStep;
                    let tempSin = cosPhase * sinStep + sinPhase * cosStep;
                    cosPhase = tempCos;
                    sinPhase = tempSin;
                    index++;
                }
            }
        }
        // Add random noise to waveField if concept complexity is high (symbolic coherence modulation)
        if (conceptComplexityNorm > 0) {
            for (let i = 0; i < N; i++) {
                if (Math.random() < 0.001) {
                    // add occasional random spikes
                    const theta = 2 * Math.PI * Math.random();
                    waveField[2 * i] += noiseAmplitude * Math.cos(theta);
                    waveField[2 * i + 1] += noiseAmplitude * Math.sin(theta);
                }
            }
        }
        // At this point, waveField contains the initial complex wave (real, imag) for the hologram.
        if (this.useWebGPU && this.device) {
            // 2. GPU path: upload data and run compute shaders for occlusion and multi-depth propagation
            const device = this.device;
            // Upload initial waveField to GPU buffer
            device.queue.writeBuffer(this.waveBufferIn, 0, waveField.buffer, waveField.byteOffset, waveField.byteLength);
            // Upload occlusion map if provided (or fill with 1.0 if none, meaning no occlusion)
            const occlusionData = new Float32Array(N);
            if (occlusionMap) {
                if (occlusionMap.length === N) {
                    occlusionData.set(occlusionMap);
                }
                else {
                    console.warn("Occlusion map size mismatch, ignoring occlusion.");
                    occlusionData.fill(1.0);
                }
            }
            else {
                occlusionData.fill(1.0);
            }
            device.queue.writeBuffer(this.occlusionBuffer, 0, occlusionData.buffer, occlusionData.byteOffset, occlusionData.byteLength);
            // Prepare uniform parameters for occlusion shader
            const cognitiveFactor = userState?.cognitiveFactor ?? 0.0;
            const phaseShiftMax = Math.PI * 0.5; // max phase shift (radians) through fully opaque occluder (e.g., could simulate half-wave delay)
            const occParams = new Float32Array([width, height, cognitiveFactor, phaseShiftMax]);
            device.queue.writeBuffer(this.occlusionParamsBuffer, 0, occParams.buffer, occParams.byteOffset, occParams.byteLength);
            // Prepare uniform parameters for multi-depth shader
            const emotion = userState?.emotion ?? 0.0;
            const proximity = userState?.proximity ?? 0.0;
            const gazeX = userState?.gazeX ?? 0.0;
            const gazeY = userState?.gazeY ?? 0.0;
            // Choose depth layers (in normalized or physical units). For demonstration, use 3 layers: near, mid, far.
            const depthLayers = [0.2, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // only first numLayers values used
            const numLayers = 3;
            // Compute a persona phase seed (deterministic value from persona embedding for layer phase randomization)
            let personaPhaseSeed = 0.5;
            if (personaEmbedding && personaEmbedding.length > 0) {
                // e.g., use normalized sum of absolute values as seed
                const sumAbs = personaEmbedding.reduce((acc, v) => acc + Math.abs(v), 0);
                personaPhaseSeed = sumAbs - Math.floor(sumAbs); // fractional part as a 0-1 seed
            }
            // Pack multi-depth uniform buffer (align to 16 bytes as needed)
            const mdParams = new Float32Array(20); // 5 vec4 (80 bytes) - will map onto the struct
            mdParams[0] = width;
            mdParams[1] = height;
            mdParams[2] = numLayers;
            mdParams[3] = 0.0; // padding
            // depths array (next 8 entries)
            for (let i = 0; i < 8; i++) {
                mdParams[4 + i] = depthLayers[i] || 0.0;
            }
            // user state floats
            mdParams[12] = emotion;
            mdParams[13] = proximity;
            mdParams[14] = gazeX;
            mdParams[15] = gazeY;
            mdParams[16] = personaPhaseSeed;
            mdParams[17] = 0.0; // padding
            mdParams[18] = 0.0; // padding
            mdParams[19] = 0.0; // padding
            device.queue.writeBuffer(this.multiDepthParamsBuffer, 0, mdParams.buffer, mdParams.byteOffset, mdParams.byteLength);
            // Dispatch compute shaders: Phase Occlusion -> Multi-Depth Synthesis
            const commandEncoder = device.createCommandEncoder();
            // Phase occlusion pass
            const occPass = commandEncoder.beginComputePass();
            occPass.setPipeline(this.phaseOcclusionPipeline);
            occPass.setBindGroup(0, this.bindGroupOcclusion);
            const wgX = Math.ceil(width / 16);
            const wgY = Math.ceil(height / 16);
            occPass.dispatchWorkgroups(wgX, wgY);
            occPass.end();
            // Multi-depth synthesis pass
            const mdPass = commandEncoder.beginComputePass();
            mdPass.setPipeline(this.multiDepthPipeline);
            mdPass.setBindGroup(0, this.bindGroupMultiDepth);
            mdPass.dispatchWorkgroups(wgX, wgY);
            mdPass.end();
            // Submit commands
            device.queue.submit([commandEncoder.finish()]);
            // Read back the final wavefield (complex values) from GPU
            const resultBuffer = this.waveBufferOut;
            // We need to copy the result buffer to a buffer that can be mapped for reading
            const readBuffer = device.createBuffer({ size: waveBufferSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            const copyEncoder = device.createCommandEncoder();
            copyEncoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, waveBufferSize);
            device.queue.submit([copyEncoder.finish()]);
            await readBuffer.mapAsync(GPUMapMode.READ);
            const arrayBuffer = readBuffer.getMappedRange();
            const finalWavefield = new Float32Array(arrayBuffer.slice(0)); // copy data
            readBuffer.unmap();
            // Optionally, compute intensity map from finalWavefield for visualization (intensity = Re^2 + Im^2)
            const intensity = new Float32Array(N);
            for (let i = 0; i < N; i++) {
                const re = finalWavefield[2 * i];
                const im = finalWavefield[2 * i + 1];
                intensity[i] = re * re + im * im;
            }
            return { real: finalWavefield.filter((_, idx) => idx % 2 === 0), imag: finalWavefield.filter((_, idx) => idx % 2 === 1), intensity };
        }
        else {
            // 3. CPU fallback: apply occlusion and multi-depth propagation in JavaScript
            const real = new Float32Array(N);
            const imag = new Float32Array(N);
            // Split interleaved waveField into real and imag for convenience
            for (let i = 0; i < N; i++) {
                real[i] = waveField[2 * i];
                imag[i] = waveField[2 * i + 1];
            }
            // Occlusion step (if occlusionMap is provided)
            const occ = occlusionMap && occlusionMap.length === N ? occlusionMap : null;
            const cognitiveFactor = userState?.cognitiveFactor ?? 0.0;
            const phaseShiftMax = Math.PI * 0.5;
            if (occ) {
                // Smooth the occlusion map edges
                const occSmooth = new Float32Array(N);
                occSmooth.set(occ);
                // simple 1-pixel radius blur
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        const i = y * width + x;
                        let sum = occ[i];
                        let count = 1;
                        if (x + 1 < width) {
                            sum += occ[i + 1];
                            count++;
                        }
                        if (x > 0) {
                            sum += occ[i - 1];
                            count++;
                        }
                        if (y + 1 < height) {
                            sum += occ[i + width];
                            count++;
                        }
                        if (y > 0) {
                            sum += occ[i - width];
                            count++;
                        }
                        occSmooth[i] = sum / count;
                    }
                }
                // Apply occlusion transparency and phase shift to wave
                for (let i = 0; i < N; i++) {
                    const trans = occSmooth[i];
                    const effectiveT = trans + cognitiveFactor * (1.0 - trans);
                    if (effectiveT <= 0.0001) {
                        // fully occluded
                        real[i] = 0;
                        imag[i] = 0;
                    }
                    else {
                        const phaseShift = (1.0 - effectiveT) * phaseShiftMax;
                        const cosP = Math.cos(phaseShift);
                        const sinP = Math.sin(phaseShift);
                        const inRe = real[i];
                        const inIm = imag[i];
                        real[i] = effectiveT * (inRe * cosP - inIm * sinP);
                        imag[i] = effectiveT * (inRe * sinP + inIm * cosP);
                    }
                }
            }
            // Multi-depth synthesis: combine wave at multiple focus depths
            const emotion = userState?.emotion ?? 0.0;
            const proximity = userState?.proximity ?? 0.0;
            const gazeX = userState?.gazeX ?? 0.0;
            const gazeY = userState?.gazeY ?? 0.0;
            const numLayers = 3;
            const depths = [0.2, 0.5, 1.0]; // example depth values
            // Compute persona-based random phase offsets for layers (to modulate coherence)
            let personaPhaseSeed = 0.5;
            if (personaEmbedding && personaEmbedding.length > 0) {
                const sumAbs = personaEmbedding.reduce((acc, v) => acc + Math.abs(v), 0);
                personaPhaseSeed = sumAbs - Math.floor(sumAbs);
            }
            const coherence = Math.max(0.0, 1.0 - emotion);
            const layerPhaseOffsets = [];
            for (let l = 0; l < numLayers; l++) {
                // simple deterministic pseudo-random from seed
                const seed = personaPhaseSeed + l * 17.654;
                const rand = Math.sin(seed) * 43758.5453;
                const randFrac = rand - Math.floor(rand);
                const randPhase = randFrac * 2 * Math.PI;
                const offset = (1.0 - coherence) * randPhase;
                layerPhaseOffsets.push(offset);
            }
            // Prepare output arrays
            const finalReal = new Float32Array(N);
            const finalImag = new Float32Array(N);
            finalReal.fill(0);
            finalImag.fill(0);
            // For each layer, propagate current wave to that depth and accumulate
            const wavelength = 0.000633;
            for (let li = 0; li < numLayers; li++) {
                const z = depths[li];
                const invDepth = 1.0 / (z + 1e-6);
                const layerOffset = layerPhaseOffsets[li];
                // Compute linear phase gradient for gaze (approximate tilt)
                const tiltFactor = 2 * Math.PI; // one full fringe across screen for gaze=1
                // We'll incorporate gaze into the per-pixel phase inside loop to avoid huge memory overhead
                for (let y = 0; y < height; y++) {
                    const dy = y - centerY;
                    for (let x = 0; x < width; x++) {
                        const dx = x - centerX;
                        const i = y * width + x;
                        // Compute Fresnel lens phase for this depth: π/(λz)*(x^2+y^2)
                        const lensPhase = Math.PI * invDepth / wavelength * (dx * dx + dy * dy);
                        const tiltPhase = (gazeX * dx + gazeY * dy) * tiltFactor / width;
                        const phase = lensPhase + tiltPhase + layerOffset;
                        const cosP = Math.cos(phase);
                        const sinP = Math.sin(phase);
                        // Rotate the current wave by this phase and add to final
                        const inRe = real[i];
                        const inIm = imag[i];
                        finalReal[i] += inRe * cosP - inIm * sinP;
                        finalImag[i] += inRe * sinP + inIm * cosP;
                    }
                }
            }
            // Final complex wave (finalReal, finalImag) is now the synthesized multi-depth hologram.
            // Compute intensity map for visualization (if needed)
            const intensity = new Float32Array(N);
            for (let i = 0; i < N; i++) {
                intensity[i] = finalReal[i] * finalReal[i] + finalImag[i] * finalImag[i];
            }
            return { real: finalReal, imag: finalImag, intensity };
        }
    }
}
