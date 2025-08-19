// Auto-generated FFT precomputed data
// Includes sizes up to 8192 for >4K diffraction fields

export interface PrecomputedFFTData {
    size: number;
    twiddlesForward: Float32Array;
    twiddlesInverse: Float32Array;
    bitReversal: Uint32Array;
    twiddleOffsets: Uint32Array;
}

// Helper to generate twiddle factors
function generateTwiddles(size: number, direction: 'forward' | 'inverse'): Float32Array {
    const sign = direction === 'forward' ? -1 : 1;
    const stages = Math.log2(size);
    const data: number[] = [];
    
    for (let stage = 0; stage < stages; stage++) {
        const stageSize = 1 << (stage + 1);
        const halfStageSize = stageSize >> 1;
        
        for (let k = 0; k < halfStageSize; k++) {
            const angle = sign * 2 * Math.PI * k / stageSize;
            data.push(Math.cos(angle), Math.sin(angle));
        }
    }
    
    return new Float32Array(data);
}

// Helper to generate bit reversal indices
function generateBitReversal(size: number): Uint32Array {
    const bits = Math.log2(size);
    const indices = new Uint32Array(size);
    
    for (let i = 0; i < size; i++) {
        let reversed = 0;
        let temp = i;
        
        for (let j = 0; j < bits; j++) {
            reversed = (reversed << 1) | (temp & 1);
            temp >>= 1;
        }
        
        indices[i] = reversed;
    }
    
    return indices;
}

// Helper to generate twiddle offsets for each stage
function generateTwiddleOffsets(size: number): Uint32Array {
    const stages = Math.log2(size);
    const offsets = new Uint32Array(stages);
    let offset = 0;
    
    for (let stage = 0; stage < stages; stage++) {
        offsets[stage] = offset;
        const stageSize = 1 << (stage + 1);
        const halfStageSize = stageSize >> 1;
        offset += halfStageSize * 2; // 2 floats per complex number
    }
    
    return offsets;
}

// Precomputed data for common sizes
const PRECOMPUTED_DATA = new Map<number, PrecomputedFFTData>();

// Generate data for sizes: 256, 512, 1024, 2048, 4096, 8192
const SIZES = [256, 512, 1024, 2048, 4096, 8192];

for (const size of SIZES) {
    PRECOMPUTED_DATA.set(size, {
        size,
        twiddlesForward: generateTwiddles(size, 'forward'),
        twiddlesInverse: generateTwiddles(size, 'inverse'),
        bitReversal: generateBitReversal(size),
        twiddleOffsets: generateTwiddleOffsets(size)
    });
}

// Export functions
export function isPrecomputedSize(size: number): boolean {
    return PRECOMPUTED_DATA.has(size);
}

export function getPrecomputedData(size: number): PrecomputedFFTData | null {
    return PRECOMPUTED_DATA.get(size) || null;
}

export function getAvailableSizes(): number[] {
    return Array.from(PRECOMPUTED_DATA.keys());
}

// For build-time generation
export function generateStaticExports(): string {
    let code = '// Static FFT data exports\n\n';
    
    for (const [size, data] of PRECOMPUTED_DATA) {
        code += `export const FFT_${size}_FORWARD = new Float32Array([${Array.from(data.twiddlesForward).join(', ')}]);\n`;
        code += `export const FFT_${size}_INVERSE = new Float32Array([${Array.from(data.twiddlesInverse).join(', ')}]);\n`;
        code += `export const FFT_${size}_BIT_REVERSAL = new Uint32Array([${Array.from(data.bitReversal).join(', ')}]);\n`;
        code += `export const FFT_${size}_OFFSETS = new Uint32Array([${Array.from(data.twiddleOffsets).join(', ')}]);\n\n`;
    }
    
    return code;
}
