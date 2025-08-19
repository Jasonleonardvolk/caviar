// fftPrecomputed.ts
// Precomputed values for common FFT sizes

export class FFTPrecomputed {
    // Cache for computed values
    private static twiddleCache = new Map<string, Float32Array>();
    private static bitReversalCache = new Map<number, Uint32Array>();
    
    // Common FFT sizes to precompute at build time
    private static readonly COMMON_SIZES = [256, 512, 1024, 2048, 4096];
    
    static getTwiddleFactors(size: number, direction: 'forward' | 'inverse'): Float32Array {
        const key = `${size}_${direction}`;
        
        // Check cache
        if (this.twiddleCache.has(key)) {
            return this.twiddleCache.get(key)!;
        }
        
        // Generate twiddle factors
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
        
        const result = new Float32Array(data);
        this.twiddleCache.set(key, result);
        return result;
    }
    
    static getBitReversalIndices(size: number): Uint32Array {
        // Check cache
        if (this.bitReversalCache.has(size)) {
            return this.bitReversalCache.get(size)!;
        }
        
        // Generate bit reversal indices
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
        
        this.bitReversalCache.set(size, indices);
        return indices;
    }
    
    // Precompute common sizes
    static precomputeCommonSizes(): void {
        for (const size of this.COMMON_SIZES) {
            this.getTwiddleFactors(size, 'forward');
            this.getTwiddleFactors(size, 'inverse');
            this.getBitReversalIndices(size);
        }
    }
    
    // Generate static data for build-time inclusion
    static generateStaticData(): string {
        let code = '// Auto-generated FFT precomputed data\n\n';
        
        // Twiddle factors
        code += 'export const PRECOMPUTED_TWIDDLES: Record<string, Float32Array> = {\n';
        for (const size of this.COMMON_SIZES) {
            for (const direction of ['forward', 'inverse'] as const) {
                const key = `${size}_${direction}`;
                const data = this.getTwiddleFactors(size, direction);
                code += `    '${key}': new Float32Array([${Array.from(data).join(', ')}]),\n`;
            }
        }
        code += '};\n\n';
        
        // Bit reversal indices
        code += 'export const PRECOMPUTED_BIT_REVERSAL: Record<number, Uint32Array> = {\n';
        for (const size of this.COMMON_SIZES) {
            const data = this.getBitReversalIndices(size);
            code += `    ${size}: new Uint32Array([${Array.from(data).join(', ')}]),\n`;
        }
        code += '};\n';
        
        return code;
    }
}

// Initialize common sizes on module load
if (typeof window !== 'undefined') {
    FFTPrecomputed.precomputeCommonSizes();
}
