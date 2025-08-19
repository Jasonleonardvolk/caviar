// fftOptimizations.ts
// Additional optimizations and utilities for FFT implementation

export class FFTOptimizations {
    /**
     * Precompute twiddle factor offsets for each stage
     * This avoids the bit-shift calculation in the shader
     */
    static getTwiddleOffsets(size: number): Uint32Array {
        const stages = Math.log2(size);
        const offsets = new Uint32Array(stages);
        
        for (let stage = 0; stage < stages; stage++) {
            // Offset formula: (2^stage - 1)
            offsets[stage] = (1 << stage) - 1;
        }
        
        return offsets;
    }
    
    /**
     * Generate optimized dispatch parameters for different FFT operations
     */
    static getDispatchParams(operation: string, size: number, batchSize: number, workgroupSize: number): {
        x: number;
        y: number;
        z: number;
    } {
        switch (operation) {
            case 'bitReversal':
            case 'normalize':
            case 'fftShift':
                // 1D dispatch for element-wise operations
                const totalElements = size * batchSize;
                return {
                    x: Math.ceil(totalElements / workgroupSize),
                    y: 1,
                    z: 1
                };
                
            case 'butterfly':
                // 1D dispatch for butterfly operations (half the elements)
                const butterflies = (size / 2) * batchSize;
                return {
                    x: Math.ceil(butterflies / workgroupSize),
                    y: 1,
                    z: 1
                };
                
            case 'transpose':
                // 2D tiled dispatch for transpose
                const tilesPerDim = Math.ceil(size / 16); // TILE_SIZE = 16
                return {
                    x: tilesPerDim,
                    y: tilesPerDim,
                    z: batchSize
                };
                
            default:
                throw new Error(`Unknown operation: ${operation}`);
        }
    }
    
    /**
     * Validate workgroup size divisibility
     */
    static validateWorkgroupSize(totalElements: number, workgroupSize: number): number {
        if (totalElements % workgroupSize === 0) {
            return workgroupSize;
        }
        
        // Find the largest divisor <= workgroupSize
        const factors = this.getFactors(totalElements);
        const validSize = factors
            .filter(f => f <= workgroupSize)
            .sort((a, b) => b - a)[0];
        
        return validSize || 1;
    }
    
    private static getFactors(n: number): number[] {
        const factors: number[] = [];
        for (let i = 1; i <= Math.sqrt(n); i++) {
            if (n % i === 0) {
                factors.push(i);
                if (i !== n / i) {
                    factors.push(n / i);
                }
            }
        }
        return factors;
    }
    
    /**
     * Check if GPU supports required features
     */
    static async checkGPUFeatures(device: GPUDevice): Promise<{
        hasTimestampQuery: boolean;
        hasFloat16: boolean;
        maxWorkgroupSizeX: number;
        maxStorageBufferSize: number;
        maxBufferSize: number;
    }> {
        return {
            hasTimestampQuery: device.features.has('timestamp-query'),
            hasFloat16: device.features.has('shader-f16'),
            maxWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
            maxStorageBufferSize: device.limits.maxStorageBufferBindingSize,
            maxBufferSize: device.limits.maxBufferSize
        };
    }
    
    /**
     * Calculate memory requirements for FFT
     */
    static calculateMemoryRequirements(config: {
        size: number;
        dimensions: 1 | 2;
        batchSize: number;
        precision: 'f32' | 'f16';
    }): {
        bufferSize: number;
        twiddleSize: number;
        bitReversalSize: number;
        totalMemory: number;
    } {
        const elementSize = config.precision === 'f32' ? 4 : 2;
        const complexSize = 2; // real + imaginary
        const totalElements = config.dimensions === 2 ? 
            config.size * config.size : config.size;
        
        const bufferSize = totalElements * complexSize * elementSize * config.batchSize;
        const twiddleSize = config.size * Math.log2(config.size) * complexSize * elementSize;
        const bitReversalSize = config.size * 4; // u32
        
        return {
            bufferSize,
            twiddleSize,
            bitReversalSize,
            totalMemory: bufferSize * 2 + twiddleSize + bitReversalSize
        };
    }
    
    /**
     * DEPRECATED: Use ShaderConstantManager.getConstantsForShader() instead
     * Generate specialization constants for shader compilation
     */
    static getSpecializationConstants(config: {
        size: number;
        workgroupSize: number;
        normalizationMode: 'dynamic' | '1/N' | '1/sqrt(N)' | 'none';
    }): Record<string, number> {
        console.warn('⚠️  FFTOptimizations.getSpecializationConstants() is deprecated. Use ShaderConstantManager instead.');
        
        // Convert normalization mode to numeric value
        let normMode = 0;
        switch (config.normalizationMode) {
            case 'dynamic': normMode = 0; break;
            case '1/N': normMode = 1; break;
            case '1/sqrt(N)': normMode = 2; break;
            case 'none': normMode = 3; break;
        }
        
        // Import and use the new manager
        // Note: This maintains backward compatibility while encouraging migration
        return {
            '0': config.workgroupSize,
            '1': normMode
        };
    }
}

// Export types for better integration
export interface FFTMemoryRequirements {
    bufferSize: number;
    twiddleSize: number;
    bitReversalSize: number;
    totalMemory: number;
}

export interface FFTGPUFeatures {
    hasTimestampQuery: boolean;
    hasFloat16: boolean;
    maxWorkgroupSizeX: number;
    maxStorageBufferSize: number;
    maxBufferSize: number;
}
