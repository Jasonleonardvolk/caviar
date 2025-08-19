// fftCompute.test.ts
// Comprehensive test suite for FFT implementation

import { FFTCompute } from './fftCompute';
import { describe, test, expect, beforeAll, afterAll } from '@jest/globals';

// Reference FFT implementation for validation
class ReferenceFFT {
    static fft(input: Float32Array, inverse: boolean = false): Float32Array {
        const n = input.length / 2;
        if (n === 1) return new Float32Array(input);
        
        // Check if n is power of 2
        if ((n & (n - 1)) !== 0) {
            throw new Error('Size must be power of 2');
        }
        
        // Bit reversal
        const reversed = new Float32Array(input.length);
        const bits = Math.log2(n);
        
        for (let i = 0; i < n; i++) {
            let rev = 0;
            let temp = i;
            for (let j = 0; j < bits; j++) {
                rev = (rev << 1) | (temp & 1);
                temp >>= 1;
            }
            reversed[rev * 2] = input[i * 2];
            reversed[rev * 2 + 1] = input[i * 2 + 1];
        }
        
        // Cooley-Tukey FFT
        const sign = inverse ? 1 : -1;
        for (let size = 2; size <= n; size *= 2) {
            const halfsize = size / 2;
            const step = n / size;
            
            for (let i = 0; i < n; i += size) {
                for (let j = 0; j < halfsize; j++) {
                    const l = i + j;
                    const r = i + j + halfsize;
                    
                    const angle = sign * 2 * Math.PI * j / size;
                    const twiddle_r = Math.cos(angle);
                    const twiddle_i = Math.sin(angle);
                    
                    const a_r = reversed[l * 2];
                    const a_i = reversed[l * 2 + 1];
                    const b_r = reversed[r * 2];
                    const b_i = reversed[r * 2 + 1];
                    
                    const b_twiddle_r = b_r * twiddle_r - b_i * twiddle_i;
                    const b_twiddle_i = b_r * twiddle_i + b_i * twiddle_r;
                    
                    reversed[l * 2] = a_r + b_twiddle_r;
                    reversed[l * 2 + 1] = a_i + b_twiddle_i;
                    reversed[r * 2] = a_r - b_twiddle_r;
                    reversed[r * 2 + 1] = a_i - b_twiddle_i;
                }
            }
        }
        
        return reversed;
    }
    
    static impulseResponse(size: number): Float32Array {
        const signal = new Float32Array(size * 2);
        signal[0] = 1; // Real part of first element
        return signal;
    }
    
    static sineWave(size: number, frequency: number, phase: number = 0): Float32Array {
        const signal = new Float32Array(size * 2);
        for (let i = 0; i < size; i++) {
            signal[i * 2] = Math.cos(2 * Math.PI * frequency * i / size + phase);
            signal[i * 2 + 1] = Math.sin(2 * Math.PI * frequency * i / size + phase);
        }
        return signal;
    }
    
    static randomSignal(size: number): Float32Array {
        const signal = new Float32Array(size * 2);
        for (let i = 0; i < size * 2; i++) {
            signal[i] = Math.random() * 2 - 1;
        }
        return signal;
    }
}

describe('FFTCompute', () => {
    let device: GPUDevice;
    
    beforeAll(async () => {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }
        
        const adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance'
        });
        if (!adapter) {
            throw new Error('No GPU adapter found');
        }
        
        device = await adapter.requestDevice({
            requiredFeatures: adapter.features.has('timestamp-query') ? ['timestamp-query'] : []
        });
    });
    
    afterAll(() => {
        device.destroy();
    });
    
    describe('Configuration Validation', () => {
        test('should reject non-power-of-2 sizes', () => {
            expect(() => new FFTCompute(device, {
                size: 100,
                precision: 'f32',
                normalization: 'none',
                direction: 'forward',
                dimensions: 1
            })).toThrow('FFT size must be power of 2');
        });
        
        test('should accept valid power-of-2 sizes', () => {
            const validSizes = [256, 512, 1024, 2048, 4096];
            
            validSizes.forEach(size => {
                const fft = new FFTCompute(device, {
                    size,
                    precision: 'f32',
                    normalization: 'none',
                    direction: 'forward',
                    dimensions: 1
                });
                expect(fft).toBeDefined();
            });
        });
        
        test('should validate input size', async () => {
            const fft = new FFTCompute(device, {
                size: 16,
                precision: 'f32',
                normalization: 'none',
                direction: 'forward',
                dimensions: 1
            });
            
            await fft.initialize();
            
            const wrongSize = new Float32Array(30); // Wrong size
            await expect(fft.execute(wrongSize)).rejects.toThrow('Input size mismatch');
            
            fft.destroy();
        });
    });
    
    describe('Core FFT Tests', () => {
        test('impulse response should give all ones', async () => {
            const size = 16;
            const fft = new FFTCompute(device, {
                size,
                precision: 'f32',
                normalization: 'none',
                direction: 'forward',
                dimensions: 1
            });
            
            await fft.initialize();
            
            const input = ReferenceFFT.impulseResponse(size);
            const result = await fft.execute(input);
            
            // FFT of impulse should be all ones
            for (let i = 0; i < size; i++) {
                expect(result![i * 2]).toBeCloseTo(1, 5);     // Real
                expect(result![i * 2 + 1]).toBeCloseTo(0, 5); // Imaginary
            }
            
            fft.destroy();
        });
        
        test('single tone should show peak at correct frequency', async () => {
            const size = 64;
            const frequency = 5; // 5 cycles in the window
            
            const fft = new FFTCompute(device, {
                size,
                precision: 'f32',
                normalization: 'forward',
                direction: 'forward',
                dimensions: 1
            });
            
            await fft.initialize();
            
            // Generate pure cosine wave
            const input = new Float32Array(size * 2);
            for (let i = 0; i < size; i++) {
                input[i * 2] = Math.cos(2 * Math.PI * frequency * i / size);
                input[i * 2 + 1] = 0; // No imaginary part
            }
            
            const result = await fft.execute(input);
            
            // Check for peaks at frequency bin
            for (let i = 0; i < size; i++) {
                const magnitude = Math.sqrt(result![i * 2] ** 2 + result![i * 2 + 1] ** 2);
                
                if (i === frequency || i === size - frequency) {
                    // Should have peak at frequency and its mirror
                    expect(magnitude).toBeGreaterThan(0.4);
                } else {
                    // Other bins should be near zero
                    expect(magnitude).toBeLessThan(0.1);
                }
            }
            
            fft.destroy();
        });
        
        test('round-trip FFT should recover original signal', async () => {
            const size = 128;
            const fft = new FFTCompute(device, {
                size,
                precision: 'f32',
                normalization: 'inverse',
                direction: 'forward',
                dimensions: 1
            });
            
            await fft.initialize();
            
            // Generate random signal
            const original = ReferenceFFT.randomSignal(size);
            
            // Forward FFT
            const spectrum = await fft.execute(original);
            
            // Change to inverse direction
            fft.resize(size); // This will reinitialize with same size
            const fftInverse = new FFTCompute(device, {
                size,
                precision: 'f32',
                normalization: 'inverse',
                direction: 'inverse',
                dimensions: 1
            });
            await fftInverse.initialize();
            
            // Inverse FFT
            const recovered = await fftInverse.execute(spectrum!);
            
            // Compare with original
            for (let i = 0; i < original.length; i++) {
                expect(recovered![i]).toBeCloseTo(original[i], 4);
            }
            
            fft.destroy();
            fftInverse.destroy();
        });
        
        test('should match reference FFT implementation', async () => {
            const size = 32;
            const fft = new FFTCompute(device, {
                size,
                precision: 'f32',
                normalization: 'none',
                direction: 'forward',
                dimensions: 1
            });
            
            await fft.initialize();
            
            // Test with complex signal
            const input = ReferenceFFT.sineWave(size, 3, Math.PI / 4);
            
            const gpuResult = await fft.execute(input);
            const cpuResult = ReferenceFFT.fft(input, false);
            
            // Compare results
            for (let i = 0; i < size * 2; i++) {
                expect(gpuResult![i]).toBeCloseTo(cpuResult[i], 4);
            }
            
            fft.destroy();
        });
    });
    
    describe('Batch Processing', () => {
        test('should handle multiple FFTs in parallel', async () => {
            const size = 64;
            const batchSize = 4;
            
            const fft = new FFTCompute(device, {
                size,
                precision: 'f32',
                normalization: 'symmetric',
                direction: 'forward',
                dimensions: 1,
                batchSize
            });
            
            await fft.initialize();
            
            // Create batched input with different signals
            const input = new Float32Array(size * 2 * batchSize);
            for (let b = 0; b < batchSize; b++) {
                const offset = b * size * 2;
                
                // Each batch gets a different frequency
                const frequency = b + 1;
                for (let i = 0; i < size; i++) {
                    input[offset + i * 2] = Math.cos(2 * Math.PI * frequency * i / size);
                    input[offset + i * 2 + 1] = 0;
                }
            }
            
            const result = await fft.execute(input);
            
            // Verify each batch has peak at correct frequency
            for (let b = 0; b < batchSize; b++) {
                const offset = b * size * 2;
                const expectedFreq = b + 1;
                
                for (let i = 0; i < size; i++) {
                    const magnitude = Math.sqrt(
                        result![offset + i * 2] ** 2 + 
                        result![offset + i * 2 + 1] ** 2
                    );
                    
                    if (i === expectedFreq || i === size - expectedFreq) {
                        expect(magnitude).toBeGreaterThan(0.3);
                    } else {
                        expect(magnitude).toBeLessThan(0.1);
                    }
                }
            }
            
            fft.destroy();
        });
    });
    
    describe('Normalization Modes', () => {
        const testNormalization = async (mode: 'none' | 'forward' | 'inverse' | 'symmetric') => {
            const size = 32;
            const fft = new FFTCompute(device, {
                size,
                precision: 'f32',
                normalization: mode,
                direction: 'forward',
                dimensions: 1
            });
            
            await fft.initialize();
            
            const input = ReferenceFFT.impulseResponse(size);
            const result = await fft.execute(input);
            
            // Check normalization
            const expectedValue = mode === 'none' ? 1.0 :
                                 mode === 'forward' ? 1.0 / size :
                                 mode === 'symmetric' ? 1.0 / Math.sqrt(size) : 1.0;
            
            expect(result![0]).toBeCloseTo(expectedValue, 5);
            
            fft.destroy();
        };
        
        test('none normalization', () => testNormalization('none'));
        test('forward normalization', () => testNormalization('forward'));
        test('symmetric normalization', () => testNormalization('symmetric'));
    });
    
    describe('Performance', () => {
        test('should track performance metrics', async () => {
            const features = device.features;
            if (!features.has('timestamp-query')) {
                console.log('Timestamp queries not supported, skipping performance test');
                return;
            }
            
            const fft = new FFTCompute(device, {
                size: 1024,
                precision: 'f32',
                normalization: 'symmetric',
                direction: 'forward',
                dimensions: 1
            });
            
            await fft.initialize();
            
            const input = new Float32Array(1024 * 2);
            
            // Warm up
            await fft.execute(input);
            
            // Run multiple times
            for (let i = 0; i < 10; i++) {
                await fft.execute(input);
            }
            
            const stats = fft.getPerformanceStats();
            expect(stats).not.toBeNull();
            expect(stats!.average).toBeGreaterThan(0);
            expect(stats!.min).toBeLessThanOrEqual(stats!.average);
            expect(stats!.max).toBeGreaterThanOrEqual(stats!.average);
            
            console.log(`FFT Performance: avg=${stats!.average.toFixed(3)}ms, min=${stats!.min.toFixed(3)}ms, max=${stats!.max.toFixed(3)}ms`);
            
            fft.destroy();
        });
    });
    
    describe('Buffer Reuse', () => {
        test('should reuse buffers when enabled', async () => {
            const fft = new FFTCompute(device, {
                size: 256,
                precision: 'f32',
                normalization: 'none',
                direction: 'forward',
                dimensions: 1,
                reuseBuffers: true
            });
            
            await fft.initialize();
            
            const input = new Float32Array(256 * 2);
            
            // Execute multiple times
            const results = [];
            for (let i = 0; i < 5; i++) {
                // Fill with different data
                for (let j = 0; j < input.length; j++) {
                    input[j] = Math.random();
                }
                
                const result = await fft.execute(input);
                results.push(result);
            }
            
            // Verify results are different (not reusing output data)
            expect(results[0]).not.toEqual(results[1]);
            
            fft.destroy();
        });
    });
    
    describe('Resize Functionality', () => {
        test('should handle size changes', async () => {
            let fft = new FFTCompute(device, {
                size: 256,
                precision: 'f32',
                normalization: 'none',
                direction: 'forward',
                dimensions: 1
            });
            
            await fft.initialize();
            
            // Execute with original size
            let input = new Float32Array(256 * 2);
            let result = await fft.execute(input);
            expect(result!.length).toBe(512);
            
            // Resize
            fft.resize(512);
            await fft.initialize();
            
            // Execute with new size
            input = new Float32Array(512 * 2);
            result = await fft.execute(input);
            expect(result!.length).toBe(1024);
            
            fft.destroy();
        });
    });
});

// Benchmark suite
describe('FFT Benchmarks', () => {
    let device: GPUDevice;
    
    beforeAll(async () => {
        const adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance'
        });
        device = await adapter!.requestDevice();
    });
    
    test('benchmark common sizes', async () => {
        const sizes = [256, 512, 1024, 2048, 4096];
        const results: Record<number, { gpu: number; cpu: number; speedup: number }> = {};
        
        for (const size of sizes) {
            // GPU FFT
            const fft = new FFTCompute(device, {
                size,
                precision: 'f32',
                normalization: 'symmetric',
                direction: 'forward',
                dimensions: 1
            });
            
            await fft.initialize();
            
            const input = ReferenceFFT.randomSignal(size);
            
            // Warmup
            await fft.execute(input);
            
            // GPU benchmark
            const gpuIterations = 50;
            const gpuStart = performance.now();
            
            for (let i = 0; i < gpuIterations; i++) {
                await fft.execute(input);
            }
            
            await device.queue.onSubmittedWorkDone();
            const gpuTime = (performance.now() - gpuStart) / gpuIterations;
            
            // CPU benchmark
            const cpuIterations = 10;
            const cpuStart = performance.now();
            
            for (let i = 0; i < cpuIterations; i++) {
                ReferenceFFT.fft(input);
            }
            
            const cpuTime = (performance.now() - cpuStart) / cpuIterations;
            
            results[size] = {
                gpu: gpuTime,
                cpu: cpuTime,
                speedup: cpuTime / gpuTime
            };
            
            console.log(`Size ${size}: GPU=${gpuTime.toFixed(3)}ms, CPU=${cpuTime.toFixed(3)}ms, Speedup=${results[size].speedup.toFixed(1)}x`);
            
            fft.destroy();
        }
        
        // Verify GPU is faster for larger sizes
        expect(results[4096].speedup).toBeGreaterThan(1);
    });
});
