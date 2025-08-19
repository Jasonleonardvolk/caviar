// scripts/benchmarkFFT.ts
// Benchmark FFT performance across different configurations

import { FFTCompute } from '../frontend/lib/webgpu/fftCompute';

interface BenchmarkResult {
    size: number;
    batchSize: number;
    dimensions: 1 | 2;
    precision: 'f32' | 'f16';
    avgTime: number;
    minTime: number;
    maxTime: number;
    throughput: number; // elements/second
}

class FFTBenchmark {
    private device: GPUDevice;
    
    constructor(device: GPUDevice) {
        this.device = device;
    }
    
    async runBenchmark(config: {
        size: number;
        batchSize: number;
        dimensions: 1 | 2;
        precision: 'f32' | 'f16';
        iterations: number;
    }): Promise<BenchmarkResult> {
        const fft = new FFTCompute(this.device, {
            size: config.size,
            precision: config.precision,
            normalization: 'symmetric',
            direction: 'forward',
            dimensions: config.dimensions,
            batchSize: config.batchSize
        });
        
        await fft.initialize();
        
        // Create test data
        const elementCount = config.dimensions === 2 ? 
            config.size * config.size : config.size;
        const dataSize = elementCount * 2 * config.batchSize; // complex
        const input = new Float32Array(dataSize);
        
        // Fill with random data
        for (let i = 0; i < dataSize; i++) {
            input[i] = Math.random() * 2 - 1;
        }
        
        // Warmup
        for (let i = 0; i < 5; i++) {
            await fft.execute(input);
        }
        
        // Benchmark
        const times: number[] = [];
        
        for (let i = 0; i < config.iterations; i++) {
            const start = performance.now();
            await fft.execute(input);
            await this.device.queue.onSubmittedWorkDone();
            const elapsed = performance.now() - start;
            times.push(elapsed);
        }
        
        // Calculate statistics
        const avgTime = times.reduce((a, b) => a + b) / times.length;
        const minTime = Math.min(...times);
        const maxTime = Math.max(...times);
        
        // Calculate throughput
        const elementsPerOp = elementCount * config.batchSize;
        const throughput = (elementsPerOp / avgTime) * 1000; // elements/second
        
        fft.destroy();
        
        return {
            size: config.size,
            batchSize: config.batchSize,
            dimensions: config.dimensions,
            precision: config.precision,
            avgTime,
            minTime,
            maxTime,
            throughput
        };
    }
    
    async runComprehensiveBenchmark(): Promise<BenchmarkResult[]> {
        const results: BenchmarkResult[] = [];
        
        // Test configurations
        const sizes = [256, 512, 1024, 2048, 4096];
        const batchSizes = [1, 4, 16];
        const dimensions: (1 | 2)[] = [1, 2];
        
        for (const dimension of dimensions) {
            for (const size of sizes) {
                // Skip very large 2D FFTs
                if (dimension === 2 && size > 2048) continue;
                
                for (const batchSize of batchSizes) {
                    console.log(`Benchmarking ${dimension}D FFT: size=${size}, batch=${batchSize}`);
                    
                    try {
                        const result = await this.runBenchmark({
                            size,
                            batchSize,
                            dimensions: dimension,
                            precision: 'f32',
                            iterations: 20
                        });
                        
                        results.push(result);
                        
                        console.log(`  Average: ${result.avgTime.toFixed(3)}ms`);
                        console.log(`  Throughput: ${(result.throughput / 1e6).toFixed(2)}M elements/s`);
                    } catch (error) {
                        console.error(`  Failed: ${error}`);
                    }
                }
            }
        }
        
        return results;
    }
    
    static formatResults(results: BenchmarkResult[]): string {
        let output = 'FFT Benchmark Results\n';
        output += '====================\n\n';
        
        // Group by dimensions
        const results1D = results.filter(r => r.dimensions === 1);
        const results2D = results.filter(r => r.dimensions === 2);
        
        output += '1D FFT Performance:\n';
        output += 'Size    Batch  Avg(ms)  Min(ms)  Max(ms)  Throughput(ME/s)\n';
        output += '------  -----  -------  -------  -------  ----------------\n';
        
        for (const r of results1D) {
            output += `${r.size.toString().padEnd(6)} `;
            output += `${r.batchSize.toString().padEnd(6)} `;
            output += `${r.avgTime.toFixed(3).padEnd(8)} `;
            output += `${r.minTime.toFixed(3).padEnd(8)} `;
            output += `${r.maxTime.toFixed(3).padEnd(8)} `;
            output += `${(r.throughput / 1e6).toFixed(2)}\n`;
        }
        
        output += '\n2D FFT Performance:\n';
        output += 'Size    Batch  Avg(ms)  Min(ms)  Max(ms)  Throughput(ME/s)\n';
        output += '------  -----  -------  -------  -------  ----------------\n';
        
        for (const r of results2D) {
            output += `${r.size.toString().padEnd(6)} `;
            output += `${r.batchSize.toString().padEnd(6)} `;
            output += `${r.avgTime.toFixed(3).padEnd(8)} `;
            output += `${r.minTime.toFixed(3).padEnd(8)} `;
            output += `${r.maxTime.toFixed(3).padEnd(8)} `;
            output += `${(r.throughput / 1e6).toFixed(2)}\n`;
        }
        
        return output;
    }
}

async function main() {
    if (!navigator.gpu) {
        console.error('WebGPU not supported');
        return;
    }
    
    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance'
    });
    
    if (!adapter) {
        console.error('No GPU adapter found');
        return;
    }
    
    const device = await adapter.requestDevice({
        requiredFeatures: adapter.features.has('timestamp-query') ? ['timestamp-query'] : []
    });
    
    console.log('GPU Adapter:', adapter.name || 'Unknown');
    console.log('Timestamp queries:', device.features.has('timestamp-query') ? 'Enabled' : 'Disabled');
    console.log('');
    
    const benchmark = new FFTBenchmark(device);
    const results = await benchmark.runComprehensiveBenchmark();
    
    console.log('\n' + FFTBenchmark.formatResults(results));
    
    // Save results to file
    const fs = require('fs');
    const resultsJson = JSON.stringify(results, null, 2);
    fs.writeFileSync('fft-benchmark-results.json', resultsJson);
    console.log('\nResults saved to fft-benchmark-results.json');
    
    device.destroy();
}

if (require.main === module) {
    main().catch(console.error);
}

export { FFTBenchmark, BenchmarkResult };
