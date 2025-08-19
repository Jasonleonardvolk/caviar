// Minimal GPU Test Utility - Isolate and test GPU components
// Useful for debugging GPU-related issues in isolation

import { FFTCompute } from './fftCompute';
import { FFTDispatchValidator } from './fftDispatchValidator';

interface TestResult {
    name: string;
    passed: boolean;
    duration: number;
    error?: string;
}

export class MinimalGPUTest {
    private results: TestResult[] = [];
    
    /**
     * Test basic WebGPU availability
     */
    async testWebGPUAvailability(): Promise<TestResult> {
        const start = performance.now();
        try {
            if (!navigator.gpu) {
                throw new Error('WebGPU not supported in this browser');
            }
            
            const adapter = await navigator.gpu.requestAdapter({
                powerPreference: 'high-performance'
            });
            
            if (!adapter) {
                throw new Error('No WebGPU adapter available');
            }
            
            const device = await adapter.requestDevice();
            if (!device) {
                throw new Error('Failed to get WebGPU device');
            }
            
            // Log device info
            console.log('[GPU Test] ✅ WebGPU available:', {
                vendor: adapter.features.has('timestamp-query') ? 'supports timestamps' : 'basic support',
                limits: {
                    maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
                    maxBufferSize: device.limits.maxBufferSize,
                    maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX
                }
            });
            
            device.destroy();
            
            return {
                name: 'WebGPU Availability',
                passed: true,
                duration: performance.now() - start
            };
        } catch (error) {
            return {
                name: 'WebGPU Availability',
                passed: false,
                duration: performance.now() - start,
                error: error instanceof Error ? error.message : String(error)
            };
        }
    }
    
    /**
     * Test minimal FFT computation
     */
    async testMinimalFFT(size: number = 64): Promise<TestResult> {
        const start = performance.now();
        let device: GPUDevice | undefined;
        
        try {
            const adapter = await navigator.gpu?.requestAdapter({ 
                powerPreference: 'high-performance' 
            });
            device = await adapter?.requestDevice();
            
            if (!device) {
                throw new Error('Failed to get WebGPU device');
            }
            
            // Create FFT instance
            const fft = new FFTCompute(device, {
                size,
                precision: 'f32',
                normalization: 'none',
                direction: 'forward',
                dimensions: 1,
                batchSize: 1,
                workgroupSize: Math.min(size, 256)
            });
            
            await fft.initialize();
            console.log(`[GPU Test] FFT initialized for size ${size} ✅`);
            
            // Create test input
            const input = new Float32Array(size * 2); // Complex numbers
            for (let i = 0; i < size; i++) {
                input[i * 2] = Math.sin(2 * Math.PI * i / size); // Real
                input[i * 2 + 1] = 0; // Imaginary
            }
            
            // Execute FFT
            const output = await fft.execute(input) || new Float32Array(0);
            console.log(`[GPU Test] FFT executed ✅`, {
                inputSample: input.slice(0, 4),
                outputSample: output ? output.slice(0, 4) : []
            });
            
            // Basic validation
            if (!output || output.length !== input.length) {
                throw new Error(`Output length mismatch: ${output?.length || 0} vs ${input.length}`);
            }
            
            fft.destroy();
            
            return {
                name: `Minimal FFT (size=${size})`,
                passed: true,
                duration: performance.now() - start
            };
        } catch (error) {
            return {
                name: `Minimal FFT (size=${size})`,
                passed: false,
                duration: performance.now() - start,
                error: error instanceof Error ? error.message : String(error)
            };
        } finally {
            device?.destroy();
        }
    }
    
    /**
     * Test buffer creation and copy
     */
    async testBufferOperations(): Promise<TestResult> {
        const start = performance.now();
        let device: GPUDevice | undefined;
        
        try {
            const adapter = await navigator.gpu?.requestAdapter();
            device = await adapter?.requestDevice();
            
            if (!device) {
                throw new Error('Failed to get WebGPU device');
            }
            
            // Create a buffer
            const size = 1024; // 1KB
            const buffer = device.createBuffer({
                size,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
            });
            
            // Write data
            const data = new Float32Array(size / 4);
            data.fill(42.0);
            device.queue.writeBuffer(buffer, 0, data.slice());
            
            // Create staging buffer for readback
            const stagingBuffer = device.createBuffer({
                size,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            
            // Copy data
            const commandEncoder = device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
            device.queue.submit([commandEncoder.finish()]);
            
            // Read back
            await stagingBuffer.mapAsync(GPUMapMode.READ);
            const readback = new Float32Array(stagingBuffer.getMappedRange());
            
            if (readback[0] !== 42.0) {
                throw new Error(`Readback failed: expected 42.0, got ${readback[0]}`);
            }
            
            stagingBuffer.unmap();
            buffer.destroy();
            stagingBuffer.destroy();
            
            console.log('[GPU Test] Buffer operations ✅');
            
            return {
                name: 'Buffer Operations',
                passed: true,
                duration: performance.now() - start
            };
        } catch (error) {
            return {
                name: 'Buffer Operations',
                passed: false,
                duration: performance.now() - start,
                error: error instanceof Error ? error.message : String(error)
            };
        } finally {
            device?.destroy();
        }
    }
    
    /**
     * Test compute shader dispatch
     */
    async testComputeDispatch(): Promise<TestResult> {
        const start = performance.now();
        let device: GPUDevice | undefined;
        
        try {
            const adapter = await navigator.gpu?.requestAdapter();
            device = await adapter?.requestDevice();
            
            if (!device) {
                throw new Error('Failed to get WebGPU device');
            }
            
            // Simple compute shader that doubles values
            const shaderCode = `
                @group(0) @binding(0) var<storage, read_write> data: array<f32>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let index = global_id.x;
                    if (index < arrayLength(&data)) {
                        data[index] = data[index] * 2.0;
                    }
                }
            `;
            
            const shaderModule = device.createShaderModule({
                code: shaderCode
            });
            
            const pipeline = device.createComputePipeline({
                layout: 'auto',
                compute: {
                    module: shaderModule,
                    entryPoint: 'main'
                }
            });
            
            // Create buffer
            const dataSize = 256;
            const buffer = device.createBuffer({
                size: dataSize * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });
            
            // Initialize data
            const inputData = new Float32Array(dataSize);
            for (let i = 0; i < dataSize; i++) {
                inputData[i] = i;
            }
            device.queue.writeBuffer(buffer, 0, inputData.slice(.buffer));
            
            // Create bind group
            const bindGroup = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [{
                    binding: 0,
                    resource: { buffer }
                }]
            });
            
            // Dispatch compute
            const commandEncoder = device.createCommandEncoder();
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(pipeline);
            computePass.setBindGroup(0, bindGroup);
            
            const workgroups = Math.ceil(dataSize / 64);
            console.log(`[GPU Test] Dispatching compute with ${workgroups} workgroups`);
            
            // Validate dispatch
            FFTDispatchValidator.validateAndLog('test_compute', { x: workgroups, y: 1, z: 1 }, device);
            
            computePass.dispatchWorkgroups(workgroups);
            computePass.end();
            
            // Submit
            device.queue.submit([commandEncoder.finish()]);
            
            console.log('[GPU Test] Compute dispatch ✅');
            
            buffer.destroy();
            
            return {
                name: 'Compute Dispatch',
                passed: true,
                duration: performance.now() - start
            };
        } catch (error) {
            return {
                name: 'Compute Dispatch',
                passed: false,
                duration: performance.now() - start,
                error: error instanceof Error ? error.message : String(error)
            };
        } finally {
            device?.destroy();
        }
    }
    
    /**
     * Run all tests
     */
    async runAllTests(): Promise<void> {
        console.log('🧪 Starting Minimal GPU Tests...\n');
        
        const tests = [
            () => this.testWebGPUAvailability(),
            () => this.testBufferOperations(),
            () => this.testComputeDispatch(),
            () => this.testMinimalFFT(64),
            () => this.testMinimalFFT(256),
            () => this.testMinimalFFT(1024)
        ];
        
        for (const test of tests) {
            const result = await test();
            this.results.push(result);
            
            if (result.passed) {
                console.log(`✅ ${result.name} - ${result.duration.toFixed(2)}ms`);
            } else {
                console.error(`❌ ${result.name} - ${result.error}`);
            }
        }
        
        this.printSummary();
    }
    
    /**
     * Print test summary
     */
    private printSummary(): void {
        const passed = this.results.filter(r => r.passed).length;
        const total = this.results.length;
        const totalDuration = this.results.reduce((sum, r) => sum + r.duration, 0);
        
        console.log('\n📊 Test Summary:');
        console.log(`   Total tests: ${total}`);
        console.log(`   Passed: ${passed}`);
        console.log(`   Failed: ${total - passed}`);
        console.log(`   Total duration: ${totalDuration.toFixed(2)}ms`);
        
        if (passed === total) {
            console.log('\n🎉 All tests passed!');
        } else {
            console.log('\n⚠️ Some tests failed. Check the logs above for details.');
        }
    }
}

// Export convenience function
export async function runMinimalGPUTests(): Promise<void> {
    const tester = new MinimalGPUTest();
    await tester.runAllTests();
}

// Allow running from command line
if (typeof window === 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
    runMinimalGPUTests().catch(console.error);
}
