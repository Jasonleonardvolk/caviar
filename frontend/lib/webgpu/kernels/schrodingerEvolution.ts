/**
 * schrodingerEvolution.ts
 * PLATINUM Edition: Main integration point for all Schrödinger evolution methods
 * 
 * This is the primary interface for using the various evolution kernels.
 * Provides a simple API while supporting advanced features.
 */

import { SchrodingerRegistry, SchrodingerKernel, KernelType } from './schrodingerKernelRegistry';
import { runSchrodingerBenchmark, BenchmarkMethod } from './schrodingerBenchmark';
import { FilterType, FilterPresets, generateSpectralFilterShader } from './spectralFiltering';
import { SplitStepOrchestrator } from './splitStepOrchestrator';
import { OnnxWaveOpRunner } from './onnxWaveOpRunner';

export interface EvolutionConfig {
    width: number;
    height: number;
    dt: number;
    method?: 'auto' | 'biharmonic' | 'splitstep' | 'onnx' | 'hybrid';
    potential?: {
        type: 'harmonic' | 'double-well' | 'custom';
        strength?: number;
        texture?: GPUTexture;
    };
    boundary?: {
        type: 'periodic' | 'absorbing' | 'reflecting' | 'pml' | 'airy';
        params?: any;
    };
    filtering?: {
        enabled: boolean;
        type?: FilterType;
        cutoff?: number;
    };
    optimization?: {
        enableSubgroups?: boolean;
        enableBatching?: boolean;
        cacheSize?: number;
    };
}

/**
 * Main Schrödinger Evolution class
 * Provides a unified interface to all evolution methods
 */
export class SchrodingerEvolution {
    private device: GPUDevice;
    private config: Required<EvolutionConfig>;
    private activeKernel: SchrodingerKernel | null = null;
    
    // Resources
    private fieldBuffer: GPUBuffer;
    private potentialTexture: GPUTexture;
    private filterPipeline?: GPUComputePipeline;
    
    // Telemetry
    private frameCount: number = 0;
    private totalTime: number = 0;
    private lastBenchmark: Map<string, any> | null = null;
    
    constructor(device: GPUDevice, config: EvolutionConfig) {
        this.device = device;
        this.config = this.fillConfig(config);
        
        // Create resources
        this.fieldBuffer = this.createFieldBuffer();
        this.potentialTexture = this.createPotentialTexture();
    }
    
    private fillConfig(config: EvolutionConfig): Required<EvolutionConfig> {
        return {
            width: config.width,
            height: config.height,
            dt: config.dt,
            method: config.method || 'auto',
            potential: config.potential || {
                type: 'harmonic',
                strength: 1.0,
            },
            boundary: config.boundary || {
                type: 'periodic',
                params: {},
            },
            filtering: config.filtering || {
                enabled: false,
                type: FilterType.Gaussian,
                cutoff: 0.8,
            },
            optimization: config.optimization || {
                enableSubgroups: true,
                enableBatching: false,
                cacheSize: 10,
            },
        };
    }
    
    /**
     * Initialize the evolution system
     */
    async initialize(): Promise<void> {
        console.log('[Evolution] Initializing Schrödinger evolution system');
        
        // Initialize registry
        await SchrodingerRegistry.initialize(this.device);
        
        // Select and activate kernel
        await this.selectKernel();
        
        // Setup filtering if enabled
        if (this.config.filtering.enabled) {
            await this.setupFiltering();
        }
        
        console.log('[Evolution] Initialization complete');
    }
    
    private async selectKernel(): Promise<void> {
        let kernelId: string;
        
        switch (this.config.method) {
            case 'biharmonic':
                kernelId = 'biharmonic-fd';
                break;
            
            case 'splitstep':
                kernelId = 'splitstep-fft-platinum';
                break;
            
            case 'onnx':
                kernelId = 'onnx-neural';
                break;
            
            case 'hybrid':
                // Use split-step for evolution, ONNX for complex potentials
                kernelId = 'splitstep-fft-platinum';
                break;
            
            case 'auto':
            default:
                // Let registry recommend based on requirements
                const recommended = SchrodingerRegistry.recommendKernel({
                    width: this.config.width,
                    height: this.config.height,
                    speed: 'fast',
                    accuracy: 'high',
                    features: this.config.optimization.enableBatching ? ['batching'] : [],
                });
                
                if (!recommended) {
                    throw new Error('No suitable kernel found');
                }
                
                kernelId = recommended.id;
                break;
        }
        
        // Prepare kernel configuration
        const kernelConfig = {
            width: this.config.width,
            height: this.config.height,
            dt: this.config.dt,
            dx: 1.0,
            dy: 1.0,
            boundaryType: this.getBoundaryType(),
            boundaryParams: this.config.boundary.params,
            enableTelemetry: true,
            batchSize: this.config.optimization.enableBatching ? 4 : 1,
        };
        
        // Activate kernel
        await SchrodingerRegistry.activateKernel(kernelId, kernelConfig);
        this.activeKernel = SchrodingerRegistry.getActiveKernel();
        
        console.log(`[Evolution] Activated kernel: ${this.activeKernel?.name}`);
    }
    
    private getBoundaryType(): number {
        const typeMap: Record<string, number> = {
            'periodic': 0,
            'absorbing': 1,
            'reflecting': 2,
            'pml': 2,
            'airy': 3,
        };
        return typeMap[this.config.boundary.type] || 0;
    }
    
    private createFieldBuffer(): GPUBuffer {
        const size = this.config.width * this.config.height * 8;  // Complex float32
        
        const buffer = this.device.createBuffer({
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: 'Wave Field Buffer',
        });
        
        // Initialize with Gaussian wave packet
        const initialData = this.createInitialWavePacket();
        this.device.queue.writeBuffer(buffer, 0, initialData.buffer as ArrayBuffer.buffer);
        
        return buffer;
    }
    
    private createInitialWavePacket(): Float32Array {
        const { width, height } = this.config;
        const data = new Float32Array(width * height * 2);
        
        const sigma = width / 10;
        const k0 = 2 * Math.PI / width * 5;
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 2;
                
                const dx = x - width / 2;
                const dy = y - height / 2;
                const r2 = dx * dx + dy * dy;
                
                const amplitude = Math.exp(-r2 / (2 * sigma * sigma));
                const phase = k0 * dx;
                
                data[idx] = amplitude * Math.cos(phase);
                data[idx + 1] = amplitude * Math.sin(phase);
            }
        }
        
        // Normalize
        let norm = 0;
        for (let i = 0; i < data.length; i += 2) {
            norm += data[i] * data[i] + data[i + 1] * data[i + 1];
        }
        norm = Math.sqrt(norm);
        
        for (let i = 0; i < data.length; i++) {
            data[i] /= norm;
        }
        
        return data;
    }
    
    private createPotentialTexture(): GPUTexture {
        const { width, height } = this.config;
        
        if (this.config.potential.texture) {
            return this.config.potential.texture;
        }
        
        const texture = this.device.createTexture({
            size: { width, height },
            format: 'rgba32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            label: 'Potential Texture',
        });
        
        // Generate potential based on type
        const data = new Float32Array(width * height * 4);
        
        switch (this.config.potential.type) {
            case 'harmonic':
                this.generateHarmonicPotential(data);
                break;
            
            case 'double-well':
                this.generateDoubleWellPotential(data);
                break;
            
            case 'custom':
                // User should provide texture
                break;
        }
        
        this.device.queue.writeTexture(
            { texture },
            data,
            { bytesPerRow: width * 16 },
            { width, height }
        );
        
        return texture;
    }
    
    private generateHarmonicPotential(data: Float32Array): void {
        const { width, height } = this.config;
        const omega = 0.01 * (this.config.potential.strength || 1.0);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                
                const dx = x - width / 2;
                const dy = y - height / 2;
                const r2 = dx * dx + dy * dy;
                
                data[idx] = 0.5 * omega * omega * r2;  // V(x,y)
                data[idx + 1] = 0;  // Imaginary part (absorption)
                data[idx + 2] = 0;  // Reserved
                data[idx + 3] = 1;  // Mask (1 = no absorption)
            }
        }
    }
    
    private generateDoubleWellPotential(data: Float32Array): void {
        const { width, height } = this.config;
        const a = 0.01 * (this.config.potential.strength || 1.0);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                
                const dx = (x - width / 2) / (width / 4);
                const dy = (y - height / 2) / (height / 4);
                
                // Double well: V(x) = a(x^2 - 1)^2
                const potential = a * Math.pow(dx * dx - 1, 2);
                
                data[idx] = potential;
                data[idx + 1] = 0;
                data[idx + 2] = 0;
                data[idx + 3] = 1;
            }
        }
    }
    
    private async setupFiltering(): Promise<void> {
        if (!this.config.filtering.enabled) return;
        
        const filterConfig = {
            width: this.config.width,
            height: this.config.height,
            dx: 1.0,
            dy: 1.0,
            filterType: this.config.filtering.type!,
            cutoffFrequency: this.config.filtering.cutoff!,
            rolloff: 0.1,
            order: 4,
        };
        
        const shaderCode = generateSpectralFilterShader(filterConfig);
        
        const shaderModule = this.device.createShaderModule({
            label: 'Spectral Filter',
            code: shaderCode,
        });
        
        this.filterPipeline = this.device.createComputePipeline({
            label: 'Filter Pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
        
        console.log('[Evolution] Spectral filtering enabled');
    }
    
    /**
     * Evolve the wave function by one time step
     */
    evolve(): void {
        const startTime = performance.now();
        
        const commandEncoder = this.device.createCommandEncoder({
            label: 'Evolution Command Encoder',
        });
        
        // Convert buffer to texture for kernel input
        const fieldTexture = this.bufferToTexture(this.fieldBuffer);
        
        // Execute evolution kernel
        if (this.activeKernel) {
            SchrodingerRegistry.execute(commandEncoder, {
                fieldTexture,
                potentialTexture: this.potentialTexture,
                dt: this.config.dt,
            });
        }
        
        // Apply spectral filtering if enabled
        if (this.filterPipeline) {
            this.applyFiltering(commandEncoder);
        }
        
        this.device.queue.submit([commandEncoder.finish()]);
        
        // Update telemetry
        this.frameCount++;
        this.totalTime += performance.now() - startTime;
    }
    
    /**
     * Evolve for multiple steps
     */
    async evolveSteps(steps: number): Promise<void> {
        console.log(`[Evolution] Evolving ${steps} steps`);
        
        const startTime = performance.now();
        
        for (let i = 0; i < steps; i++) {
            this.evolve();
            
            if ((i + 1) % 100 === 0) {
                console.log(`  Step ${i + 1}/${steps}`);
            }
        }
        
        await this.device.queue.onSubmittedWorkDone();
        
        const totalTime = performance.now() - startTime;
        console.log(`[Evolution] Completed in ${totalTime.toFixed(2)}ms`);
        console.log(`  Average: ${(totalTime / steps).toFixed(3)}ms per step`);
    }
    
    private bufferToTexture(buffer: GPUBuffer): GPUTexture {
        // In production, this would be cached
        const texture = this.device.createTexture({
            size: { width: this.config.width, height: this.config.height },
            format: 'rg32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToTexture(
            { buffer, bytesPerRow: this.config.width * 8 },
            { texture },
            { width: this.config.width, height: this.config.height }
        );
        this.device.queue.submit([commandEncoder.finish()]);
        
        return texture;
    }
    
    private applyFiltering(commandEncoder: GPUCommandEncoder): void {
        if (!this.filterPipeline) return;
        
        const computePass = commandEncoder.beginComputePass({
            label: 'Spectral Filtering',
        });
        
        computePass.setPipeline(this.filterPipeline);
        // Set bind groups...
        
        const workgroups = Math.ceil((this.config.width * this.config.height) / 256);
        computePass.dispatchWorkgroups(workgroups);
        
        computePass.end();
    }
    
    /**
     * Run benchmark comparison
     */
    async benchmark(): Promise<Map<string, any>> {
        console.log('[Evolution] Running benchmark...');
        
        const results = await runSchrodingerBenchmark(this.device, {
            width: this.config.width,
            height: this.config.height,
            dt: this.config.dt,
            steps: 100,
            warmupSteps: 10,
            iterations: 20,
            methods: [BenchmarkMethod.All],
            exportResults: true,
        });
        
        this.lastBenchmark = results;
        return results;
    }
    
    /**
     * Get current wave function data
     */
    async getWaveFunction(): Promise<Float32Array> {
        const readBuffer = this.device.createBuffer({
            size: this.fieldBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            this.fieldBuffer,
            0,
            readBuffer,
            0,
            this.fieldBuffer.size
        );
        this.device.queue.submit([commandEncoder.finish()]);
        
        await readBuffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(readBuffer.getMappedRange().slice());
        readBuffer.unmap();
        
        return data;
    }
    
    /**
     * Set wave function data
     */
    setWaveFunction(data: Float32Array): void {
        this.device.queue.writeBuffer(this.fieldBuffer, 0, data.buffer as ArrayBuffer.buffer);
    }
    
    /**
     * Calculate observables
     */
    async calculateObservables(): Promise<{
        probability: number;
        energy: number;
        momentum: [number, number];
        position: [number, number];
    }> {
        const data = await this.getWaveFunction();
        
        let probability = 0;
        let xMean = 0;
        let yMean = 0;
        
        const { width, height } = this.config;
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 2;
                const prob = data[idx] * data[idx] + data[idx + 1] * data[idx + 1];
                
                probability += prob;
                xMean += x * prob;
                yMean += y * prob;
            }
        }
        
        xMean /= probability;
        yMean /= probability;
        
        // Energy and momentum would require more complex calculations
        
        return {
            probability,
            energy: 0,  // Placeholder
            momentum: [0, 0],  // Placeholder
            position: [xMean, yMean],
        };
    }
    
    /**
     * Get performance statistics
     */
    getPerformanceStats(): {
        averageFrameTime: number;
        framesPerSecond: number;
        totalFrames: number;
    } {
        const averageFrameTime = this.frameCount > 0 ? this.totalTime / this.frameCount : 0;
        
        return {
            averageFrameTime,
            framesPerSecond: averageFrameTime > 0 ? 1000 / averageFrameTime : 0,
            totalFrames: this.frameCount,
        };
    }
    
    /**
     * Switch to a different evolution method
     */
    async switchMethod(method: string): Promise<void> {
        this.config.method = method as any;
        await this.selectKernel();
    }
    
    /**
     * Cleanup resources
     */
    destroy(): void {
        this.fieldBuffer?.destroy();
        this.potentialTexture?.destroy();
        SchrodingerRegistry.destroy();
        
        console.log('[Evolution] Resources cleaned up');
    }
}

/**
 * Factory function for easy instantiation
 */
export async function createSchrodingerEvolution(
    device: GPUDevice,
    config: EvolutionConfig
): Promise<SchrodingerEvolution> {
    const evolution = new SchrodingerEvolution(device, config);
    await evolution.initialize();
    return evolution;
}

/**
 * Quick start example
 */
export async function quickStart(device: GPUDevice): Promise<void> {
    console.log('=== Schrödinger Evolution PLATINUM Quick Start ===');
    
    // Create evolution system with automatic method selection
    const evolution = await createSchrodingerEvolution(device, {
        width: 512,
        height: 512,
        dt: 0.01,
        method: 'auto',
        potential: {
            type: 'harmonic',
            strength: 1.0,
        },
        boundary: {
            type: 'absorbing',
        },
        filtering: {
            enabled: true,
            type: FilterType.Gaussian,
            cutoff: 0.9,
        },
    });
    
    // Run evolution
    console.log('Evolving wave function...');
    await evolution.evolveSteps(100);
    
    // Calculate observables
    const observables = await evolution.calculateObservables();
    console.log('Observables:', observables);
    
    // Get performance stats
    const stats = evolution.getPerformanceStats();
    console.log('Performance:', stats);
    
    // Run benchmark
    console.log('Running benchmark comparison...');
    const benchmarkResults = await evolution.benchmark();
    
    // Cleanup
    evolution.destroy();
    
    console.log('=== Quick Start Complete ===');
}