/**
 * schrodingerKernelRegistry.ts
 * PLATINUM Edition: Unified kernel registration and management
 * 
 * Central registry for all Schrödinger evolution kernels:
 * - Biharmonic Finite Difference (existing)
 * - Split-Step Fourier (new)
 * - ONNX Neural Operator (new)
 * - Custom user kernels
 * 
 * Features:
 * - Automatic kernel selection based on problem characteristics
 * - Performance profiling and kernel recommendation
 * - Hot-swapping kernels at runtime
 * - Kernel composition and chaining
 */

import { KernelSpec } from '../types';
import { SplitStepOrchestrator, createSplitStepOrchestrator } from './splitStepOrchestrator';
import { OnnxWaveOpRunner, createOnnxWaveRunner } from './onnxWaveOpRunner';
import { SchrodingerBenchmark, BenchmarkMethod, runSchrodingerBenchmark } from './schrodingerBenchmark';

export interface SchrodingerKernel {
    id: string;
    name: string;
    description: string;
    type: KernelType;
    implementation: KernelImplementation;
    capabilities: KernelCapabilities;
    performance: KernelPerformance;
    initialize: (device: GPUDevice, config: any) => Promise<void>;
    execute: (commandEncoder: GPUCommandEncoder, inputs: KernelInputs) => void;
    destroy: () => void;
}

export enum KernelType {
    FiniteDifference = 'finite-difference',
    SpectralFFT = 'spectral-fft',
    NeuralOperator = 'neural-operator',
    Hybrid = 'hybrid',
    Custom = 'custom',
}

export interface KernelImplementation {
    splitStep?: SplitStepOrchestrator;
    onnxRunner?: OnnxWaveOpRunner;
    pipeline?: GPUComputePipeline;
    bindGroup?: GPUBindGroup;
    implementation?: any;
}

export interface KernelCapabilities {
    supportsAnisotropic: boolean;
    supportsBoundaries: string[];  // ['periodic', 'absorbing', 'reflecting']
    supportsNonlinear: boolean;
    supportsBatching: boolean;
    requiresPowerOfTwo: boolean;
    maxDimensions: { width: number; height: number     implementation?: any;
};
}

export interface KernelPerformance {
    flopsPerElement: number;
    memoryBandwidth: number;  // GB/s
    optimalSize: { width: number; height: number     implementation?: any;
};
    scalingBehavior: 'linear' | 'nlogn' | 'quadratic';
}

export interface KernelInputs {
    fieldTexture: GPUTexture;
    potentialTexture: GPUTexture;
    dt: number;
    params?: any;
    implementation?: any;
}

/**
 * Singleton registry for all Schrödinger kernels
 */
export class SchrodingerKernelRegistry {
    private static instance: SchrodingerKernelRegistry;
    private kernels: Map<string, SchrodingerKernel> = new Map();
    private activeKernel: SchrodingerKernel | null = null;
    private device: GPUDevice | null = null;
    private performanceCache: Map<string, KernelPerformance> = new Map();
    
    private constructor() {
        // Register built-in kernels
        this.registerBuiltinKernels();
    }
    
    static getInstance(): SchrodingerKernelRegistry {
        if (!SchrodingerKernelRegistry.instance) {
            SchrodingerKernelRegistry.instance = new SchrodingerKernelRegistry();
        }
        return SchrodingerKernelRegistry.instance;
    }
    
    /**
     * Initialize registry with GPU device
     */
    async initialize(device: GPUDevice): Promise<void> {
        this.device = device;
        console.log('[Registry] Initializing with GPU device');
        
        // Auto-detect best kernel for device
        await this.profileKernels();
        
        // Set default kernel
        const recommended = this.recommendKernel();
        if (recommended) {
            await this.activateKernel(recommended.id);
        }
    }
    
    /**
     * Register built-in kernels
     */
    private registerBuiltinKernels(): void {
        // Biharmonic Finite Difference
        this.register({
            id: 'biharmonic-fd',
            name: 'Biharmonic Finite Difference',
            description: 'Fourth-order accurate finite difference with biharmonic operator',
            type: KernelType.FiniteDifference,
            implementation: {},
            capabilities: {
                supportsAnisotropic: false,
                supportsBoundaries: ['periodic', 'absorbing'],
                supportsNonlinear: true,
                supportsBatching: false,
                requiresPowerOfTwo: false,
                maxDimensions: { width: 4096, height: 4096 },
            },
            performance: {
                flopsPerElement: 25,
                memoryBandwidth: 200,
                optimalSize: { width: 512, height: 512 },
                scalingBehavior: 'linear',
            },
            initialize: async (device: GPUDevice, config: any) => {
                // Load and compile biharmonic shader
                const shaderSource = await fetch('/lib/webgpu/shaders/schrodinger_biharmonic.wgsl')
                    .then(r => r.text());
                
                const shaderModule = device.createShaderModule({
                    label: 'Biharmonic FD',
                    code: shaderSource,
                });
                
                const pipeline = device.createComputePipeline({
                    label: 'Biharmonic Pipeline',
                    layout: 'auto',
                    compute: {
                        module: shaderModule,
                        entryPoint: 'main',
                    },
                });
                
                // Store in implementation
                (this as any).implementation.pipeline = pipeline;
            },
            execute: (commandEncoder: GPUCommandEncoder, inputs: KernelInputs) => {
                if (!(this as any).implementation.pipeline) return;
                
                const computePass = commandEncoder.beginComputePass({
                    label: 'Biharmonic FD Pass',
                });
                
                computePass.setPipeline((this as any).implementation.pipeline);
                // Set bind groups...
                computePass.dispatchWorkgroups(64, 64);  // Example
                computePass.end();
            },
            destroy: () => {
                // Cleanup
            },
        });
        
        // Split-Step Fourier
        this.register({
            id: 'splitstep-fft-platinum',
            name: 'Split-Step Fourier (PLATINUM)',
            description: 'High-performance split-step method with FFT acceleration',
            type: KernelType.SpectralFFT,
            implementation: {},
            capabilities: {
                supportsAnisotropic: true,
                supportsBoundaries: ['periodic', 'absorbing', 'pml', 'airy'],
                supportsNonlinear: true,
                supportsBatching: true,
                requiresPowerOfTwo: true,
                maxDimensions: { width: 2048, height: 2048 },
            },
            performance: {
                flopsPerElement: 80,
                memoryBandwidth: 150,
                optimalSize: { width: 512, height: 512 },
                scalingBehavior: 'nlogn',
            },
            initialize: async (device: GPUDevice, config: any) => {
                const orchestrator = await createSplitStepOrchestrator(device, config);
                (this as any).implementation.splitStep = orchestrator;
            },
            execute: (commandEncoder: GPUCommandEncoder, inputs: KernelInputs) => {
                if (!(this as any).implementation.splitStep) return;
                
                (this as any).implementation.splitStep.execute(
                    commandEncoder,
                    inputs.fieldTexture,
                    inputs.potentialTexture
                );
            },
            destroy: () => {
                (this as any).implementation.splitStep?.destroy();
            },
        });
        
        // ONNX Neural Operator
        this.register({
            id: 'onnx-neural',
            name: 'ONNX Neural Operator',
            description: 'Machine learning-based evolution operator',
            type: KernelType.NeuralOperator,
            implementation: {},
            capabilities: {
                supportsAnisotropic: true,
                supportsBoundaries: ['learned'],
                supportsNonlinear: true,
                supportsBatching: true,
                requiresPowerOfTwo: false,
                maxDimensions: { width: 1024, height: 1024 },
            },
            performance: {
                flopsPerElement: 1000,  // Much higher due to neural network
                memoryBandwidth: 100,
                optimalSize: { width: 256, height: 256 },
                scalingBehavior: 'quadratic',
            },
            initialize: async (device: GPUDevice, config: any) => {
                const runner = await createOnnxWaveRunner(config, device);
                (this as any).implementation.onnxRunner = runner;
            },
            execute: (commandEncoder: GPUCommandEncoder, inputs: KernelInputs) => {
                // ONNX runs separately, not in command encoder
                console.warn('[Registry] ONNX kernel requires async execution');
            },
            destroy: () => {
                (this as any).implementation.onnxRunner?.destroy();
            },
        });
    }
    
    /**
     * Register a custom kernel
     */
    register(kernel: SchrodingerKernel): void {
        if (this.kernels.has(kernel.id)) {
            console.warn(`[Registry] Kernel ${kernel.id} already registered, overwriting`);
        }
        
        this.kernels.set(kernel.id, kernel);
        console.log(`[Registry] Registered kernel: ${kernel.name}`);
    }
    
    /**
     * Activate a kernel for use
     */
    async activateKernel(kernelId: string, config?: any): Promise<void> {
        if (!this.device) {
            throw new Error('Registry not initialized with device');
        }
        
        const kernel = this.kernels.get(kernelId);
        if (!kernel) {
            throw new Error(`Kernel ${kernelId} not found`);
        }
        
        // Deactivate current kernel
        if (this.activeKernel) {
            this.activeKernel.destroy();
        }
        
        // Initialize new kernel
        await kernel.initialize(this.device, config || {});
        this.activeKernel = kernel;
        
        console.log(`[Registry] Activated kernel: ${kernel.name}`);
    }
    
    /**
     * Execute active kernel
     */
    execute(commandEncoder: GPUCommandEncoder, inputs: KernelInputs): void {
        if (!this.activeKernel) {
            throw new Error('No active kernel');
        }
        
        this.activeKernel.execute(commandEncoder, inputs);
    }
    
    /**
     * Profile all registered kernels
     */
    async profileKernels(): Promise<void> {
        if (!this.device) return;
        
        console.log('[Registry] Profiling kernels...');
        
        // Run mini-benchmark for each kernel
        for (const [id, kernel] of this.kernels) {
            try {
                // Quick performance test
                const testSize = 256;
                const results = await runSchrodingerBenchmark(this.device, {
                    width: testSize,
                    height: testSize,
                    steps: 10,
                    warmupSteps: 2,
                    iterations: 5,
                    methods: [this.kernelTypeToBenchmarkMethod(kernel.type)],
                    exportResults: false,
                });
                
                // Update performance cache
                const result = results.values().next().value;
                if (result) {
                    this.performanceCache.set(id, {
                        ...kernel.performance,
                        memoryBandwidth: result.performance.bandwidth,
                    });
                }
            } catch (error) {
                console.warn(`[Registry] Failed to profile kernel ${id}:`, error);
            }
        }
        
        console.log('[Registry] Profiling complete');
    }
    
    private kernelTypeToBenchmarkMethod(type: KernelType): BenchmarkMethod {
        switch (type) {
            case KernelType.FiniteDifference:
                return BenchmarkMethod.BiharmonicFD;
            case KernelType.SpectralFFT:
                return BenchmarkMethod.SplitStepFFT;
            case KernelType.NeuralOperator:
                return BenchmarkMethod.ONNX;
            default:
                return BenchmarkMethod.BiharmonicFD;
        }
    }
    
    /**
     * Recommend best kernel for current configuration
     */
    recommendKernel(requirements?: {
        width?: number;
        height?: number;
        accuracy?: 'low' | 'medium' | 'high';
        speed?: 'slow' | 'medium' | 'fast';
        features?: string[];
    }): SchrodingerKernel | null {
        let bestKernel: SchrodingerKernel | null = null;
        let bestScore = -Infinity;
        
        for (const kernel of this.kernels.values()) {
            let score = 0;
            
            // Check requirements
            if (requirements) {
                // Size compatibility
                if (requirements.width && requirements.height) {
                    const isPowerOfTwo = (n: number) => (n & (n - 1)) === 0;
                    
                    if (kernel.capabilities.requiresPowerOfTwo) {
                        if (!isPowerOfTwo(requirements.width) || !isPowerOfTwo(requirements.height)) {
                            continue;  // Skip this kernel
                        }
                    }
                    
                    if (requirements.width > kernel.capabilities.maxDimensions.width ||
                        requirements.height > kernel.capabilities.maxDimensions.height) {
                        continue;  // Too large
                    }
                }
                
                // Accuracy preference
                if (requirements.accuracy === 'high') {
                    if (kernel.type === KernelType.SpectralFFT) {
                        score += 10;
                    }
                } else if (requirements.accuracy === 'low') {
                    if (kernel.type === KernelType.FiniteDifference) {
                        score += 5;
                    }
                }
                
                // Speed preference
                if (requirements.speed === 'fast') {
                    score -= kernel.performance.flopsPerElement / 10;
                }
                
                // Feature requirements
                if (requirements.features) {
                    for (const feature of requirements.features) {
                        if (feature === 'anisotropic' && kernel.capabilities.supportsAnisotropic) {
                            score += 5;
                        }
                        if (feature === 'batching' && kernel.capabilities.supportsBatching) {
                            score += 5;
                        }
                    }
                }
            }
            
            // Use cached performance if available
            const perf = this.performanceCache.get(kernel.id) || kernel.performance;
            score += perf.memoryBandwidth / 10;
            
            if (score > bestScore) {
                bestScore = score;
                bestKernel = kernel;
            }
        }
        
        if (bestKernel) {
            console.log(`[Registry] Recommended kernel: ${bestKernel.name} (score: ${bestScore})`);
        }
        
        return bestKernel;
    }
    
    /**
     * Get kernel by ID
     */
    getKernel(id: string): SchrodingerKernel | undefined {
        return this.kernels.get(id);
    }
    
    /**
     * List all registered kernels
     */
    listKernels(): SchrodingerKernel[] {
        return Array.from(this.kernels.values());
    }
    
    /**
     * Get active kernel
     */
    getActiveKernel(): SchrodingerKernel | null {
        return this.activeKernel;
    }
    
    /**
     * Create kernel chain for multi-stage evolution
     */
    createKernelChain(kernelIds: string[]): KernelChain {
        const kernels = kernelIds.map(id => {
            const kernel = this.kernels.get(id);
            if (!kernel) {
                throw new Error(`Kernel ${id} not found`);
            }
            return kernel;
        });
        
        return new KernelChain(kernels, this.device!);
    }
    
    /**
     * Export registry configuration
     */
    exportConfig(): string {
        const config = {
            kernels: Array.from(this.kernels.values()).map(k => ({
                id: k.id,
                name: k.name,
                type: k.type,
                capabilities: k.capabilities,
                performance: this.performanceCache.get(k.id) || k.performance,
            })),
            activeKernel: this.activeKernel?.id || null,
            timestamp: new Date().toISOString(),
        };
        
        return JSON.stringify(config, null, 2);
    }
    
    /**
     * Import registry configuration
     */
    async importConfig(configJson: string): Promise<void> {
        const config = JSON.parse(configJson);
        
        // Update performance cache
        for (const kernelConfig of config.kernels) {
            if (kernelConfig.performance) {
                this.performanceCache.set(kernelConfig.id, kernelConfig.performance);
            }
        }
        
        // Activate kernel if specified
        if (config.activeKernel) {
            await this.activateKernel(config.activeKernel);
        }
    }
    
    /**
     * Cleanup all resources
     */
    destroy(): void {
        for (const kernel of this.kernels.values()) {
            kernel.destroy();
        }
        
        this.kernels.clear();
        this.performanceCache.clear();
        this.activeKernel = null;
        
        console.log('[Registry] All kernels destroyed');
    }
}

/**
 * Kernel chain for multi-stage evolution
 */
export class KernelChain {
    private kernels: SchrodingerKernel[];
    private device: GPUDevice;
    private initialized: boolean = false;
    
    constructor(kernels: SchrodingerKernel[], device: GPUDevice) {
        this.kernels = kernels;
        this.device = device;
    }
    
    async initialize(configs?: any[]): Promise<void> {
        for (let i = 0; i < this.kernels.length; i++) {
            const config = configs?.[i] || {};
            await this.kernels[i].initialize(this.device, config);
        }
        this.initialized = true;
    }
    
    execute(commandEncoder: GPUCommandEncoder, inputs: KernelInputs): void {
        if (!this.initialized) {
            throw new Error('Kernel chain not initialized');
        }
        
        // Execute kernels in sequence
        for (const kernel of this.kernels) {
            kernel.execute(commandEncoder, inputs);
        }
    }
    
    destroy(): void {
        for (const kernel of this.kernels) {
            kernel.destroy();
        }
    }
}

/**
 * Global registry instance accessor
 */
export const SchrodingerRegistry = SchrodingerKernelRegistry.getInstance();