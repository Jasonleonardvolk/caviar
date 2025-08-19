import { createSchrodingerEvolution, SchrodingerRegistry, type SchrodingerEvolution } from "./schrodinger";

/**
 * index.ts
 * PLATINUM Edition: Main export file for Schrödinger evolution kernels
 * 
 * Clean exports for all the PLATINUM components
 */

// Main evolution interface
export {
    SchrodingerEvolution,
    createSchrodingerEvolution,
    quickStart,
    type EvolutionConfig,
} from './schrodingerEvolution';

// Kernel registry
export {
    SchrodingerRegistry,
    SchrodingerKernelRegistry,
    KernelChain,
    type SchrodingerKernel,
    type KernelCapabilities,
    type KernelPerformance,
    type KernelInputs,
    KernelType,
} from './schrodingerKernelRegistry';

// Split-step orchestrator
export {
    SplitStepOrchestrator,
    createSplitStepOrchestrator,
    BoundaryType,
} from './splitStepOrchestrator';

export type {
    SplitStepConfig,
    PerformanceTelemetry,
} from './splitStepOrchestrator';

// ONNX integration
export {
    OnnxWaveOpRunner,
    createOnnxWaveRunner,
    prepareOnnxModel,
    type OnnxConfig,
    type PerformanceMetrics,
} from './onnxWaveOpRunner';

// Spectral filtering
export {
    generateSpectralFilterShader,
    createFilterTexture,
    FilterPresets,
    FilterType,
    type SpectralFilterConfig,
} from './spectralFiltering';

// Benchmarking
export {
    SchrodingerBenchmark,
    runSchrodingerBenchmark,
    BenchmarkMethod,
    type BenchmarkConfig,
    type BenchmarkResult,
    type TimingStats,
    type AccuracyMetrics,
} from './schrodingerBenchmark';

// Convenience re-exports
export type { KernelSpec } from '../types';

/**
 * Version information
 */
export const PLATINUM_VERSION = '2.0.0';
export const PLATINUM_BUILD_DATE = '2025-08-10';
export const PLATINUM_FEATURES = [
    'split-step-fft',
    'onnx-integration',
    'spectral-filtering',
    'subgroup-optimization',
    'zero-copy-binding',
    'lru-caching',
    'benchmarking',
    'auto-kernel-selection',
];

/**
 * Check WebGPU compatibility
 */
export async function checkCompatibility(): Promise<{
    supported: boolean;
    features: string[];
    limits: any;
    suggestions: string[];
}> {
    const suggestions: string[] = [];
    
    if (!navigator.gpu) {
        return {
            supported: false,
            features: [],
            limits: {},
            suggestions: ['WebGPU is not supported in this browser'],
        };
    }
    
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        return {
            supported: false,
            features: [],
            limits: {},
            suggestions: ['No GPU adapter found'],
        };
    }
    
    const features = Array.from(adapter.features);
    const limits = adapter.limits;
    
    // Check for recommended features
    if (!features.includes('timestamp-query' as GPUFeatureName)) {
        suggestions.push('Timestamp queries not supported - performance profiling will be limited');
    }
    
    if (!features.includes('subgroups' as GPUFeatureName)) {
        suggestions.push('Subgroup operations not supported - using fallback FFT implementation');
    }
    
    // Check limits
    if (limits.maxStorageBufferBindingSize < 128 * 1024 * 1024) {
        suggestions.push('Limited storage buffer size - may affect large simulations');
    }
    
    return {
        supported: true,
        features,
        limits,
        suggestions,
    };
}

/**
 * Initialize PLATINUM system
 */
export async function initializePlatinum(
    device?: GPUDevice
): Promise<{
    device: GPUDevice;
    evolution: SchrodingerEvolution;
    ready: boolean;
}> {
    console.log(`
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   SCHRÖDINGER EVOLUTION - PLATINUM EDITION v${PLATINUM_VERSION}      ║
║                                                           ║
║   Features:                                               ║
║   • Split-Step FFT with subgroup optimization            ║
║   • ONNX Neural Operators with zero-copy binding         ║
║   • Advanced spectral filtering                          ║
║   • Comprehensive benchmarking suite                     ║
║   • Automatic kernel selection                           ║
║                                                           ║
║   Build Date: ${PLATINUM_BUILD_DATE}                              ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    `);
    
    // Check compatibility
    const compatibility = await checkCompatibility();
    if (!compatibility.supported) {
        throw new Error('WebGPU not supported');
    }
    
    if (compatibility.suggestions.length > 0) {
        console.log('⚠️ Compatibility notes:');
        compatibility.suggestions.forEach(s => console.log(`  - ${s}`));
    }
    
    // Get or create device
    if (!device) {
        const adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance',
        });
        
        if (!adapter) {
            throw new Error('No GPU adapter found');
        }
        
        // Request device with optional features
        const requiredFeatures: GPUFeatureName[] = [];
        const optionalFeatures = [
            'timestamp-query',
            'subgroups',
        ] as GPUFeatureName[];
        
        for (const feature of optionalFeatures) {
            if (adapter.features.has(feature)) {
                requiredFeatures.push(feature);
            }
        }
        
        device = await adapter.requestDevice({
            requiredFeatures,
            requiredLimits: {
                maxStorageBufferBindingSize: 256 * 1024 * 1024,  // 256MB
                maxComputeWorkgroupSizeX: 256,
                maxComputeWorkgroupSizeY: 256,
            },
        });
    }
    
    // Initialize evolution system
    const evolution = await createSchrodingerEvolution(device, {
        gridSize: 512
    });
    
    console.log('✅ PLATINUM system initialized and ready!');
    
    return {
        device,
        evolution,
        ready: true,
    };
}

/**
 * Default export for convenience
 */
export default {
    initializePlatinum,
    checkCompatibility,
    createSchrodingerEvolution,
    SchrodingerRegistry,
    PLATINUM_VERSION,
    PLATINUM_FEATURES,
};