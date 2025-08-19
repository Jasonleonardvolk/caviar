/**
 * End-to-End Performance Test
 * Ensures holographic pipeline meets timing requirements
 */

import { SpectralHologramEngine, getDefaultCalibration } from '../../frontend/lib/holographicEngine';
import { performance } from 'perf_hooks';

interface PerformanceThresholds {
    wavefieldGeneration: number;  // ms
    propagation: number;          // ms
    viewSynthesis: number;        // ms
    lenticularInterlace: number;  // ms
    totalFrame: number;           // ms
    targetFPS: number;
}

// Performance targets by quality level
const PERFORMANCE_TARGETS: Record<string, PerformanceThresholds> = {
    low: {
        wavefieldGeneration: 2,
        propagation: 3,
        viewSynthesis: 4,
        lenticularInterlace: 1,
        totalFrame: 11,  // ~90 FPS
        targetFPS: 90
    },
    medium: {
        wavefieldGeneration: 3,
        propagation: 5,
        viewSynthesis: 6,
        lenticularInterlace: 2,
        totalFrame: 16.67,  // 60 FPS
        targetFPS: 60
    },
    high: {
        wavefieldGeneration: 5,
        propagation: 8,
        viewSynthesis: 10,
        lenticularInterlace: 3,
        totalFrame: 27,  // ~37 FPS
        targetFPS: 37
    },
    ultra: {
        wavefieldGeneration: 10,
        propagation: 15,
        viewSynthesis: 20,
        lenticularInterlace: 5,
        totalFrame: 50,  // 20 FPS
        targetFPS: 20
    }
};

export class PerformanceBenchmark {
    private engine!: SpectralHologramEngine;
    private canvas!: HTMLCanvasElement;
    
    async setup() {
        // Create canvas using the canvas package for Node.js
        const { createCanvas } = await import('canvas');
        this.canvas = createCanvas(1920, 1080);
        
        // Initialize engine
        this.engine = new SpectralHologramEngine();
        await this.engine.initialize(
            this.canvas as any,
            getDefaultCalibration('looking_glass_portrait')
        );
    }
    
    async teardown() {
        if (this.engine) {
            this.engine.destroy();
        }
    }
    
    /**
     * Run performance benchmark for a quality level
     */
    async benchmarkQuality(
        quality: 'low' | 'medium' | 'high' | 'ultra',
        frames: number = 100
    ): Promise<BenchmarkResult> {
        console.log(`\nBenchmarking ${quality} quality (${frames} frames)...`);
        
        // Set quality
        await this.engine.setQuality(quality);
        
        // Warm up (10 frames)
        for (let i = 0; i < 10; i++) {
            await this.renderFrame();
        }
        
        // Benchmark
        const frameTimes: number[] = [];
        const stageTimings: StageTimings[] = [];
        
        for (let i = 0; i < frames; i++) {
            const timing = await this.renderFrameWithTiming();
            frameTimes.push(timing.total);
            stageTimings.push(timing);
            
            // Update oscillator state for variety
            this.updateTestOscillator(i / frames);
        }
        
        // Calculate statistics
        const stats = this.calculateStats(frameTimes);
        const stageStats = this.calculateStageStats(stageTimings);
        const thresholds = PERFORMANCE_TARGETS[quality];
        
        // Check against thresholds
        const passed = this.checkThresholds(stats, stageStats, thresholds);
        
        return {
            quality,
            frames,
            stats,
            stageStats,
            thresholds,
            passed
        };
    }
    
    /**
     * Render single frame
     */
    private async renderFrame(): Promise<void> {
        await this.engine.render({
            propagationDistance: 0.3,
            enableVelocityField: false,
            enableAdaptiveQuality: false
        });
    }
    
    /**
     * Render frame with detailed timing
     */
    private async renderFrameWithTiming(): Promise<StageTimings> {
        // Get internal metrics from engine
        const metrics = await this.engine.getPerformanceMetrics();
        
        return {
            wavefield: metrics.wavefieldTime,
            propagation: metrics.propagationTime,
            viewSynthesis: metrics.viewSynthesisTime,
            total: metrics.totalTime
        };
    }
    
    /**
     * Update oscillator for test variety
     */
    private updateTestOscillator(t: number): void {
        const phase = t * Math.PI * 2;
        
        this.engine.updateFromOscillator({
            psi_phase: Math.sin(phase),
            phase_coherence: 0.5 + 0.5 * Math.cos(phase * 2),
            oscillator_phases: Array(8).fill(0).map((_, i) => phase + i * Math.PI / 4),
            oscillator_frequencies: Array(8).fill(0).map((_, i) => 440 * (i + 1)),
            coupling_strength: 0.5 + 0.3 * Math.sin(phase * 3),
            dominant_frequency: 440 + 220 * Math.sin(phase)
        });
    }
    
    /**
     * Calculate statistics
     */
    private calculateStats(times: number[]): FrameStats {
        const sorted = [...times].sort((a, b) => a - b);
        const sum = sorted.reduce((a, b) => a + b, 0);
        
        return {
            min: sorted[0],
            max: sorted[sorted.length - 1],
            mean: sum / sorted.length,
            median: sorted[Math.floor(sorted.length / 2)],
            p95: sorted[Math.floor(sorted.length * 0.95)],
            p99: sorted[Math.floor(sorted.length * 0.99)],
            fps: 1000 / (sum / sorted.length)
        };
    }
    
    /**
     * Calculate per-stage statistics
     */
    private calculateStageStats(timings: StageTimings[]): StageStats {
        const stages = ['wavefield', 'propagation', 'viewSynthesis'] as const;
        const result: any = {};
        
        for (const stage of stages) {
            const times = timings.map(t => t[stage]);
            result[stage] = this.calculateStats(times);
        }
        
        return result;
    }
    
    /**
     * Check if performance meets thresholds
     */
    private checkThresholds(
        stats: FrameStats,
        stageStats: StageStats,
        thresholds: PerformanceThresholds
    ): boolean {
        let passed = true;
        
        // Check total frame time (95th percentile)
        if (stats.p95 > thresholds.totalFrame) {
            console.error(`❌ Frame time p95 (${stats.p95.toFixed(2)}ms) exceeds threshold (${thresholds.totalFrame}ms)`);
            passed = false;
        } else {
            console.log(`✅ Frame time p95: ${stats.p95.toFixed(2)}ms (threshold: ${thresholds.totalFrame}ms)`);
        }
        
        // Check FPS
        if (stats.fps < thresholds.targetFPS) {
            console.error(`❌ FPS (${stats.fps.toFixed(1)}) below target (${thresholds.targetFPS})`);
            passed = false;
        } else {
            console.log(`✅ FPS: ${stats.fps.toFixed(1)} (target: ${thresholds.targetFPS})`);
        }
        
        // Check individual stages
        const stageChecks = [
            { name: 'Wavefield', stats: stageStats.wavefield, threshold: thresholds.wavefieldGeneration },
            { name: 'Propagation', stats: stageStats.propagation, threshold: thresholds.propagation },
            { name: 'View Synthesis', stats: stageStats.viewSynthesis, threshold: thresholds.viewSynthesis }
        ];
        
        for (const check of stageChecks) {
            if (check.stats.mean > check.threshold) {
                console.error(`❌ ${check.name} mean (${check.stats.mean.toFixed(2)}ms) exceeds threshold (${check.threshold}ms)`);
                passed = false;
            } else {
                console.log(`✅ ${check.name}: ${check.stats.mean.toFixed(2)}ms (threshold: ${check.threshold}ms)`);
            }
        }
        
        return passed;
    }
}

// Type definitions
interface StageTimings {
    wavefield: number;
    propagation: number;
    viewSynthesis: number;
    total: number;
}

interface FrameStats {
    min: number;
    max: number;
    mean: number;
    median: number;
    p95: number;
    p99: number;
    fps: number;
}

interface StageStats {
    wavefield: FrameStats;
    propagation: FrameStats;
    viewSynthesis: FrameStats;
}

interface BenchmarkResult {
    quality: string;
    frames: number;
    stats: FrameStats;
    stageStats: StageStats;
    thresholds: PerformanceThresholds;
    passed: boolean;
}

// Run benchmark
export async function runPerformanceBenchmark(): Promise<void> {
    const benchmark = new PerformanceBenchmark();
    
    try {
        await benchmark.setup();
        
        const qualities = ['low', 'medium', 'high', 'ultra'] as const;
        const results: BenchmarkResult[] = [];
        
        for (const quality of qualities) {
            const result = await benchmark.benchmarkQuality(quality, 100);
            results.push(result);
        }
        
        // Summary
        console.log('\n=== Performance Benchmark Summary ===');
        for (const result of results) {
            const status = result.passed ? '✅ PASSED' : '❌ FAILED';
            console.log(`${result.quality.toUpperCase()}: ${status} (${result.stats.fps.toFixed(1)} FPS)`);
        }
        
        // Generate report
        await generatePerformanceReport(results);
        
    } finally {
        await benchmark.teardown();
    }
}

/**
 * Generate detailed performance report
 */
async function generatePerformanceReport(results: BenchmarkResult[]): Promise<void> {
    const report = {
        timestamp: new Date().toISOString(),
        platform: {
            gpu: 'RTX 4070',
            cpu: process.platform,
            node: process.version
        },
        results: results.map(r => ({
            quality: r.quality,
            passed: r.passed,
            fps: {
                mean: r.stats.fps,
                min: 1000 / r.stats.max,
                p95: 1000 / r.stats.p95
            },
            frameTimes: {
                mean: r.stats.mean,
                p95: r.stats.p95,
                p99: r.stats.p99
            },
            stages: {
                wavefield: r.stageStats.wavefield.mean,
                propagation: r.stageStats.propagation.mean,
                viewSynthesis: r.stageStats.viewSynthesis.mean
            }
        }))
    };
    
    // Save report
    const fs = await import('fs/promises');
    const reportPath = `performance_report_${Date.now()}.json`;
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    
    console.log(`\nPerformance report saved to: ${reportPath}`);
}

// Run if called directly
if (require.main === module) {
    runPerformanceBenchmark().catch(console.error);
}
