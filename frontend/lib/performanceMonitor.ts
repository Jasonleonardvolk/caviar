export class HologramPerformanceMonitor {
    private frameTimings: number[] = [];
    private readonly maxSamples = 100;
    
    measureFrame(callback: () => void): void {
        const start = performance.now();
        callback();
        const end = performance.now();
        
        this.frameTimings.push(end - start);
        if (this.frameTimings.length > this.maxSamples) {
            this.frameTimings.shift();
        }
    }
    
    async measureFrameAsync(callback: () => Promise<void>): Promise<void> {
        const start = performance.now();
        await callback();
        const end = performance.now();
        
        this.frameTimings.push(end - start);
        if (this.frameTimings.length > this.maxSamples) {
            this.frameTimings.shift();
        }
    }
    
    getMetrics() {
        const avg = this.average(this.frameTimings);
        return {
            avgFrameTime: avg,
            fps: avg > 0 ? 1000 / avg : 0,
            percentile95: this.percentile(this.frameTimings, 0.95),
            percentile99: this.percentile(this.frameTimings, 0.99),
            min: Math.min(...this.frameTimings),
            max: Math.max(...this.frameTimings),
            sampleCount: this.frameTimings.length
        };
    }
    
    getDetailedMetrics() {
        if (this.frameTimings.length === 0) {
            return {
                avgFrameTime: 0,
                fps: 0,
                percentiles: {},
                histogram: {},
                summary: "No data collected"
            };
        }
        
        const avg = this.average(this.frameTimings);
        const sorted = [...this.frameTimings].sort((a, b) => a - b);
        
        return {
            avgFrameTime: avg,
            fps: avg > 0 ? 1000 / avg : 0,
            percentiles: {
                p50: this.percentile(this.frameTimings, 0.50),
                p75: this.percentile(this.frameTimings, 0.75),
                p90: this.percentile(this.frameTimings, 0.90),
                p95: this.percentile(this.frameTimings, 0.95),
                p99: this.percentile(this.frameTimings, 0.99),
                p999: this.percentile(this.frameTimings, 0.999)
            },
            histogram: this.createHistogram(sorted),
            summary: this.generateSummary(avg)
        };
    }
    
    private average(arr: number[]): number {
        if (arr.length === 0) return 0;
        return arr.reduce((sum, v) => sum + v, 0) / arr.length;
    }
    
    private percentile(arr: number[], p: number): number {
        if (arr.length === 0) return 0;
        const sorted = [...arr].sort((a, b) => a - b);
        const idx = Math.min(sorted.length - 1, Math.ceil(p * sorted.length) - 1);
        return sorted[idx];
    }
    
    private createHistogram(sorted: number[]): Record<string, number> {
        const buckets: Record<string, number> = {
            "<8ms": 0,
            "8-16ms": 0,
            "16-33ms": 0,
            "33-50ms": 0,
            "50-100ms": 0,
            ">100ms": 0
        };
        
        for (const time of sorted) {
            if (time < 8) buckets["<8ms"]++;
            else if (time < 16) buckets["8-16ms"]++;
            else if (time < 33) buckets["16-33ms"]++;
            else if (time < 50) buckets["33-50ms"]++;
            else if (time < 100) buckets["50-100ms"]++;
            else buckets[">100ms"]++;
        }
        
        return buckets;
    }
    
    private generateSummary(avgFrameTime: number): string {
        const fps = 1000 / avgFrameTime;
        
        if (fps >= 120) return "Excellent (120+ FPS)";
        if (fps >= 60) return "Good (60-120 FPS)";
        if (fps >= 30) return "Acceptable (30-60 FPS)";
        if (fps >= 15) return "Poor (15-30 FPS)";
        return "Unacceptable (<15 FPS)";
    }
    
    reset(): void {
        this.frameTimings = [];
    }
    
    // Export metrics to CSV for analysis
    exportToCSV(): string {
        const header = "Frame,Time(ms)\n";
        const rows = this.frameTimings.map((time, i) => `${i + 1},${time.toFixed(2)}`).join("\n");
        return header + rows;
    }
    
    // Get rolling average over last N frames
    getRollingAverage(windowSize: number = 10): number {
        const window = this.frameTimings.slice(-windowSize);
        return this.average(window);
    }
    
    // Detect performance spikes
    detectSpikes(threshold: number = 2.0): number[] {
        const avg = this.average(this.frameTimings);
        const spikeIndices: number[] = [];
        
        this.frameTimings.forEach((time, index) => {
            if (time > avg * threshold) {
                spikeIndices.push(index);
            }
        });
        
        return spikeIndices;
    }
}

// Helper class for GPU performance monitoring
export class GPUPerformanceMonitor {
    private gpuTimings: Map<string, number[]> = new Map();
    private readonly maxSamples = 100;
    
    async measureGPUOperation(
        label: string, 
        operation: () => Promise<void>
    ): Promise<void> {
        // Note: Real GPU timing requires timestamp queries
        // This is a simplified version using CPU timing
        const start = performance.now();
        await operation();
        const end = performance.now();
        
        if (!this.gpuTimings.has(label)) {
            this.gpuTimings.set(label, []);
        }
        
        const timings = this.gpuTimings.get(label)!;
        timings.push(end - start);
        
        if (timings.length > this.maxSamples) {
            timings.shift();
        }
    }
    
    getGPUMetrics(label?: string): Record<string, any> {
        if (label) {
            const timings = this.gpuTimings.get(label) || [];
            return this.calculateMetrics(label, timings);
        }
        
        // Return metrics for all labels
        const allMetrics: Record<string, any> = {};
        for (const [label, timings] of this.gpuTimings) {
            allMetrics[label] = this.calculateMetrics(label, timings);
        }
        return allMetrics;
    }
    
    private calculateMetrics(label: string, timings: number[]) {
        if (timings.length === 0) {
            return { label, avgTime: 0, count: 0 };
        }
        
        const avg = timings.reduce((sum, v) => sum + v, 0) / timings.length;
        const sorted = [...timings].sort((a, b) => a - b);
        
        return {
            label,
            avgTime: avg,
            minTime: sorted[0],
            maxTime: sorted[sorted.length - 1],
            p95Time: sorted[Math.floor(sorted.length * 0.95)],
            count: timings.length
        };
    }
    
    reset(label?: string): void {
        if (label) {
            this.gpuTimings.delete(label);
        } else {
            this.gpuTimings.clear();
        }
    }
}

// Combined monitor for hologram rendering
export class HologramRenderMonitor {
    private frameMonitor = new HologramPerformanceMonitor();
    private gpuMonitor = new GPUPerformanceMonitor();
    private startTime = performance.now();
    private frameCount = 0;
    
    async measureRenderFrame(renderCallback: () => Promise<void>): Promise<void> {
        await this.frameMonitor.measureFrameAsync(async () => {
            await renderCallback();
            this.frameCount++;
        });
    }
    
    async measureGPUStage(stage: string, operation: () => Promise<void>): Promise<void> {
        await this.gpuMonitor.measureGPUOperation(stage, operation);
    }
    
    getFullReport() {
        const totalTime = (performance.now() - this.startTime) / 1000; // seconds
        const frameMetrics = this.frameMonitor.getDetailedMetrics();
        const gpuMetrics = this.gpuMonitor.getGPUMetrics();
        
        return {
            summary: {
                totalFrames: this.frameCount,
                totalTime: totalTime,
                averageFPS: this.frameCount / totalTime,
                targetFPS: frameMetrics.fps
            },
            framePerformance: frameMetrics,
            gpuPerformance: gpuMetrics,
            recommendations: this.generateRecommendations(frameMetrics, gpuMetrics)
        };
    }
    
    private generateRecommendations(frameMetrics: any, gpuMetrics: any): string[] {
        const recommendations: string[] = [];
        
        if (frameMetrics.fps < 30) {
            recommendations.push("Consider reducing hologram resolution");
        }
        
        if (frameMetrics.percentiles?.p95 > 33) {
            recommendations.push("Frame time spikes detected - check for blocking operations");
        }
        
        // Check GPU metrics for bottlenecks
        for (const [stage, metrics] of Object.entries(gpuMetrics)) {
            if ((metrics as any).avgTime > 10) {
                recommendations.push(`GPU stage '${stage}' is slow (${(metrics as any).avgTime.toFixed(1)}ms)`);
            }
        }
        
        if (recommendations.length === 0) {
            recommendations.push("Performance is optimal");
        }
        
        return recommendations;
    }
    
    reset(): void {
        this.frameMonitor.reset();
        this.gpuMonitor.reset();
        this.frameCount = 0;
        this.startTime = performance.now();
    }
}

// Export for use in hologram renderer
export default HologramPerformanceMonitor;
