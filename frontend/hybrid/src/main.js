/**
 * main.ts - Complete integration example for TORI Hybrid Holographic System
 *
 * This demonstrates how all components work together:
 * - Adaptive rendering based on device capabilities
 * - Mobile head tracking for parallax viewing
 * - WebGPU light field composition
 * - UI controls for real-time parameter adjustment
 */
import { AdaptiveRenderer, ParallaxController } from '@hybrid/lib';
import { lightFieldComposerEnhancedWGSL } from '@hybrid/wgsl';
// Main application class
export class HolographicApp {
    constructor() {
        this.device = null;
        this.pipeline = null;
        this.currentViewIndices = { h: 0, v: 0 };
        // Rendering parameters
        this.blendRatio = 0.5;
        this.phaseMode = 'Kerr';
        this.viewMode = 'quilt';
        this.adaptiveRenderer = new AdaptiveRenderer(60); // Target 60 FPS
        this.parallaxController = new ParallaxController({
            viewCountHoriz: 9,
            viewCountVert: 1,
            onUpdate: this.handleViewUpdate.bind(this)
        });
        this.settings = this.adaptiveRenderer.settings;
    }
    async initialize() {
        try {
            // Check WebGPU support
            if (!navigator.gpu) {
                console.error('WebGPU not supported');
                return false;
            }
            // Request adapter and device
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.error('No GPU adapter found');
                return false;
            }
            this.device = await adapter.requestDevice();
            // Initialize adaptive rendering
            this.settings = await this.adaptiveRenderer.initialize();
            console.log('Adaptive settings:', this.settings);
            // Update parallax controller with detected view count
            this.parallaxController = new ParallaxController({
                viewCountHoriz: this.settings.viewCount,
                viewCountVert: 1,
                onUpdate: this.handleViewUpdate.bind(this)
            });
            // Create compute pipeline
            await this.createPipeline();
            // Start mobile head tracking if available
            if (this.isMobile()) {
                await this.enableHeadTracking();
            }
            console.log('✅ Holographic app initialized successfully');
            return true;
        }
        catch (error) {
            console.error('Initialization failed:', error);
            return false;
        }
    }
    async createPipeline() {
        if (!this.device)
            return;
        // Create shader module from enhanced light field composer
        const shaderModule = this.device.createShaderModule({
            label: 'Light Field Composer Enhanced',
            code: lightFieldComposerEnhancedWGSL
        });
        // Create compute pipeline
        this.pipeline = this.device.createComputePipeline({
            label: 'Holographic Pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });
        console.log('✅ WebGPU pipeline created');
    }
    isMobile() {
        return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
    }
    async enableHeadTracking() {
        try {
            // Request permission on iOS
            await ParallaxController.requestPermission();
            // Start tracking
            this.parallaxController.start();
            console.log('✅ Head tracking enabled');
        }
        catch (error) {
            console.warn('Head tracking not available:', error);
        }
    }
    handleViewUpdate(hIndex, vIndex) {
        this.currentViewIndices = { h: hIndex, v: vIndex };
        console.log(`View updated: H=${hIndex}, V=${vIndex}`);
        // Trigger re-render with new view
        this.render();
    }
    updateParameters(params) {
        if (params.blendRatio !== undefined) {
            this.blendRatio = params.blendRatio;
        }
        if (params.phaseMode !== undefined) {
            this.phaseMode = params.phaseMode;
        }
        if (params.viewMode !== undefined) {
            this.viewMode = params.viewMode;
        }
        // Re-render with new parameters
        this.render();
    }
    render() {
        if (!this.device || !this.pipeline)
            return;
        const commandEncoder = this.device.createCommandEncoder();
        // Create compute pass
        const computePass = commandEncoder.beginComputePass({
            label: 'Holographic Render Pass'
        });
        computePass.setPipeline(this.pipeline);
        // Calculate workgroup dispatch based on output size and adaptive settings
        const outputWidth = 1920 * this.settings.resolutionScale;
        const outputHeight = 1080 * this.settings.resolutionScale;
        const workgroupsX = Math.ceil(outputWidth / 8);
        const workgroupsY = Math.ceil(outputHeight / 8);
        computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
        computePass.end();
        // Submit commands
        this.device.queue.submit([commandEncoder.finish()]);
    }
    getDiagnostics() {
        return {
            'Device Tier': this.settings.shaderPrecision === 'float32' ? 'High' : 'Medium/Low',
            'View Count': this.settings.viewCount,
            'Resolution Scale': `${(this.settings.resolutionScale * 100).toFixed(0)}%`,
            'Current View': `H:${this.currentViewIndices.h} V:${this.currentViewIndices.v}`,
            'Blend Ratio': this.blendRatio.toFixed(2),
            'Phase Mode': this.phaseMode,
            'View Mode': this.viewMode,
            'WebGPU': this.device ? 'Active' : 'Inactive',
            'Head Tracking': this.isMobile() ? 'Enabled' : 'Desktop'
        };
    }
    shutdown() {
        // Stop head tracking
        this.parallaxController.stop();
        // Destroy GPU resources
        if (this.device) {
            this.device.destroy();
        }
        console.log('✅ Holographic app shutdown complete');
    }
}
// Initialize on page load
async function main() {
    console.log('🚀 Starting TORI Hybrid Holographic System...');
    const app = new HolographicApp();
    const initialized = await app.initialize();
    if (initialized) {
        // Initial render
        app.render();
        // Log diagnostics
        console.log('System Diagnostics:', app.getDiagnostics());
        // Expose to window for debugging
        window.holographicApp = app;
        console.log('✨ System ready! Use window.holographicApp to interact.');
    }
    else {
        console.error('❌ Failed to initialize holographic system');
    }
}
// Run when DOM is ready
if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', main);
    }
    else {
        main();
    }
}
// Export for use in other modules
export { main };
