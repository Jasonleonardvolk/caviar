/**
 * WASM Fallback Renderer - CPU-based holographic rendering when WebGPU is unavailable
 */
import { getDeviceTier } from './lib/deviceDetect';
import { handleRenderError } from './lib/errorHandler';
export class WASMFallbackRenderer {
    constructor(canvas) {
        this.wasmModule = null;
        this.animationId = null;
        this.viewIndices = { h: 4, v: 0 }; // Center view by default
        this.quiltSize = { width: 2048, height: 2048 };
        this.viewCount = { h: 9, v: 1 };
        this.canvas = canvas;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            throw new Error('Failed to get 2D context');
        }
        this.ctx = ctx;
        this.deviceTier = getDeviceTier();
    }
    async initialize() {
        try {
            // Load WASM module (placeholder - would load actual WASM in production)
            await this.loadWASMModule();
            // Adjust quality based on device tier
            this.adjustQualityForDevice();
            // Set canvas size
            this.canvas.width = 800;
            this.canvas.height = 600;
            // Show fallback message
            this.showFallbackMessage();
            // Start render loop after brief delay
            setTimeout(() => {
                this.startRenderLoop();
            }, 1000);
        }
        catch (error) {
            handleRenderError(error);
        }
    }
    async loadWASMModule() {
        // In production, this would load actual WASM module
        // For now, simulate with a promise
        return new Promise((resolve) => {
            setTimeout(() => {
                this.wasmModule = {
                    renderQuiltView: (h, v) => {
                        // Simulated WASM rendering
                        return this.simulateQuiltView(h, v);
                    }
                };
                resolve();
            }, 100);
        });
    }
    adjustQualityForDevice() {
        switch (this.deviceTier) {
            case 'low':
                this.quiltSize = { width: 1024, height: 1024 };
                this.viewCount = { h: 5, v: 1 };
                break;
            case 'medium':
                this.quiltSize = { width: 1536, height: 1536 };
                this.viewCount = { h: 7, v: 1 };
                break;
            case 'high':
                // Keep defaults
                break;
        }
    }
    showFallbackMessage() {
        this.ctx.fillStyle = '#1a1a2e';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        // Draw gradient background
        const gradient = this.ctx.createLinearGradient(0, 0, this.canvas.width, this.canvas.height);
        gradient.addColorStop(0, '#1a1a2e');
        gradient.addColorStop(1, '#16213e');
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        // Draw message
        this.ctx.fillStyle = '#fbbf24';
        this.ctx.font = '20px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('WASM Fallback Mode', this.canvas.width / 2, 50);
        this.ctx.fillStyle = '#e2e8f0';
        this.ctx.font = '14px sans-serif';
        this.ctx.fillText('Limited features - WebGPU not available', this.canvas.width / 2, 80);
    }
    simulateQuiltView(h, v) {
        // Create a simple pattern based on view indices
        const viewWidth = Math.floor(this.quiltSize.width / this.viewCount.h);
        const viewHeight = Math.floor(this.quiltSize.height / this.viewCount.v);
        const imageData = this.ctx.createImageData(viewWidth, viewHeight);
        const data = imageData.data;
        // Generate a simple holographic-like pattern
        for (let y = 0; y < viewHeight; y++) {
            for (let x = 0; x < viewWidth; x++) {
                const index = (y * viewWidth + x) * 4;
                // Create interference pattern
                const centerX = viewWidth / 2;
                const centerY = viewHeight / 2;
                const dx = x - centerX;
                const dy = y - centerY;
                const distance = Math.sqrt(dx * dx + dy * dy);
                // Vary color based on view index and distance
                const hueShift = h * 40;
                const intensity = Math.sin(distance * 0.1 + h * 0.5) * 127 + 128;
                // Simple HSL to RGB conversion (approximation)
                const hue = (hueShift + distance * 0.5) % 360;
                const r = intensity * (1 + Math.sin((hue - 0) * Math.PI / 180)) / 2;
                const g = intensity * (1 + Math.sin((hue - 120) * Math.PI / 180)) / 2;
                const b = intensity * (1 + Math.sin((hue - 240) * Math.PI / 180)) / 2;
                data[index] = r;
                data[index + 1] = g;
                data[index + 2] = b;
                data[index + 3] = 255;
            }
        }
        return imageData;
    }
    startRenderLoop() {
        const render = () => {
            // Clear canvas
            this.ctx.fillStyle = '#000';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            // Render current view
            if (this.wasmModule) {
                const viewData = this.wasmModule.renderQuiltView(this.viewIndices.h, this.viewIndices.v);
                // Draw to canvas (centered)
                const x = (this.canvas.width - viewData.width) / 2;
                const y = (this.canvas.height - viewData.height) / 2;
                this.ctx.putImageData(viewData, x, y);
            }
            // Draw overlay info
            this.drawOverlay();
            // Continue animation
            this.animationId = requestAnimationFrame(render);
        };
        render();
    }
    drawOverlay() {
        // Draw performance info
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        this.ctx.fillRect(10, 10, 200, 80);
        this.ctx.fillStyle = '#10b981';
        this.ctx.font = '12px monospace';
        this.ctx.textAlign = 'left';
        this.ctx.fillText(`WASM Renderer`, 20, 30);
        this.ctx.fillText(`View: ${this.viewIndices.h},${this.viewIndices.v}`, 20, 50);
        this.ctx.fillText(`Quality: ${this.deviceTier}`, 20, 70);
    }
    updateView(h, v) {
        this.viewIndices.h = Math.max(0, Math.min(this.viewCount.h - 1, h));
        this.viewIndices.v = Math.max(0, Math.min(this.viewCount.v - 1, v));
    }
    destroy() {
        if (this.animationId !== null) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        this.wasmModule = null;
    }
}
// Export initialization function
export async function init() {
    const canvas = document.getElementById('hologram-canvas');
    if (!canvas) {
        throw new Error('Canvas element not found');
    }
    const renderer = new WASMFallbackRenderer(canvas);
    await renderer.initialize();
    // Listen for view updates
    window.addEventListener('controlUpdate', (event) => {
        // Map control state to view indices (simplified)
        const h = Math.floor(event.detail.blendRatio * 8);
        renderer.updateView(h, 0);
    });
    // Store renderer reference
    window.wasmRenderer = renderer;
    console.log('[WASM] Fallback renderer initialized');
}
