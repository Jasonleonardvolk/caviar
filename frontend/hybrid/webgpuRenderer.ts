// WebGPU Renderer
export class WebGPURenderer {
    private canvas: HTMLCanvasElement;
    private device: GPUDevice | null = null;
    private context: GPUCanvasContext | null = null;
    
    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
    }
    
    async initialize() {
        if (!navigator.gpu) {
            throw new Error('WebGPU not supported');
        }
        
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error('No GPU adapter found');
        }
        
        this.device = await adapter.requestDevice();
        this.context = this.canvas.getContext('webgpu') as GPUCanvasContext;
        
        if (!this.context) {
            throw new Error('Failed to get WebGPU context');
        }
        
        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: presentationFormat,
        });
    }
    
    render() {
        if (!this.device || !this.context) return;
        
        // Basic render pass
        const commandEncoder = this.device.createCommandEncoder();
        const textureView = this.context.getCurrentTexture().createView();
        
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });
        
        renderPass.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }
    
    dispose() {
        this.context?.unconfigure();
    }
}

export default WebGPURenderer;

// Module init function for main.ts compatibility
let renderer: WebGPURenderer | null = null;

export async function init() {
    const canvas = document.getElementById('canvas') as HTMLCanvasElement;
    if (!canvas) {
        throw new Error('Canvas element not found');
    }
    
    renderer = new WebGPURenderer(canvas);
    await renderer.initialize();
    
    // Start render loop
    function animate() {
        renderer?.render();
        requestAnimationFrame(animate);
    }
    animate();
    
    return renderer;
}

export function getRenderer() {
    return renderer;
}
