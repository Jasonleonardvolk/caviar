/**
 * lightFieldComposerPipeline.ts
 *
 * TypeScript integration for the enhanced light field composer shader
 * Bridges tensor/phase backend with WebGPU rendering pipeline
 */
import { ALBERT } from './albert/tensorSystem';
import { SolitonComputer } from './soliton/dynamics';
import { PhaseEncoder } from './phase/encoder';
var ComposerMode;
(function (ComposerMode) {
    ComposerMode[ComposerMode["QUILT_PHASE_BLEND"] = 0] = "QUILT_PHASE_BLEND";
    ComposerMode[ComposerMode["DEPTH_LAYERS_TENSOR"] = 1] = "DEPTH_LAYERS_TENSOR";
    ComposerMode[ComposerMode["STEREO_SOLITON"] = 2] = "STEREO_SOLITON";
    ComposerMode[ComposerMode["TENSOR_FIELD_VIZ"] = 3] = "TENSOR_FIELD_VIZ";
    ComposerMode[ComposerMode["PHASE_COHERENT_HOLOGRAM"] = 4] = "PHASE_COHERENT_HOLOGRAM";
})(ComposerMode || (ComposerMode = {}));
export class LightFieldComposerPipeline {
    constructor(config) {
        this.device = config.device;
        this.config = config;
        this.startTime = performance.now();
        // Initialize physics backends
        this.albert = new ALBERT({
            mass: 1.0,
            spin: 0.9, // Near-maximal Kerr black hole
            device: this.device
        });
        this.soliton = new SolitonComputer({
            amplitude: 1.0,
            wavelength: 10.0,
            velocity: 1.0,
            nonlinearity: 0.5,
            device: this.device
        });
        this.phaseEncoder = new PhaseEncoder({
            device: this.device
        });
    }
    async initialize() {
        // Load enhanced shader
        const shaderCode = await fetch('/wgsl/lightFieldComposerEnhanced.wgsl')
            .then(r => r.text());
        const shaderModule = this.device.createShaderModule({
            label: 'Light Field Composer Enhanced',
            code: shaderCode
        });
        // Create pipeline
        this.pipeline = this.device.createComputePipeline({
            label: 'Light Field Composer Pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });
        // Initialize textures and buffers
        await this.createResources();
    }
    async createResources() {
        const { width, height, viewCount, tileCountX, tileCountY } = this.config;
        // Create texture arrays for multi-view content
        const textureDesc = {
            size: { width, height, depthOrArrayLayers: viewCount },
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        };
        const baseTex = this.device.createTexture({ ...textureDesc, label: 'Base Views' });
        const occTex = this.device.createTexture({ ...textureDesc, label: 'Occlusion' });
        const phaseTex = this.device.createTexture({ ...textureDesc, label: 'Phase' });
        // Output texture (full quilt size)
        const outputTex = this.device.createTexture({
            size: { width: width * tileCountX, height: height * tileCountY },
            format: 'rgba8unorm',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
            label: 'Light Field Output'
        });
        // Create uniform buffers
        const paramsBuffer = this.createParamsBuffer();
        const tensorBuffer = this.createTensorBuffer();
        const solitonBuffer = this.createSolitonBuffer();
        // Create bind group
        this.bindGroup = this.device.createBindGroup({
            label: 'Light Field Composer Bind Group',
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: baseTex.createView() },
                { binding: 1, resource: occTex.createView() },
                { binding: 2, resource: outputTex.createView() },
                { binding: 3, resource: { buffer: paramsBuffer } },
                { binding: 4, resource: phaseTex.createView() },
                { binding: 5, resource: { buffer: tensorBuffer } },
                { binding: 6, resource: { buffer: solitonBuffer } }
            ]
        });
    }
    createParamsBuffer() {
        const params = new Float32Array([
            this.config.width,
            this.config.height,
            this.config.viewCount,
            this.config.tileCountX,
            this.config.tileCountY,
            this.config.mode,
            0, // time (updated each frame)
            0, // phaseShift (updated each frame)
            1.0, // tensorStrength
            0.5 // coherenceRadius
        ]);
        const buffer = this.device.createBuffer({
            size: params.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Composer Params'
        });
        this.device.queue.writeBuffer(buffer, 0, params);
        return buffer;
    }
    createTensorBuffer() {
        // Get tensor field from ALBERT system
        const tensorData = this.albert.computeKerrMetric();
        const buffer = this.device.createBuffer({
            size: tensorData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Tensor Field'
        });
        this.device.queue.writeBuffer(buffer, 0, tensorData);
        return buffer;
    }
    createSolitonBuffer() {
        // Get soliton parameters
        const solitonData = this.soliton.getParameters();
        const buffer = this.device.createBuffer({
            size: solitonData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'Soliton Params'
        });
        this.device.queue.writeBuffer(buffer, 0, solitonData);
        return buffer;
    }
    render(commandEncoder) {
        // Update time-based parameters
        const currentTime = (performance.now() - this.startTime) / 1000.0;
        this.updateTimeParams(currentTime);
        // Update tensor field if needed
        if (this.config.mode === ComposerMode.TENSOR_FIELD_VIZ) {
            this.updateTensorField();
        }
        // Dispatch compute shader
        const computePass = commandEncoder.beginComputePass({
            label: 'Light Field Composition Pass'
        });
        computePass.setPipeline(this.pipeline);
        computePass.setBindGroup(0, this.bindGroup);
        const workgroupsX = Math.ceil((this.config.width * this.config.tileCountX) / 8);
        const workgroupsY = Math.ceil((this.config.height * this.config.tileCountY) / 8);
        computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
        computePass.end();
    }
    updateTimeParams(time) {
        // Update time and phase shift in params buffer
        const timeData = new Float32Array([
            time,
            Math.sin(time * 0.5) * Math.PI // Oscillating phase shift
        ]);
        // Write to params buffer at time offset (6 * 4 bytes)
        this.device.queue.writeBuffer(this.bindGroup.entries[3].resource.buffer, 6 * 4, timeData);
    }
    updateTensorField() {
        // Dynamically update tensor field for animation
        const rotationAngle = performance.now() * 0.001;
        this.albert.setRotation(rotationAngle);
        const tensorData = this.albert.computeKerrMetric();
        this.device.queue.writeBuffer(this.bindGroup.entries[5].resource.buffer, 0, tensorData);
    }
    /**
     * Integrate with gyroscope for mobile head tracking
     */
    enableHeadTracking() {
        if ('DeviceOrientationEvent' in window) {
            // Request permission on iOS
            if (typeof DeviceOrientationEvent.requestPermission === 'function') {
                DeviceOrientationEvent.requestPermission()
                    .then(response => {
                    if (response === 'granted') {
                        this.setupOrientationListener();
                    }
                });
            }
            else {
                this.setupOrientationListener();
            }
        }
    }
    setupOrientationListener() {
        window.addEventListener('deviceorientation', (event) => {
            if (event.alpha === null || event.beta === null || event.gamma === null)
                return;
            // Convert device orientation to view angle
            const viewAngle = this.orientationToViewAngle(event.alpha, event.beta, event.gamma);
            // Update shader parameters based on head position
            this.updateViewAngle(viewAngle);
        });
    }
    orientationToViewAngle(alpha, beta, gamma) {
        // Simple mapping - can be enhanced with sensor fusion
        const normalizedAngle = gamma / 90.0; // -1 to 1
        return (normalizedAngle + 1.0) * 0.5 * this.config.viewCount;
    }
    updateViewAngle(viewAngle) {
        // This would update the shader to interpolate between views
        // based on the current head position
        console.log(`View angle: ${viewAngle}`);
    }
    /**
     * Export composed light field for display
     */
    async exportLightField() {
        const { width, height } = this.config;
        const outputWidth = width * this.config.tileCountX;
        const outputHeight = height * this.config.tileCountY;
        // Create staging buffer for readback
        const bytesPerRow = Math.ceil(outputWidth * 4 / 256) * 256;
        const bufferSize = bytesPerRow * outputHeight;
        const stagingBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        // Copy texture to buffer
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyTextureToBuffer({ texture: this.bindGroup.entries[2].resource }, { buffer: stagingBuffer, bytesPerRow }, { width: outputWidth, height: outputHeight });
        this.device.queue.submit([commandEncoder.finish()]);
        // Read back data
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const data = new Uint8ClampedArray(stagingBuffer.getMappedRange());
        const imageData = new ImageData(data, outputWidth, outputHeight);
        stagingBuffer.unmap();
        return imageData;
    }
}
// Usage example
export async function initializeLightFieldRenderer() {
    // Check WebGPU support
    if (!navigator.gpu) {
        throw new Error('WebGPU not supported');
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error('No GPU adapter found');
    }
    const device = await adapter.requestDevice();
    // Configure for Looking Glass-style 5x9 quilt
    const config = {
        width: 420, // Each view width
        height: 560, // Each view height
        viewCount: 45, // 5x9 grid
        tileCountX: 9,
        tileCountY: 5,
        mode: ComposerMode.PHASE_COHERENT_HOLOGRAM,
        device
    };
    const pipeline = new LightFieldComposerPipeline(config);
    await pipeline.initialize();
    // Enable head tracking on mobile
    pipeline.enableHeadTracking();
    return pipeline;
}
