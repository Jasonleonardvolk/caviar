// phaseRingBuffer.ts - Temporal accumulation with warp-before-accumulate
import { getTexturePool } from '../../../lib/webgpu/utils/texturePool';
export class PhaseRingBuffer {
    constructor(device, config) {
        this.fields = [];
        // Pipeline for accumulation
        this.accumulatePipeline = null;
        this.shaderModule = null;
        this.device = device;
        this.depth = config.depth || 3;
        this.width = config.width;
        this.height = config.height;
        this.format = config.format || 'rg16float'; // Complex field storage
        // Initialize exponentially decaying weights
        this.weights = new Float32Array(this.depth);
        for (let i = 0; i < this.depth; i++) {
            this.weights[i] = Math.exp(-0.5 * i);
        }
        // Normalize weights
        const sum = this.weights.reduce((a, b) => a + b, 0);
        for (let i = 0; i < this.depth; i++) {
            this.weights[i] /= sum;
        }
        // Create parameter buffer
        this.paramsBuffer = device.createBuffer({
            size: 48, // 12 * 4 bytes for the AccumParams struct
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        // Create output buffer
        const bufferSize = this.width * this.height * 8; // 2 floats per pixel * 4 bytes
        this.outputBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        this.initPipeline();
    }
    async initPipeline() {
        // Load shader code
        const shaderCode = await fetch('/shaders/temporal/phasor_accumulate.wgsl').then(r => r.text());
        this.shaderModule = this.device.createShaderModule({
            code: shaderCode
        });
        this.accumulatePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.shaderModule,
                entryPoint: 'accumulate_with_motion' // Use motion-aware variant
            }
        });
    }
    push(field) {
        // Add new field to front
        this.fields.unshift(field);
        // Remove oldest if over capacity
        if (this.fields.length > this.depth) {
            const old = this.fields.pop();
            // Don't destroy here - let TexturePool manage it
        }
    }
    async accumulate(viewDelta, z, wavelength = 0.000532 // Green laser default
    ) {
        if (this.fields.length === 0) {
            throw new Error('PhaseRingBuffer: No fields to accumulate');
        }
        if (!this.accumulatePipeline) {
            await this.initPipeline();
        }
        // Get texture pool
        const pool = getTexturePool();
        // Create output texture
        const outputTexture = pool.create({
            size: [this.width, this.height],
            format: this.format,
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        });
        // Update parameters
        const params = new Float32Array([
            viewDelta.dx,
            viewDelta.dy,
            z,
            1.0 / wavelength, // invLambda
            this.weights[0] || 1.0,
            this.weights[1] || 0.0,
            this.weights[2] || 0.0,
            Math.min(this.fields.length, 3), // numBuffers
            this.width,
            this.height,
            0, // padding
            0 // padding
        ]);
        this.device.queue.writeBuffer(this.paramsBuffer, 0, params);
        // Create storage buffers for field data
        const fieldBuffers = [];
        for (let i = 0; i < Math.min(3, this.fields.length); i++) {
            const buffer = this.device.createBuffer({
                size: this.width * this.height * 8,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });
            // Copy texture to buffer (simplified - in practice use a copy encoder)
            // This is a placeholder - implement proper texture-to-buffer copy
            fieldBuffers.push(buffer);
        }
        // Pad with empty buffers if needed
        while (fieldBuffers.length < 3) {
            fieldBuffers.push(this.device.createBuffer({
                size: this.width * this.height * 8,
                usage: GPUBufferUsage.STORAGE
            }));
        }
        // Create bind group
        const bindGroup = this.device.createBindGroup({
            layout: this.accumulatePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.paramsBuffer } },
                { binding: 1, resource: { buffer: fieldBuffers[0] } },
                { binding: 2, resource: { buffer: fieldBuffers[1] } },
                { binding: 3, resource: { buffer: fieldBuffers[2] } },
                { binding: 4, resource: { buffer: this.outputBuffer } }
            ]
        });
        // Run compute pass
        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.accumulatePipeline);
        computePass.setBindGroup(0, bindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
        computePass.end();
        // Copy output buffer to texture
        commandEncoder.copyBufferToTexture({
            buffer: this.outputBuffer,
            bytesPerRow: this.width * 8,
            rowsPerImage: this.height
        }, { texture: outputTexture }, [this.width, this.height]);
        this.device.queue.submit([commandEncoder.finish()]);
        // Clean up temporary buffers
        for (const buffer of fieldBuffers) {
            buffer.destroy();
        }
        return outputTexture;
    }
    clear() {
        this.fields = [];
    }
    getDepth() {
        return this.depth;
    }
    getCurrentBufferCount() {
        return this.fields.length;
    }
}
