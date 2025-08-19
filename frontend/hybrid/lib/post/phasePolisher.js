// ${IRIS_ROOT}\frontend\hybrid\lib\post\phasePolisher.ts
import { device } from '@/lib/webgpu/context/device';
import phasePolisherWgsl from '@/lib/webgpu/shaders/post/phasePolisher.wgsl?raw';
export class PhasePolisher {
    async init() {
        this.module = device.createShaderModule({ code: phasePolisherWgsl });
        this.pipeline = await device.createComputePipelineAsync({
            layout: 'auto',
            compute: { module: this.module, entryPoint: 'main' },
        });
    }
    // reBuf/imBuf/maskBuf are STORAGE buffers of length width*height floats
    run(encoder, reBuf, imBuf, params, maskBuf) {
        const { width, height } = params;
        const tv = params.tvLambda ?? 0.08;
        const maxc = params.maxCorrection ?? 0.25;
        const useMask = (params.useMask && maskBuf) ? 1 : 0;
        // Params uniform: [width(u32), height(u32), tv_lambda(f32), max_correction(f32), use_mask(u32), pad(u32)]
        const paramArray = new ArrayBuffer(24);
        const u32v = new Uint32Array(paramArray);
        const f32v = new Float32Array(paramArray);
        u32v[0] = width;
        u32v[1] = height;
        f32v[2] = tv;
        f32v[3] = maxc;
        u32v[8 / 4 + 0] = useMask; // offset 8 bytes after first two f32s (keep alignment simple)
        this.paramBuf ?? (this.paramBuf = device.createBuffer({
            size: 24, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        }));
        device.queue.writeBuffer(this.paramBuf, 0, paramArray);
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: reBuf } },
                { binding: 1, resource: { buffer: imBuf } },
                { binding: 2, resource: { buffer: maskBuf ?? reBuf } }, // dummy if unused
                { binding: 3, resource: { buffer: this.paramBuf } },
            ]
        }));
        const wgX = Math.ceil(width / 8);
        const wgY = Math.ceil(height / 8);
        pass.dispatchWorkgroups(wgX, wgY, 1);
        pass.end();
    }
}
