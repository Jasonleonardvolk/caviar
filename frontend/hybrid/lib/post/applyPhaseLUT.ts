// ${IRIS_ROOT}\frontend\hybrid\lib\post\applyPhaseLUT.ts
import { device } from "@/lib/webgpu/context/device";
import applyLutWgsl from "@/lib/webgpu/shaders/post/applyPhaseLUT.wgsl?raw";

const PARAMS_SIZE = 64;

export type LutMeta = {
  shape_hw: [number, number];        // [H, W] of LUT
  max_correction?: number;           // optional hint
  generated_utc?: string;
};

export class ApplyPhaseLUT {
  private module!: GPUShaderModule;
  private pipeline!: GPUComputePipeline;
  private paramsBuf!: GPUBuffer;
  private dphiBuf!: GPUBuffer;
  private lutW = 0;
  private lutH = 0;

  async init() {
    this.module = device.createShaderModule({ code: applyLutWgsl });
    this.pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: this.module, entryPoint: "main" },
    });
    this.paramsBuf = device.createBuffer({
      size: PARAMS_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  async loadFromUrls(binUrl: string, metaUrl?: string) {
    const [binRes, metaRes] = await Promise.all([
      fetch(binUrl),
      metaUrl ? fetch(metaUrl) : Promise.resolve(undefined as any),
    ]);
    if (!binRes.ok) throw new Error(`Dphi bin fetch failed: ${binRes.status} ${binUrl}`);
    const bin = new Float32Array(await binRes.arrayBuffer());

    let meta: LutMeta | undefined;
    if (metaUrl) {
      if (!metaRes.ok) throw new Error(`Dphi meta fetch failed: ${metaRes.status} ${metaUrl}`);
      meta = (await metaRes.json()) as LutMeta;
    }

    // Infer H,W if not in meta - assume square if perfect square length
    if (meta?.shape_hw) {
      this.lutH = meta.shape_hw[0] | 0;
      this.lutW = meta.shape_hw[1] | 0;
    } else {
      const n = bin.length;
      const s = Math.floor(Math.sqrt(n));
      if (s * s !== n) throw new Error(`Cannot infer LUT shape (len=${n}); provide meta.json`);
      this.lutH = s; this.lutW = s;
    }

    if (bin.length !== this.lutW * this.lutH) {
      throw new Error(`Dphi length ${bin.length} != lutW*lutH ${this.lutW*this.lutH}`);
    }

    // Create / upload STORAGE buffer
    this.dphiBuf?.destroy();
    this.dphiBuf = device.createBuffer({
      size: bin.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.dphiBuf.getMappedRange()).set(bin);
    this.dphiBuf.unmap();
  }

  run(encoder: GPUCommandEncoder, reBuf: GPUBuffer, imBuf: GPUBuffer, width: number, height: number, opts?: {
    gain?: number;
    maxCorrection?: number;
    mapping?: {
      scaleX?: number; scaleY?: number;
      offsetX?: number; offsetY?: number;
    };
  }) {
    if (!this.dphiBuf) throw new Error("Dphi LUT not loaded");

    // Default mapping: align edges exactly (identity when sizes match)
    const scaleX = opts?.mapping?.scaleX ?? ((this.lutW - 1) / Math.max(1, (width - 1)));
    const scaleY = opts?.mapping?.scaleY ?? ((this.lutH - 1) / Math.max(1, (height - 1)));
    const offsetX = opts?.mapping?.offsetX ?? 0.0;
    const offsetY = opts?.mapping?.offsetY ?? 0.0;

    const gain = opts?.gain ?? 1.0;
    const maxC = opts?.maxCorrection ?? 0.30;

    // Pack uniform
    const buf = new ArrayBuffer(PARAMS_SIZE);
    const dv  = new DataView(buf);
    dv.setUint32(0,  width,  true);
    dv.setUint32(4,  height, true);
    dv.setUint32(8,  this.lutW, true);
    dv.setUint32(12, this.lutH, true);
    dv.setFloat32(16, gain,  true);
    dv.setFloat32(20, maxC,  true);
    dv.setFloat32(24, scaleX, true);
    dv.setFloat32(28, scaleY, true);
    dv.setFloat32(32, offsetX, true);
    dv.setFloat32(36, offsetY, true);
    // rest padded to zero
    device.queue.writeBuffer(this.paramsBuf, 0, buf.slice());

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: reBuf } },
        { binding: 1, resource: { buffer: imBuf } },
        { binding: 2, resource: { buffer: this.dphiBuf } },
        { binding: 3, resource: { buffer: this.paramsBuf } },
      ],
    }));
    pass.dispatchWorkgroups(Math.ceil(width/8), Math.ceil(height/8), 1);
    pass.end();
  }

  destroy() {
    this.paramsBuf?.destroy();
    this.dphiBuf?.destroy();
  }
}
