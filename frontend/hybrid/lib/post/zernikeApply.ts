// ${IRIS_ROOT}\frontend\hybrid\lib\post\zernikeApply.ts
import { device } from "@/lib/webgpu/context/device";
import zernikeWgsl from "@/lib/webgpu/shaders/post/zernikeApply.wgsl?raw";

export type ZernikeCoeffs = {
  tipX?: number; 
  tiltY?: number; 
  defocus?: number;
  astig0?: number; 
  astig45?: number; 
  comaX?: number; 
  comaY?: number; 
  spherical?: number;
};

export type ZernikeParams = {
  width: number; 
  height: number;
  cx?: number; 
  cy?: number;   // default: width/2, height/2 (px)
  ax?: number; 
  ay?: number;   // default: width/2, height/2 (px)
  maxCorrection?: number;     // radians, default 0.25
  softness?: number;          // 0..0.5, default 0.05
  outsideBehavior?: 0 | 1 | 2; // 0=zero, 1=attenuate, 2=hold (default 1)
};

const COEFFS_LEN = 8;            // [tipX, tiltY, defocus, astig0, astig45, comaX, comaY, spherical]
const PARAMS_SIZE = 64;          // pad to 64B for cross-driver safety

export class ZernikeApply {
  private module!: GPUShaderModule;
  private pipeline!: GPUComputePipeline;
  private paramsBuf!: GPUBuffer;
  private coeffsBuf!: GPUBuffer;

  async init() {
    this.module = device.createShaderModule({ code: zernikeWgsl });
    this.pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: this.module, entryPoint: "main" },
    });
    this.paramsBuf = device.createBuffer({
      size: PARAMS_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.coeffsBuf = device.createBuffer({
      size: COEFFS_LEN * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // Default zero coeffs
    const zeros = new Float32Array(COEFFS_LEN);
    device.queue.writeBuffer(this.coeffsBuf, 0, zeros.slice());
  }

  setCoeffs(c: ZernikeCoeffs) {
    const arr = new Float32Array(COEFFS_LEN);
    arr[0] = c.tipX ?? 0;
    arr[1] = c.tiltY ?? 0;
    arr[2] = c.defocus ?? 0;
    arr[3] = c.astig0 ?? 0;
    arr[4] = c.astig45 ?? 0;
    arr[5] = c.comaX ?? 0;
    arr[6] = c.comaY ?? 0;
    arr[7] = c.spherical ?? 0;
    device.queue.writeBuffer(this.coeffsBuf, 0, arr.slice());
  }

  async setCoeffsFromUrl(url: string) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`zernike coeffs fetch failed: ${res.status} ${url}`);
    const json = await res.json();
    this.setCoeffs(json as ZernikeCoeffs);
  }

  run(encoder: GPUCommandEncoder, reBuf: GPUBuffer, imBuf: GPUBuffer, p: ZernikeParams) {
    const width = p.width, height = p.height;
    // Defaults: centered circular aperture, gentle edge attenuation
    const cx = p.cx ?? (width * 0.5);
    const cy = p.cy ?? (height * 0.5);
    const ax = p.ax ?? (width * 0.5);
    const ay = p.ay ?? (height * 0.5);
    const maxC = p.maxCorrection ?? 0.25;
    const soft = p.softness ?? 0.05;
    const outside = p.outsideBehavior ?? 1;

    // Pack Params struct (matches WGSL layout):
    // u32 width, u32 height,
    // f32 cx, f32 cy, f32 ax, f32 ay, f32 max_correction, f32 softness,
    // u32 outside_behavior, u32 pad0, u32 pad1, u32 pad2
    const buf = new ArrayBuffer(PARAMS_SIZE);
    const dv  = new DataView(buf);
    dv.setUint32(0,  width, true);
    dv.setUint32(4,  height, true);
    dv.setFloat32(8,  cx, true);
    dv.setFloat32(12, cy, true);
    dv.setFloat32(16, ax, true);
    dv.setFloat32(20, ay, true);
    dv.setFloat32(24, maxC, true);
    dv.setFloat32(28, soft, true);
    dv.setUint32(32, outside, true);
    // padding left as zero
    device.queue.writeBuffer(this.paramsBuf, 0, buf.slice());

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: reBuf } },
        { binding: 1, resource: { buffer: imBuf } },
        { binding: 2, resource: { buffer: this.coeffsBuf } },
        { binding: 3, resource: { buffer: this.paramsBuf } },
      ],
    }));
    const wgX = Math.ceil(width / 8);
    const wgY = Math.ceil(height / 8);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  }

  destroy() {
    this.paramsBuf.destroy();
    this.coeffsBuf.destroy();
    // pipeline/module are GC'd with device
  }
}
