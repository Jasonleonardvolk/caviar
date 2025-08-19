// slm_export.ts - PNG exporters for phase-only and Lee off-axis masks
// Expects Float32Array amp (A in [0,1]) and phi (phi in [-pi, pi]) of size W*H.

import { ensureArrayBuffer } from '../../../frontend/lib/utils/bufferUtils';

function buf(device: GPUDevice, size: number, usage: GPUBufferUsageFlags) {
  return device.createBuffer({ size, usage });
}

async function downloadGrayPNG(pixels: Uint8ClampedArray, w: number, h: number, filename: string) {
  const off = (typeof OffscreenCanvas !== 'undefined')
    ? new OffscreenCanvas(w, h)
    : (() => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; })() as any;
  // @ts-ignore
  const ctx: CanvasRenderingContext2D = off.getContext('2d');
  const img = ctx.createImageData(w, h);
  for (let i = 0, j = 0; i < pixels.length; i++, j += 4) {
    const v = pixels[i];
    img.data[j] = v; img.data[j + 1] = v; img.data[j + 2] = v; img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  // @ts-ignore
  const blob: Blob = off.convertToBlob ? await off.convertToBlob({ type: 'image/png' }) : await new Promise<Blob>(res => (off as HTMLCanvasElement).toBlob(b => res(b!), 'image/png'));
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

export async function exportPhaseOnly(device: GPUDevice, amp: Float32Array, phi: Float32Array, w: number, h: number, filename = `phase_only_${w*2}x${h}.png`) {
  const src = await import('../shaderSources');
  const code = (src as any).phase_only_encode_wgsl as string;
  const module = device.createShaderModule({ code });
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });

  const ampBuf = buf(device, amp.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(ampBuf, 0, ensureArrayBuffer(amp) as ArrayBufferView & { buffer: ArrayBuffer });
  const phiBuf = buf(device, phi.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(phiBuf, 0, ensureArrayBuffer(phi) as ArrayBufferView & { buffer: ArrayBuffer });

  const outLen = w * 2 * h;
  const outBuf = buf(device, outLen * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  // Params (16B): width, height, pad, pad
  const params = new Uint32Array([w, h, 0, 0]);
  const ubo = buf(device, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(ubo, 0, params);

  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: ampBuf } },
      { binding: 1, resource: { buffer: phiBuf } },
      { binding: 2, resource: { buffer: outBuf } },
      { binding: 3, resource: { buffer: ubo } }
    ]
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
  pass.end();

  const cpu = buf(device, outLen * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  enc.copyBufferToBuffer(outBuf, 0, cpu, 0, outLen * 4);
  device.queue.submit([enc.finish()]);
  await cpu.mapAsync(GPUMapMode.READ);
  const f32 = new Float32Array(cpu.getMappedRange().slice(0));
  const bytes = new Uint8ClampedArray(outLen);
  const TWO_PI = Math.PI * 2;
  for (let i = 0; i < outLen; i++) bytes[i] = Math.min(255, Math.max(0, Math.round((f32[i] / TWO_PI) * 255)));
  await downloadGrayPNG(bytes, w * 2, h, filename);
}

export async function exportLeeAmplitude(
  device: GPUDevice,
  amp: Float32Array, phi: Float32Array, w: number, h: number,
  opts?: { fx?: number; fy?: number; binary?: boolean; bias?: number; scale?: number; filename?: string }
) {
  const src = await import('../shaderSources');
  const code = (src as any).lee_offaxis_encode_wgsl as string;
  const module = device.createShaderModule({ code });
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });

  const ampBuf = buf(device, amp.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(ampBuf, 0, ensureArrayBuffer(amp) as ArrayBufferView & { buffer: ArrayBuffer });
  const phiBuf = buf(device, phi.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(phiBuf, 0, ensureArrayBuffer(phi) as ArrayBufferView & { buffer: ArrayBuffer });
  const outBuf = buf(device, w * h * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

  const fx = opts?.fx ?? 0.25;
  const fy = opts?.fy ?? 0.0;
  const binary = opts?.binary ? 1 : 0;
  const bias = opts?.bias ?? 0.5;
  const scale = opts?.scale ?? 0.5;

  // Params layout (32B): u32 w,h,binary,pad | f32 fx,fy,bias,scale
  const u8 = new Uint8Array(32);
  const dv = new DataView(u8.buffer);
  dv.setUint32(0, w, true);
  dv.setUint32(4, h, true);
  dv.setUint32(8, binary, true);
  dv.setUint32(12, 0, true);       // pad
  dv.setFloat32(16, fx, true);
  dv.setFloat32(20, fy, true);
  dv.setFloat32(24, bias, true);
  dv.setFloat32(28, scale, true);
  const ubo = buf(device, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(ubo, 0, u8);

  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: ampBuf } },
      { binding: 1, resource: { buffer: phiBuf } },
      { binding: 2, resource: { buffer: outBuf } },
      { binding: 3, resource: { buffer: ubo } }
    ]
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
  pass.end();

  const cpu = buf(device, w * h * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  enc.copyBufferToBuffer(outBuf, 0, cpu, 0, w * h * 4);
  device.queue.submit([enc.finish()]);
  await cpu.mapAsync(GPUMapMode.READ);
  const f32 = new Float32Array(cpu.getMappedRange().slice(0));
  const bytes = new Uint8ClampedArray(w * h);
  for (let i = 0; i < bytes.length; i++) bytes[i] = Math.round(Math.min(1, Math.max(0, f32[i])) * 255);
  await downloadGrayPNG(bytes, w, h, opts?.filename ?? `lee_amp_${w}x${h}${binary ? '_bin' : ''}.png`);
}