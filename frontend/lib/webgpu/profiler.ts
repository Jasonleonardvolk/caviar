// ${IRIS_ROOT}\frontend\lib\webgpu\profiler.ts
export type Sample = { name: string; ns: number };

export class GpuProfiler {
  private device: GPUDevice;
  private supported: boolean;
  private qs?: GPUQuerySet;
  private resolveBuf?: GPUBuffer;
  private writeIndex = 0;
  private capacity = 256; // pairs -> up to 128 passes

  constructor(device: GPUDevice, supported: boolean) {
    this.device = device;
    this.supported = supported;
    if (supported) {
      this.qs = device.createQuerySet({ type: 'timestamp', count: this.capacity });
      this.resolveBuf = device.createBuffer({
        size: BigInt(this.capacity * 8) as any, // 8 bytes per u64
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }
  }

  beginComputePass(encoder: GPUCommandEncoder, label: string) {
    const start = this.supported ? this.writeIndex++ : -1;
    const end   = this.supported ? this.writeIndex++ : -1;
    const pass = encoder.beginComputePass({
      label,
      timestampWrites: this.supported && this.qs ? {
        querySet: this.qs,
        beginningOfPassWriteIndex: start,
        endOfPassWriteIndex: end,
      } : undefined,
    });
    return { pass, start, end, label };
  }

  beginRenderPass(encoder: GPUCommandEncoder, desc: GPURenderPassDescriptor, label: string) {
    const start = this.supported ? this.writeIndex++ : -1;
    const end   = this.supported ? this.writeIndex++ : -1;
    const rpd: GPURenderPassDescriptor = { ...desc };
    if (this.supported && this.qs) {
      (rpd as any).timestampWrites = {
        querySet: this.qs,
        beginningOfPassWriteIndex: start,
        endOfPassWriteIndex: end,
      };
    }
    const pass = encoder.beginRenderPass(rpd);
    (pass as any).__label = label;
    return { pass, start, end, label };
  }

  async resolve(encoder: GPUCommandEncoder): Promise<void> {
    if (!this.supported || !this.qs || !this.resolveBuf) return;
    const count = this.writeIndex;
    if (count > 0) encoder.resolveQuerySet(this.qs, 0, count, this.resolveBuf, 0);
  }

  async read(): Promise<Sample[]> {
    if (!this.supported || !this.resolveBuf) return [];
    await this.device.queue.onSubmittedWorkDone();
    await this.resolveBuf.mapAsync(GPUMapMode.READ);
    const arr = new BigUint64Array(this.resolveBuf.getMappedRange());
    const out: Sample[] = [];
    for (let i = 0; i < this.writeIndex; i += 2) {
      const start = arr[i];
      const end = arr[i + 1];
      // Spec: timestamp results resolve to nanoseconds (u64). If 0, ignore.
      // See W3C WD text.
      if (start !== 0n && end !== 0n && end > start) {
        const ns = Number(end - start);
        out.push({ name: `pass#${(i/2)|0}`, ns });
      }
    }
    this.resolveBuf.unmap();
    this.writeIndex = 0;
    return out;
  }
}
