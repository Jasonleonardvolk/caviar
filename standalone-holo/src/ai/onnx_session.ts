// onnx_session.ts - tiny ORT wrapper with webgpu->wasm fallback.
export type OrtBackend = 'webgpu' | 'wasm';

type OrtNS = typeof import('onnxruntime-web');
let ortPromise: Promise<OrtNS> | null = null;

async function getOrt(): Promise<OrtNS> {
  if (!ortPromise) ortPromise = import('onnxruntime-web');
  return ortPromise!;
}

export class OrtSession {
  private ort!: OrtNS;
  private session!: any;
  private provider!: OrtBackend;

  static async load(modelUrl: string, prefer: OrtBackend = 'webgpu'): Promise<OrtSession> {
    const inst = new OrtSession();
    inst.ort = await getOrt();

    const tryProviders: OrtBackend[] = prefer === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
    let lastErr: any = null;

    for (const ep of tryProviders) {
      try {
        // @ts-ignore: runtime options are loosely typed in onnxruntime-web
        inst.session = await inst.ort.InferenceSession.create(modelUrl, {
          executionProviders: [ep],
          graphOptimizationLevel: 'all',
        });
        inst.provider = ep;
        return inst;
      } catch (e) {
        lastErr = e;
      }
    }
    throw new Error(`Failed to init ONNX session for ${modelUrl}: ${lastErr}`);
  }

  getProvider(): OrtBackend { return this.provider; }

  getInputNames(): string[] { return this.session.inputNames as string[]; }
  getOutputNames(): string[] { return this.session.outputNames as string[]; }

  async run(feeds: Record<string, any>): Promise<Record<string, any>> {
    // Feeds should be ort.Tensor or plain TypedArray; we wrap as needed.
    const wrapped: Record<string, any> = {};
    for (const [k, v] of Object.entries(feeds)) {
      if (v && typeof (v as any).dims !== 'undefined') { wrapped[k] = v; continue; }
      // Assume v = { data: Float32Array|* , dims: number[] }
      if (ArrayBuffer.isView((v as any).data) && Array.isArray((v as any).dims)) {
        wrapped[k] = new this.ort.Tensor((v as any).data.constructor, (v as any).data, (v as any).dims);
      } else {
        throw new Error(`Feed ${k} must be ort.Tensor or {data,dims}`);
      }
    }
    const results = await this.session.run(wrapped);
    return results as Record<string, any>;
  }

  // Utility: convert ImageBitmap/ImageData to NCHW float32 [0..1]
  async imageToNCHW(img: ImageBitmap | ImageData, targetW: number, targetH: number): Promise<{ data: Float32Array, dims: number[] }> {
    const canvas = typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(targetW, targetH)
      : Object.assign(document.createElement('canvas'), { width: targetW, height: targetH }) as HTMLCanvasElement;

    // @ts-ignore
    canvas.width = targetW; canvas.height = targetH;
    // @ts-ignore
    const ctx = canvas.getContext('2d')!;
    if (img instanceof ImageData) ctx.putImageData(img, 0, 0);
    else ctx.drawImage(img as any, 0, 0, targetW, targetH);

    const rgba = ctx.getImageData(0, 0, targetW, targetH).data;
    const chw = new Float32Array(3 * targetH * targetW);
    // NCHW: [1,3,H,W]
    let r = 0, g = targetH * targetW, b = 2 * targetH * targetW;
    for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
      const R = rgba[i] / 255, G = rgba[i + 1] / 255, B = rgba[i + 2] / 255;
      chw[r++] = R; chw[g++] = G; chw[b++] = B;
    }
    return { data: chw, dims: [1, 3, targetH, targetW] };
  }
}