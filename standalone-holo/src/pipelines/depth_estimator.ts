// depth_estimator.ts - monocular depth runner (generic), outputs Float32Array depth in [0..1]
import { OrtSession } from '../ai/onnx_session';

let depthSession: OrtSession | null = null;

export type DepthOut = { depth: Float32Array, width: number, height: number };

// Synthetic depth generator for testing without model
function syntheticDepth(img: ImageBitmap | ImageData, w: number, h: number): Float32Array {
  // Create a simple depth map based on image luminance + radial gradient
  const canvas = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(w, h)
    : Object.assign(document.createElement('canvas'), { width: w, height: h }) as HTMLCanvasElement;
  
  // @ts-ignore
  const ctx = canvas.getContext('2d')!;
  if (img instanceof ImageData) {
    ctx.putImageData(img, 0, 0);
  } else {
    ctx.drawImage(img as any, 0, 0, w, h);
  }
  
  const imageData = ctx.getImageData(0, 0, w, h);
  const depth = new Float32Array(w * h);
  const cx = w / 2, cy = h / 2;
  const maxDist = Math.sqrt(cx * cx + cy * cy);
  
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      // Luminance from RGB
      const lum = (imageData.data[i] * 0.299 + imageData.data[i+1] * 0.587 + imageData.data[i+2] * 0.114) / 255;
      // Radial gradient for 3D effect
      const dx = x - cx, dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy) / maxDist;
      // Combine luminance and radial for depth
      depth[y * w + x] = lum * 0.7 + (1 - dist) * 0.3;
    }
  }
  
  return depth;
}

export async function estimateDepth(
  img: ImageBitmap | ImageData,
  opts?: { modelUrl?: string, inputSize?: { w: number, h: number }, inputName?: string, outputName?: string, useSynthetic?: boolean }
): Promise<DepthOut> {
  const modelUrl = opts?.modelUrl ?? '/models/depth_estimator.onnx';
  const inputSize = opts?.inputSize ?? { w: 256, h: 256 }; // adjust to your model
  const inputName = opts?.inputName; // optional override
  const outputName = opts?.outputName; // optional override
  const useSynthetic = opts?.useSynthetic ?? false;

  // Check if model exists or use synthetic
  try {
    if (useSynthetic) throw new Error('Using synthetic depth');
    
    if (!depthSession) {
      // Try to load model
      const response = await fetch(modelUrl, { method: 'HEAD' });
      if (!response.ok) throw new Error('Model not found');
      depthSession = await OrtSession.load(modelUrl, 'webgpu');
    }

    const { data, dims } = await depthSession.imageToNCHW(img, inputSize.w, inputSize.h);
    const feedName = inputName ?? (depthSession.getInputNames()[0] ?? 'input');
    const feeds: Record<string, any> = {};
    feeds[feedName] = { data, dims };

    const result = await depthSession.run(feeds);
    const outName = outputName ?? (depthSession.getOutputNames()[0] ?? 'depth');
    const tensor = result[outName];
    if (!tensor) throw new Error(`Depth model missing output '${outName}'. Available: ${depthSession.getOutputNames().join(', ')}`);

    // Normalize to [0..1]
    const raw = tensor.data as Float32Array | Float32Array;
    let min = Number.POSITIVE_INFINITY, max = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < raw.length; i++) { const v = raw[i]; if (v < min) min = v; if (v > max) max = v; }
    const inv = (max > min) ? 1 / (max - min) : 1;
    const norm = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) norm[i] = (raw[i] - min) * inv;

    // Assume H,W = dims tail; common shapes: [1,1,H,W] or [1,H,W]
    let H = inputSize.h, W = inputSize.w;
    const d = tensor.dims as number[] | undefined;
    if (d && d.length >= 2) { H = d[d.length - 2]; W = d[d.length - 1]; }

    return { depth: norm, width: W, height: H };
    
  } catch (err) {
    // Fallback to synthetic depth
    console.log('Using synthetic depth (model not available):', err.message);
    const depth = syntheticDepth(img, inputSize.w, inputSize.h);
    return { depth, width: inputSize.w, height: inputSize.h };
  }
}