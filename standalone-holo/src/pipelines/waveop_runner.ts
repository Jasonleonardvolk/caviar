// waveop_runner.ts - Penrose ONNX-core wrapper: depth -> (amp, phi). Falls back to deterministic mapper.
import { OrtSession } from '../ai/onnx_session';
import { fieldFromDepth } from './field_from_depth';

let waveSession: OrtSession | null = null;

export async function runWaveOp(
  depth: Float32Array, w: number, h: number,
  opts?: { modelUrl?: string, inputName?: string, ampName?: string, phiName?: string }
): Promise<{ amp: Float32Array, phi: Float32Array }> {
  const modelUrl = opts?.modelUrl ?? '/models/waveop_fno_v1.onnx';
  if (!waveSession) waveSession = await OrtSession.load(modelUrl, 'webgpu');

  const inName = opts?.inputName ?? (waveSession.getInputNames()[0] ?? 'in');
  // Shape to [1,1,H,W]
  const data = new Float32Array(1 * 1 * h * w);
  data.set(depth, 0);
  const feeds: Record<string, any> = {};
  feeds[inName] = { data, dims: [1, 1, h, w] };

  try {
    const result = await waveSession.run(feeds);
    const ampName = opts?.ampName ?? (waveSession.getOutputNames().find(n => /amp/i.test(n)) ?? waveSession.getOutputNames()[0]);
    const phiName = opts?.phiName ?? (waveSession.getOutputNames().find(n => /phi|phase/i.test(n)) ?? waveSession.getOutputNames()[1]);

    const ampT = result[ampName], phiT = result[phiName];
    if (!ampT || !phiT) throw new Error('Missing amp/phi outputs');
    // Ensure flat arrays of length w*h
    const amp = new Float32Array(w * h); amp.set(ampT.data.slice(0, w * h));
    const phi = new Float32Array(w * h); phi.set(phiT.data.slice(0, w * h));
    return { amp, phi };
  } catch {
    // Fallback
    return fieldFromDepth(depth, w, h);
  }
}