// D:\Dev\kha\frontend\lib\ml\ort.ts
import * as ort from 'onnxruntime-web';
import { InferenceSession, Tensor, env } from 'onnxruntime-web';

export { ort };

export { Tensor, InferenceSession };

export async function createSession(model: ArrayBuffer | Uint8Array) {
  // You can tweak these per-device if needed
  env.wasm.numThreads = navigator.hardwareConcurrency ?? 4;
  env.wasm.simd = true;

  const sessionOptions = {
    executionProviders: ['wasm'] // or 'webgpu' if/when you add the EP package
  };

  // Handle both ArrayBuffer and Uint8Array
  const modelData = model instanceof ArrayBuffer ? new Uint8Array(model) : model;
  
  const session = await InferenceSession.create(modelData, sessionOptions);
  return session;
}

export async function run(session: InferenceSession, feeds: Record<string, Tensor>, outputs?: string[]) {
  // Direct run - no outputNames in options
  if (outputs && outputs.length > 0) {
    // Pass output names as second parameter
    return session.run(feeds, outputs);
  }
  return session.run(feeds);
}
