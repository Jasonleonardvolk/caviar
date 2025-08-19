// ${IRIS_ROOT}\frontend\hybrid\lib\parallaxController.ts
import { HeadPosePredictor } from './tracking/HeadPosePredictor';
import { LatencyModel } from './tracking/LatencyModel';
import { OutlierRejector } from './tracking/OutlierRejector';
import type { Pose, PredictedPose } from './tracking/types';

export type ParallaxOptions = {
  strength?: number;
  smoothing?: number;
  mode?: 'abg' | 'kalman' | 'euro';
};

export class ParallaxController {
  private canvas: HTMLCanvasElement;
  private options: ParallaxOptions;
  private enabled: boolean = false;
  
  // Static method for requesting permissions (iOS devices)
  static async requestPermission(): Promise<void> {
    if ('DeviceOrientationEvent' in window && 
        typeof (DeviceOrientationEvent as any).requestPermission === 'function') {
      const response = await (DeviceOrientationEvent as any).requestPermission();
      if (response !== 'granted') {
        throw new Error('Permission denied for device orientation');
      }
    }
    // For non-iOS or browsers without permission API, this is a no-op
  }
  
  constructor(canvas: HTMLCanvasElement, options: ParallaxOptions = {}) {
    this.canvas = canvas;
    this.options = {
      strength: 1.0,
      smoothing: 0.5,
      mode: 'abg',
      ...options
    };
  }
  
  start() {
    this.enabled = true;
    // Initialize tracking
  }
  
  stop() {
    this.enabled = false;
  }
  
  update() {
    if (!this.enabled) return;
    // Update parallax
  }
}

const latency = new LatencyModel();
latency.start(); // EMA estimate via rAF

const predictor = new HeadPosePredictor({
  mode: 'abg',
  alpha: 0.85, beta: 0.45, gamma: 0.06,
  euroMinCutoff: 1.0, euroBeta: 0.25, euroDerivCutoff: 1.0,
  confidenceFloor: 0.2,
});

const rejectX = new OutlierRejector(4, 0.05);
const rejectY = new OutlierRejector(4, 0.05);
const rejectZ = new OutlierRejector(4, 0.05);

// ---- camera apply seam ----
type ApplyFn = (pred: PredictedPose) => void;
let apply: ApplyFn = (pred) => {
  window.dispatchEvent(new CustomEvent('parallax:pose', { detail: pred }));
};
export function setParallaxApply(fn: ApplyFn) { apply = fn; }

// ---- external latency refinement (optional) ----
export function setExternalLatency(deltaMs: number) { latency.set(deltaMs); }
export function getLatencyMs(): number { return latency.get(); }

// ---- public API ----
export function ingestInput(meas: Pose) {
  latency.markInput(meas.t);
  const clean: Pose = {
    ...meas,
    p: [rejectX.accept(meas.p[0]), rejectY.accept(meas.p[1]), rejectZ.accept(meas.p[2])] as any,
  };
  const predicted = predictor.update(clean, latency.get());
  apply(predicted);
  (window as any).__PARALLAX_DEBUG = (window as any).__PARALLAX_DEBUG || {};
  (window as any).__PARALLAX_DEBUG.getPredictedPose = () => predictor.peek();
}

// Optional hooks for future staging, kept as no-ops for now
export function onRaf() {}
export function onSubmitted() {}
export function onPresented() {}

// Expose tuning API
export function updatePredictorParams(p: Partial<{
  alpha:number; beta:number; gamma:number;
  euroMinCutoff:number; euroBeta:number; euroDerivCutoff:number;
}>) {
  predictor.setParams(p);
}

export function getPredictorParams() {
  return predictor.getParams() ?? {};
}
