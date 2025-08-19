// ${IRIS_ROOT}\frontend\hybrid\lib\parallaxController.ts
import { HeadPosePredictor } from './tracking/HeadPosePredictor';
import { LatencyModel } from './tracking/LatencyModel';
import { OutlierRejector } from './tracking/OutlierRejector';
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
let apply = (pred) => {
    window.dispatchEvent(new CustomEvent('parallax:pose', { detail: pred }));
};
export function setParallaxApply(fn) { apply = fn; }
// ---- external latency refinement (optional) ----
export function setExternalLatency(deltaMs) { latency.set(deltaMs); }
export function getLatencyMs() { return latency.get(); }
// ---- public API ----
export function ingestInput(meas) {
    latency.markInput(meas.t);
    const clean = {
        ...meas,
        p: [rejectX.accept(meas.p[0]), rejectY.accept(meas.p[1]), rejectZ.accept(meas.p[2])],
    };
    const predicted = predictor.update(clean, latency.get());
    apply(predicted);
    window.__PARALLAX_DEBUG = window.__PARALLAX_DEBUG || {};
    window.__PARALLAX_DEBUG.getPredictedPose = () => predictor.peek();
}
// Optional hooks for future staging, kept as no-ops for now
export function onRaf() { }
export function onSubmitted() { }
export function onPresented() { }
// Expose tuning API
export function updatePredictorParams(p) {
    predictor.setParams(p);
}
export function getPredictorParams() {
    return predictor.getParams() ?? {};
}
