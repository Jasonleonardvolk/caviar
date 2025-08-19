// ${IRIS_ROOT}\frontend\hybrid\lib\tracking\LatencyModel.ts
// EMA of end-to-end latency Delta approximately (present - lastInput). Uses rAF as a proxy for present.
export class LatencyModel {
    constructor() {
        this.ema = 24; // ms baseline
        this.alpha = 0.15; // smoothing
        this.lastInput = 0;
        this.rafHandle = 0;
        // approximate "presented" time using next rAF tick
        this.tick = () => {
            const now = performance.now();
            if (this.lastInput > 0) {
                const delta = now - this.lastInput;
                this.ema = (1 - this.alpha) * this.ema + this.alpha * delta;
            }
            this.rafHandle = requestAnimationFrame(this.tick);
        };
    }
    markInput(t = performance.now()) { this.lastInput = t; }
    start() { if (!this.rafHandle)
        this.rafHandle = requestAnimationFrame(this.tick); }
    stop() { if (this.rafHandle)
        cancelAnimationFrame(this.rafHandle); this.rafHandle = 0; }
    set(deltaMs) { this.ema = (1 - this.alpha) * this.ema + this.alpha * deltaMs; }
    get() { return this.ema; }
}
