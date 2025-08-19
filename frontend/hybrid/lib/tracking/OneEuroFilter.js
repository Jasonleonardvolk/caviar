// ${IRIS_ROOT}\frontend\hybrid\lib\tracking\OneEuroFilter.ts
export class OneEuro {
    constructor(minCutoff = 1.0, // Hz
    beta = 0.1, // speed coefficient
    dCutoff = 1.0 // derivative cutoff
    ) {
        this.minCutoff = minCutoff;
        this.beta = beta;
        this.dCutoff = dCutoff;
        this.dxPrev = 0;
        this.xPrev = 0;
        this.tPrev = 0;
        this.initialized = false;
    }
    alpha(cutoff, dt) {
        const tau = 1 / (2 * Math.PI * cutoff);
        return 1 / (1 + tau / dt);
    }
    update(x, tMs) {
        if (!this.initialized) {
            this.xPrev = x;
            this.tPrev = tMs;
            this.dxPrev = 0;
            this.initialized = true;
            return x;
        }
        const dt = Math.max(1e-3, (tMs - this.tPrev) / 1000); // seconds
        const dx = (x - this.xPrev) / dt;
        const aD = this.alpha(this.dCutoff, dt);
        const dxHat = aD * dx + (1 - aD) * this.dxPrev;
        const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
        const aX = this.alpha(cutoff, dt);
        const xHat = aX * x + (1 - aX) * this.xPrev;
        this.tPrev = tMs;
        this.xPrev = xHat;
        this.dxPrev = dxHat;
        return xHat;
    }
}
