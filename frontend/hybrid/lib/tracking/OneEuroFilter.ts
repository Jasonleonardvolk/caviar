// ${IRIS_ROOT}\frontend\hybrid\lib\tracking\OneEuroFilter.ts
export class OneEuro {
  private dxPrev = 0;
  private xPrev = 0;
  private tPrev = 0;
  private initialized = false;

  constructor(
    public minCutoff = 1.0,   // Hz
    public beta = 0.1,        // speed coefficient
    public dCutoff = 1.0      // derivative cutoff
  ) {}

  private alpha(cutoff: number, dt: number) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  update(x: number, tMs: number): number {
    if (!this.initialized) {
      this.xPrev = x; this.tPrev = tMs; this.dxPrev = 0; this.initialized = true;
      return x;
    }
    const dt = Math.max(1e-3, (tMs - this.tPrev) / 1000); // seconds
    const dx = (x - this.xPrev) / dt;
    const aD = this.alpha(this.dCutoff, dt);
    const dxHat = aD * dx + (1 - aD) * this.dxPrev;

    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const aX = this.alpha(cutoff, dt);
    const xHat = aX * x + (1 - aX) * this.xPrev;

    this.tPrev = tMs; this.xPrev = xHat; this.dxPrev = dxHat;
    return xHat;
  }
}
