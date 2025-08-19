// ${IRIS_ROOT}\frontend\hybrid\lib\tracking\OutlierRejector.ts
export class OutlierRejector {
  private mean = 0; private var = 1; private n = 0;
  constructor(private readonly maxSigma = 4, private readonly clamp: number | null = null) {}

  accept(x: number): number {
    this.n++;
    const k = 1 / this.n;
    const delta = x - this.mean;
    this.mean += k * delta;
    this.var += (x - this.mean) * delta; // Welford
    const sigma = Math.sqrt(Math.max(1e-6, this.var / Math.max(1, this.n - 1)));
    const z = sigma > 1e-6 ? Math.abs((x - this.mean) / sigma) : 0;

    let y = x;
    if (z > this.maxSigma) y = this.mean;
    if (this.clamp != null) {
      const d = y - this.mean;
      if (Math.abs(d) > this.clamp) y = this.mean + Math.sign(d) * this.clamp;
    }
    return y;
  }
}
