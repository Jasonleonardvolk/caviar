type Cfg = { targetFps:number; emaAlpha:number; upHysteresisMs:number; downHysteresisMs:number };
export class ThermalGovernor {
  private ema = 0;
  private budgetMs: number;
  private cfg: Cfg;
  constructor(cfg: Partial<Cfg> = {}) {
    this.cfg = { targetFps: 60, emaAlpha: 0.1, upHysteresisMs: 2.0, downHysteresisMs: 1.0, ...cfg };
    this.budgetMs = 1000 / this.cfg.targetFps;
  }
  tick(frameMs: number) {
    this.ema = this.cfg.emaAlpha * frameMs + (1 - this.cfg.emaAlpha) * this.ema;
  }
  advise(currentN: number) {
    const over = this.ema > (this.budgetMs + this.cfg.downHysteresisMs);
    const under = this.ema < (this.budgetMs - this.cfg.upHysteresisMs);
    if (over && currentN > 256) return { action: 'decrease', nextN: Math.max(256, currentN >> 1) };
    if (over && currentN > 192)  return { action: 'decrease', nextN: 192 };
    if (under && currentN < 512) return { action: 'increase', nextN: Math.min(512, currentN << 1) };
    return { action: 'hold', nextN: currentN };
  }
}