// tori_ui_svelte/src/lib/cognitive/dynamics.ts
// -----------------------------------------------------------------------------
// Production‑grade implementations for the missing dynamic‑systems utilities
// referenced by GhostSolitonIntegration.ts: KoopmanOperator and LyapunovAnalyzer.
// Each class ingests raw time‑series signals, performs the requisite analysis,
// and emits CustomEvents that GhostSolitonIntegration already subscribes to.
//
//  • KoopmanOperator   →  dispatches  "tori-koopman-update"  (spectral snapshot)
//  • LyapunovAnalyzer  →  dispatches  "tori-lyapunov-spike" (chaos detector)
// -----------------------------------------------------------------------------

import * as math from 'mathjs';

/** ---------------------------------------------------------------------------
 * Types
 * --------------------------------------------------------------------------*/
export interface KoopmanUpdate {
  eigenmodes: Array<{ frequency: number; amplitude: number; phase: number }>;
  spectralGap: number;
  dominantMode: string;
}

export interface LyapunovSpike {
  exponent: number;           // Largest Lyapunov exponent (λ₁)
  instabilityLevel: number;   // Normalised 0‑1 measure of chaos severity
  divergenceRate: number;     // Instantaneous divergence per timestep
  timeHorizon: number;        // Samples used to estimate λ₁
}

/** ---------------------------------------------------------------------------
 * Utility – DOM event helper
 * --------------------------------------------------------------------------*/
function emit<T>(type: string, detail: T) {
  document.dispatchEvent(new CustomEvent<T>(type, { detail }));
}

/** ---------------------------------------------------------------------------
 * KoopmanOperator – Dynamic Mode Decomposition (DMD) wrapper
 * --------------------------------------------------------------------------*/
export class KoopmanOperator {
  private buffer: number[][] = [];
  private readonly window: number;
  private readonly maxModes: number;

  constructor(opts: { window?: number; maxModes?: number } = {}) {
    this.window = opts.window ?? 128;     // sliding window size
    this.maxModes = opts.maxModes ?? 10;  // modes to keep in report
  }

  /** Ingest a vector observation (multi‑dimensional state snapshot). */
  ingest(sample: number[]): void {
    this.buffer.push(sample);
    if (this.buffer.length >= this.window) {
      this.analyze();
      // slide window by half to balance cost vs. latency
      this.buffer = this.buffer.slice(this.window / 2);
    }
  }

  /** Perform a basic DMD to approximate Koopman spectral components. */
  private analyze() {
    const X = math.matrix(this.buffer.slice(0, -1));
    const Y = math.matrix(this.buffer.slice(1));

    // Solve Y ≈ K · X  →  K = Y · X⁺ (pseudo‑inverse)
    const pinvX = math.pinv(X);
    const K = math.multiply(Y, pinvX) as math.Matrix;

    // Eigen‑decomposition
    const eigsResult = math.eigs(K);
    const eigvals = eigsResult.values as any[];
    const eigenvectors = eigsResult.eigenvectors as any[];
    
    // Process eigenvalues and eigenvectors
    const modeData = eigenvectors.map((item: any, i: number) => {
      const eigenvalue = item.value;
      const eigenvector = item.vector;
      const mag = typeof eigenvalue === 'number' ? Math.abs(eigenvalue) : 
                   (math.abs(eigenvalue as math.Complex) as unknown) as number;
      
      return {
        lambda: eigenvalue,
        vec: eigenvector,
        mag: mag,
        frequency: typeof eigenvalue === 'number' ? 0 : 
                   Math.atan2((eigenvalue as any).im || 0, (eigenvalue as any).re || eigenvalue) / (2 * Math.PI),
        phase: typeof eigenvalue === 'number' ? 0 :
               Math.atan2((eigenvalue as any).im || 0, (eigenvalue as any).re || eigenvalue)
      };
    });

    // Sort by magnitude desc and take top modes
    const sorted = modeData
      .sort((a: any, b: any) => b.mag - a.mag)
      .slice(0, this.maxModes);

    const dominant = sorted[0];
    const spectralGap = sorted.length > 1 ? dominant.mag - sorted[1].mag : dominant.mag;

    const eigenmodes = sorted.map((m) => ({
      frequency: m.frequency,
      amplitude: m.mag,
      phase: m.phase,
    }));

    const update: KoopmanUpdate = {
      eigenmodes,
      spectralGap,
      dominantMode: eigenmodes[0]?.frequency.toFixed(3) ?? 'n/a',
    };

    emit<KoopmanUpdate>('tori-koopman-update', update);
  }
}

/** Singleton instance (import { koopmanOperator } …) */
export const koopmanOperator = new KoopmanOperator();

/** ---------------------------------------------------------------------------
 * LyapunovAnalyzer – Largest Lyapunov exponent estimator (Rosenstein method)
 * --------------------------------------------------------------------------*/
export class LyapunovAnalyzer {
  private series: number[] = [];
  private readonly dim: number;
  private readonly delay: number;
  private readonly horizon: number;
  private readonly spikeThreshold: number;

  constructor(opts: { dim?: number; delay?: number; horizon?: number; spikeThreshold?: number } = {}) {
    this.dim = opts.dim ?? 3;                 // embedding dimension
    this.delay = opts.delay ?? 2;             // delay (samples)
    this.horizon = opts.horizon ?? 64;        // samples for λ₁ estimate
    this.spikeThreshold = opts.spikeThreshold ?? 0.1; // chaos detection
  }

  ingest(value: number): void {
    this.series.push(value);
    if (this.series.length >= this.horizon + this.dim * this.delay) {
      const spike = this.analyze();
      if (spike.instabilityLevel >= this.spikeThreshold) {
        emit<LyapunovSpike>('tori-lyapunov-spike', spike);
      }
      // keep last horizon samples to slide forward
      this.series = this.series.slice(-this.horizon);
    }
  }

  /** Estimate λ₁ via average divergence of nearest neighbours. */
  private analyze(): LyapunovSpike {
    // Build delay‑embedding matrix
    const m = this.series.length - (this.dim - 1) * this.delay;
    const embedded: number[][] = Array.from({ length: m }, (_, i) =>
      Array.from({ length: this.dim }, (_, j) => this.series[i + j * this.delay])
    );

    // Find nearest neighbour for each row (excluding self & temporal neighbours)
    const divergences: number[] = [];
    for (let i = 0; i < m; i++) {
      let minDist = Infinity;
      let nnIdx = -1;
      for (let j = 0; j < m; j++) {
        if (Math.abs(i - j) < this.delay) continue; // Theiler window
        const dist = math.distance(embedded[i], embedded[j]) as number;
        if (dist < minDist) {
          minDist = dist;
          nnIdx = j;
        }
      }
      if (nnIdx === -1) continue;

      // track divergence over horizon (or until end)
      const maxT = Math.min(this.horizon, m - Math.max(i, nnIdx));
      for (let t = 1; t < maxT; t++) {
        const d = math.distance(embedded[i + t], embedded[nnIdx + t]) as number;
        if (d > 0) divergences.push(Math.log(d));
      }
    }

    const lambda = divergences.length ? math.mean(divergences) / (this.horizon * 1.0) : 0;
    const instability = Math.min(1, Math.max(0, 1 - Math.exp(-lambda))); // squash 0‑1

    return {
      exponent: lambda,
      instabilityLevel: instability,
      divergenceRate: lambda > 0 ? lambda : 0,
      timeHorizon: this.horizon,
    } as LyapunovSpike;
  }
}

/** Singleton instance (import { lyapunovAnalyzer } …) */
export const lyapunovAnalyzer = new LyapunovAnalyzer();

// -----------------------------------------------------------------------------
// Quick usage example (development only – remove in prod):
// -----------------------------------------------------------------------------
// const ko = koopmanOperator;
// setInterval(() => {
//   ko.ingest([Math.random(), Math.sin(Date.now() / 1000), Math.cos(Date.now() / 700)]);
// }, 50);
// const la = lyapunovAnalyzer;
// setInterval(() => la.ingest(Math.random()), 33);
// -----------------------------------------------------------------------------
