// field_from_depth.ts - deterministic fallback: map depth -> (amp, phi)
export type Field = { amp: Float32Array, phi: Float32Array };

export function fieldFromDepth(depth: Float32Array, w: number, h: number, opts?: { kScale?: number }): Field {
  const n = w * h;
  const amp = new Float32Array(n);
  const phi = new Float32Array(n);
  // Center depth then map to phase
  let sum = 0;
  for (let i = 0; i < n; i++) sum += depth[i];
  const mean = sum / n;
  const kScale = opts?.kScale ?? 8.0; // tweakable radians per normalized depth unit
  const PI = Math.PI;

  for (let i = 0; i < n; i++) {
    amp[i] = 1.0; // flat amplitude; refine later
    let p = kScale * (depth[i] - mean);
    // wrap to [-pi, pi]
    p = ((p + PI) % (2 * PI) + (2 * PI)) % (2 * PI) - PI;
    phi[i] = p;
  }
  return { amp, phi };
}