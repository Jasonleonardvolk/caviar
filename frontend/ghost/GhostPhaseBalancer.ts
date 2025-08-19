/**
 * GhostPhaseBalancer – generates a dark soliton shadow trace for a given bright memory.
 * The shadow has phase shifted by π (out-of-phase) and a small negative amplitude.
 */
export function createShadowTrace(bright: { phaseTag: number; amplitude: number; polarity?: string }) {
  // Compute phase-shifted tag (add π, wrap to [0, 2π))
  const shadowPhase = (bright.phaseTag + Math.PI) % (2 * Math.PI);
  // Use a 10% amplitude inversion for the shadow
  const shadowAmp = -0.1 * bright.amplitude;
  return {
    phaseTag: shadowPhase,
    amplitude: shadowAmp,
    polarity: 'dark'
  };
}
