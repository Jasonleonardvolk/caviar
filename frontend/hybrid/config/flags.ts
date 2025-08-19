// ${IRIS_ROOT}\frontend\hybrid\config\flags.ts
/**
 * Feature flags for phase correction derivatives
 * Enable one at a time to avoid stacking corrections
 */

// Phase correction options (choose one)
export const ENABLE_PHASE_POLISHER = true;      // TV-based smoothing (recommended to start)
export const ENABLE_ZERNIKE_MICRO = false;      // Zernike polynomial correction
export const ENABLE_STATIC_DPHI_LUT = false;    // Pre-baked phase offset LUT

// Phase correction parameters
export const PHASE_CORRECTION_CONFIG = {
  // TV Polisher settings
  polisher: {
    tvLambda: 0.08,        // Smoothing strength (0.04-0.10)
    maxCorrection: 0.25,   // Max phase change in radians
    useMask: false,        // Use edge mask if available
  },
  
  // Zernike settings
  zernike: {
    maxCorrection: 0.25,   // Max phase change in radians
    softness: 0.05,        // Edge attenuation (0-0.5)
    outsideBehavior: 1,    // 0=zero, 1=attenuate, 2=hold
  },
  
  // LUT settings
  lut: {
    gain: 1.0,             // Scale factor for LUT values
    maxCorrection: 0.30,   // Max phase change in radians
  },
  
  // Budget splitting when stacking multiple corrections
  stackingBudget: {
    maxTotalCorrection: 0.30,  // Total budget across all stages
    perStageLimit: 0.15,       // Max per stage when stacking
  }
};

// Performance monitoring
export const ENABLE_PHASE_METRICS = true;  // Log phase correction metrics
export const PHASE_METRICS_INTERVAL = 60;  // Frames between metric logs
