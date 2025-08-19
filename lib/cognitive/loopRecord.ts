// Enhanced LoopRecord structure for Phase 2 braid memory

export interface LoopRecord {
  // Core identification
  id: string;
  prompt: string;
  
  // Execution trace
  glyphPath: string[];                    // Full glyph sequence
  phaseTrace: number[];                   // Φ(t) trajectory over time
  coherenceTrace: number[];               // C(t) trajectory over time
  contradictionTrace: number[];           // Π(t) trajectory over time
  
  // Closure state
  returnGlyph?: string;                   // R_i: final closure glyph
  closed: boolean;                        // successful closure flag
  scarFlag: boolean;                      // unresolved contradiction flag
  timestamp: Date;                        // creation time
  processingTime: number;                 // milliseconds to complete
  
  // Phase 2: Compression & Memory
  coreGlyphs?: string[];                  // Ĉ: compressed essential glyphs
  cInfinity?: number;                     // C∞: final stable coherence
  digest?: string;                        // SHA-256 fingerprint
  reentryCount?: number;                  // how many times this pattern occurred
  memoryEchoFlag?: boolean;               // flagged as repeating pattern
  
  // Braid topology
  crossingRefs?: string[];                // IDs of loops this crosses with
  braidPosition?: {                       // 3D position in memory braid
    x: number;
    y: number; 
    z: number;
  };
  
  // Integration metadata
  metadata?: {
    createdByPersona?: string;            // Ghost that generated this loop
    scriptId?: string;                    // ELFIN++ script that triggered it
    conceptFootprint?: string[];          // Concepts touched by this loop
    phaseGateHits?: string[];            // Gates that were active during execution
    contradictionPeaks?: number[];       // Timestamps of Π(t) spikes
    coherenceGains?: Array<{             // Major coherence improvement points
      glyphIndex: number;
      deltaC: number;
    }>;
    [key: string]: any;
  };
}

export interface LoopCrossing {
  id: string;
  glyph: string;                          // Shared glyph that creates crossing
  loops: [string, string];               // [loopIdA, loopIdB]
  type: 'paradox' | 'harmony';           // Crossing classification
  timestamp: Date;                        // When crossing was detected
  weight: number;                         // (coherence gain + contradiction drop) / 2
  order: number;                          // Temporal order of crossing detection
  position?: {                            // 3D position of crossing in space
    x: number;
    y: number;
    z: number;
  };
  metadata?: {
    contradictionBefore?: number;         // Π(t) before crossing
    contradictionAfter?: number;          // Π(t) after crossing
    coherenceGain?: number;               // C(t) improvement from crossing
    phaseAlignment?: number;              // Φ(t) at crossing moment
  };
}

export interface BraidMemoryStats {
  totalLoops: number;
  closedLoops: number;
  scarredLoops: number;
  memoryEchoes: number;
  crossings: number;
  paradoxCrossings: number;
  harmonyCrossings: number;
  compressionRatio: number;               // compressed size / original size
  topDigests: Array<{                     // Most frequent patterns
    digest: string;
    count: number;
    lastSeen: Date;
  }>;
}

// Compression configuration
export const CompressionConfig = {
  // Multi-signal compression thresholds
  COHERENCE_DELTA_THRESHOLD: 0.1,        // ε₁: minimum dC for core glyph
  CONTRADICTION_DROP_THRESHOLD: 0.15,    // ε₂: minimum ΔΠ drop for core glyph
  PHASE_GATE_BONUS: 0.3,                 // bonus weight for phase-gated glyphs
  GLYPH_TYPE_BONUSES: {                   // bonus weights by glyph type
    'return': 0.5,
    'anchor': 0.4,
    'scar-repair': 0.6,
    'intent-bifurcation': 0.3,
    'meta-echo:reflect': 0.4
  },
  
  // Compression weights (α, β coefficients)
  COHERENCE_WEIGHT: 1.0,                 // α: weight for dC signal
  CONTRADICTION_WEIGHT: 1.2,             // β: weight for ΔΠ signal  
  MINIMUM_CORE_WEIGHT: 0.2,              // threshold for core glyph inclusion
  
  // Reentry thresholds
  REENTRY_MONITOR_THRESHOLD: 2,          // start monitoring at count 2
  NOVELTY_INJECTION_THRESHOLD: 3,       // inject novelty at count 3
  PERSONA_SWAP_THRESHOLD: 4,             // swap ghost at count 4
  ECHO_COLLAPSE_THRESHOLD: 5             // collapse to digest summary at count 5+
} as const;

// Novelty glyph types for reentry optimization
export const NoveltyGlyphs = [
  'interruptor',                          // breaks repetitive patterns
  'scar-sealer',                          // attempts to heal contradictions
  'intent-bifurcation',                   // explores alternative paths
  'meta-echo:reflect',                    // self-reflective analysis
  'phase-drift',                          // intentional phase desynchronization
  'paradox-embrace',                      // lean into contradiction
  'coherence-boost',                      // explicit coherence enhancement
  'memory-anchor'                         // create strong memory reference
] as const;

export type NoveltyGlyph = typeof NoveltyGlyphs[number];

// Helper functions for loop analysis
export function calculateLoopWeight(
  coherenceDelta: number,
  contradictionDelta: number, 
  phaseGateAlignment: number,
  glyphType: string
): number {
  const coherenceComponent = coherenceDelta * CompressionConfig.COHERENCE_WEIGHT;
  const contradictionComponent = -contradictionDelta * CompressionConfig.CONTRADICTION_WEIGHT;
  const phaseComponent = phaseGateAlignment * CompressionConfig.PHASE_GATE_BONUS;
  const typeBonus = CompressionConfig.GLYPH_TYPE_BONUSES[glyphType] || 0;
  
  return coherenceComponent + contradictionComponent + phaseComponent + typeBonus;
}

export function classifyCrossingType(
  contradictionBefore: number,
  contradictionAfter: number,
  coherenceGain: number
): 'paradox' | 'harmony' {
  const contradictionIncrease = contradictionAfter - contradictionBefore;
  
  // If contradiction increased significantly or coherence dropped, it's a paradox crossing
  if (contradictionIncrease > 0.2 || coherenceGain < -0.1) {
    return 'paradox';
  }
  
  // If contradiction decreased and coherence improved, it's harmony
  if (contradictionIncrease < -0.1 && coherenceGain > 0.1) {
    return 'harmony';
  }
  
  // Default to harmony for neutral crossings
  return 'harmony';
}

export function selectNoveltyGlyph(
  reentryCount: number,
  contradictionLevel: number,
  coherenceLevel: number,
  scarCount: number
): NoveltyGlyph {
  // Select novelty glyph based on system state
  if (contradictionLevel > 0.7) {
    return scarCount > 2 ? 'scar-sealer' : 'paradox-embrace';
  }
  
  if (coherenceLevel < 0.3) {
    return 'coherence-boost';
  }
  
  if (reentryCount >= CompressionConfig.ECHO_COLLAPSE_THRESHOLD) {
    return 'meta-echo:reflect';
  }
  
  if (reentryCount >= CompressionConfig.PERSONA_SWAP_THRESHOLD) {
    return 'intent-bifurcation';
  }
  
  // Default interruptor for general repetition
  return 'interruptor';
}

console.log('🧠 Enhanced LoopRecord structure loaded for Phase 2 braid memory');