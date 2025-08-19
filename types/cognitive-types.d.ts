/**
 * Cognitive System Type Definitions
 * Central type definitions for TORI's cognitive layer
 */

import type { Writable } from 'svelte/store';

// ============================================
// Core Cognitive Types
// ============================================

export interface ConceptDiffState {
  // Concept mesh state
  activeConcepts: string[];
  predictionError: number;
  lastUpdate: Date;
  
  // Symbolic cognitive fields
  phasePhi: number;              // Φ(t) - current phase intention [0, 2π]
  contradictionPi: number;       // Π(t) - contradiction index
  coherenceC: number;            // C(t) - symbolic coherence
  volatilitySigma: number;       // σ_s - scar volatility
  gateActive: boolean;           // Current phase gate status
  
  // Loop tracking
  activeLoopId: string | null;
  loopDepth: number;
  scarCount: number;
  
  // Additional fields for complete state
  content?: string;
  digest?: string;
  concepts?: Array<{ concept: string; type: string; score: number }>;
  timestamp?: number;
  loopId?: string;
}

export interface LoopRecord {
  id: string;
  entries: ConceptDiffState[];
  createdAt: number;
  closedAt?: number;
  weight: number;
  scarred: boolean;
  compressed?: boolean;
  crossings?: LoopCrossing[];
}

export interface LoopCrossing {
  id: string;
  sourceLoopId: string;
  targetLoopId: string;
  type: 'reentry' | 'echo' | 'resonance' | 'paradox';
  glyph: string;
  timestamp: number;
}

export interface BraidMemoryStats {
  totalLoops: number;
  activeLoops: number;
  closedLoops: number;
  scarredLoops: number;
  crossings: number;
  memoryEchoes: number;
  compressionRatio: number;
}

// ============================================
// Memory Metrics Types
// ============================================

export interface MemoryMetrics {
  rhoM: number;                  // ρ_M - Loop density
  kappaI: number;                // κ_I - Information curvature
  thetaDelta: number;            // θ_Δ - Phase transition rate
  scarRatio: number;
  memoryPressure: number;
  godelianCollapseRisk: boolean;
}

// ============================================
// Paradox Analysis Types
// ============================================

export interface ParadoxEvent {
  id: string;
  timestamp: number;
  associatorResult: AssociatorResult;
  cognitiveState: ConceptDiffState;
  resolved: boolean;
}

export interface AssociatorResult {
  type: 'left' | 'right' | 'jacobi' | 'moufang' | 'valid';
  expression: string;
  operations: Operation[];
}

export interface Operation {
  left: string;
  right: string;
  result: string;
}

// ============================================
// Closure Types
// ============================================

export interface ClosureResult {
  canClose: boolean;
  reason?: string;
  suggestedAction?: string;
  metrics?: {
    coherence: number;
    contradiction: number;
    volatility: number;
  };
}

export interface FeedbackOptions {
  strengthenConcepts?: string[];
  weakenConcepts?: string[];
  adjustPhase?: number;
  healScars?: boolean;
}

// ============================================
// Engine Configuration Types
// ============================================

export interface CognitiveEngineConfig {
  enableBraidMemory?: boolean;
  enableParadoxAnalysis?: boolean;
  enableMemoryMetrics?: boolean;
  maxLoopDepth?: number;
  autoHealScars?: boolean;
  compressionEnabled?: boolean;
  phase?: number;
}

// ============================================
// Compression Types
// ============================================

export interface CompressionConfig {
  maxLoopSize: number;
  compressionRatio: number;
  preserveKeyframes: boolean;
  keepCrossings: boolean;
}

// ============================================
// Store Types
// ============================================

export interface CognitiveStores {
  cognitiveState: Writable<ConceptDiffState>;
  braidMemory: Writable<BraidMemory>;
  memoryMetrics: Writable<MemoryMetrics>;
  paradoxEvents: Writable<ParadoxEvent[]>;
}

export interface BraidMemory {
  loops: Map<string, LoopRecord>;
  digestMap: Map<string, string[]>;
  activeLoop: LoopRecord | null;
  stats: BraidMemoryStats;
}

// ============================================
// Constants
// ============================================

export const NoveltyGlyphs = {
  STAR: '⟡',
  CROSS: '☩',
  SPIRAL: '⟆',
  DIAMOND: '♢',
  CIRCLE: '⊙'
} as const;

export const CognitiveThresholds = {
  PI_LOW: 0.2,
  PI_MEDIUM: 0.5,
  PI_HIGH: 0.8,
  PI_CRITICAL: 1.0,
  SIGMA_STABLE: 0.1,
  SIGMA_ALERT: 0.3,
  SIGMA_CRITICAL: 0.6,
  COHERENCE_MIN: 0.4,
  COHERENCE_GOOD: 0.7,
  COHERENCE_EXCELLENT: 0.9,
  GATE_WIDTH_DEFAULT: 0.5,
  GATE_ALIGNMENT_TIMEOUT: 10,
  DENSITY_MIN: 0.3,
  CURVATURE_MAX: 0.8
} as const;

// ============================================
// Type Guards
// ============================================

export function isLoopRecord(obj: any): obj is LoopRecord {
  return obj && typeof obj.id === 'string' && Array.isArray(obj.entries);
}

export function isParadoxEvent(obj: any): obj is ParadoxEvent {
  return obj && typeof obj.id === 'string' && obj.associatorResult;
}

export function isConceptDiffState(obj: any): obj is ConceptDiffState {
  return obj && Array.isArray(obj.activeConcepts) && typeof obj.phasePhi === 'number';
}
