export type LoopRecord = {
  id: string;
  timestamp: number;
  data: any;
};

// Enhanced ConceptDiff state representation with symbolic cognitive fields
import { writable } from 'svelte/store';

export interface ConceptDiffState {
  // Existing concept mesh state
  activeConcepts: string[];
  predictionError: number;
  lastUpdate: Date;
  
  // New symbolic cognitive fields
  phasePhi: number;              // Φ(t) - current phase intention [0, 2π]
  contradictionPi: number;       // Π(t) - contradiction index
  coherenceC: number;            // C(t) - symbolic coherence
  volatilitySigma: number;       // σ_s - scar volatility over time window
  gateActive: boolean;           // Current phase gate status
  
  // Loop tracking
  activeLoopId: string | null;   // Current loop being processed
  loopDepth: number;            // Nested loop depth
  scarCount: number;            // Number of unresolved scars
}

// Thresholds for cognitive control
export const CognitiveThresholds = {
  // Contradiction thresholds
  PI_LOW: 0.2,               // Low contradiction - safe operation
  PI_MEDIUM: 0.5,            // Medium contradiction - caution
  PI_HIGH: 0.8,              // High contradiction - intervention needed
  PI_CRITICAL: 1.0,          // Critical contradiction - emergency stop
  
  // Volatility thresholds  
  SIGMA_STABLE: 0.1,         // Stable volatility
  SIGMA_ALERT: 0.3,          // Scar alert threshold
  SIGMA_CRITICAL: 0.6,       // Critical instability
  
  // Coherence thresholds
  COHERENCE_MIN: 0.4,        // Minimum coherence for closure
  COHERENCE_GOOD: 0.7,       // Good coherence threshold
  COHERENCE_EXCELLENT: 0.9,  // Excellent coherence
  
  // Phase gate defaults
  GATE_WIDTH_DEFAULT: 0.5,   // Default phase gate width (radians)
  GATE_ALIGNMENT_TIMEOUT: 10, // Max ticks to wait for phase alignment
  
  // Memory metrics
  DENSITY_MIN: 0.3,          // Minimum loop density ρ_M
  CURVATURE_MAX: 0.8         // Maximum memory curvature κ_I
} as const;

// Initialize cognitive state store
const initialCognitiveState: ConceptDiffState = {
  activeConcepts: [],
  predictionError: 0,
  lastUpdate: new Date(),
  phasePhi: 0,
  contradictionPi: 0,
  coherenceC: 0.8,
  volatilitySigma: 0,
  gateActive: false,
  activeLoopId: null,
  loopDepth: 0,
  scarCount: 0
};

export const cognitiveState = writable<ConceptDiffState>(initialCognitiveState);

// Helper functions for cognitive state management
export function updateCognitiveState(updates: Partial<ConceptDiffState>) {
  cognitiveState.update(state => ({
    ...state,
    ...updates,
    lastUpdate: new Date()
  }));
}

export function incrementLoopDepth() {
  cognitiveState.update(state => ({
    ...state,
    loopDepth: state.loopDepth + 1
  }));
}

export function decrementLoopDepth() {
  cognitiveState.update(state => ({
    ...state,
    loopDepth: Math.max(0, state.loopDepth - 1)
  }));
}

export function addScar() {
  cognitiveState.update(state => ({
    ...state,
    scarCount: state.scarCount + 1
  }));
}

export function healScar() {
  cognitiveState.update(state => ({
    ...state,
    scarCount: Math.max(0, state.scarCount - 1)
  }));
}

console.log('🧠 Cognitive state system initialized');