// Types for TORI concept mesh and system state management

export interface ConceptDiff {
  id: string;
  title: string;
  summary?: string;
  concepts: string[];
  type: 'document' | 'chat' | 'manual' | 'system';
  timestamp: Date;
  metadata?: {
    processedBy?: string;
    fallbackMode?: boolean;
    confidence?: number;
    source?: string;
    [key: string]: any;
  };
}

export interface SystemState {
  status: 'idle' | 'processing' | 'analyzing' | 'error';
  coherenceLevel: number;
  lastUpdate?: Date;
  activeProcesses?: string[];
  errorMessages?: string[];
}

export interface ConceptConnection {
  target: string;
  strength: number;
  type: 'semantic' | 'temporal' | 'causal' | 'contextual';
  established: Date;
}

export interface ConceptNode {
  id: string;
  name: string;
  strength: number;
  type: 'general' | 'technical' | 'domain' | 'keyword' | 'entity' | 'restored';
  position: {
    x: number;
    y: number;
    z: number;
  };
  highlighted: boolean;
  connections: ConceptConnection[];
  lastActive: Date;
  contradictionLevel?: number;
  coherenceContribution?: number;
  loopReferences?: string[];
  // Enhanced properties for ELFIN integration
  psiCoord?: [number, number]; // [theta, lambda] - phase coordinates
  justUpdated?: boolean;
  entangled?: boolean;
  stability?: number;
  wavelength?: number;
  conceptType?: 'memory' | 'idea' | 'emotion' | 'system' | 'pattern';
  lyapunovValue?: number;
}

// Enhanced concept for rich analysis
export interface EnhancedConcept {
  eigenfunction_id: string;
  name: string;
  confidence: number;
  context: string;
  cluster_id?: number | string;
  title?: string;
  timestamp?: string;
  strength?: number;
  type?: string;
  metadata?: Record<string, any>;
}

// System coherence and entropy tracking
export interface CoherenceMetrics {
  level: number;
  trend: 'increasing' | 'decreasing' | 'stable';
  lastUpdate: Date;
  factors: {
    conceptAlignment: number;
    connectionStrength: number;
    activityLevel: number;
    contradictionLevel: number;
  };
}

export interface EntropyMetrics {
  level: number;
  sources: {
    newConcepts: number;
    contradictions: number;
    systemLoad: number;
    userActivity: number;
  };
  lastCalculated: Date;
}

// Event types for system communication
export type ConceptMeshEvent = 
  | { type: 'concept:added'; concept: string; metadata?: any }
  | { type: 'concept:connected'; from: string; to: string; strength: number }
  | { type: 'concept:highlighted'; concepts: string[] }
  | { type: 'concept:activated'; concept: string }
  | { type: 'system:coherence_change'; level: number; delta: number }
  | { type: 'system:entropy_change'; level: number; delta: number }
  | { type: 'ghost:triggered'; persona: string; concepts: string[] }
  | { type: 'mesh:cleared' }
  | { type: 'mesh:loaded'; diffCount: number; conceptCount: number };

// Storage and persistence types
export interface ConceptMeshStorage {
  version: string;
  timestamp: string;
  diffs: ConceptDiff[];
  nodeCount: number;
  metadata?: {
    sessionId?: string;
    userId?: string;
    systemVersion?: string;
  };
}

// Network analysis types
export interface NetworkStats {
  nodeCount: number;
  connectionCount: number;
  density: number;
  avgConnectionsPerNode: number;
  clusters: number;
  isolatedNodes: number;
  stronglyConnectedComponents: number;
}

// Ghost interaction types
export interface GhostInteraction {
  persona: string;
  action: 'trigger' | 'respond' | 'focus' | 'dismiss';
  concepts: string[];
  timestamp: Date;
  metadata?: {
    confidence?: number;
    reasoning?: string;
    followUp?: boolean;
  };
}

// Thoughtspace visualization types
export interface ThoughtspaceState {
  visible: boolean;
  focus: {
    concept: string | null;
    position: { x: number; y: number; z: number } | null;
    zoom: number;
  };
  animation: {
    active: boolean;
    type: 'orbit' | 'focus' | 'network' | 'idle';
    duration: number;
  };
  filters: {
    minStrength: number;
    maxAge: number; // in milliseconds
    types: string[];
    hideIsolated: boolean;
  };
}

// Error types
export class ConceptMeshError extends Error {
  constructor(
    message: string,
    public code: 'STORAGE_ERROR' | 'CONCEPT_ERROR' | 'CONNECTION_ERROR' | 'SYSTEM_ERROR',
    public context?: any
  ) {
    super(message);
    this.name = 'ConceptMeshError';
  }
}

// Type guards
export function isConceptDiff(obj: any): obj is ConceptDiff {
  return (
    obj &&
    typeof obj.id === 'string' &&
    typeof obj.title === 'string' &&
    Array.isArray(obj.concepts) &&
    obj.timestamp instanceof Date
  );
}

export function isConceptNode(obj: any): obj is ConceptNode {
  return (
    obj &&
    typeof obj.id === 'string' &&
    typeof obj.name === 'string' &&
    typeof obj.strength === 'number' &&
    obj.position &&
    typeof obj.position.x === 'number' &&
    typeof obj.position.y === 'number' &&
    typeof obj.position.z === 'number'
  );
}

// Utility types
export type ConceptMeshSubscriber = (diffs: ConceptDiff[]) => void;
export type SystemStateSubscriber = (state: SystemState) => void;
export type ConceptNodesSubscriber = (nodes: Record<string, ConceptNode>) => void;

// Configuration types
export interface ConceptMeshConfig {
  storage: {
    enabled: boolean;
    key: string;
    maxEntries: number;
    compressionEnabled: boolean;
  };
  visualization: {
    autoDecay: boolean;
    decayInterval: number;
    highlightDuration: number;
    connectionThreshold: number;
  };
  analysis: {
    coherenceUpdateInterval: number;
    entropyCalculationMethod: 'simple' | 'shannon' | 'weighted';
    connectionStrengthThreshold: number;
  };
  integration: {
    ghostTriggerEnabled: boolean;
    thoughtspaceUpdateEnabled: boolean;
    eventDispatchEnabled: boolean;
  };
}

export default {
  ConceptDiff,
  SystemState,
  ConceptNode,
  EnhancedConcept,
  CoherenceMetrics,
  EntropyMetrics,
  ConceptMeshEvent,
  ConceptMeshStorage,
  NetworkStats,
  GhostInteraction,
  ThoughtspaceState,
  ConceptMeshError,
  ConceptMeshConfig
};
