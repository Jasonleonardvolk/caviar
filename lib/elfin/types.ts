// Enhanced ELFIN++ Types with Existing System Integration
export interface ElfinCommand {
  type: 'ghost' | 'vault' | 'project' | 'conditional' | 'koopman' | 'stability' | 'memory';
  raw: string;
  params: Record<string, any>;
  timestamp: Date;
  conceptIds?: string[];
}

export interface ElfinContext {
  variables: Record<string, any>;
  lastResult?: any;
  ghostState?: GhostState;
  conceptGraph?: ConceptNode[];
  phaseMetrics?: PhaseMetrics;
  session: {
    id: string;
    startTime: Date;
    duration: number;
    messageCount: number;
  };
}

// Ghost-related types
export interface GhostState {
  activePersona: string | null;
  coherence: number;
  entropy: number;
  auraIntensity: number;
  focusConceptId?: string;
  lastEmergence?: Date;
  triggerHistory: GhostTrigger[];
}

export interface GhostTrigger {
  persona: string;
  probability: number;
  reason: string;
  phaseSignature?: string;
  wavelength: number;
  intensity: number;
  timestamp: Date;
}

// Concept-related types
export interface ConceptNode {
  id: string;
  name: string;
  psiCoord: [number, number]; // [theta, lambda] - phase coordinates
  justUpdated: boolean;
  entangled: boolean;
  stability: number;
  lastAccessed: Date;
  wavelength: number;
  conceptType: 'memory' | 'idea' | 'emotion' | 'system' | 'pattern';
  connections: string[];
  lyapunovValue?: number;
}

// Phase and stability metrics
export interface PhaseMetrics {
  coherence: number;
  entropy: number;
  drift: number;
  lyapunovStability: number;
  signature: 'resonance' | 'chaos' | 'drift' | 'stable' | 'neutral';
  wavelength: number;
  lastUpdate: Date;
}

// Enhanced command types
export interface GhostCommand {
  persona: string;
  action: 'emerge' | 'focus' | 'search' | 'project' | 'morph' | 'dismiss';
  input: string;
  options?: {
    intensity?: number;
    duration?: number;
    conceptIds?: string[];
    trigger?: GhostTrigger;
  };
}

export interface VaultCommand {
  operation: 'save' | 'load' | 'seal' | 'unseal' | 'link' | 'search';
  key?: string;
  variable?: string;
  value?: any;
  arcId?: string;
  reason?: string;
  metadata?: {
    emotionalWeight?: number;
    conceptIds?: string[];
    timestamp?: Date;
    userId?: string;
  };
}

export interface ProjectionCommand {
  concept: string;
  target?: 'Thoughtspace' | 'VaultBrowser' | 'ConceptMap';
  position?: string | [number, number]; // Named position or coordinates
  duration?: number;
  intensity?: number;
  wavelength?: number;
}

export interface ConditionalCommand {
  condition: string;
  operator: '<' | '>' | '<=' | '>=' | '==' | '!=';
  threshold: number;
  action: string;
  variables?: string[];
}

// Koopman operator types (from existing system)
export interface KoopmanCommand {
  operation: 'eigenfunction' | 'mode' | 'phase_bind' | 'transform';
  eigenfunctions?: string[];
  modes?: Record<string, number>;
  phaseBinding?: Record<string, any>;
  target?: string;
}

// Stability analysis types
export interface StabilityCommand {
  operation: 'lyapunov' | 'verify' | 'monitor' | 'adapt';
  target: string;
  method?: 'sos' | 'sampling' | 'milp' | 'smt';
  options?: Record<string, any>;
  threshold?: number;
}

// Memory gating types (from existing memory systems)
export interface MemoryCommand {
  operation: 'gate' | 'sculpt' | 'introspect' | 'cluster';
  target: string;
  parameters?: {
    gatingThreshold?: number;
    sculptingMode?: 'compress' | 'expand' | 'refactor';
    clusterMethod?: 'spectral' | 'kmeans' | 'hierarchical';
  };
}

// ELFIN++ execution result
export interface ElfinResult {
  success: boolean;
  command: ElfinCommand;
  result?: any;
  error?: string;
  context: ElfinContext;
  executionTime: number;
  sideEffects?: {
    ghostStateChange?: Partial<GhostState>;
    conceptUpdates?: ConceptNode[];
    vaultOperations?: string[];
    thoughtspaceProjections?: ProjectionCommand[];
  };
}

// Enhanced system state
export interface SystemState {
  phase: PhaseMetrics;
  ghost: GhostState;
  concepts: ConceptNode[];
  vault: {
    entries: number;
    sealed: string[];
    lastOperation: Date;
  };
  session: {
    id: string;
    duration: number;
    messageCount: number;
    errorStreak: number;
    frustrationLevel: number;
  };
}

// Event types for system integration
export type ToriEvent = 
  | { type: 'ghost:emerge'; persona: string; trigger: GhostTrigger }
  | { type: 'ghost:dismiss'; persona: string }
  | { type: 'vault:sealed'; arcId: string; reason: string }
  | { type: 'concept:updated'; concept: ConceptNode }
  | { type: 'phase:changed'; metrics: PhaseMetrics }
  | { type: 'thoughtspace:projected'; projection: ProjectionCommand }
  | { type: 'stability:alert'; target: string; value: number }
  | { type: 'elfin:executed'; command: ElfinCommand; result: ElfinResult };

// Variable types for ELFIN++ context
export type ElfinVariable = 
  | string 
  | number 
  | boolean 
  | ConceptNode 
  | ConceptNode[]
  | GhostTrigger
  | PhaseMetrics
  | any[];

// Error types
export class ElfinExecutionError extends Error {
  constructor(
    message: string,
    public command: ElfinCommand,
    public context: ElfinContext,
    public originalError?: Error
  ) {
    super(message);
    this.name = 'ElfinExecutionError';
  }
}

export class ElfinParseError extends Error {
  constructor(
    message: string,
    public line: string,
    public position?: number
  ) {
    super(message);
    this.name = 'ElfinParseError';
  }
}

// Export all types
export type {
  ElfinCommand,
  ElfinContext,
  ElfinResult,
  SystemState,
  ToriEvent,
  ElfinVariable
};