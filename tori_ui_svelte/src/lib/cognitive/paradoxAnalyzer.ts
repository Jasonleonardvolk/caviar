// paradoxAnalyzer.ts - Phase 3 Associator Bracket Logic & Paradox Detection
import type { LoopRecord } from './loopRecord';
import { get } from 'svelte/store';
import { cognitiveState } from './cognitiveState';
import { conceptNodes } from '$lib/stores/conceptMesh';

export type Operation = (state: CognitiveState) => CognitiveState;

export interface CognitiveState {
  coherence: number;
  contradiction: number;
  conceptActivations: Map<string, number>;
  topology: {
    nodes: Set<string>;
    edges: Map<string, Set<string>>;
  };
}

export interface AssociatorResult {
  value: number;              // |[X,Y,Z]| - magnitude of non-associativity
  type: 'zero' | 'linear' | 'cyclic' | 'chaotic';
  paradoxClass?: 'godel' | 'condorcet' | 'arrow' | 'unknown';
  suggestedResolution?: string;
}

export interface ParadoxEvent {
  id: string;
  operations: [string, string, string];
  associatorResult: AssociatorResult;
  timestamp: Date;
  resolved: boolean;
  resolutionLoop?: string;
}

export class ParadoxAnalyzer {
  private paradoxRegistry = new Map<string, ParadoxEvent>();
  private operationCache = new Map<string, Operation>();
  private stateDistanceWeights = {
    alpha: 0.4,    // coherence weight
    beta: 0.3,     // contradiction weight
    gamma: 0.2,    // concept activation weight
    delta: 0.1     // topology weight
  };
  private callbacks: Array<(event: ParadoxEvent) => void> = [];

  constructor() {
    console.log('🔺 ParadoxAnalyzer initialized for Phase 3 associator bracket logic');
  }

  /**
   * Measure associator bracket [X,Y,Z] for three operations
   */
  measureAssociator(
    opX: string | Operation,
    opY: string | Operation,
    opZ: string | Operation,
    initialState?: CognitiveState
  ): AssociatorResult {
    // Get or create operations
    const X = this.getOperation(opX);
    const Y = this.getOperation(opY);
    const Z = this.getOperation(opZ);
    
    // Get initial state or current cognitive state
    const state0 = initialState || this.getCurrentCognitiveState();
    
    // Calculate (X·Y)·Z
    const stateXY = Y(X(this.cloneState(state0)));
    const stateXYZ = Z(stateXY);
    
    // Calculate X·(Y·Z)
    const stateYZ = Z(Y(this.cloneState(state0)));
    const stateXYZ_alt = X(stateYZ);
    
    // Measure distance between results
    const distance = this.stateDistance(stateXYZ, stateXYZ_alt);
    
    // Classify associator type
    const result: AssociatorResult = {
      value: distance,
      type: this.classifyAssociator(distance),
      paradoxClass: this.classifyParadox(distance, stateXYZ, stateXYZ_alt),
      suggestedResolution: this.suggestResolution(distance, opX, opY, opZ)
    };
    
    // Log significant paradoxes
    if (result.type !== 'zero' && result.type !== 'linear') {
      console.log(`🔺 Paradox detected: [X,Y,Z] = ${distance.toFixed(3)} (${result.type})`);
      this.registerParadox(opX, opY, opZ, result);
    }
    
    return result;
  }

  /**
   * State distance calculation using weighted composite
   */
  private stateDistance(a: CognitiveState, b: CognitiveState): number {
    const { alpha, beta, gamma, delta } = this.stateDistanceWeights;
    
    // Coherence delta
    const coherenceDelta = Math.abs(a.coherence - b.coherence);
    
    // Contradiction delta
    const contradictionDelta = Math.abs(a.contradiction - b.contradiction);
    
    // Concept activation difference (vector distance)
    let activationDiff = 0;
    const allConcepts = new Set([...a.conceptActivations.keys(), ...b.conceptActivations.keys()]);
    allConcepts.forEach(concept => {
      const aVal = a.conceptActivations.get(concept) || 0;
      const bVal = b.conceptActivations.get(concept) || 0;
      activationDiff += Math.pow(aVal - bVal, 2);
    });
    activationDiff = Math.sqrt(activationDiff);
    
    // Topology difference (Jaccard distance)
    const aEdgeCount = Array.from(a.topology.edges.values()).reduce((sum, edges) => sum + edges.size, 0);
    const bEdgeCount = Array.from(b.topology.edges.values()).reduce((sum, edges) => sum + edges.size, 0);
    const topologyDelta = Math.abs(aEdgeCount - bEdgeCount) / Math.max(aEdgeCount, bEdgeCount, 1);
    
    // Weighted composite
    return alpha * coherenceDelta + 
           beta * contradictionDelta + 
           gamma * activationDiff + 
           delta * topologyDelta;
  }

  /**
   * Classify associator type based on magnitude
   */
  private classifyAssociator(distance: number): AssociatorResult['type'] {
    if (distance < 0.01) return 'zero';        // Perfect associativity
    if (distance < 0.1) return 'linear';       // Near-associative
    if (distance < 0.5) return 'cyclic';       // Cyclic paradox
    return 'chaotic';                          // Chaotic non-associativity
  }

  /**
   * Classify paradox type based on state analysis
   */
  private classifyParadox(
    distance: number, 
    stateA: CognitiveState, 
    stateB: CognitiveState
  ): AssociatorResult['paradoxClass'] {
    // Gödel-type: high contradiction with coherence collapse
    if (stateA.contradiction > 0.8 && stateB.coherence < 0.2) {
      return 'godel';
    }
    
    // Condorcet cycle: preference reversal pattern
    const activationReversal = this.detectActivationReversal(stateA, stateB);
    if (activationReversal && distance > 0.3) {
      return 'condorcet';
    }
    
    // Arrow impossibility: no consistent ordering
    if (distance > 0.5 && this.detectInconsistentOrdering(stateA, stateB)) {
      return 'arrow';
    }
    
    return 'unknown';
  }

  /**
   * Detect activation reversal (Condorcet-like)
   */
  private detectActivationReversal(a: CognitiveState, b: CognitiveState): boolean {
    const concepts = Array.from(a.conceptActivations.keys());
    if (concepts.length < 3) return false;
    
    // Check for cyclic preferences: A > B > C > A
    for (let i = 0; i < Math.min(3, concepts.length - 2); i++) {
      const c1 = concepts[i];
      const c2 = concepts[i + 1];
      const c3 = concepts[i + 2];
      
      const aOrder = (a.conceptActivations.get(c1)! > a.conceptActivations.get(c2)!) &&
                     (a.conceptActivations.get(c2)! > a.conceptActivations.get(c3)!);
      const bOrder = (b.conceptActivations.get(c3)! > b.conceptActivations.get(c1)!);
      
      if (aOrder && bOrder) return true;
    }
    
    return false;
  }

  /**
   * Detect inconsistent ordering (Arrow-like)
   */
  private detectInconsistentOrdering(a: CognitiveState, b: CognitiveState): boolean {
    // Check if topology orderings are incompatible
    const aNodeCount = a.topology.nodes.size;
    const bNodeCount = b.topology.nodes.size;
    
    return Math.abs(aNodeCount - bNodeCount) > aNodeCount * 0.3;
  }

  /**
   * Suggest resolution strategy
   */
  private suggestResolution(
    distance: number,
    opX: string | Operation,
    opY: string | Operation,
    opZ: string | Operation
  ): string {
    const type = this.classifyAssociator(distance);
    
    switch (type) {
      case 'zero':
      case 'linear':
        return 'No resolution needed';
        
      case 'cyclic':
        return 'Insert phase gate between operations or reorder sequence';
        
      case 'chaotic':
        return 'Spawn reflective loop with anchor: inverse(' + this.getOpName(opX) + ')';
        
      default:
        return 'Unknown paradox type';
    }
  }

  /**
   * Register paradox event
   */
  private registerParadox(
    opX: string | Operation,
    opY: string | Operation,
    opZ: string | Operation,
    result: AssociatorResult
  ): void {
    const event: ParadoxEvent = {
      id: `PDX_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`,
      operations: [this.getOpName(opX), this.getOpName(opY), this.getOpName(opZ)],
      associatorResult: result,
      timestamp: new Date(),
      resolved: false
    };
    
    this.paradoxRegistry.set(event.id, event);
    
    // Emit event
    this.callbacks.forEach(callback => {
      try {
        callback(event);
      } catch (error: any) {
        console.error('Error in paradox event callback:', error);
      }
    });
    
    // Global event
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('tori:paradox-detected', {
        detail: event
      }));
    }
  }

  /**
   * Get or create operation from string/function
   */
  private getOperation(op: string | Operation): Operation {
    if (typeof op === 'function') return op;
    
    // Check cache
    if (this.operationCache.has(op)) {
      return this.operationCache.get(op)!;
    }
    
    // Create operation based on string
    const operation = this.createOperationFromString(op);
    this.operationCache.set(op, operation);
    return operation;
  }

  /**
   * Create operation from string identifier
   */
  private createOperationFromString(opName: string): Operation {
    // Common cognitive operations
    switch (opName) {
      case 'focus':
        return (state) => ({
          ...state,
          coherence: Math.min(1, state.coherence * 1.2),
          contradiction: state.contradiction * 0.9
        });
        
      case 'explore':
        return (state) => ({
          ...state,
          coherence: state.coherence * 0.95,
          contradiction: Math.min(1, state.contradiction * 1.1)
        });
        
      case 'compress':
        return (state) => ({
          ...state,
          coherence: Math.min(1, state.coherence * 1.1),
          conceptActivations: this.compressActivations(state.conceptActivations)
        });
        
      case 'expand':
        return (state) => ({
          ...state,
          contradiction: Math.min(1, state.contradiction * 1.2),
          conceptActivations: this.expandActivations(state.conceptActivations)
        });
        
      default:
        // Identity operation
        return (state) => state;
    }
  }

  /**
   * Compress concept activations
   */
  private compressActivations(activations: Map<string, number>): Map<string, number> {
    const compressed = new Map<string, number>();
    activations.forEach((value, key) => {
      compressed.set(key, value * 0.8); // Simple compression
    });
    return compressed;
  }

  /**
   * Expand concept activations
   */
  private expandActivations(activations: Map<string, number>): Map<string, number> {
    const expanded = new Map<string, number>();
    activations.forEach((value, key) => {
      expanded.set(key, Math.min(1, value * 1.2)); // Simple expansion
    });
    return expanded;
  }

  /**
   * Get current cognitive state
   */
  private getCurrentCognitiveState(): CognitiveState {
    const state = get(cognitiveState);
    const nodes = get(conceptNodes);
    
    // Build concept activations
    const activations = new Map<string, number>();
    Object.entries(nodes).forEach(([name, node]) => {
      activations.set(name, node.strength || 0.5);
    });
    
    // Build topology
    const topology = {
      nodes: new Set(Object.keys(nodes)),
      edges: new Map<string, Set<string>>()
    };
    
    Object.entries(nodes).forEach(([name, node]) => {
      if (node.connections && node.connections.length > 0) {
        topology.edges.set(name, new Set(node.connections));
      }
    });
    
    return {
      coherence: state.coherenceC,
      contradiction: state.contradictionPi,
      conceptActivations: activations,
      topology
    };
  }

  /**
   * Clone cognitive state
   */
  private cloneState(state: CognitiveState): CognitiveState {
    return {
      coherence: state.coherence,
      contradiction: state.contradiction,
      conceptActivations: new Map(state.conceptActivations),
      topology: {
        nodes: new Set(state.topology.nodes),
        edges: new Map(Array.from(state.topology.edges.entries()).map(
          ([k, v]) => [k, new Set(v)]
        ))
      }
    };
  }

  /**
   * Get operation name
   */
  private getOpName(op: string | Operation): string {
    return typeof op === 'string' ? op : 'custom_op';
  }

  /**
   * Check for Condorcet cycles in current state
   */
  checkCondorcetCycle(concepts: string[]): boolean {
    if (concepts.length < 3) return false;
    
    const state = this.getCurrentCognitiveState();
    
    // Check all permutations for cycles
    for (let i = 0; i < concepts.length; i++) {
      const a = concepts[i];
      const b = concepts[(i + 1) % concepts.length];
      const c = concepts[(i + 2) % concepts.length];
      
      const aVal = state.conceptActivations.get(a) || 0;
      const bVal = state.conceptActivations.get(b) || 0;
      const cVal = state.conceptActivations.get(c) || 0;
      
      // Check for cycle: a > b > c > a
      if (aVal > bVal && bVal > cVal && cVal > aVal) {
        console.log(`🔺 Condorcet cycle detected: ${a} > ${b} > ${c} > ${a}`);
        return true;
      }
    }
    
    return false;
  }

  /**
   * Spawn reflective loop for paradox resolution
   */
  async spawnReflectiveLoop(paradoxId: string): Promise<string | null> {
    const paradox = this.paradoxRegistry.get(paradoxId);
    if (!paradox) return null;
    
    console.log(`🔺 Spawning reflective loop for paradox ${paradoxId}`);
    
    // Create reflective prompt
    const prompt = `Paradox resolution: [${paradox.operations.join(', ')}] = ${paradox.associatorResult.value.toFixed(3)}`;
    
    // Emit event for cognitive engine to handle
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('tori:spawn-reflective-loop', {
        detail: { 
          paradoxId,
          prompt,
          type: paradox.associatorResult.type,
          paradoxClass: paradox.associatorResult.paradoxClass
        }
      }));
    }
    
    return paradoxId;
  }

  /**
   * Mark paradox as resolved
   */
  resolveParadox(paradoxId: string, resolutionLoopId: string): void {
    const paradox = this.paradoxRegistry.get(paradoxId);
    if (paradox) {
      paradox.resolved = true;
      paradox.resolutionLoop = resolutionLoopId;
      console.log(`🔺 Paradox ${paradoxId} resolved by loop ${resolutionLoopId}`);
    }
  }

  /**
   * Get unresolved paradoxes
   */
  getUnresolvedParadoxes(): ParadoxEvent[] {
    return Array.from(this.paradoxRegistry.values()).filter(p => !p.resolved);
  }

  /**
   * Register callback for paradox events
   */
  onParadox(callback: (event: ParadoxEvent) => void): void {
    this.callbacks.push(callback);
  }

  /**
   * Get paradox statistics
   */
  getStats(): {
    total: number;
    resolved: number;
    byType: Record<string, number>;
    byClass: Record<string, number>;
  } {
    const paradoxes = Array.from(this.paradoxRegistry.values());
    const stats = {
      total: paradoxes.length,
      resolved: paradoxes.filter(p => p.resolved).length,
      byType: {} as Record<string, number>,
      byClass: {} as Record<string, number>
    };
    
    paradoxes.forEach(p => {
      // Count by type
      stats.byType[p.associatorResult.type] = (stats.byType[p.associatorResult.type] || 0) + 1;
      
      // Count by class
      if (p.associatorResult.paradoxClass) {
        stats.byClass[p.associatorResult.paradoxClass] = 
          (stats.byClass[p.associatorResult.paradoxClass] || 0) + 1;
      }
    });
    
    return stats;
  }

  /**
   * Reset analyzer (useful for testing)
   */
  reset(): void {
    this.paradoxRegistry.clear();
    this.operationCache.clear();
    console.log('🔺 ParadoxAnalyzer reset');
  }
}

// Singleton instance
export const paradoxAnalyzer = new ParadoxAnalyzer();

console.log('🔺 ParadoxAnalyzer ready for Phase 3 metacognitive paradox detection');
