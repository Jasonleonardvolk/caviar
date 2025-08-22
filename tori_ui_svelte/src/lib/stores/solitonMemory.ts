// src/lib/stores/solitonMemory.ts
// Soliton Memory Store - Quantum consciousness memory system
import { writable, derived, get, type Writable } from 'svelte/store';

// Types
export interface MemoryItem {
  id: string;
  title: string;
  content: any;
  timestamp: number;
  phase: string;
  coherence: number;
  tags: string[];
  relationships: string[];
}

export interface MemoryStats {
  totalMemories: number;
  activePhases: number;
  averageCoherence: number;
  quantumEntanglement: number;
}

export interface QuantumState {
  superposition: number;
  decoherence: number;
  entanglement: number;
  collapse: boolean;
}

// Core stores
export const solitonMemory = writable<MemoryItem[]>([]);
export const currentPhase = writable<string>('exploration');
export const phaseCoherence = writable<number>(0.75);
export const isLearning = writable<boolean>(false);
export const conceptMesh = writable<any[]>([]);

// Quantum state store
export const quantumState = writable<QuantumState>({
  superposition: 0.5,
  decoherence: 0.1,
  entanglement: 0.3,
  collapse: false
});

// Derived memory statistics
export const memoryStats = derived(
  solitonMemory,
  ($memories) => {
    const stats: MemoryStats = {
      totalMemories: $memories.length,
      activePhases: new Set($memories.map(m => m.phase)).size,
      averageCoherence: $memories.length > 0 
        ? $memories.reduce((sum, m) => sum + m.coherence, 0) / $memories.length 
        : 0,
      quantumEntanglement: get(quantumState).entanglement
    };
    return stats;
  }
);

// Helper functions
export function addMemory(memory: Partial<MemoryItem>) {
  const newMemory: MemoryItem = {
    id: `memory-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    title: memory.title || 'Untitled Memory',
    content: memory.content || '',
    timestamp: memory.timestamp || Date.now(),
    phase: memory.phase || get(currentPhase),
    coherence: memory.coherence || get(phaseCoherence),
    tags: memory.tags || [],
    relationships: memory.relationships || []
  };
  
  solitonMemory.update(memories => [...memories, newMemory]);
  return newMemory;
}

export function removeMemory(id: string) {
  solitonMemory.update(memories => memories.filter(m => m.id !== id));
}

export function clearMemories() {
  solitonMemory.set([]);
}

export function updatePhase(newPhase: string, coherence?: number) {
  currentPhase.set(newPhase);
  if (coherence !== undefined) {
    phaseCoherence.set(coherence);
  }
}

export function setQuantumState(state: Partial<QuantumState>) {
  quantumState.update(current => ({
    ...current,
    ...state
  }));
}

export function collapseWaveFunction() {
  quantumState.update(state => ({
    ...state,
    superposition: 0,
    collapse: true
  }));
}

export function resetQuantumState() {
  quantumState.set({
    superposition: 0.5,
    decoherence: 0.1,
    entanglement: 0.3,
    collapse: false
  });
}

// Phase transitions
export const phaseTransitions = writable<any[]>([]);

export function recordPhaseTransition(from: string, to: string, trigger: string) {
  const transition = {
    from,
    to,
    timestamp: Date.now(),
    trigger,
    coherenceDelta: get(phaseCoherence) - 0.5
  };
  
  phaseTransitions.update(transitions => [...transitions, transition]);
  currentPhase.set(to);
}

// Learning mode helpers
export function startLearning() {
  isLearning.set(true);
  updatePhase('learning', 0.9);
}

export function stopLearning() {
  isLearning.set(false);
  updatePhase('integration', 0.7);
}

// Concept mesh integration
export function addConcept(concept: any) {
  conceptMesh.update(mesh => [...mesh, concept]);
}

export function linkConcepts(conceptId1: string, conceptId2: string, weight: number = 0.5) {
  // Link concepts in the mesh
  conceptMesh.update(mesh => {
    // Implementation for linking concepts
    return mesh;
  });
}

// Default export for convenience
export default {
  solitonMemory,
  memoryStats,
  currentPhase,
  phaseCoherence,
  quantumState,
  isLearning,
  conceptMesh,
  addMemory,
  removeMemory,
  clearMemories,
  updatePhase,
  setQuantumState,
  collapseWaveFunction,
  resetQuantumState,
  recordPhaseTransition,
  startLearning,
  stopLearning,
  addConcept,
  linkConcepts
};
