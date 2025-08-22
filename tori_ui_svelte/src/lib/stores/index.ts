// Main store exports
export { 
  concepts, 
  addConceptDiff, 
  setActiveConcept, 
  systemCoherence, 
  conceptMesh 
} from './conceptMesh';

export type { ConceptDiff, ConceptNode } from './conceptMesh';

export { ghostPersona, Ghost, setLastTriggeredGhost } from './ghostPersona';

// Soliton Memory exports
export {
  solitonMemory,
  memoryStats,
  currentPhase,
  phaseCoherence,
  quantumState,
  isLearning,
  addMemory,
  removeMemory,
  clearMemories,
  updatePhase,
  setQuantumState,
  collapseWaveFunction,
  resetQuantumState,
  recordPhaseTransition,
  startLearning,
  stopLearning
} from './solitonMemory';

export type { MemoryItem, MemoryStats, QuantumState } from './solitonMemory';

// Additional functions that commands are looking for
export const activateConcept = (conceptId: string) => {
  console.log('Activating concept:', conceptId);
  // Implementation or re-export from conceptMesh
};

export const focusConcept = (conceptId: string) => {
  console.log('Focusing concept:', conceptId);
  // Implementation or re-export from conceptMesh
};

export const addConcept = (concept: any) => {
  console.log('Adding concept:', concept);
  // Implementation or re-export from conceptMesh
};

export const linkConcepts = (concepts: string[]) => {
  console.log('Linking concepts:', concepts);
  // Implementation or re-export from conceptMesh
};

// Create proper store implementations
import { writable } from 'svelte/store';

// Vault Entry interface for proper typing
export interface VaultEntry {
  id: string;
  key?: string;
  title: string;
  content: string;
  value?: string;
  sealed: boolean;
  timestamp: Date;
  conceptIds?: string[];
  emotionalWeight?: number;
  metadata?: any;
}

export const ghostState = writable({});
export const activeAgents = writable({});
export const conversationLog = writable<any[]>([]);
export const vaultEntries = writable<VaultEntry[]>([]);
export const sealedArcs = writable<string[]>([]);

// Other exports
export * from './types';
export * from './persistence';
export * from './session';
export * from './toriStorage';
