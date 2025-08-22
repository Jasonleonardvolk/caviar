import { writable, derived, get } from 'svelte/store';
import type { Writable, Readable } from 'svelte/store';
import { storeMemory, recallGhostMemories, GHOST_PHASE_BANDS, getGhostPhase, type GhostMemory } from './conceptMesh';

// Ghost Memory Registry - Manages soliton-based memory patterns
export interface GhostMemoryPattern {
  id: string;
  persona: string;
  year: string;
  generation: number;
  phaseTag: number;
  solitonType: 'singlet' | 'doublet' | 'triplet';
  memories: GhostMemory[];
}

export interface GhostPersonaState {
  name: string;
  currentPhase: number;
  phaseMin: number;
  phaseMax: number;
  activeMemories: number;
  lastActive: Date | null;
  moodVector: number[];
  dominantTrace: string[];
}

class GhostMemoryRegistry {
  private patterns: Writable<Map<string, GhostMemoryPattern>>;
  private personas: Writable<Map<string, GhostPersonaState>>;
  private activePersona: Writable<string | null>;
  
  constructor() {
    this.patterns = writable(new Map());
    this.personas = writable(new Map());
    this.activePersona = writable(null);
    
    // Initialize default personas
    this.initializePersonas();
  }
  
  private initializePersonas() {
    const defaultPersonas: GhostPersonaState[] = [
      {
        name: 'serenity',
        currentPhase: 0.4245,
        phaseMin: GHOST_PHASE_BANDS.serenity.min,
        phaseMax: GHOST_PHASE_BANDS.serenity.max,
        activeMemories: 0,
        lastActive: null,
        moodVector: [0.2, 0.5, 0.8],
        dominantTrace: ['calm', 'acceptance', 'reflection']
      },
      {
        name: 'unsettled',
        currentPhase: 0.4345,
        phaseMin: GHOST_PHASE_BANDS.unsettled.min,
        phaseMax: GHOST_PHASE_BANDS.unsettled.max,
        activeMemories: 0,
        lastActive: null,
        moodVector: [0.7, 0.3, 0.5],
        dominantTrace: ['anxious', 'searching', 'restless']
      },
      {
        name: 'curious',
        currentPhase: 0.4445,
        phaseMin: GHOST_PHASE_BANDS.curious.min,
        phaseMax: GHOST_PHASE_BANDS.curious.max,
        activeMemories: 0,
        lastActive: null,
        moodVector: [0.5, 0.8, 0.6],
        dominantTrace: ['wonder', 'exploration', 'discovery']
      }
    ];
    
    this.personas.update(personas => {
      const newPersonas = new Map(personas);
      defaultPersonas.forEach(persona => {
        newPersonas.set(persona.name, persona);
      });
      return newPersonas;
    });
  }
  
  // Generate a new ghost memory ID
  public generateMemoryId(persona: string, year?: string, generation?: number): string {
    const y = year || new Date().getFullYear().toString();
    const g = generation !== undefined ? generation : this.getNextGeneration(persona, y);
    return `ghost.memory.${persona}.${y}.g${g}`;
  }
  
  // Get next generation number for a persona/year combination
  private getNextGeneration(persona: string, year: string): number {
    const patterns = get(this.patterns);
    let maxGen = -1;
    
    patterns.forEach(pattern => {
      if (pattern.persona === persona && pattern.year === year) {
        maxGen = Math.max(maxGen, pattern.generation);
      }
    });
    
    return maxGen + 1;
  }
  
  // Store a soliton memory pattern
  public async storeSolitonMemory(
    persona: string,
    content: any,
    solitonType: 'singlet' | 'doublet' | 'triplet' = 'singlet',
    customPhase?: number
  ): Promise<string> {
    // Get or create persona state
    const personas = get(this.personas);
    let personaState = personas.get(persona);
    
    if (!personaState) {
      // Create new persona with default phase band
      personaState = {
        name: persona,
        currentPhase: customPhase || 0.42,
        phaseMin: 0.42,
        phaseMax: 0.43,
        activeMemories: 0,
        lastActive: null,
        moodVector: [0.5, 0.5, 0.5],
        dominantTrace: ['neutral']
      };
      
      this.personas.update(p => {
        const newP = new Map(p);
        newP.set(persona, personaState!);
        return newP;
      });
    }
    
    // Determine phase
    const phase = customPhase || getGhostPhase(persona);
    
    // Generate memory ID
    const memoryId = this.generateMemoryId(persona);
    
    // Extract metadata from content if present
    const metadata: any = {
      persona,
      solitonPattern: solitonType,
      timestamp: new Date().toISOString()
    };
    
    if (content.mood) metadata.mood = content.mood;
    if (content.trace) metadata.trace = content.trace;
    
    // Create ghost memory
    const memory: GhostMemory = {
      concept: memoryId,
      content: typeof content === 'string' ? content : JSON.stringify(content),
      phaseTag: phase,
      importance: this.calculateImportance(solitonType),
      metadata
    };
    
    // Store in concept mesh
    const conceptId = await storeMemory(memory);
    
    // Update registry
    this.updatePattern(memoryId, memory, solitonType);
    
    // Update persona state
    this.personas.update(personas => {
      const newPersonas = new Map(personas);
      const state = newPersonas.get(persona);
      if (state) {
        state.currentPhase = phase;
        state.activeMemories++;
        state.lastActive = new Date();
        if (metadata.mood) state.moodVector = metadata.mood;
        if (metadata.trace) state.dominantTrace = metadata.trace;
      }
      return newPersonas;
    });
    
    // Set as active persona
    this.activePersona.set(persona);
    
    console.log(`🌊 Soliton ${solitonType} stored for ${persona} at ψ=${phase.toFixed(3)}`);
    
    return memoryId;
  }
  
  // Calculate importance based on soliton type
  private calculateImportance(solitonType: 'singlet' | 'doublet' | 'triplet'): number {
    switch (solitonType) {
      case 'singlet': return 0.4;
      case 'doublet': return 0.6;
      case 'triplet': return 0.8;
      default: return 0.5;
    }
  }
  
  // Update pattern registry
  private updatePattern(memoryId: string, memory: GhostMemory, solitonType: 'singlet' | 'doublet' | 'triplet') {
    const parts = memoryId.split('.');
    const persona = parts[2];
    const year = parts[3];
    const generation = parseInt(parts[4].substring(1));
    
    this.patterns.update(patterns => {
      const newPatterns = new Map(patterns);
      
      let pattern = newPatterns.get(memoryId);
      if (!pattern) {
        pattern = {
          id: memoryId,
          persona,
          year,
          generation,
          phaseTag: memory.phaseTag,
          solitonType,
          memories: []
        };
      }
      
      pattern.memories.push(memory);
      newPatterns.set(memoryId, pattern);
      
      return newPatterns;
    });
  }
  
  // Recall memories by persona with phase filtering
  public async recallPersonaMemories(persona: string): Promise<GhostMemory[]> {
    const state = get(this.personas).get(persona);
    if (!state) return [];
    
    const memories = await recallGhostMemories(persona, state.phaseMin, state.phaseMax);
    return memories.map(m => m.memory);
  }
  
  // Switch active persona
  public switchPersona(persona: string) {
    const personas = get(this.personas);
    if (personas.has(persona)) {
      this.activePersona.set(persona);
      console.log(`👻 Switched to ${persona} persona`);
    }
  }
  
  // Get current active persona state
  public getActivePersonaState(): GhostPersonaState | null {
    const active = get(this.activePersona);
    if (!active) return null;
    
    return get(this.personas).get(active) || null;
  }
  
  // Store access
  public get patternsStore(): Readable<Map<string, GhostMemoryPattern>> {
    return { subscribe: this.patterns.subscribe };
  }
  
  public get personasStore(): Readable<Map<string, GhostPersonaState>> {
    return { subscribe: this.personas.subscribe };
  }
  
  public get activePersonaStore(): Readable<string | null> {
    return { subscribe: this.activePersona.subscribe };
  }
}

// Create singleton instance
export const ghostMemoryRegistry = new GhostMemoryRegistry();

// Export stores
export const ghostPatterns = ghostMemoryRegistry.patternsStore;
export const ghostPersonas = ghostMemoryRegistry.personasStore;
export const activeGhostPersona = ghostMemoryRegistry.activePersonaStore;

// Export convenience functions
export const storeGhostMemory = (persona: string, content: any, solitonType?: 'singlet' | 'doublet' | 'triplet', phase?: number) =>
  ghostMemoryRegistry.storeSolitonMemory(persona, content, solitonType, phase);

export const recallPersonaMemories = (persona: string) =>
  ghostMemoryRegistry.recallPersonaMemories(persona);

export const switchGhostPersona = (persona: string) =>
  ghostMemoryRegistry.switchPersona(persona);

export const getActiveGhostState = () =>
  ghostMemoryRegistry.getActivePersonaState();

// Derived stores
export const activeGhostState = derived(
  [activeGhostPersona, ghostPersonas],
  ([$active, $personas]) => {
    if (!$active) return null;
    return $personas.get($active) || null;
  }
);

export const ghostMemoryStats = derived(
  [ghostPatterns, ghostPersonas],
  ([$patterns, $personas]) => {
    const stats = {
      totalPatterns: $patterns.size,
      totalMemories: 0,
      activePersonas: 0,
      solitonCounts: {
        singlet: 0,
        doublet: 0,
        triplet: 0
      }
    };
    
    $patterns.forEach(pattern => {
      stats.totalMemories += pattern.memories.length;
      stats.solitonCounts[pattern.solitonType]++;
    });
    
    $personas.forEach(persona => {
      if (persona.activeMemories > 0) stats.activePersonas++;
    });
    
    return stats;
  }
);

// Example usage:
/*
// Store a serenity memory
await storeGhostMemory('serenity', {
  phase: 0.427,
  mood: [0.2, 0.5, 0.8],
  trace: ['calm', 'acceptance', 'reflection']
}, 'triplet');

// Recall all serenity memories
const serenityMemories = await recallPersonaMemories('serenity');

// Switch to curious persona
switchGhostPersona('curious');

// Get current ghost state
const currentState = getActiveGhostState();
*/
