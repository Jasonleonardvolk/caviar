import { writable } from 'svelte/store';
import type { ConceptDiff, SystemState, ConceptNode } from './types';

// Enhanced concept structure for ConceptInspector compatibility
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

// Concept interface matching your implementation - WITH DEFENSIVE STRUCTURE
export interface Concept {
  name: string;
  score: number;
  method?: string;
  source?: any;
  context?: string;
  metadata?: Record<string, any>;
}

// Enhanced ConceptDiff with rich concept support
export interface EnhancedConceptDiff extends ConceptDiff {
  enrichedConcepts?: EnhancedConcept[];
}

const initialDiffs: ConceptDiff[] = [];
const initialSystemState: SystemState = {
  status: 'idle',
  coherenceLevel: 0.8
};

export const conceptMesh = writable<ConceptDiff[]>(initialDiffs);
export const systemState = writable<SystemState>(initialSystemState);
export const conceptNodes = writable<Record<string, ConceptNode>>({});

// Derived stores for UI consumption
export const systemCoherence = writable<number>(0.8);
export const thoughtspaceVisible = writable<boolean>(true);
export const activeConcept = writable<string | null>(null);
export const lastTriggeredGhost = writable<string | null>(null);
export const systemEntropy = writable<number>(20);

// Storage management constants
const STORAGE_KEY = 'tori-concept-mesh';

/**
 * Save concept mesh to localStorage
 */
function saveConceptMeshToMemory(diffs: ConceptDiff[]): void {
  try {
    const payload = {
      version: '1.0.0',
      timestamp: new Date().toISOString(),
      diffs: diffs,
      nodeCount: diffs.length
    };
    
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    console.log(`💾 Concept mesh saved to memory system - ${diffs.length} diffs`);
    
  } catch (error) {
    console.warn('❌ Failed to save concept mesh to memory:', error);
    
    // Try to clear corrupted data if quota exceeded
    if (error instanceof Error && error.name === 'QuotaExceededError') {
      console.warn('🗑️ Storage quota exceeded, clearing old data');
      localStorage.removeItem(STORAGE_KEY);
    }
  }
}

/**
 * Load concept mesh from localStorage
 */
function loadConceptMeshFromMemory(): void {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    
    if (!stored) {
      console.log('📊 No previous concept mesh found in memory system');
      return;
    }
    
    const parsed = JSON.parse(stored);
    
    // Validate stored data structure
    if (!parsed.diffs || !Array.isArray(parsed.diffs)) {
      console.warn('⚠️ Invalid concept mesh data structure, skipping load');
      return;
    }
    
    // Convert timestamps back to Date objects
    const restoredDiffs: ConceptDiff[] = parsed.diffs.map((diff: any) => ({
      ...diff,
      timestamp: new Date(diff.timestamp)
    }));
    
    // Set conceptMesh with restored data
    conceptMesh.set(restoredDiffs);
    
    // Rebuild concept nodes from loaded data
    const allConcepts = restoredDiffs.flatMap(diff => 
      Array.isArray(diff.concepts) ? diff.concepts : []
    );
    
    rebuildConceptNodes(allConcepts);
    
    console.log(`📊 Concept mesh loaded from memory system - ${restoredDiffs.length} diffs, ${allConcepts.length} concepts`);
    
  } catch (error) {
    console.warn('❌ Failed to load concept mesh from memory:', error);
    localStorage.removeItem(STORAGE_KEY);
    console.log('🗑️ Corrupted memory data cleared');
  }
}

/**
 * Rebuild concept nodes from loaded concepts
 */
function rebuildConceptNodes(concepts: string[]): void {
  const nodeMap: Record<string, ConceptNode> = {};
  
  concepts.forEach((conceptName, index) => {
    if (typeof conceptName === 'string' && conceptName.trim()) {
      nodeMap[conceptName] = {
        id: `restored_concept_${index}_${Date.now()}`,
        name: conceptName,
        strength: 0.6 + Math.random() * 0.3,
        type: 'restored',
        position: {
          x: (Math.random() - 0.5) * 4,
          y: (Math.random() - 0.5) * 4,
          z: (Math.random() - 0.5) * 4
        },
        highlighted: false,
        connections: [],
        lastActive: new Date(),
        contradictionLevel: 0,
        coherenceContribution: 0.1,
        loopReferences: []
      };
    }
  });
  
  conceptNodes.set(nodeMap);
  console.log(`🔄 Rebuilt ${Object.keys(nodeMap).length} concept nodes from memory`);
}

/**
 * Validate and normalize concept input
 */
function validateAndNormalizeConcepts(concepts: any): Concept[] {
  if (!concepts) {
    console.warn('🚨 Concepts parameter is null or undefined');
    return [];
  }
  
  if (!Array.isArray(concepts)) {
    console.warn('🚨 Concepts parameter is not an array:', typeof concepts, concepts);
    return [];
  }
  
  return concepts.map((concept, index) => {
    // Handle different concept formats
    if (typeof concept === 'string') {
      return {
        name: concept,
        score: 0.5, // Default score for string concepts
        method: 'string_conversion'
      };
    }
    
    if (typeof concept === 'object' && concept !== null) {
      return {
        name: concept.name || concept.toString() || `Unnamed Concept ${index}`,
        score: typeof concept.score === 'number' ? concept.score : 
               typeof concept.confidence === 'number' ? concept.confidence : 0.5,
        method: concept.method || 'unknown',
        source: concept.source,
        context: concept.context,
        metadata: concept.metadata
      };
    }
    
    // Fallback for unexpected types
    console.warn(`🚨 Unexpected concept type at index ${index}:`, typeof concept, concept);
    return {
      name: `Unknown Concept ${index}`,
      score: 0.1,
      method: 'fallback'
    };
  });
}

// Load any saved concept diffs from previous sessions (persistent storage)
if (typeof window !== 'undefined') {
  loadConceptMeshFromMemory();
}

/**
 * 🧠 YOUR CLEAN IMPLEMENTATION - Enhanced with full mesh integration + CRASH PROTECTION
 */
export function addConceptDiff(docId: string, concepts: any, metadata?: Record<string, any>) {
  // 🛡️ DEFENSIVE GUARDS - Prevent all runtime crashes
  if (!docId || typeof docId !== 'string') {
    console.warn("❌ addConceptDiff received invalid docId:", docId);
    return;
  }
  
  // Validate and normalize concepts
  const validatedConcepts = validateAndNormalizeConcepts(concepts);
  
  if (validatedConcepts.length === 0) {
    console.warn("❌ addConceptDiff received no valid concepts for docId:", docId);
    console.warn("   Original input:", concepts);
    return;
  }
  
  // 🔧 Your verification logging - NOW CRASH-PROOF
  console.log("✅ ConceptMesh Updated:", validatedConcepts.length, "concepts added from", docId);
  
  if (validatedConcepts.length < 5) {
    console.warn("⚠️ Low concept yield. Consider adjusting extract thresholds or rerunning vector pass.");
  }
  
  validatedConcepts.forEach((c, index) => {
    const score = typeof c.score === 'number' ? c.score.toFixed(3) : 'N/A';
    console.debug(`🧠 Concept ${index + 1}: ${c.name}, Score: ${score}`);
  });
  
  try {
    // ✅ ACTUAL MESH UPDATE LOGIC - Integrated with your interface
    const enhancedDiff: ConceptDiff = {
      id: `diff_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: metadata?.source === 'pdf_ingest' ? 'document' : 'general',
      title: docId,
      concepts: validatedConcepts.map(c => c.name), // Convert to string array for mesh compatibility
      summary: `Extracted ${validatedConcepts.length} concepts from ${docId}${validatedConcepts.length >= 5 ? ' - Good yield' : ' - Low yield'}`,
      timestamp: new Date(),
      metadata: {
        ...metadata,
        conceptScores: validatedConcepts.reduce((acc, c) => ({ ...acc, [c.name]: c.score }), {}),
        averageScore: validatedConcepts.reduce((sum, c) => sum + c.score, 0) / validatedConcepts.length,
        highConfidenceConcepts: validatedConcepts.filter(c => c.score > 0.8).length,
        boostingApplied: validatedConcepts.some(c => c.method === 'file_storage_boosted'),
        extractionMethod: validatedConcepts.find(c => c.method)?.method || 'standard',
        inputValidation: {
          originalType: typeof concepts,
          originalLength: Array.isArray(concepts) ? concepts.length : 'N/A',
          normalizedLength: validatedConcepts.length,
          validationApplied: true
        }
      }
    };
    
    // Enrich concepts for ConceptInspector compatibility
    const enrichedConcepts: EnhancedConcept[] = validatedConcepts.map((concept, index) => ({
      eigenfunction_id: `${enhancedDiff.id}_concept_${index}`,
      name: concept.name,
      confidence: concept.score,
      context: concept.context || `Extracted from "${docId}" - ${concept.name}`,
      cluster_id: Math.floor(concept.score * 5) + 1, // Score-based clustering (1-5)
      title: docId,
      timestamp: enhancedDiff.timestamp.toISOString(),
      strength: concept.score,
      type: concept.method === 'file_storage_boosted' ? 'boosted' : 'extracted',
      metadata: {
        source: metadata?.source || 'unknown',
        extractionMethod: concept.method || 'standard',
        diffId: enhancedDiff.id,
        originalScore: concept.score,
        ...concept.metadata
      }
    }));
    
    // Store enriched concepts
    (enhancedDiff as EnhancedConceptDiff).enrichedConcepts = enrichedConcepts;
    
    // Update the mesh
    conceptMesh.update((current) => {
      const updated = [...current, enhancedDiff];
      saveConceptMeshToMemory(updated);
      console.log(`📊 ConceptDiff added to mesh [${validatedConcepts.length} concepts] for "${docId}"`, enhancedDiff);
      return updated;
    });
    
    // Update concept nodes for 3D visualization
    updateConceptNodes(validatedConcepts.map(c => c.name));
    
    // Dispatch event for cognitive processing
    if (typeof window !== 'undefined') {
      const contradictionDelta = calculateContradictionDelta(enhancedDiff);
      window.dispatchEvent(new CustomEvent('tori:concept-diff', {
        detail: { delta: contradictionDelta, diff: enhancedDiff, concepts: validatedConcepts }
      }));
    }
    
    console.log(`🧠 Mesh integration complete - ${validatedConcepts.length} concepts from "${docId}" now in cognitive system`);
    
  } catch (error) {
    console.error(`❌ Error during mesh integration for "${docId}":`, error);
    console.error('   Concepts that caused error:', validatedConcepts);
    // Don't re-throw - we want the system to continue functioning
  }
}

/**
 * Alternative function signature for backward compatibility
 */
export function addConceptDiffLegacy(newDiff: Omit<ConceptDiff, 'id' | 'timestamp'>) {
  if (!newDiff || !newDiff.concepts) {
    console.warn("❌ addConceptDiffLegacy received invalid diff:", newDiff);
    return;
  }
  
  const enhancedDiff: ConceptDiff = {
    ...newDiff,
    id: `diff_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    timestamp: new Date()
  };
  
  const conceptCount = Array.isArray(enhancedDiff.concepts) ? enhancedDiff.concepts.length : 0;
  
  console.log("✅ ConceptMesh Updated:", conceptCount, "concepts added from", enhancedDiff.title);
  
  if (conceptCount < 5) {
    console.warn("⚠️ Low concept yield. Consider adjusting extract thresholds or rerunning vector pass.");
  }
  
  try {
    conceptMesh.update((current) => {
      const updated = [...current, enhancedDiff];
      saveConceptMeshToMemory(updated);
      return updated;
    });
    
    if (Array.isArray(enhancedDiff.concepts)) {
      updateConceptNodes(enhancedDiff.concepts);
    }
  } catch (error) {
    console.error("❌ Error in addConceptDiffLegacy:", error);
  }
}

/**
 * Calculate contradiction delta for concept diff processing
 */
function calculateContradictionDelta(diff: ConceptDiff): number {
  let delta = 0;
  
  // Base contradiction from concept count
  delta += Math.min(0.5, diff.concepts.length * 0.05);
  
  // Higher contradiction for document processing vs chat
  if (diff.type === 'document') {
    delta += 0.1;
  }
  
  // Check for complexity indicators in title/summary
  const text = (diff.title + ' ' + (diff.summary || '')).toLowerCase();
  if (text.includes('complex') || text.includes('paradox') || text.includes('conflict')) {
    delta += 0.2;
  }
  
  // Low yield adds uncertainty
  if (diff.concepts.length < 5) {
    delta += 0.2;
  }
  
  // Boosting reduces contradiction (more confident concepts)
  if (diff.metadata?.boostingApplied) {
    delta -= 0.1;
  }
  
  return Math.min(1.0, Math.max(0.0, delta));
}

/**
 * Update concept nodes for 3D visualization with cognitive awareness
 */
function updateConceptNodes(concepts: string[]) {
  if (!Array.isArray(concepts)) {
    console.warn("❌ updateConceptNodes received non-array:", concepts);
    return;
  }
  
  conceptNodes.update(current => {
    const updated = { ...current };
    
    concepts.forEach(conceptName => {
      if (typeof conceptName === 'string' && conceptName.trim()) {
        if (!updated[conceptName]) {
          // Create new concept node with cognitive fields
          updated[conceptName] = {
            id: `concept_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`,
            name: conceptName,
            strength: 0.7,
            type: 'general',
            position: {
              x: (Math.random() - 0.5) * 4,
              y: (Math.random() - 0.5) * 4,
              z: (Math.random() - 0.5) * 4
            },
            highlighted: true,
            connections: [],
            lastActive: new Date(),
            contradictionLevel: 0,
            coherenceContribution: 0.1,
            loopReferences: []
          };
        } else {
          // Update existing concept
          updated[conceptName].lastActive = new Date();
          updated[conceptName].highlighted = true;
          updated[conceptName].strength = Math.min(1.0, updated[conceptName].strength + 0.1);
          updated[conceptName].contradictionLevel = Math.random() * 0.1;
        }
      }
    });
    
    return updated;
  });
}

/**
 * Connect two concepts with a specified strength
 */
export function connectConcepts(concept1: string, concept2: string, strength: number = 0.5): void {
  console.log(`🔗 Connecting concepts: ${concept1} ↔ ${concept2} (strength: ${strength})`);
  
  conceptNodes.update(current => {
    const updated = { ...current };
    
    // Ensure both concepts exist
    if (!updated[concept1]) {
      updated[concept1] = {
        id: `concept_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`,
        name: concept1,
        strength: 0.7,
        type: 'general',
        position: {
          x: (Math.random() - 0.5) * 4,
          y: (Math.random() - 0.5) * 4,
          z: (Math.random() - 0.5) * 4
        },
        highlighted: false,
        connections: [],
        lastActive: new Date(),
        contradictionLevel: 0,
        coherenceContribution: 0.1,
        loopReferences: []
      };
    }
    
    if (!updated[concept2]) {
      updated[concept2] = {
        id: `concept_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`,
        name: concept2,
        strength: 0.7,
        type: 'general',
        position: {
          x: (Math.random() - 0.5) * 4,
          y: (Math.random() - 0.5) * 4,
          z: (Math.random() - 0.5) * 4
        },
        highlighted: false,
        connections: [],
        lastActive: new Date(),
        contradictionLevel: 0,
        coherenceContribution: 0.1,
        loopReferences: []
      };
    }
    
    // Add connection from concept1 to concept2
    if (!updated[concept1].connections.some(conn => conn.target === concept2)) {
      updated[concept1].connections.push({
        target: concept2,
        strength: strength,
        type: 'semantic',
        established: new Date()
      });
    }
    
    // Add connection from concept2 to concept1 (bidirectional)
    if (!updated[concept2].connections.some(conn => conn.target === concept1)) {
      updated[concept2].connections.push({
        target: concept1,
        strength: strength,
        type: 'semantic',
        established: new Date()
      });
    }
    
    // Update both concepts' last active time
    updated[concept1].lastActive = new Date();
    updated[concept2].lastActive = new Date();
    
    return updated;
  });
  
  // Update system coherence based on new connections
  systemCoherence.update(current => Math.min(1.0, current + strength * 0.1));
}

/**
 * Set the currently active concept
 */
export function setActiveConcept(concept: string | null): void {
  activeConcept.set(concept);
  console.log('🎯 Active concept set to:', concept);
}

/**
 * Set the last triggered ghost
 */
export function setLastTriggeredGhost(ghost: string | null): void {
  lastTriggeredGhost.set(ghost);
  console.log('👻 Last triggered ghost set to:', ghost);
}

/**
 * Update system entropy level with cognitive feedback
 */
export function updateSystemEntropy(delta: number): void {
  systemEntropy.update(current => {
    const newEntropy = Math.max(0, Math.min(100, current + delta));
    
    // Update coherence (inverse of entropy)
    const newCoherence = (100 - newEntropy) / 100;
    systemCoherence.set(newCoherence);
    
    console.log(`📊 System entropy: ${current} → ${newEntropy} (Δ${delta})`);
    return newEntropy;
  });
}

/**
 * Activate a specific concept
 */
export function activateConcept(conceptName: string): void {
  console.log('🎯 Activating concept:', conceptName);
  setActiveConcept(conceptName);
  updateSystemEntropy(-2); // Reduce entropy when concept is activated
}

/**
 * Highlight specific concepts in the 3D visualization
 */
export async function highlightConcepts(concepts: Array<{ name: string; strength?: number; type?: string }>): Promise<void> {
  if (!Array.isArray(concepts)) {
    console.warn("❌ highlightConcepts received non-array:", concepts);
    return;
  }
  
  console.log('✨ Highlighting concepts in Thoughtspace:', concepts);
  
  conceptNodes.update(current => {
    const updated = { ...current };
    
    concepts.forEach(concept => {
      if (concept && typeof concept.name === 'string') {
        const conceptName = concept.name;
        
        if (!updated[conceptName]) {
          updated[conceptName] = {
            id: `concept_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`,
            name: conceptName,
            strength: concept.strength || 0.7,
            type: concept.type || 'general',
            position: {
              x: (Math.random() - 0.5) * 4,
              y: (Math.random() - 0.5) * 4,
              z: (Math.random() - 0.5) * 4
            },
            highlighted: true,
            connections: [],
            lastActive: new Date(),
            contradictionLevel: 0,
            coherenceContribution: concept.strength || 0.7,
            loopReferences: []
          };
        } else {
          updated[conceptName].highlighted = true;
          updated[conceptName].lastActive = new Date();
          if (concept.strength) {
            updated[conceptName].strength = Math.max(updated[conceptName].strength, concept.strength);
          }
        }
      }
    });
    
    return updated;
  });
  
  if (concepts.length > 0 && concepts[0] && concepts[0].name) {
    setActiveConcept(concepts[0].name);
  }
  
  // Remove highlights after delay
  setTimeout(() => {
    conceptNodes.update(current => {
      const updated = { ...current };
      concepts.forEach(concept => {
        if (concept && concept.name && updated[concept.name]) {
          updated[concept.name].highlighted = false;
        }
      });
      return updated;
    });
  }, 5000);
}

/**
 * Remove a concept diff by ID
 */
export function removeConceptDiff(id: string) {
  if (!id || typeof id !== 'string') {
    console.warn("❌ removeConceptDiff received invalid id:", id);
    return;
  }
  
  conceptMesh.update(currentList => {
    const updated = currentList.filter(diff => diff.id !== id);
    saveConceptMeshToMemory(updated);
    return updated;
  });
}

/**
 * Clear all concept diffs and reset system
 */
export function clearConceptMesh() {
  conceptMesh.set([]);
  conceptNodes.set({});
  activeConcept.set(null);
  lastTriggeredGhost.set(null);
  systemEntropy.set(20);
  systemCoherence.set(0.8);
  
  if (typeof localStorage !== 'undefined') {
    localStorage.removeItem(STORAGE_KEY);
  }
  
  console.log('🗑️ Concept mesh cleared and system reset');
}

/**
 * Get concept network statistics
 */
export function getNetworkStats(): { nodeCount: number; connectionCount: number; density: number } {
  let nodeCount = 0;
  let connectionCount = 0;
  
  conceptNodes.subscribe(nodes => {
    nodeCount = Object.keys(nodes).length;
    connectionCount = Object.values(nodes).reduce((sum, node) => sum + node.connections.length, 0) / 2;
  })();
  
  const density = nodeCount > 1 ? connectionCount / (nodeCount * (nodeCount - 1) / 2) : 0;
  
  return { nodeCount, connectionCount, density };
}

/**
 * Derived store for ConceptInspector - flattens concepts with metadata
 */
export const inspectorConcepts = writable<EnhancedConcept[]>([]);

// Update inspector concepts whenever conceptMesh changes
conceptMesh.subscribe($mesh => {
  const flattened: EnhancedConcept[] = $mesh.flatMap(diff => {
    // Use enriched concepts if available, otherwise create from simple concepts
    if ((diff as EnhancedConceptDiff).enrichedConcepts) {
      return (diff as EnhancedConceptDiff).enrichedConcepts!;
    } else {
      return diff.concepts.map((concept, index) => ({
        eigenfunction_id: `${diff.id}_concept_${index}`,
        name: typeof concept === 'string' ? concept : concept.name || concept.toString(),
        confidence: 0.7 + Math.random() * 0.3,
        context: `From "${diff.title}" - ${typeof concept === 'string' ? concept : concept.name}`,
        cluster_id: Math.floor(Math.random() * 5) + 1,
        title: diff.title,
        timestamp: diff.timestamp.toISOString(),
        strength: 0.8,
        type: diff.type
      }));
    }
  });
  
  inspectorConcepts.set(flattened);
});

// Auto-decay highlighted concepts over time
if (typeof window !== 'undefined') {
  setInterval(() => {
    conceptNodes.update(current => {
      const updated = { ...current };
      const now = new Date();
      
      Object.values(updated).forEach(node => {
        const timeSinceActive = now.getTime() - node.lastActive.getTime();
        
        if (timeSinceActive > 30000 && node.highlighted) {
          node.highlighted = false;
        }
        
        if (timeSinceActive > 60000) {
          node.strength = Math.max(0.1, node.strength * 0.95);
        }
        
        if (node.contradictionLevel !== undefined) {
          node.contradictionLevel = Math.max(0, node.contradictionLevel * 0.98);
        }
      });
      
      return updated;
    });
  }, 5000);
}

console.log('🧠 Enhanced ConceptMesh system initialized - CRASH-PROOF with input validation and defensive guards');
