// src/lib/stores/persistence.ts - Memory persistence utilities for TORI cognitive system
import type { ConceptDiff } from './types';
import { conceptMesh, conceptNodes } from './conceptMesh';

const STORAGE_KEY = 'tori-concept-mesh';
const STORAGE_VERSION = '1.0.0';

interface StoredConceptMesh {
  version: string;
  timestamp: string;
  diffs: ConceptDiff[];
  nodeCount: number;
}

/**
 * Save concept mesh to localStorage with versioning and error handling
 */
export function saveConceptMeshToMemory(diffs: ConceptDiff[]): void {
  try {
    const payload: StoredConceptMesh = {
      version: STORAGE_VERSION,
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
 * Load concept mesh from localStorage with validation and migration
 */
export function loadConceptMeshFromMemory(): void {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    
    if (!stored) {
      console.log('📊 No previous concept mesh found in memory system');
      return;
    }
    
    const parsed = JSON.parse(stored) as StoredConceptMesh;
    
    // Validate stored data structure
    if (!parsed.diffs || !Array.isArray(parsed.diffs)) {
      console.warn('⚠️ Invalid concept mesh data structure, skipping load');
      return;
    }
    
    // Convert timestamps back to Date objects
    const restoredDiffs: ConceptDiff[] = parsed.diffs.map(diff => ({
      ...diff,
      timestamp: new Date(diff.timestamp)
    }));
    
    // Set conceptMesh with restored data
    conceptMesh.set(restoredDiffs);
    
    // Rebuild concept nodes from loaded data
    const allConcepts = restoredDiffs.flatMap(diff => 
      Array.isArray(diff.concepts) ? diff.concepts : []
    );
    
    rebuildConceptNodes(allConcepts, restoredDiffs);
    
    console.log(`📊 Concept mesh loaded from memory system - ${restoredDiffs.length} diffs, ${allConcepts.length} concepts`);
    console.log(`🕒 Last saved: ${parsed.timestamp} (version: ${parsed.version})`);
    
  } catch (error) {
    console.warn('❌ Failed to load concept mesh from memory:', error);
    
    // Clear corrupted data
    localStorage.removeItem(STORAGE_KEY);
    console.log('🗑️ Corrupted memory data cleared');
  }
}

/**
 * Rebuild concept nodes from loaded concepts
 */
function rebuildConceptNodes(concepts: string[], diffs: ConceptDiff[]): void {
  const nodeMap: Record<string, any> = {};
  
  concepts.forEach((conceptName, index) => {
    if (typeof conceptName === 'string' && conceptName.trim()) {
      nodeMap[conceptName] = {
        id: `restored_concept_${index}_${Date.now()}`,
        name: conceptName,
        strength: 0.6 + Math.random() * 0.3, // Restored concepts start with medium strength
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
 * Clear all persisted concept mesh data
 */
export function clearConceptMeshMemory(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
    console.log('🗑️ Concept mesh memory cleared');
  } catch (error) {
    console.warn('❌ Failed to clear concept mesh memory:', error);
  }
}

/**
 * Get memory storage statistics
 */
export function getMemoryStats(): { 
  hasData: boolean; 
  lastSaved?: string; 
  version?: string; 
  estimatedSize: number;
  diffsCount: number;
} {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    
    if (!stored) {
      return { hasData: false, estimatedSize: 0, diffsCount: 0 };
    }
    
    const parsed = JSON.parse(stored) as StoredConceptMesh;
    
    return {
      hasData: true,
      lastSaved: parsed.timestamp,
      version: parsed.version,
      estimatedSize: new Blob([stored]).size, // Approximate size in bytes
      diffsCount: parsed.nodeCount || 0
    };
    
  } catch (error) {
    console.warn('❌ Failed to get memory stats:', error);
    return { hasData: false, estimatedSize: 0, diffsCount: 0 };
  }
}

/**
 * Export concept mesh data for backup/migration
 */
export function exportConceptMeshData(): string | null {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return null;
    
    const parsed = JSON.parse(stored);
    return JSON.stringify(parsed, null, 2); // Pretty formatted for export
    
  } catch (error) {
    console.warn('❌ Failed to export concept mesh data:', error);
    return null;
  }
}

/**
 * Import concept mesh data from backup
 */
export function importConceptMeshData(jsonData: string): boolean {
  try {
    const parsed = JSON.parse(jsonData) as StoredConceptMesh;
    
    // Validate structure
    if (!parsed.diffs || !Array.isArray(parsed.diffs)) {
      throw new Error('Invalid data structure');
    }
    
    // Save imported data
    localStorage.setItem(STORAGE_KEY, JSON.stringify(parsed));
    
    // Reload from memory to update stores
    loadConceptMeshFromMemory();
    
    console.log(`📥 Imported concept mesh data - ${parsed.diffs.length} diffs`);
    return true;
    
  } catch (error) {
    console.warn('❌ Failed to import concept mesh data:', error);
    return false;
  }
}

console.log('💾 TORI persistence module loaded');
