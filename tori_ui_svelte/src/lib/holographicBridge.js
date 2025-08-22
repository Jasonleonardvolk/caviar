// holographicBridge.js - Import bridge with Concept Mesh support
import { RealGhostEngine } from './realGhostEngine_v2.js';

// For backward compatibility - alias RealGhostEngine as GhostEngine
export class GhostEngine extends RealGhostEngine {
    constructor() {
        super();
        console.log('🎉 Using REAL Ghost Engine with Concept Mesh visualization!');
    }
}

// Also export the real engine directly
export { RealGhostEngine };

// Export a singleton instance for components that expect it
export const ghostEngine = new GhostEngine();

// Auto-initialize if we're in a browser with WebGPU
if (typeof window !== 'undefined' && navigator.gpu) {
    window.TORI_GHOST_ENGINE = ghostEngine;
    console.log('✨ TORI Ghost Engine with Concept Mesh available globally');
    
    // Auto-enable concept mesh if hologram mode is on
    window.TORI_CONCEPT_MESH_ENABLED = true;
}

console.log('🌉 Holographic Bridge with Concept Mesh loaded!');
