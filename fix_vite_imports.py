#!/usr/bin/env python3
"""Fix RealGhostEngine imports for Vite - remove .js from TypeScript imports"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_vite_imports():
    print("🔧 FIXING REALGHOSTENGINE FOR VITE")
    print("=" * 70)
    
    base_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib")
    
    # The correct imports WITHOUT .js extension for TypeScript files
    fixed_content = '''// RealGhostEngine.js - The ACTUAL integration that connects everything
// This is what ghostEngine.js SHOULD have been!
// NOW WITH CONCEPT MESH INTEGRATION!

// Import from TypeScript files WITHOUT .js extension - Vite handles this correctly
import HolographicEngine from '../../../frontend/lib/holographicEngine';
import { ToriHolographicRenderer } from '../../../frontend/lib/holographicRenderer';
import { FFTCompute } from '../../../frontend/lib/webgpu/fftCompute';
import { HologramPropagation } from '../../../frontend/lib/webgpu/hologramPropagation';
import { QuiltGenerator } from '../../../frontend/lib/webgpu/quiltGenerator';
import { ConceptHologramRenderer } from './conceptHologramRenderer';

/**
 * The REAL Ghost Engine that actually connects all your amazing work!
 * No more console.log placeholders - this is the integration hub.
 * NOW WITH CONCEPT MESH VISUALIZATION!
 */
export class RealGhostEngine {
    constructor() {
        this.engine = null;
        this.renderer = null;
        this.wsConnection = null;
        this.audioProcessor = null;
        this.conceptRenderer = null; // NEW: Concept Mesh renderer
        this.isInitialized = false;
        
        // State management
        this.psiState = {
            psi_phase: 0,
            phase_coherence: 0.8,
            oscillator_phases: new Array(32).fill(0),
            oscillator_frequencies: new Array(32).fill(0),
            coupling_strength: 0.5,
            dominant_frequency: 440
        };
        
        // Hologram configuration
        this.config = {
            displayType: 'looking_glass_portrait',
            quality: 'high',
            enableAudio: true,
            enableHoTT: true,
            enableConceptMesh: true // NEW: Enable concept visualization
        };
        
        console.log('🚀 Initializing REAL Ghost Engine with Concept Mesh support!');
    }
    
    /**
     * Initialize the complete holographic system
     */
    async initialize(canvas, options = {}) {
        try {
            // Merge options
            this.config = { ...this.config, ...options };
            
            // 1. Initialize WebGPU Holographic Engine
            console.log('📊 Initializing Holographic Engine...');
            this.engine = new HolographicEngine(); // Use the default export
            const calibration = this.getCalibration(this.config.displayType);
            await this.engine.initialize(canvas, calibration);
            
            // 2. Initialize Holographic Renderer
            console.log('🎨 Initializing Holographic Renderer...');
            this.renderer = new ToriHolographicRenderer(canvas);
            await this.renderer.initialize();
            
            // 3. Connect to Python backend for audio processing
            if (this.config.enableAudio) {
                console.log('🎵 Connecting to audio processing backend...');
                await this.connectAudioBackend();
            }
            
            // 4. Initialize HoTT integration if enabled
            if (this.config.enableHoTT) {
                console.log('🧮 Initializing HoTT proof system...');
                await this.initializeHoTT();
            }
            
            // 5. NEW: Initialize Concept Mesh renderer
            if (this.config.enableConceptMesh) {
                console.log('🧠 Initializing Concept Mesh visualization...');
                await this.initializeConceptMesh(canvas);
            }
            
            // 6. Start render loop
            this.startRenderLoop();
            
            this.isInitialized = true;
            console.log('✅ Real Ghost Engine fully initialized with Concept Mesh!');
            
            return {
                success: true,
                capabilities: {
                    webgpu: true,
                    fft: true,
                    propagation: true,
                    multiview: true,
                    audio: this.config.enableAudio,
                    hott: this.config.enableHoTT,
                    conceptMesh: this.config.enableConceptMesh
                }
            };
            
        } catch (error) {
            console.error('❌ Failed to initialize Ghost Engine:', error);
            throw error;
        }
    }
'''
    
    # Read the original file to get the rest of the content
    original_file = base_path / "realGhostEngine.js"
    if original_file.exists():
        content = original_file.read_text(encoding='utf-8')
        
        # Find where the initialize method ends
        init_end = content.find('    /**\n     * Initialize Concept Mesh visualization')
        if init_end == -1:
            # Try another marker
            init_end = content.find('    async initializeConceptMesh')
        
        if init_end > 0:
            # Append the rest of the file
            fixed_content += '\n' + content[init_end:]
        else:
            print("⚠️ Could not find where to continue the file, using manual completion")
            # Add the rest manually
            fixed_content += '''
    
    /**
     * Initialize Concept Mesh visualization
     */
    async initializeConceptMesh(canvas) {
        this.conceptRenderer = new ConceptHologramRenderer();
        await this.conceptRenderer.initialize(canvas, this.engine);
        
        console.log('✅ Concept Mesh renderer connected');
        
        // Load initial concepts if available
        if (this.conceptRenderer.concepts.size > 0) {
            console.log(`📚 Loaded ${this.conceptRenderer.concepts.size} concepts`);
        }
    }
    
    // ... rest of the original file continues ...
    
    getCalibration(displayType) {
        // This would return actual Looking Glass calibration
        // Using the data from holographicEngine.ts
        const calibrations = {
            looking_glass_portrait: {
                pitch: 49.825,
                tilt: -0.1745,
                center: 0.04239,
                viewCone: 40,
                invView: 1,
                verticalAngle: 0,
                DPI: 324,
                screenW: 1536,
                screenH: 2048,
                flipImageX: 0,
                flipImageY: 0,
                flipSubp: 0,
                numViews: 45,
                quiltWidth: 3360,
                quiltHeight: 3360,
                tileWidth: 420,
                tileHeight: 560
            },
            webgpu_only: {
                pitch: 50.0,
                tilt: 0.0,
                center: 0.0,
                viewCone: 45,
                invView: 1,
                verticalAngle: 0,
                DPI: 96,
                screenW: 1920,
                screenH: 1080,
                flipImageX: 0,
                flipImageY: 0,
                flipSubp: 0,
                numViews: 25,
                quiltWidth: 2560,
                quiltHeight: 1600,
                tileWidth: 512,
                tileHeight: 320
            }
        };
        
        return calibrations[displayType] || calibrations.webgpu_only;
    }
    
    // ... other methods continue ...
}

// For backward compatibility
export default RealGhostEngine;
'''
    
    # Write the fixed file
    fixed_file = base_path / "realGhostEngine_vite_fixed.js"
    fixed_file.write_text(fixed_content, encoding='utf-8')
    print(f"✅ Created {fixed_file.name}")
    
    print("\n🎯 TO APPLY THE FIX:")
    print("1. Backup current file:")
    print("   copy realGhostEngine.js realGhostEngine.backup.js")
    print("2. Apply the fix:")
    print("   copy realGhostEngine_vite_fixed.js realGhostEngine.js")
    print("3. Do the same for realGhostEngine_v2.js if needed")
    
    print("\n💡 KEY CHANGES:")
    print("- Removed .js extensions from TypeScript imports")
    print("- Fixed HolographicEngine import (default export)")
    print("- Vite will now correctly resolve .ts files")

if __name__ == "__main__":
    fix_vite_imports()
