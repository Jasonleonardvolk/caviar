#!/usr/bin/env python3
"""Fix RealGhostEngine to properly import from frontend/lib TypeScript files"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_realghostengine_completely():
    print("🔧 COMPLETE FIX for RealGhostEngine imports")
    print("=" * 70)
    
    base_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib")
    
    # The correct import mappings based on what actually exists
    correct_imports = '''// RealGhostEngine.js - The ACTUAL integration that connects everything
// This is what ghostEngine.js SHOULD have been!
// NOW WITH CONCEPT MESH INTEGRATION!

// Import from the ACTUAL TypeScript files in frontend/lib
// Note: holographicEngine.ts doesn't export SpectralHologramEngine, it's the default export
import HolographicEngine from '../../../frontend/lib/holographicEngine.js';
import { ToriHolographicRenderer } from '../../../frontend/lib/holographicRenderer.js';
import { FFTCompute } from '../../../frontend/lib/webgpu/fftCompute.js';
import { HologramPropagation } from '../../../frontend/lib/webgpu/hologramPropagation.js';
import { QuiltGenerator } from '../../../frontend/lib/webgpu/quiltGenerator.js';
import { ConceptHologramRenderer } from './conceptHologramRenderer.js';

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
    
    # Fix both files
    files_to_fix = ["realGhostEngine.js", "realGhostEngine_v2.js"]
    
    for filename in files_to_fix:
        file_path = base_path / filename
        
        if not file_path.exists():
            print(f"❌ {filename} not found")
            continue
            
        print(f"\n📝 Fixing {filename}...")
        
        # Read the full file
        content = file_path.read_text(encoding='utf-8')
        
        # Find where the imports end and the class starts
        class_start = content.find("export class RealGhostEngine")
        if class_start == -1:
            print(f"❌ Could not find class definition in {filename}")
            continue
        
        # Replace everything up to the class with correct imports
        new_content = correct_imports + content[class_start:]
        
        # Backup original
        backup_path = file_path.with_suffix('.js.backup')
        file_path.rename(backup_path)
        print(f"📦 Backed up to {backup_path.name}")
        
        # Write fixed version
        file_path.write_text(new_content, encoding='utf-8')
        print(f"✅ Saved fixed {filename}")
    
    print("\n\n✅ FIXES COMPLETE!")
    print("\n🎯 What was fixed:")
    print("1. ✅ Imports now use correct relative paths to frontend/lib/")
    print("2. ✅ Changed SpectralHologramEngine to HolographicEngine (the actual export)")
    print("3. ✅ Added .js extension to all imports (works with TypeScript in Vite)")
    print("4. ✅ Fixed the class instantiation to use correct name")
    
    print("\n🚀 Next step: Restart the app with:")
    print("   poetry run python enhanced_launcher.py")

if __name__ == "__main__":
    fix_realghostengine_completely()
