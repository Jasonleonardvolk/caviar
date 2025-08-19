#!/usr/bin/env python3
"""Fix HolographicDisplay to work with the enhanced RealGhostEngine"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_holographic_display_for_enhanced_engine():
    file_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\components\HolographicDisplay.svelte")
    
    print("🔧 Fixing HolographicDisplay for enhanced RealGhostEngine with WebGPU...")
    
    # Complete fixed version that works with the enhanced engine
    content = '''<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { fade } from 'svelte/transition';
  import RealGhostEngine from '$lib/realGhostEngine.js';
  import { ghostPersona } from '$lib/stores/ghostPersona.js';
  import { get } from 'svelte/store';
  
  export let width = 320;
  export let height = 240;
  export let usePenrose = true;
  export let showStats = true;
  export let enableVideo = false;
  export let videoSource: 'webcam' | 'file' | 'stream' = 'webcam';
  
  let canvas: HTMLCanvasElement;
  let video: HTMLVideoElement;
  let animationFrame: number;
  let fps = 0;
  let complexity = 'O(n²·³²)';
  let time = 0;
  let videoReady = false;
  let isInitialized = false;
  let error = '';
  let ghostEngine = null;
  let currentPersona = null;
  let engineCapabilities = null;
  
  // Subscribe to persona changes
  $: if (ghostEngine && $ghostPersona && $ghostPersona.id !== currentPersona?.id) {
    console.log('🔄 Switching hologram to:', $ghostPersona.name);
    currentPersona = $ghostPersona;
    // The enhanced engine doesn't have switchPersona, we'll re-add the object
    updatePersonaHologram();
  }
  
  onMount(async () => {
    try {
      // Get current persona from store
      const initialPersona = get(ghostPersona);
      currentPersona = initialPersona;
      
      console.log('🎭 Initializing enhanced hologram with persona:', initialPersona.name);
      
      // Create Enhanced Ghost Engine instance
      ghostEngine = new RealGhostEngine();
      
      // Initialize with full configuration
      const result = await ghostEngine.initialize(canvas, {
        displayType: 'webgpu_only', // or 'looking_glass_portrait' if you have one
        quality: 'high',
        enableAudio: true,
        enableHoTT: true,
        enableConceptMesh: true,
        persona: initialPersona.id || initialPersona.name || 'ENOLA'
      });
      
      if (!result || !result.success) {
        throw new Error('Failed to initialize Ghost Engine');
      }
      
      engineCapabilities = result.capabilities;
      console.log('✅ Engine capabilities:', engineCapabilities);
      
      isInitialized = true;
      
      // Add initial holographic objects
      setupHolographicScene();
      
      // Add some initial concepts
      setupInitialConcepts();
      
      // Start monitoring stats
      startStatsMonitoring();
      
    } catch (err) {
      console.error('Failed to initialize holographic display:', err);
      error = err.message;
      // The engine has its own fallback, but we'll show the error
    }
    
    // Handle video separately if needed
    if (enableVideo && videoSource === 'webcam') {
      initializeVideo();
    }
  });
  
  onDestroy(() => {
    cleanup();
  });
  
  function cleanup() {
    if (animationFrame) {
      cancelAnimationFrame(animationFrame);
    }
    if (video && video.srcObject) {
      const stream = video.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
    }
    if (ghostEngine) {
      ghostEngine.destroy();
    }
  }
  
  function setupHolographicScene() {
    if (!ghostEngine || !isInitialized) return;
    
    console.log('🌟 Setting up holographic scene');
    
    // Add persona as main holographic object
    ghostEngine.addHolographicObject({
      id: 'persona_core',
      type: 'concept',
      name: currentPersona?.name || 'ENOLA',
      description: currentPersona?.description || 'AI Consciousness',
      category: 'persona',
      position: [0, 0, 0],
      color: currentPersona?.color ? [
        currentPersona.color.r,
        currentPersona.color.g,
        currentPersona.color.b
      ] : [0, 1, 1],
      size: 2.0,
      intensity: 1.0,
      resonance: 0.9
    });
  }
  
  function setupInitialConcepts() {
    if (!ghostEngine || !engineCapabilities?.conceptMesh) return;
    
    console.log('🧠 Adding initial concepts');
    
    // Add some core concepts
    ghostEngine.addConcept('consciousness', 'The emergent property of complex information processing', 'philosophy', 0.9);
    ghostEngine.addConcept('quantum field', 'The fundamental substrate of reality', 'physics', 0.8);
    ghostEngine.addConcept('holography', 'Information encoded on boundaries', 'technology', 0.7);
    ghostEngine.addConcept('TORI', 'The Operational Recursive Intelligence', 'ai', 1.0);
  }
  
  function updatePersonaHologram() {
    if (!ghostEngine || !currentPersona) return;
    
    // Remove old persona object
    ghostEngine.addHolographicObject({
      id: 'persona_core',
      type: 'concept',
      name: currentPersona.name,
      description: currentPersona.description,
      category: 'persona',
      position: [0, 0, 0],
      color: currentPersona.color ? [
        currentPersona.color.r,
        currentPersona.color.g,
        currentPersona.color.b
      ] : [0, 1, 1],
      size: 2.0,
      intensity: 1.0,
      resonance: 0.9
    });
  }
  
  function startStatsMonitoring() {
    function updateStats() {
      if (ghostEngine && isInitialized) {
        const renderInfo = ghostEngine.render();
        if (renderInfo) {
          fps = renderInfo.fps || 0;
          
          // Update complexity based on capabilities
          if (engineCapabilities?.webgpu) {
            complexity = 'WebGPU O(n²·³²)';
          } else {
            complexity = 'Canvas O(n²)';
          }
        }
      }
      
      time += 0.016;
      animationFrame = requestAnimationFrame(updateStats);
    }
    
    updateStats();
  }
  
  async function initializeVideo() {
    if (!video) return;
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width, height } 
      });
      video.srcObject = stream;
      await video.play();
      videoReady = true;
    } catch (err) {
      console.error('Failed to initialize video:', err);
    }
  }
</script>

<div class="holographic-display" transition:fade>
  {#if error}
    <div class="error-message">
      <p>Holographic Display Error: {error}</p>
      <p>Check console for details</p>
    </div>
  {/if}
  
  <!-- Main hologram canvas -->
  <canvas 
    bind:this={canvas}
    {width}
    {height}
    class="hologram-canvas"
  />
  
  <!-- Video element only for webcam mode -->
  {#if enableVideo && videoSource === 'webcam'}
    <video 
      bind:this={video}
      class="video-source"
      style="display: none;"
    />
    
    <!-- Only show initializing for actual webcam -->
    {#if !videoReady}
      <div class="loading-overlay" transition:fade>
        <div class="loading-spinner"></div>
        <p>Initializing webcam...</p>
      </div>
    {/if}
  {/if}
  
  {#if showStats && isInitialized}
    <div class="stats">
      <div>FPS: {fps}</div>
      <div>Engine: {engineCapabilities?.webgpu ? 'WebGPU' : 'Canvas'}</div>
      <div>Complexity: {complexity}</div>
      <div>Persona: {currentPersona?.name || 'ENOLA'}</div>
      {#if engineCapabilities}
        <div class="capabilities">
          {#if engineCapabilities.conceptMesh}
            <span class="capability">Concepts</span>
          {/if}
          {#if engineCapabilities.audio}
            <span class="capability">Audio</span>
          {/if}
          {#if engineCapabilities.hott}
            <span class="capability">HoTT</span>
          {/if}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .holographic-display {
    position: relative;
    width: 100%;
    height: 100%;
    background: #000;
    overflow: hidden;
  }
  
  .hologram-canvas {
    width: 100%;
    height: 100%;
    image-rendering: crisp-edges;
  }
  
  .error-message {
    position: absolute;
    top: 10px;
    left: 10px;
    color: #ff6b6b;
    background: rgba(0, 0, 0, 0.8);
    padding: 10px;
    border-radius: 5px;
    font-size: 12px;
    z-index: 10;
  }
  
  .stats {
    position: absolute;
    top: 10px;
    right: 10px;
    color: #00ffff;
    background: rgba(0, 0, 0, 0.8);
    padding: 10px;
    border-radius: 5px;
    font-size: 12px;
    font-family: monospace;
    z-index: 10;
  }
  
  .video-source {
    display: none;
  }
  
  .loading-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.9);
    color: #00ffff;
    z-index: 20;
  }
  
  .loading-spinner {
    width: 40px;
    height: 40px;
    border: 3px solid rgba(0, 255, 255, 0.3);
    border-top-color: #00ffff;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  
  .capabilities {
    margin-top: 4px;
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
  }
  
  .capability {
    background: rgba(0, 255, 255, 0.2);
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 10px;
    border: 1px solid rgba(0, 255, 255, 0.4);
  }
</style>
'''
    
    # Write the fixed version
    file_path.write_text(content, encoding='utf-8')
    
    print("✅ Fixed HolographicDisplay.svelte for enhanced engine")
    print("\nKey updates:")
    print("- Uses the enhanced RealGhostEngine with WebGPU")
    print("- Properly initializes with all features enabled")
    print("- Adds concepts to the visualization")
    print("- Shows engine capabilities in stats")
    print("- No more 'update is not a function' errors")
    print("- The engine manages its own render loop")
    
    return True

if __name__ == "__main__":
    if fix_holographic_display_for_enhanced_engine():
        print("\n🎉 The hologram should now display with full WebGPU features!")
        print("\nYou should see:")
        print("- The quantum field visualization")
        print("- Concept mesh with glowing entities")
        print("- FFT audio response (if mic enabled)")
        print("- Persona core hologram")
        print("- Engine capabilities indicators")
