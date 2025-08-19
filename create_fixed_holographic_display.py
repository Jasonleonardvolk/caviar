#!/usr/bin/env python3
"""Create a proper HolographicDisplay.svelte with all fixes"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def create_fixed_holographic_display():
    file_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\components\HolographicDisplay.svelte")
    
    print("🔧 Creating properly fixed HolographicDisplay.svelte...")
    
    # Complete fixed version with all requirements
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
  
  // Subscribe to persona changes
  $: if (ghostEngine && $ghostPersona && $ghostPersona.id !== currentPersona?.id) {
    console.log('🔄 Switching hologram to:', $ghostPersona.name);
    currentPersona = $ghostPersona;
    ghostEngine.switchPersona($ghostPersona);
  }
  
  onMount(async () => {
    try {
      // Get current persona from store
      const initialPersona = get(ghostPersona);
      currentPersona = initialPersona;
      
      console.log('🎭 Initializing hologram with persona:', initialPersona.name);
      
      // Create Ghost Engine instance
      ghostEngine = new RealGhostEngine({ 
        persona: initialPersona.id || initialPersona.name || 'ENOLA' 
      });
      
      // Initialize Ghost Engine with canvas
      await ghostEngine.initialize(canvas, {
        enableSync: true,
        targetFPS: 60
      });
      
      console.log('✅ Ghost Engine initialized successfully!');
      isInitialized = true;
      
      // Auto-start hologram immediately (no video dependency)
      startHologram();
      
    } catch (err) {
      console.error('Failed to initialize holographic display:', err);
      error = err.message;
      // Fallback to simple canvas animation
      startFallbackAnimation();
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
      ghostEngine.stop();
      ghostEngine.dispose();
    }
  }
  
  async function startHologram() {
    if (!ghostEngine || !isInitialized) return;
    
    console.log('🌟 Starting hologram visualization');
    
    // Add holographic objects
    setupHolographicScene();
    
    // Start engine
    ghostEngine.start();
    
    // Start render loop
    startRenderLoop();
  }
  
  function setupHolographicScene() {
    if (!ghostEngine) return;
    
    // Add central holographic core
    ghostEngine.addHolographicObject({
      id: 'core',
      type: 'sphere',
      position: { x: 0, y: 0, z: 0 },
      radius: 1,
      material: {
        type: 'holographic',
        color: currentPersona?.color || '#00ffff',
        opacity: 0.8,
        refractiveIndex: 1.5
      }
    });
    
    // Add particle field
    ghostEngine.addHolographicObject({
      id: 'particles',
      type: 'particleField',
      count: 1000,
      bounds: { x: 5, y: 5, z: 5 },
      behavior: 'orbital'
    });
  }
  
  function startRenderLoop() {
    let lastTime = performance.now();
    let frameCount = 0;
    let fpsTime = 0;
    
    function render() {
      const currentTime = performance.now();
      const deltaTime = currentTime - lastTime;
      lastTime = currentTime;
      
      // Update FPS
      frameCount++;
      fpsTime += deltaTime;
      if (fpsTime >= 1000) {
        fps = Math.round(frameCount * 1000 / fpsTime);
        frameCount = 0;
        fpsTime = 0;
      }
      
      // Update time
      time += deltaTime * 0.001;
      
      // Update hologram
      if (ghostEngine && isInitialized) {
        ghostEngine.update(deltaTime);
        ghostEngine.render();
      }
      
      animationFrame = requestAnimationFrame(render);
    }
    
    render();
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
  
  function startFallbackAnimation() {
    console.warn('Starting fallback 2D animation');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const animate = () => {
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, width, height);
      
      // Simple holographic effect
      ctx.strokeStyle = '#00ffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(width/2, height/2, 50 + Math.sin(time * 0.001) * 20, 0, Math.PI * 2);
      ctx.stroke();
      
      time += 16;
      animationFrame = requestAnimationFrame(animate);
    };
    
    animate();
  }
</script>

<div class="holographic-display" transition:fade>
  {#if error}
    <div class="error-message">
      <p>Holographic Display Error: {error}</p>
      <p>Running in fallback mode</p>
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
  
  {#if showStats}
    <div class="stats">
      <div>FPS: {fps}</div>
      <div>Complexity: {complexity}</div>
      <div>Persona: {currentPersona?.name || 'ENOLA'}</div>
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
</style>
'''
    
    # Write the fixed version
    file_path.write_text(content, encoding='utf-8')
    
    print("✅ Created fixed HolographicDisplay.svelte")
    print("\nKey fixes applied:")
    print("1. ✅ 'Initializing video...' only shows for webcam mode")
    print("2. ✅ Hologram starts immediately without video dependency")
    print("3. ✅ Auto-start hologram in onMount")
    print("4. ✅ Video initialization is separate and optional")
    
    return True

if __name__ == "__main__":
    if create_fixed_holographic_display():
        print("\n🎉 HolographicDisplay.svelte is now properly fixed!")
        print("The hologram should display immediately without 'Initializing video...'")
