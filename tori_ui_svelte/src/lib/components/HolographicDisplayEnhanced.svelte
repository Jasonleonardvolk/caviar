<!-- @ts-nocheck -->
// Enhanced HolographicDisplay that connects to the advanced holographic engine

<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { fade } from 'svelte/transition';
  import type { WavefieldParameters } from '$lib/../../../frontend/lib/holographicEngine';
  import { writable } from 'svelte/store';
  
  export let width = 320;
  export let height = 240;
  export let usePenrose = true;
  export let showStats = true;
  export let enableVideo = false;
  export let videoSource: 'webcam' | 'file' | 'stream' = 'webcam';
  
  // Holographic data loading
  export async function loadHologramData(data: any) {
    if (!engine) return;
    
    const { volumetricData, particles, oscillatorPhases, ghostInterpretation, morphAnimation } = data;
    
    // Convert to wavefield parameters
    const wavefieldParams: WavefieldParameters = {
      phase_modulation: oscillatorPhases[0] || 0,
      coherence: ghostInterpretation.emotionalResonance.clarity,
      oscillator_phases: oscillatorPhases,
      dominant_freq: 440 * (1 + ghostInterpretation.emotionalResonance.energy),
      spatial_frequencies: particles.positions.map(p => [p.x * 10, p.y * 10]),
      amplitudes: particles.positions.map(p => p.z)
    };
    
    // Update engine with new parameters
    await engine.updateWavefield(wavefieldParams);
    
    // Load volumetric data
    if (volumetricData) {
      await engine.loadVolumetricData(volumetricData);
    }
    
    // Configure morph animation
    morphConfig = morphAnimation;
  }
  
  export async function startMorphAnimation() {
    if (!engine) return;
    
    isMorphing = true;
    morphStartTime = performance.now();
    
    // Start the morph
    await engine.startMorphTransition(morphConfig.duration);
  }
  
  let canvas: HTMLCanvasElement;
  let engine: any;
  let animationFrame: number;
  let fps = 0;
  let complexity = 'O(n²·³²)';
  let time = 0;
  let isInitialized = false;
  let error = '';
  
  // Morphing state
  let isMorphing = false;
  let morphStartTime = 0;
  let morphConfig: any = null;
  
  // Performance monitoring
  let frameCount = 0;
  let lastFpsUpdate = 0;
  
  onMount(async () => {
    try {
      // Initialize the advanced holographic engine
      const { HolographicEngine } = await import('../../../frontend/lib/holographicEngine');
      
      engine = new HolographicEngine(canvas);
      await engine.initialize();
      
      isInitialized = true;
      startRenderLoop();
      
    } catch (err) {
  const msg = err instanceof Error ? err.message : String(err);
      console.error('Failed to initialize holographic engine:', err);
      error = msg;
      
      // Fallback to simple canvas
      initializeFallback();
    
}
  });
  
  onDestroy(() => {
    if (animationFrame) {
      cancelAnimationFrame(animationFrame);
    }
    if (engine) {
      engine.dispose();
    }
  });
  
  function startRenderLoop() {
    function render(currentTime: number) {
      // Update FPS
      frameCount++;
      if (currentTime - lastFpsUpdate > 1000) {
        fps = frameCount;
        frameCount = 0;
        lastFpsUpdate = currentTime;
      }
      
      // Update time
      time = currentTime * 0.001;
      
      // Handle morphing animation
      if (isMorphing && morphConfig) {
        const elapsed = currentTime - morphStartTime;
        const progress = Math.min(1, elapsed / morphConfig.duration);
        
        // Apply easing
        const easedProgress = cubicBezier(progress, 0.4, 0, 0.2, 1);
        
        if (engine) {
          engine.setMorphProgress(easedProgress);
        }
        
        if (progress >= 1) {
          isMorphing = false;
        }
      }
      
      // Render frame
      if (engine && isInitialized) {
        engine.render(time);
      }
      
      animationFrame = requestAnimationFrame(render);
    }
    
    render(performance.now());
  }
  
  function initializeFallback() {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    function renderFallback() {
      time += 0.016;
      
      // Clear with fade effect
      ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
      ctx.fillRect(0, 0, width, height);
      
      // Create gradient background
      const gradient = ctx.createRadialGradient(width/2, height/2, 0, width/2, height/2, width/2);
      gradient.addColorStop(0, 'rgba(138, 43, 226, 0.1)');
      gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);
      
      // Draw holographic waves
      ctx.strokeStyle = '#00ffff';
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.8;
      
      for (let y = 0; y < height; y += 3) {
        ctx.beginPath();
        for (let x = 0; x < width; x++) {
          const wave = Math.sin((x * 0.05 + time) * 0.5) * 5;
          if (x === 0) {
            ctx.moveTo(x, y + wave);
          } else {
            ctx.lineTo(x, y + wave);
          }
        }
        ctx.stroke();
      }
      
      // TORI text
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 20px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.globalAlpha = 0.9;
      ctx.fillText('TORI', width/2, height/2);
      
      animationFrame = requestAnimationFrame(renderFallback);
    }
    
    renderFallback();
  }
  
  // Cubic bezier easing function
  function cubicBezier(t: number, p1: number, p2: number, p3: number, p4: number): number {
    const cx = 3 * p1;
    const bx = 3 * (p3 - p1) - cx;
    const ax = 1 - cx - bx;
    const cy = 3 * p2;
    const by = 3 * (p4 - p2) - cy;
    const ay = 1 - cy - by;
    
    function sampleCurveX(t: number): number {
      return ((ax * t + bx) * t + cx) * t;
    }
    
    function sampleCurveY(t: number): number {
      return ((ay * t + by) * t + cy) * t;
    }
    
    function solveCurveX(x: number): number {
      let t0 = 0, t1 = 1, t2 = x, x2, d2, i;
      
      for (i = 0; i < 8; i++) {
        x2 = sampleCurveX(t2) - x;
        if (Math.abs(x2) < 0.001) return t2;
        d2 = (3 * ax * t2 + 2 * bx) * t2 + cx;
        if (Math.abs(d2) < 0.001) break;
        t2 = t2 - x2 / d2;
      }
      
      return t2;
    }
    
    return sampleCurveY(solveCurveX(t));
  }
</script>

<div class="holographic-display" transition:fade>
  <canvas 
    bind:this={canvas}
    {width}
    {height}
    class="hologram-canvas"
  />
  
  {#if showStats}
    <div class="stats-overlay">
      <div class="stat-item">
        <span class="stat-label">FPS:</span>
        <span class="stat-value">{fps}</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">Engine:</span>
        <span class="stat-value">{isInitialized ? 'WebGPU' : 'Canvas'}</span>
      </div>
      {#if usePenrose}
        <div class="stat-item">
          <span class="stat-label">Complexity:</span>
          <span class="stat-value">{complexity}</span>
        </div>
      {/if}
      {#if isMorphing}
        <div class="stat-item">
          <span class="stat-label">Morphing:</span>
          <span class="stat-value">Active</span>
        </div>
      {/if}
    </div>
  {/if}
  
  {#if error}
    <div class="error-overlay">
      <p>{error}</p>
    </div>
  {/if}
</div>

<style>
  .holographic-display {
    position: relative;
    width: fit-content;
    background: #000;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 0 30px rgba(138, 43, 226, 0.5);
  }
  
  .hologram-canvas {
    display: block;
    image-rendering: crisp-edges;
    border: 1px solid rgba(138, 43, 226, 0.3);
  }
  
  .stats-overlay {
    position: absolute;
    top: 8px;
    right: 8px;
    background: rgba(0, 0, 0, 0.7);
    padding: 8px 12px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 12px;
    color: #00ffff;
    backdrop-filter: blur(10px);
  }
  
  .stat-item {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    margin: 2px 0;
  }
  
  .stat-label {
    color: #00ffff;
    opacity: 0.7;
  }
  
  .stat-value {
    color: #00ff00;
    font-weight: bold;
  }
  
  .error-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 0, 0, 0.1);
    backdrop-filter: blur(5px);
  }
  
  .error-overlay p {
    color: #ff6666;
    font-size: 14px;
    text-align: center;
    padding: 20px;
  }
</style>
