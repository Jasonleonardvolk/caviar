<script lang="ts">
  import { onMount } from 'svelte';
  import HologramRecorder from './HologramRecorder.svelte';
  
  let canvas: HTMLCanvasElement;
  let animationId: number;
  let preset = 'neon'; // Current visual preset
  
  // Visual presets for different looks
  const presets = {
    neon: {
      color1: '#00ffff',
      color2: '#ff00ff',
      speed: 0.02,
      intensity: 1.2
    },
    ghost: {
      color1: '#ffffff',
      color2: '#88ccff',
      speed: 0.01,
      intensity: 0.7
    },
    film: {
      color1: '#ff8800',
      color2: '#0088ff',
      speed: 0.015,
      intensity: 1.0
    }
  };
  
  onMount(() => {
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d')!;
    canvas.width = 1080;
    canvas.height = 1920;
    
    let time = 0;
    
    function animate() {
      const config = presets[preset];
      
      // Clear with fade effect
      ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw holographic effect
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      
      // Create gradient
      const gradient = ctx.createRadialGradient(
        centerX + Math.sin(time) * 100,
        centerY + Math.cos(time) * 100,
        0,
        centerX,
        centerY,
        400
      );
      
      gradient.addColorStop(0, config.color1);
      gradient.addColorStop(0.5, config.color2);
      gradient.addColorStop(1, 'transparent');
      
      // Draw multiple layers for depth
      for (let i = 0; i < 5; i++) {
        ctx.save();
        ctx.globalAlpha = config.intensity * (1 - i * 0.2);
        ctx.fillStyle = gradient;
        ctx.translate(centerX, centerY);
        ctx.rotate(time + i * 0.5);
        
        // Draw geometric shapes
        ctx.beginPath();
        for (let j = 0; j < 6; j++) {
          const angle = (j / 6) * Math.PI * 2;
          const radius = 200 + Math.sin(time * 2 + j) * 50;
          const x = Math.cos(angle) * radius;
          const y = Math.sin(angle) * radius;
          if (j === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      }
      
      // Add scan lines for holographic feel
      ctx.strokeStyle = config.color1;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.3;
      for (let y = 0; y < canvas.height; y += 4) {
        if (Math.sin(y * 0.01 + time * 10) > 0.5) {
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(canvas.width, y);
          ctx.stroke();
        }
      }
      
      // Add text overlay
      ctx.globalAlpha = 1;
      ctx.fillStyle = config.color1;
      ctx.font = 'bold 48px Inter, Arial, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('iRis Hologram Studio', centerX, 100);
      
      ctx.font = '24px Inter, Arial, sans-serif';
      ctx.fillStyle = config.color2;
      ctx.fillText('Physics-Native • WebGPU Powered', centerX, 140);
      
      time += config.speed;
      animationId = requestAnimationFrame(animate);
    }
    
    animate();
    
    return () => {
      if (animationId) cancelAnimationFrame(animationId);
    };
  });
  
  function switchPreset(newPreset: string) {
    preset = newPreset;
  }
</script>

<style>
  .container {
    width: 100vw;
    height: 100vh;
    background: #000;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    overflow: hidden;
  }
  
  canvas {
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
    display: block;
  }
  
  .preset-selector {
    position: fixed;
    top: 20px;
    left: 20px;
    z-index: 1000;
    display: flex;
    gap: 10px;
  }
  
  .preset-btn {
    padding: 8px 16px;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.3);
    color: white;
    border-radius: 8px;
    cursor: pointer;
    backdrop-filter: blur(10px);
    transition: all 0.2s;
  }
  
  .preset-btn:hover {
    background: rgba(255, 255, 255, 0.2);
  }
  
  .preset-btn.active {
    background: rgba(16, 185, 129, 0.5);
    border-color: #10b981;
  }
</style>

<div class="container">
  <canvas bind:this={canvas} id="holo-canvas"></canvas>
  
  <div class="preset-selector">
    <button class="preset-btn {preset === 'neon' ? 'active' : ''}" on:click={() => switchPreset('neon')}>
      Neon
    </button>
    <button class="preset-btn {preset === 'ghost' ? 'active' : ''}" on:click={() => switchPreset('ghost')}>
      Ghost Fade
    </button>
    <button class="preset-btn {preset === 'film' ? 'active' : ''}" on:click={() => switchPreset('film')}>
      Film Look
    </button>
  </div>
  
  <HologramRecorder sourceCanvasId="holo-canvas" />
</div>
