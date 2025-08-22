<script>
  import { onMount, onDestroy } from 'svelte';
  import { browser } from '$app/environment';
  
  export let persona = null;
  export let enabled = true;
  
  let canvas;
  let animationId;
  let particles = [];
  
  class Particle {
    constructor(x, y) {
      this.x = x;
      this.y = y;
      this.vx = (Math.random() - 0.5) * 2;
      this.vy = (Math.random() - 0.5) * 2;
      this.size = Math.random() * 3 + 1;
      this.life = 1.0;
      this.decay = Math.random() * 0.02 + 0.005;
    }
    
    update() {
      this.x += this.vx;
      this.y += this.vy;
      this.life -= this.decay;
      
      // Add some wave motion
      this.vx += Math.sin(Date.now() * 0.001) * 0.1;
      this.vy += Math.cos(Date.now() * 0.001) * 0.1;
    }
    
    draw(ctx, color) {
      ctx.globalAlpha = this.life;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  
  function animate() {
    if (!canvas || !enabled) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear with fade effect
    ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
    ctx.fillRect(0, 0, width, height);
    
    // Get persona color
    const color = persona?.color || { r: 0.1, g: 0.5, b: 1.0 };
    const rgbColor = `rgb(${color.r * 255}, ${color.g * 255}, ${color.b * 255})`;
    
    // Add new particles
    if (Math.random() < 0.1) {
      particles.push(new Particle(width / 2, height / 2));
    }
    
    // Update and draw particles
    particles = particles.filter(particle => {
      particle.update();
      
      if (particle.life > 0 && 
          particle.x > 0 && particle.x < width &&
          particle.y > 0 && particle.y < height) {
        particle.draw(ctx, rgbColor);
        return true;
      }
      return false;
    });
    
    // Draw center hologram effect
    const time = Date.now() * 0.001;
    ctx.globalAlpha = 0.5 + Math.sin(time) * 0.3;
    ctx.strokeStyle = rgbColor;
    ctx.lineWidth = 2;
    
    // Draw rotating rings
    for (let i = 0; i < 3; i++) {
      ctx.save();
      ctx.translate(width / 2, height / 2);
      ctx.rotate(time * (i + 1) * 0.5);
      ctx.beginPath();
      ctx.ellipse(0, 0, 50 + i * 20, 30 + i * 15, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
    
    // Draw persona name
    ctx.globalAlpha = 1;
    ctx.fillStyle = rgbColor;
    ctx.font = '18px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(persona?.name || 'TORI', width / 2, height - 20);
    
    animationId = requestAnimationFrame(animate);
  }
  
  onMount(() => {
    if (!browser) return;
    
    // Start animation
    animate();
    
    // Listen for events
    const handleToggle = (e) => {
      enabled = e.detail.enabled;
      if (enabled) animate();
    };
    
    const handleStart = (e) => {
      enabled = true;
      animate();
    };
    
    window.addEventListener('toggle-hologram-visualization', handleToggle);
    window.addEventListener('start-hologram-visualization', handleStart);
    
    return () => {
      window.removeEventListener('toggle-hologram-visualization', handleToggle);
      window.removeEventListener('start-hologram-visualization', handleStart);
    };
  });
  
  onDestroy(() => {
    if (animationId) {
      cancelAnimationFrame(animationId);
    }
  });
</script>

<div class="hologram-container" class:hidden={!enabled}>
  <canvas
    bind:this={canvas}
    width="300"
    height="300"
    class="hologram-canvas"
  />
</div>

<style>
  .hologram-container {
    width: 100%;
    height: 300px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.9);
    border-radius: 8px;
    position: relative;
    overflow: hidden;
  }
  
  .hologram-container.hidden {
    display: none;
  }
  
  .hologram-canvas {
    width: 100%;
    height: 100%;
    max-width: 300px;
    max-height: 300px;
  }
</style>
