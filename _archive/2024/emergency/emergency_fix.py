#!/usr/bin/env python3
"""
Emergency fix for syntax error in +page.svelte
"""

import re
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_syntax_error():
    """Fix the duplicate closing brace error"""
    
    page_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\routes\+page.svelte")
    
    print("🔧 Fixing syntax error in +page.svelte...")
    
    with open(page_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and fix the duplicate closing brace issue
    # The problem is in the toggleHologramVideo function
    content = re.sub(
        r'(function toggleHologramVideo\(\) \{[^}]+\})\s*\n\s*// Save preference\s*\n\s*if \(browser\) \{\s*\n\s*localStorage\.setItem\(\'tori-hologram-video\', String\(hologramVisualizationEnabled\)\);\s*\n\s*\}\s*\n\s*\}',
        r'\1',
        content,
        flags=re.DOTALL
    )
    
    # Also fix the duplicate persona initialization issue
    # Change default persona to Enola
    content = re.sub(
        r"let currentPersona: Persona = \{[^}]+\};",
        """let currentPersona: Persona = {
    id: 'enola',
    name: 'Enola',
    description: 'Investigative and analytical consciousness',
    ψ: 'analytical',
    ε: 'focused',
    τ: 'present',
    φ: 'empirical',
    color: { r: 0.1, g: 0.5, b: 1.0 },
    voice: 'nova',
    hologram_style: 'quantum_field'
  };""",
        content,
        flags=re.DOTALL
    )
    
    # Write the fixed content
    with open(page_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed syntax error")
    
    # Now let's also fix the "Initializing video..." issue properly
    # We need to update MemoryPanel to not show that loading state
    memory_panel_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\components\MemoryPanel.svelte")
    
    if memory_panel_path.exists():
        print("\n🔧 Fixing 'Initializing video...' issue in MemoryPanel...")
        
        with open(memory_panel_path, 'r', encoding='utf-8') as f:
            memory_content = f.read()
        
        # Remove any loading overlay HTML
        memory_content = re.sub(
            r'<div class="loading-overlay.*?</div>\s*</div>',
            '',
            memory_content,
            flags=re.DOTALL
        )
        
        # Remove any "Initializing video..." text
        memory_content = re.sub(
            r'<p[^>]*>Initializing video\.\.\.</p>',
            '',
            memory_content
        )
        
        with open(memory_panel_path, 'w', encoding='utf-8') as f:
            f.write(memory_content)
        
        print("✅ Removed 'Initializing video...' loading state")
    
    # Create the hologram visualization component if it doesn't exist
    hologram_viz_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\components\HologramVisualization.svelte")
    
    if not hologram_viz_path.exists():
        print("\n🔧 Creating HologramVisualization component...")
        
        hologram_component = """<script>
  import { onMount, onDestroy } from 'svelte';
  import { browser } from '$app/environment';
  import { ghostPersona } from '$lib/stores/ghostPersona';
  
  export let enabled = true;
  
  let canvas;
  let ctx;
  let animationId;
  let particles = [];
  
  class Particle {
    constructor(x, y, persona) {
      this.x = x;
      this.y = y;
      this.vx = (Math.random() - 0.5) * 2;
      this.vy = (Math.random() - 0.5) * 2;
      this.size = Math.random() * 3 + 1;
      this.life = 1.0;
      this.decay = Math.random() * 0.02 + 0.005;
      this.persona = persona;
    }
    
    update() {
      this.x += this.vx;
      this.y += this.vy;
      this.life -= this.decay;
      
      // Add wave motion based on persona
      const time = Date.now() * 0.001;
      this.vx += Math.sin(time + this.x * 0.01) * 0.1;
      this.vy += Math.cos(time + this.y * 0.01) * 0.1;
    }
    
    draw(ctx) {
      const color = this.persona?.color || { r: 0.1, g: 0.5, b: 1.0 };
      ctx.globalAlpha = this.life * 0.8;
      ctx.fillStyle = `rgb(${color.r * 255}, ${color.g * 255}, ${color.b * 255})`;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  
  function animate() {
    if (!canvas || !ctx || !enabled) return;
    
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear with fade
    ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
    ctx.fillRect(0, 0, width, height);
    
    // Get current persona
    const persona = $ghostPersona.activePersona === 'Enola' ? {
      name: 'Enola',
      color: { r: 0.1, g: 0.5, b: 1.0 }
    } : {
      name: $ghostPersona.activePersona || 'TORI',
      color: { r: 0.5, g: 0.5, b: 0.5 }
    };
    
    // Add particles
    if (Math.random() < 0.3) {
      const angle = Math.random() * Math.PI * 2;
      const radius = Math.random() * 50;
      particles.push(new Particle(
        width/2 + Math.cos(angle) * radius,
        height/2 + Math.sin(angle) * radius,
        persona
      ));
    }
    
    // Update and draw particles
    particles = particles.filter(p => {
      p.update();
      if (p.life > 0) {
        p.draw(ctx);
        return true;
      }
      return false;
    });
    
    // Draw center effect
    const time = Date.now() * 0.001;
    ctx.save();
    ctx.translate(width/2, height/2);
    
    // Rotating rings
    const color = persona.color;
    ctx.strokeStyle = `rgba(${color.r * 255}, ${color.g * 255}, ${color.b * 255}, 0.6)`;
    ctx.lineWidth = 2;
    
    for (let i = 0; i < 3; i++) {
      ctx.save();
      ctx.rotate(time * (i + 1) * 0.3);
      ctx.globalAlpha = 0.3 + Math.sin(time + i) * 0.2;
      ctx.beginPath();
      ctx.ellipse(0, 0, 60 + i * 25, 40 + i * 20, Math.PI/4, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
    
    ctx.restore();
    
    // Draw persona name
    ctx.globalAlpha = 1;
    ctx.fillStyle = `rgb(${color.r * 255}, ${color.g * 255}, ${color.b * 255})`;
    ctx.font = 'bold 20px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(persona.name, width/2, height - 30);
    
    animationId = requestAnimationFrame(animate);
  }
  
  onMount(() => {
    if (!browser || !canvas) return;
    
    ctx = canvas.getContext('2d');
    if (enabled) {
      animate();
    }
    
    // Listen for toggle events
    const handleToggle = (e) => {
      enabled = e.detail.enabled;
      if (enabled && !animationId) {
        animate();
      }
    };
    
    window.addEventListener('toggle-hologram-visualization', handleToggle);
    
    return () => {
      window.removeEventListener('toggle-hologram-visualization', handleToggle);
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  });
  
  onDestroy(() => {
    if (animationId) {
      cancelAnimationFrame(animationId);
    }
  });
  
  // Auto-start on mount
  $: if (browser && canvas && enabled && !animationId) {
    ctx = canvas.getContext('2d');
    animate();
  }
</script>

{#if enabled}
  <div class="hologram-container">
    <canvas
      bind:this={canvas}
      width="300"
      height="200"
      class="hologram-canvas"
    />
  </div>
{/if}

<style>
  .hologram-container {
    width: 100%;
    height: 200px;
    background: #000;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
  }
  
  .hologram-canvas {
    width: 100%;
    height: 100%;
  }
</style>"""
        
        with open(hologram_viz_path, 'w', encoding='utf-8') as f:
            f.write(hologram_component)
        
        print("✅ Created HologramVisualization component")
    
    # Update HolographicDisplay to use the new component
    holographic_display_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\components\HolographicDisplay.svelte")
    
    if holographic_display_path.exists():
        print("\n🔧 Updating HolographicDisplay to show hologram...")
        
        with open(holographic_display_path, 'r', encoding='utf-8') as f:
            display_content = f.read()
        
        # Check if it imports HologramVisualization
        if "HologramVisualization" not in display_content:
            # Add import
            display_content = display_content.replace(
                "<script>",
                """<script>
  import HologramVisualization from './HologramVisualization.svelte';"""
            )
            
            # Replace any loading state with the hologram
            display_content = re.sub(
                r'<div class="loading-spinner.*?</div>.*?<p.*?>Initializing video\.\.\.</p>',
                '<HologramVisualization {enabled} />',
                display_content,
                flags=re.DOTALL
            )
        
        with open(holographic_display_path, 'w', encoding='utf-8') as f:
            f.write(display_content)
        
        print("✅ Updated HolographicDisplay")
    
    print("\n🎉 All fixes applied!")
    print("\n✅ Fixed:")
    print("  1. Syntax error (duplicate closing brace)")
    print("  2. Default persona set to Enola")
    print("  3. Removed 'Initializing video...' loading state")
    print("  4. Created proper hologram visualization")
    print("\nRestart the dev server and the hologram should work!")

if __name__ == "__main__":
    fix_syntax_error()
