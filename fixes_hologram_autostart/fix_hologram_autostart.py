#!/usr/bin/env python3
"""
Fix hologram auto-start with Enola persona
This fixes the issue where the Video button tries to use webcam instead of showing hologram
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_hologram_autostart():
    """Fix the hologram to auto-start with Enola instead of trying to use webcam"""
    
    # Paths
    project_root = Path(r"{PROJECT_ROOT}")
    page_svelte = project_root / "tori_ui_svelte" / "src" / "routes" / "+page.svelte"
    backup_dir = project_root / "backups" / f"hologram_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("🔧 Fixing hologram auto-start with Enola...")
    
    # Create backup
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    if page_svelte.exists():
        shutil.copy2(page_svelte, backup_dir / "+page.svelte.backup")
        print(f"✅ Backed up +page.svelte")
    
    # Read the current page content
    with open(page_svelte, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Change default persona to Enola
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
    
    # Fix 2: Change hologramVideoEnabled to true by default and rename it
    content = content.replace(
        "let hologramVideoEnabled = false; // Hologram shows visual content",
        "let hologramVisualizationEnabled = true; // Hologram shows quantum visualization (NOT webcam)"
    )
    
    # Fix 3: Add Enola to available personas if not there
    if "'enola'" not in content:
        # Find availablePersonas array and add Enola
        personas_match = re.search(r"let availablePersonas: Persona\[\] = \[(.*?)\];", content, re.DOTALL)
        if personas_match:
            personas_content = personas_match.group(1)
            # Add Enola as first persona
            enola_persona = """{
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
    },
    """
            new_personas = enola_persona + personas_content
            content = content.replace(
                f"let availablePersonas: Persona[] = [{personas_content}];",
                f"let availablePersonas: Persona[] = [{new_personas}];"
            )
    
    # Fix 4: Update the toggleHologramVideo function to not try to access webcam
    content = re.sub(
        r"function toggleHologramVideo\(\) \{[^}]+\}",
        """function toggleHologramVideo() {
    hologramVisualizationEnabled = !hologramVisualizationEnabled;
    
    // Show/hide hologram visualization (NOT webcam!)
    if (browser) {
      // Dispatch event to show hologram
      window.dispatchEvent(new CustomEvent('toggle-hologram-visualization', {
        detail: { enabled: hologramVisualizationEnabled, persona: currentPersona }
      }));
      
      // Remove any webcam initialization attempts
      const loadingOverlays = document.querySelectorAll('.loading-overlay');
      loadingOverlays.forEach(el => el.remove());
    }
    
    // Save preference
    if (browser) {
      localStorage.setItem('tori-hologram-visualization', String(hologramVisualizationEnabled));
    }
  }""",
        content,
        flags=re.DOTALL
    )
    
    # Fix 5: Add auto-initialization in onMount
    # Find the onMount section
    mount_match = re.search(r"onMount\(\(\) => \{(.*?)// Poll for memory stats", content, re.DOTALL)
    if mount_match:
        mount_content = mount_match.group(1)
        
        # Add hologram auto-start code after avatar websocket connection
        hologram_init = """
      // 🌟 AUTO-START HOLOGRAM WITH ENOLA
      setTimeout(() => {
        console.log('🌟 Auto-starting hologram visualization with Enola...');
        
        // Set Enola as active persona
        const enolaPersona = availablePersonas.find(p => p.id === 'enola');
        if (enolaPersona) {
          switchPersona(enolaPersona);
        }
        
        // Enable hologram visualization
        hologramVisualizationEnabled = true;
        
        // Dispatch event to start hologram
        if (browser) {
          window.dispatchEvent(new CustomEvent('start-hologram-visualization', {
            detail: { 
              persona: currentPersona,
              autoStart: true,
              mode: 'quantum_field'
            }
          }));
          
          // Remove any loading overlays
          document.querySelectorAll('.loading-overlay, .loading-spinner').forEach(el => el.remove());
        }
        
        console.log('✅ Hologram auto-start complete');
      }, 1000); // Wait 1 second for everything to initialize
"""
        
        # Insert after avatar websocket connection
        new_mount_content = mount_content.replace(
            "connectAvatarWebSocket();",
            "connectAvatarWebSocket();" + hologram_init
        )
        
        content = content.replace(mount_content, new_mount_content)
    
    # Fix 6: Update all references from hologramVideoEnabled to hologramVisualizationEnabled
    content = content.replace("hologramVideoEnabled", "hologramVisualizationEnabled")
    
    # Fix 7: Load correct preference key
    content = content.replace(
        'const savedVideoPref = localStorage.getItem(\'tori-hologram-video\');',
        'const savedVideoPref = localStorage.getItem(\'tori-hologram-visualization\');'
    )
    
    # Write the fixed content
    with open(page_svelte, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed +page.svelte")
    
    # Now create a component to handle the hologram display in the left panel
    hologram_component_path = project_root / "tori_ui_svelte" / "src" / "lib" / "components" / "HologramVisualization.svelte"
    
    hologram_component = """<script>
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
"""
    
    with open(hologram_component_path, 'w', encoding='utf-8') as f:
        f.write(hologram_component)
    
    print("✅ Created HologramVisualization.svelte component")
    
    # Find and update the left panel component to include the hologram
    left_panel_search = list(project_root.rglob("**/LeftPanel*.svelte"))
    memory_panel_search = list(project_root.rglob("**/MemoryPanel*.svelte"))
    
    panel_to_update = None
    if memory_panel_search:
        panel_to_update = memory_panel_search[0]
    elif left_panel_search:
        panel_to_update = left_panel_search[0]
    
    if panel_to_update:
        print(f"📝 Updating {panel_to_update.name}...")
        
        with open(panel_to_update, 'r', encoding='utf-8') as f:
            panel_content = f.read()
        
        # Add import if not present
        if "HologramVisualization" not in panel_content:
            # Add import at the top of script section
            panel_content = panel_content.replace(
                "<script>",
                """<script>
  import HologramVisualization from './HologramVisualization.svelte';"""
            )
            
            # Add the component where the video initialization was happening
            panel_content = panel_content.replace(
                '<div class="loading-spinner"></div>',
                '<HologramVisualization {persona} {enabled} />'
            )
            
            panel_content = panel_content.replace(
                'Initializing video...',
                '<!-- Hologram visualization replaces video initialization -->'
            )
        
        with open(panel_to_update, 'w', encoding='utf-8') as f:
            f.write(panel_content)
        
        print(f"✅ Updated {panel_to_update.name}")
    
    print("\n🎉 Hologram auto-start fix complete!")
    print("📝 Changes made:")
    print("  1. Changed default persona to Enola")
    print("  2. Renamed hologramVideoEnabled to hologramVisualizationEnabled (default true)")
    print("  3. Added Enola to available personas")
    print("  4. Fixed toggleHologramVideo to show visualization, not webcam")
    print("  5. Added auto-start code in onMount")
    print("  6. Created HologramVisualization.svelte component")
    print("\n🚀 Restart the frontend to see the hologram auto-start with Enola!")

if __name__ == "__main__":
    fix_hologram_autostart()
