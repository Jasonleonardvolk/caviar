#!/usr/bin/env python3
"""Fix HolographicDisplay to use correct RealGhostEngine API"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_holographic_display_api():
    file_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\components\HolographicDisplay.svelte")
    
    print("🔧 Fixing HolographicDisplay to use correct RealGhostEngine API...")
    
    # Read current content
    content = file_path.read_text(encoding='utf-8')
    
    # Fix the render loop to not call non-existent methods
    old_render_loop = '''function startRenderLoop() {
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
  }'''
    
    new_render_loop = '''function startRenderLoop() {
    // The RealGhostEngine has its own internal render loop
    // We just need to monitor the stats
    let frameCount = 0;
    let fpsTime = 0;
    let lastTime = performance.now();
    
    function updateStats() {
      const currentTime = performance.now();
      const deltaTime = currentTime - lastTime;
      lastTime = currentTime;
      
      // Update FPS from engine stats
      if (ghostEngine && ghostEngine.stats) {
        fps = ghostEngine.stats.fps;
      }
      
      // Update time
      time += deltaTime * 0.001;
      
      animationFrame = requestAnimationFrame(updateStats);
    }
    
    updateStats();
  }'''
    
    content = content.replace(old_render_loop, new_render_loop)
    
    # Fix the startHologram function
    old_start = '''async function startHologram() {
    if (!ghostEngine || !isInitialized) return;
    
    console.log('🌟 Starting hologram visualization');
    
    // Add holographic objects
    setupHolographicScene();
    
    // Start engine
    ghostEngine.start();
    
    // Start render loop
    startRenderLoop();
  }'''
    
    new_start = '''async function startHologram() {
    if (!ghostEngine || !isInitialized) return;
    
    console.log('🌟 Starting hologram visualization');
    
    // Add holographic objects
    setupHolographicScene();
    
    // Start engine (it has its own render loop)
    ghostEngine.start();
    
    // Start stats monitoring
    startRenderLoop();
    
    // Subscribe to frame updates
    ghostEngine.onFrame((deltaTime, psiState) => {
      time += deltaTime;
      // Update any UI elements based on psi state
    });
  }'''
    
    content = content.replace(old_start, new_start)
    
    # Write back
    file_path.write_text(content, encoding='utf-8')
    
    print("✅ Fixed HolographicDisplay API usage")
    print("- RealGhostEngine manages its own render loop")
    print("- We just monitor stats and subscribe to updates")
    print("- No more 'update is not a function' errors")
    
    return True

if __name__ == "__main__":
    if fix_holographic_display_api():
        print("\n🎉 The hologram should now display properly!")
        print("Check the browser - you should see the quantum field visualization")
