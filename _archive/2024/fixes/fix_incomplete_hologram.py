#!/usr/bin/env python3
"""Fix the incomplete HolographicDisplay.svelte file"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_incomplete_file():
    file_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\components\HolographicDisplay.svelte")
    
    print("🔧 Fixing incomplete HolographicDisplay.svelte...")
    
    if not file_path.exists():
        print("❌ File not found!")
        return False
    
    # Read the current content
    content = file_path.read_text(encoding='utf-8')
    
    # Check if it's already complete
    if '</style>' in content:
        print("✅ File appears to be complete")
        return True
    
    # Add the missing closing parts
    missing_parts = '''">
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
</style>
'''
    
    # Append the missing parts
    content += missing_parts
    
    # Write back
    file_path.write_text(content, encoding='utf-8')
    
    print("✅ Fixed incomplete file")
    return True

if __name__ == "__main__":
    if fix_incomplete_file():
        print("\n🎉 HolographicDisplay.svelte is now complete!")
        print("The dev server should auto-reload.")
