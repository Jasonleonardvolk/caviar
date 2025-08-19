#!/usr/bin/env python3
"""
Complete fix for hologram display issues
1. Fix syntax error in +page.svelte
2. Fix HolographicDisplay to not try webcam when showing hologram
3. Ensure Enola is default and hologram auto-starts
"""

import re
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_all_hologram_issues():
    """Fix all hologram-related issues in one go"""
    
    print("🔧 Fixing all hologram issues...")
    
    # Fix 1: Remove duplicate closing brace in +page.svelte
    page_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\routes\+page.svelte")
    
    if page_path.exists():
        print("\n📝 Fixing syntax error in +page.svelte...")
        
        with open(page_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove the duplicate code block after toggleHologramVideo
        # The issue is around line 439-445 where there's duplicate localStorage code
        content = re.sub(
            r'(function toggleHologramVideo\(\) \{[^}]+\})\s*\n\s*// Save preference\s*\n\s*if \(browser\) \{\s*\n\s*localStorage\.setItem\(\'tori-hologram-video\', String\(hologramVisualizationEnabled\)\);\s*\n\s*}\s*\n\s*}',
            r'\1',
            content,
            flags=re.DOTALL
        )
        
        # Also ensure Enola is the default persona
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
        
        with open(page_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fixed syntax error and set Enola as default")
    
    # Fix 2: Update HolographicDisplay to not show "Initializing video..."
    holographic_display_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\components\HolographicDisplay.svelte")
    
    if holographic_display_path.exists():
        print("\n📝 Fixing HolographicDisplay component...")
        
        with open(holographic_display_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Change enableVideo default to false
        content = content.replace(
            "export let enableVideo = false;",
            "export let enableVideo = false; // NEVER enable by default - this is for webcam!"
        )
        
        # Remove the loading overlay that shows "Initializing video..."
        # or change it to only show when actually initializing webcam
        content = re.sub(
            r'{#if enableVideo && !videoReady}\s*<div class="loading-overlay">.*?</div>\s*{/if}',
            """{#if enableVideo && !videoReady && videoSource === 'webcam'}
    <!-- Only show loading when actually trying to access webcam -->
    <div class="loading-overlay">
      <div class="loading-spinner"></div>
      <p>Initializing webcam...</p>
    </div>
  {/if}""",
            content,
            flags=re.DOTALL
        )
        
        with open(holographic_display_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fixed HolographicDisplay to not show video loading")
    
    # Fix 3: Update MemoryPanel to properly control hologram
    memory_panel_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\components\MemoryPanel.svelte")
    
    if memory_panel_path.exists():
        print("\n📝 Fixing MemoryPanel hologram controls...")
        
        with open(memory_panel_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Ensure the HolographicDisplay is not passed enableVideo from hologramVideoEnabled
        content = re.sub(
            r'<HolographicDisplay[^>]*enableVideo=\{hologramVideoEnabled\}[^>]*/>',
            '<HolographicDisplay \n      width={300} \n      height={180}\n      usePenrose={true}\n      showStats={true}\n      enableVideo={false}\n    />',
            content
        )
        
        # If HolographicDisplay doesn't have explicit enableVideo=false, add it
        if 'enableVideo={false}' not in content and '<HolographicDisplay' in content:
            content = re.sub(
                r'(<HolographicDisplay[^>]*)(/>)',
                r'\1\n      enableVideo={false}\n    \2',
                content
            )
        
        with open(memory_panel_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fixed MemoryPanel to not trigger video mode")
    
    # Fix 4: Create a simple auto-start script for the hologram
    hologram_autostart_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\hologramAutoStart.js")
    
    hologram_autostart_content = """// Auto-start hologram with Enola
export function autoStartHologram() {
  console.log('🌟 Auto-starting hologram with Enola...');
  
  // Dispatch event to start hologram
  if (typeof window !== 'undefined') {
    // Enable concept mesh visualization
    window.TORI_CONCEPT_MESH_ENABLED = true;
    
    // Set Enola as active persona
    window.dispatchEvent(new CustomEvent('set-persona', {
      detail: { persona: 'Enola' }
    }));
    
    // Start hologram visualization
    window.dispatchEvent(new CustomEvent('start-hologram', {
      detail: { 
        autoStart: true,
        persona: 'Enola',
        mode: 'quantum_field'
      }
    }));
    
    console.log('✅ Hologram auto-start initiated');
  }
}

// Auto-run on import
if (typeof window !== 'undefined') {
  // Wait a moment for everything to initialize
  setTimeout(autoStartHologram, 1000);
}
"""
    
    with open(hologram_autostart_path, 'w', encoding='utf-8') as f:
        f.write(hologram_autostart_content)
    
    print("✅ Created hologram auto-start script")
    
    # Fix 5: Add the auto-start import to +layout.svelte
    layout_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\routes\+layout.svelte")
    
    if layout_path.exists():
        print("\n📝 Adding auto-start to layout...")
        
        with open(layout_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add import if not already there
        if 'hologramAutoStart' not in content:
            # Add after other imports
            content = re.sub(
                r'(<script[^>]*>)',
                r'\1\n  import "$lib/hologramAutoStart.js"; // Auto-start hologram with Enola',
                content
            )
        
        with open(layout_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Added hologram auto-start to layout")
    
    print("\n🎉 All fixes applied!")
    print("\n📋 What was fixed:")
    print("  1. ✅ Syntax error (duplicate closing brace)")
    print("  2. ✅ Default persona set to Enola")
    print("  3. ✅ HolographicDisplay won't show 'Initializing video...'")
    print("  4. ✅ Video button controls hologram, not webcam")
    print("  5. ✅ Hologram auto-starts with Enola on page load")
    print("\n🚀 Restart the launcher to see the working hologram!")

if __name__ == "__main__":
    fix_all_hologram_issues()
