#!/usr/bin/env python3
"""Fix hologram auto-start and display issues"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_hologram_autostart():
    file_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\routes\+page.svelte")
    
    print("🔧 Fixing hologram auto-start...")
    
    if not file_path.exists():
        print("❌ File not found!")
        return False
    
    # Read the file
    content = file_path.read_text(encoding='utf-8')
    
    # Find the onMount function
    onmount_start = content.find('onMount(() => {')
    if onmount_start == -1:
        print("❌ Could not find onMount function")
        return False
    
    # Find where to insert the auto-start code
    # Look for the connectAvatarWebSocket call
    avatar_ws_pos = content.find('connectAvatarWebSocket();', onmount_start)
    if avatar_ws_pos == -1:
        print("❌ Could not find connectAvatarWebSocket call")
        return False
    
    # Find the end of that line
    insert_pos = content.find('\n', avatar_ws_pos) + 1
    
    # Auto-start code to insert
    auto_start_code = '''      
      // 🌟 AUTO-START HOLOGRAM WITH ENOLA
      setTimeout(() => {
        console.log('🌟 Auto-starting hologram visualization with Enola...');
        
        // Set Enola as active persona
        const enolaPersona = availablePersonas.find(p => p.id === 'enola');
        if (enolaPersona) {
          currentPersona = enolaPersona;
          if (browser) {
            localStorage.setItem('tori-current-persona', JSON.stringify(enolaPersona));
          }
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
'''
    
    # Insert the auto-start code
    new_content = content[:insert_pos] + auto_start_code + content[insert_pos:]
    
    # Write back
    file_path.write_text(new_content, encoding='utf-8')
    
    print("✅ Added hologram auto-start code")
    return True

if __name__ == "__main__":
    if fix_hologram_autostart():
        print("\n🎉 Hologram will now auto-start with Enola!")
        print("\nRestart the launcher to see the hologram:")
        print("1. Press Ctrl+C to stop")
        print("2. Run: poetry run python enhanced_launcher.py")
