#!/usr/bin/env python3
"""Restore HolographicDisplay.svelte from backup or fix syntax"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def restore_holographic_display():
    file_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\components\HolographicDisplay.svelte")
    backup_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\components\HolographicDisplay.svelte.backup")
    
    print("🔧 Checking for backup file...")
    
    # First, try to restore from backup
    if backup_path.exists():
        print("✅ Found backup file, restoring...")
        content = backup_path.read_text(encoding='utf-8')
        file_path.write_text(content, encoding='utf-8')
        print("✅ Restored from backup")
        return True
    
    # If no backup, check if we can fix the current file
    if file_path.exists():
        print("⚠️ No backup found, attempting to fix current file...")
        content = file_path.read_text(encoding='utf-8')
        
        # The issue is a malformed onMount - let's find and fix it
        # Look for the broken structure
        if 'ghostEngine = new RealGhostEngine({' in content:
            # Find the problematic section
            engine_start = content.find('ghostEngine = new RealGhostEngine({')
            if engine_start > 0:
                # Find the matching closing for the RealGhostEngine
                pos = engine_start + len('ghostEngine = new RealGhostEngine({')
                brace_count = 1
                while pos < len(content) and brace_count > 0:
                    if content[pos] == '{':
                        brace_count += 1
                    elif content[pos] == '}':
                        brace_count -= 1
                    pos += 1
                
                # Now fix the structure
                before = content[:pos]
                after = content[pos:]
                
                # Remove the misplaced auto-start code
                if '// Auto-start hologram if not using webcam' in before:
                    start_idx = before.find('// Auto-start hologram if not using webcam')
                    end_idx = before.find('});', start_idx)
                    if end_idx > start_idx:
                        before = before[:start_idx] + before[end_idx+3:]
                
                # Reconstruct properly
                content = before + ''';\n      
      // Initialize Ghost Engine with canvas
      await ghostEngine.initialize(canvas, {
        enableSync: true,
        targetFPS: 60
      });
      
      isInitialized = true;''' + after
                
                file_path.write_text(content, encoding='utf-8')
                print("✅ Fixed file structure")
                return True
    
    print("❌ Could not fix the file")
    return False

if __name__ == "__main__":
    restore_holographic_display()
