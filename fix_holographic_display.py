#!/usr/bin/env python3
"""Fix the HolographicDisplay component to not show 'Initializing video'"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def fix_holographic_display():
    file_path = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\components\HolographicDisplay.svelte")
    
    print("🔧 Fixing HolographicDisplay component...")
    
    if not file_path.exists():
        print("❌ File not found!")
        return False
    
    # Read the file
    content = file_path.read_text(encoding='utf-8')
    
    # Fix 1: Change the loading overlay condition
    # It should NOT show "Initializing video" for hologram mode
    old_overlay = '{#if enableVideo && !videoReady}'
    new_overlay = '{#if enableVideo && videoSource === "webcam" && !videoReady}'
    
    if old_overlay in content:
        content = content.replace(old_overlay, new_overlay)
        print("✅ Fixed loading overlay condition")
    
    # Fix 2: Make sure hologram starts even without video
    # Find the startHologram function
    if 'async function startHologram()' in content:
        # Make sure it doesn't depend on video being ready
        content = content.replace(
            'if (enableVideo && !videoReady) return;',
            '// Video not required for hologram'
        )
        print("✅ Removed video dependency for hologram")
    
    # Fix 3: Auto-start hologram when component mounts
    # Find onMount
    if 'onMount(async () => {' in content:
        # Add auto-start logic
        mount_end = content.find('});', content.find('onMount(async () => {'))
        if mount_end > 0:
            insert_pos = mount_end
            auto_start = '''
    // Auto-start hologram if not using webcam
    if (!enableVideo || videoSource !== 'webcam') {
      console.log('🌟 Auto-starting hologram display');
      startHologram();
    }
    '''
            content = content[:insert_pos] + auto_start + '\n  ' + content[insert_pos:]
            print("✅ Added hologram auto-start")
    
    # Write back
    file_path.write_text(content, encoding='utf-8')
    
    print("✅ HolographicDisplay component fixed")
    return True

if __name__ == "__main__":
    if fix_holographic_display():
        print("\n🎉 The 'Initializing video...' message should no longer appear!")
        print("Hologram will display immediately.")
