#!/usr/bin/env python3
"""Use the FIXED version of HolographicDisplay"""

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import shutil

def use_fixed_version():
    current_file = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\components\HolographicDisplay.svelte")
    fixed_file = Path(r"{PROJECT_ROOT}\tori_ui_svelte\src\lib\components\HolographicDisplay_FIXED.svelte")
    
    print("🔧 Using the FIXED version of HolographicDisplay...")
    
    if not fixed_file.exists():
        print("❌ Fixed file not found!")
        return False
    
    # Backup current broken file
    backup_path = current_file.with_suffix('.svelte.broken')
    shutil.copy2(current_file, backup_path)
    print(f"📦 Backed up broken file to {backup_path.name}")
    
    # Copy fixed version
    shutil.copy2(fixed_file, current_file)
    print("✅ Copied HolographicDisplay_FIXED.svelte to HolographicDisplay.svelte")
    
    return True

if __name__ == "__main__":
    if use_fixed_version():
        print("\n🎉 Success! The hologram display should work now.")
        print("\nThe dev server should auto-reload with the fix.")
        print("If not, restart with: poetry run python enhanced_launcher.py")
