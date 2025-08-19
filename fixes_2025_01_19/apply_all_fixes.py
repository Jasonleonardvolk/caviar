#!/usr/bin/env python3
"""
Master Fix Script - Run all fixes in sequence
"""

import os
import sys
import subprocess
from pathlib import Path

def run_fix(script_name):
    """Run a fix script and return success status"""
    print(f"\n{'='*60}")
    print(f"🔧 Running {script_name}...")
    print(f"{'='*60}")
    
    script_path = Path(__file__).parent / script_name
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        
        if result.stderr:
            print("Errors:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run {script_name}: {e}")
        return False

def main():
    """Run all fixes"""
    print("🚀 TORI Master Fix Script")
    print("This will apply all fixes to resolve:")
    print("1. Enola not showing as default persona")
    print("2. Hologram display missing")
    print("3. ScholarSphere upload errors")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    fixes = [
        "fix_persona_selector.py",
        "fix_hologram_display.py", 
        "fix_scholarsphere_upload.py"
    ]
    
    results = []
    for fix in fixes:
        success = run_fix(fix)
        results.append((fix, success))
    
    print(f"\n{'='*60}")
    print("📊 FIX SUMMARY")
    print(f"{'='*60}")
    
    for fix, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{status}: {fix}")
    
    # Run integration test
    print(f"\n{'='*60}")
    print("🧪 Running integration tests...")
    print(f"{'='*60}")
    
    run_fix("test_integration.py")
    
    print("\n✨ Fix process complete!")
    print("\n📝 Manual steps required:")
    print("1. Add HologramPersonaDisplay to your main layout:")
    print("   import HologramPersonaDisplay from '$lib/components/HologramPersonaDisplay.svelte';")
    print("   <HologramPersonaDisplay />")
    print("\n2. Restart the TORI launcher:")
    print("   poetry run python enhanced_launcher.py")
    print("\n3. Test the following:")
    print("   - Enola should appear as the default persona")
    print("   - Hologram should display the active persona")
    print("   - Upload a PDF and check for better error messages")

if __name__ == "__main__":
    main()
