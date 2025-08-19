#!/usr/bin/env python3
"""
Integration Test Script - Verify all fixes are working
"""

import os
import sys
from pathlib import Path

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_enola_in_persona_selector():
    """Test if Enola was added to PersonaSelector"""
    print("\n[CHECK] Testing Enola in PersonaSelector...")
    
    persona_selector_path = project_root / "tori_ui_svelte/src/lib/components/PersonaSelector.svelte"
    
    if not persona_selector_path.exists():
        print("[FAIL] PersonaSelector.svelte not found")
        return False
    
    with open(persona_selector_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'Enola' in content:
        print("[PASS] Enola found in PersonaSelector")
        
        # Check for specific attributes
        checks = [
            ("name: 'Enola'", "Enola name definition"),
            ("investigative", "Investigative cognitive mode"),
            ("#2563eb", "Investigation blue color"),
            ("analytical", "Analytical mood")
        ]
        
        for check_str, description in checks:
            if check_str in content:
                print(f"  [PASS] {description} found")
            else:
                print(f"  [FAIL] {description} NOT found")
        
        return True
    else:
        print("[FAIL] Enola NOT found in PersonaSelector")
        return False

def test_hologram_display_exists():
    """Test if HologramPersonaDisplay component was created"""
    print("\n[CHECK] Testing HologramPersonaDisplay component...")
    
    hologram_path = project_root / "tori_ui_svelte/src/lib/components/HologramPersonaDisplay.svelte"
    
    if hologram_path.exists():
        print("[PASS] HologramPersonaDisplay.svelte exists")
        
        with open(hologram_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key features
        checks = [
            ("ghostPersona", "Ghost persona store import"),
            ("currentPersona", "Current persona tracking"),
            ("hologram-container", "Hologram container element"),
            ("persona-icon", "Persona icon display"),
            ("Enola", "Enola persona support")
        ]
        
        for check_str, description in checks:
            if check_str in content:
                print(f"  [PASS] {description} found")
            else:
                print(f"  [FAIL] {description} NOT found")
        
        return True
    else:
        print("[FAIL] HologramPersonaDisplay.svelte does not exist")
        return False

def test_upload_debug_wrapper():
    """Test if upload debug wrapper was created"""
    print("\n[CHECK] Testing upload debug wrapper...")
    
    wrapper_path = project_root / "api/upload_debug_wrapper.py"
    
    if wrapper_path.exists():
        print("[PASS] upload_debug_wrapper.py exists")
        
        # Try to import it
        try:
            from api.upload_debug_wrapper import UploadDebugger, get_debugger
            print("  [PASS] Successfully imported UploadDebugger")
            
            # Test basic functionality
            debugger = UploadDebugger()
            debugger.log_step("test", {"message": "Testing"})
            print("  [PASS] Debugger logging works")
            
            return True
        except ImportError as e:
            print(f"  [FAIL] Failed to import: {e}")
            return False
    else:
        print("[FAIL] upload_debug_wrapper.py does not exist")
        return False

def test_ghost_persona_store():
    """Test if ghost persona store has Enola as default"""
    print("\n[CHECK] Testing ghost persona store...")
    
    ghost_store_path = project_root / "tori_ui_svelte/src/lib/stores/ghostPersona.ts"
    
    if not ghost_store_path.exists():
        print("[FAIL] ghostPersona.ts not found")
        return False
    
    with open(ghost_store_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ("persona: 'Enola'", "Enola as default persona"),
        ("activePersona: 'Enola'", "Enola as active persona"),
        ('"Enola": new GhostPersona("Enola")', "Enola in ghost registry")
    ]
    
    all_good = True
    for check_str, description in checks:
        if check_str in content:
            print(f"[PASS] {description} found")
        else:
            print(f"[FAIL] {description} NOT found")
            all_good = False
    
    return all_good

def main():
    """Run all integration tests"""
    print("TORI Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_enola_in_persona_selector,
        test_hologram_display_exists,
        test_upload_debug_wrapper,
        test_ghost_persona_store
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Test {test.__name__} failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! TORI is ready to launch!")
        print("\nNext steps:")
        print("1. Run the fix scripts to apply changes:")
        print("   python fix_persona_selector.py")
        print("   python fix_hologram_display.py")
        print("   python fix_scholarsphere_upload.py")
        print("\n2. Add HologramPersonaDisplay to your main layout")
        print("3. Test upload with a PDF file")
        print("4. Verify Enola appears as default persona")
    else:
        print("\n[WARNING] Some tests failed. Please run the fix scripts first.")
        print("\nTo apply all fixes at once, run:")
        print("   python apply_all_fixes.py")

if __name__ == "__main__":
    main()
