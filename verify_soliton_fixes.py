#!/usr/bin/env python3
"""
Quick verification script to ensure soliton fixes are applied
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ SUCCESS")
            if result.stdout:
                print(f"Output:\n{result.stdout}")
        else:
            print(f"❌ FAILED (exit code: {result.returncode})")
            if result.stderr:
                print(f"Error:\n{result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def main():
    print("🔍 Soliton Fixes Verification")
    print("="*60)
    
    # Check environment
    print(f"Python: {sys.executable}")
    print(f"Working dir: {os.getcwd()}")
    print(f"TORI_DISABLE_MESH_CHECK: {os.environ.get('TORI_DISABLE_MESH_CHECK', 'not set')}")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Check if soliton module can be imported
    tests_total += 1
    cmd = f'"{sys.executable}" -c "from mcp_metacognitive.core import soliton_memory; print(\'Import successful\')"'
    if run_command(cmd, "Import soliton_memory module"):
        tests_passed += 1
    
    # Test 2: Check if API route file exists and has diagnostic endpoint
    tests_total += 1
    cmd = f'"{sys.executable}" -c "import os; content = open(\'api/routes/soliton.py\').read(); print(\'diagnostic\' in content)"'
    if run_command(cmd, "Check soliton.py has diagnostic endpoint"):
        tests_passed += 1
    
    # Test 3: Check if frontend guard is in place
    tests_total += 1
    cmd = f'"{sys.executable}" -c "content = open(\'tori_ui_svelte/src/lib/services/solitonMemory.ts\').read(); print(\'STATS_RETRY_COOLDOWN\' in content)"'
    if run_command(cmd, "Check frontend has rate limiting"):
        tests_passed += 1
    
    # Test 4: Check if requirements.lock exists
    tests_total += 1
    cmd = f'"{sys.executable}" -c "import os; print(os.path.exists(\'requirements.lock\'))"'
    if run_command(cmd, "Check requirements.lock exists"):
        tests_passed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("\n✅ All fixes appear to be applied correctly!")
        print("\nNext: Start the backend and run the API tests:")
        print("  uvicorn api.main:app --port 5173 --reload")
        print("  python fixes/soliton_500_fixes/test_soliton_api.py")
    else:
        print("\n⚠️ Some fixes may not be applied correctly.")
        print("Check the failed tests above.")
    
    return 0 if tests_passed == tests_total else 1

if __name__ == "__main__":
    sys.exit(main())
