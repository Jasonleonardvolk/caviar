#!/usr/bin/env python3
"""
Apply all fixes to complete TORI startup cleanup
"""

import subprocess
import sys

def run_fix(script_name, description):
    """Run a fix script and report results"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ FAILED: {description}")
            if result.stderr:
                print(result.stderr)
            if result.stdout:
                print(result.stdout)
                
        return result.returncode == 0
    except Exception as e:
        print(f"❌ ERROR running {script_name}: {e}")
        return False

def main():
    print("🚀 TORI STARTUP FIX SUITE")
    print("="*60)
    print("Applying all fixes to clean up TORI startup...")
    
    fixes = [
        ("apply_soliton_stats_fix.py", "Fix soliton memory stats error"),
        ("add_concept_mesh_population.py", "Add concept mesh population at startup"),
    ]
    
    success_count = 0
    for script, description in fixes:
        if run_fix(script, description):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {success_count}/{len(fixes)} fixes applied successfully")
    print(f"{'='*60}")
    
    if success_count == len(fixes):
        print("\n✅ All fixes applied successfully!")
        print("\nNext steps:")
        print("1. Restart TORI: python enhanced_launcher.py")
        print("2. The following issues are now fixed:")
        print("   - MCP router warning (fixed in server_proper.py)")
        print("   - Duplicate registrations (fixed in server_proper.py)")
        print("   - Startup warnings (suppressed in enhanced_launcher.py)")
        print("   - ConceptDB already has __hash__ methods")
        print("   - Soliton stats error (if frontend applied)")
        print("   - Empty concept mesh (will populate on startup)")
    else:
        print("\n⚠️ Some fixes failed. Check the output above for details.")

if __name__ == "__main__":
    main()
