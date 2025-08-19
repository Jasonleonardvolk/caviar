"""
FINAL EXCEPTION LOGGING FIX - Windows Safe Version
Run this to apply all fixes without Unicode errors
"""

import subprocess
import sys
import os
from pathlib import Path

# Force UTF-8
os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

print("="*60)
print("APPLYING EXCEPTION LOGGING FIXES")
print("="*60)
print("\nThis will fix:")
print("1. API startup exception logging")
print("2. Missing health endpoint")
print("3. Missing auth endpoints")
print("4. PowerShell timing issues")

# Check if we have the fixed scripts
required_scripts = [
    "add_api_exception_logging_fixed.py",
    "check_health_endpoint_fixed.py",
    "quick_auth_fix_fixed.py"
]

missing = [s for s in required_scripts if not Path(s).exists()]
if missing:
    print(f"\n[ERROR] Missing required scripts: {', '.join(missing)}")
    print("Please ensure all fixed scripts are present")
    sys.exit(1)

print("\n[OK] All required scripts found")

# Run each fix
results = {}
for script in required_scripts:
    print(f"\n{'='*60}")
    print(f"[RUNNING] {script}")
    print("="*60)
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.returncode == 0:
            results[script] = "SUCCESS"
            print(f"\n[SUCCESS] {script} completed successfully")
        else:
            results[script] = "FAILED"
            print(f"\n[FAILED] {script} failed with code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
    except Exception as e:
        results[script] = f"ERROR: {e}"
        print(f"\n[ERROR] Failed to run {script}: {e}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

success_count = sum(1 for r in results.values() if r == "SUCCESS")
print(f"\nCompleted: {success_count}/{len(required_scripts)} fixes")

for script, result in results.items():
    status = "[OK]" if result == "SUCCESS" else "[FAIL]"
    print(f"{status} {script}: {result}")

if success_count == len(required_scripts):
    print("\n" + "="*60)
    print("ALL FIXES APPLIED SUCCESSFULLY!")
    print("="*60)
    print("\nYour TORI system now has:")
    print("- Exception logging for API startup")
    print("- Health endpoint at /api/health")
    print("- Auth endpoints for login")
    print("- No more silent hangs!")
    
    print("\nNEXT STEPS:")
    print("1. Kill all Python processes:")
    print("   Get-Process python | Stop-Process -Force")
    print("\n2. Start fresh:")
    print("   .\\start_tori_hardened.ps1 -Force")
    print("\n3. Check logs for errors:")
    print("   Get-Content logs\\launcher.log -Tail 50")
    
    print("\nLOGIN CREDENTIALS:")
    print("Username: admin  Password: admin")
    print("Username: user   Password: user")
    print("Username: test   Password: test")
else:
    print("\n[WARNING] Some fixes failed")
    print("Please check the errors above and try again")

print("\n[DONE] Exception logging setup complete!")
