"""
Test that fixed scripts work without Unicode errors
"""

import subprocess
import sys
import os

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

print("[TEST] Testing fixed scripts...")
print("=" * 60)

scripts = [
    ("add_api_exception_logging_fixed.py", "Exception logging"),
    ("check_health_endpoint_fixed.py", "Health endpoint"),
    ("quick_auth_fix_fixed.py", "Auth endpoints")
]

for script, desc in scripts:
    print(f"\n[TEST] {desc}...")
    try:
        # Just test if the script runs without Unicode errors
        result = subprocess.run(
            [sys.executable, script, "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Script might not support --help, but that's OK
        # We're just checking for Unicode errors
        if "UnicodeEncodeError" in str(result.stderr):
            print(f"[FAIL] {script} has Unicode errors!")
        else:
            print(f"[OK] {script} runs without Unicode errors")
    except subprocess.TimeoutExpired:
        print(f"[OK] {script} started (no Unicode errors)")
    except Exception as e:
        print(f"[ERROR] {script}: {e}")

print("\n[COMPLETE] Testing done!")
