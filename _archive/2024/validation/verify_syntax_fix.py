#!/usr/bin/env python3
"""Verify the syntax fix in prajna_api.py"""

import subprocess
import sys

print("🔍 Verifying Python syntax in prajna_api.py...")

result = subprocess.run(
    [sys.executable, "-m", "py_compile", "prajna_api.py"],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("✅ SUCCESS! Python syntax is now valid!")
    print("\n🎉 The syntax error has been fixed!")
    print("\nYou can now run:")
    print("   poetry run python enhanced_launcher.py")
else:
    print("❌ Syntax error still present:")
    print(result.stderr)
    
    # Try to parse the error
    if result.stderr:
        lines = result.stderr.split('\n')
        for line in lines:
            if 'line' in line:
                print(f"\n⚠️ Error location: {line}")
