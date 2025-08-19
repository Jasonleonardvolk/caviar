from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
#!/usr/bin/env python3
"""
Run BPS Import Test
"""

import subprocess
import sys

print("Running BPS import test...")
print("=" * 60)

result = subprocess.run(
    [sys.executable, "test_bps_imports.py"],
    capture_output=True,
    text=True,
    cwd=r"{PROJECT_ROOT}"
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

if result.returncode == 0:
    print("\n✅ Test passed! Ready to proceed with integration.")
else:
    print(f"\n❌ Test failed with return code {result.returncode}")
    print("Please check the errors above.")
