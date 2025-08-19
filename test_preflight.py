#!/usr/bin/env python3
"""Quick test of the preflight check"""
import subprocess
import sys

result = subprocess.run(["node", "tools/runtime/preflight.mjs"], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print(result.stderr)
sys.exit(result.returncode)
