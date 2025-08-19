#!/usr/bin/env python3
"""
Quick runner for test_concept_mesh_3 with correct port configuration.
This ensures the test uses port 8002 (where your API is actually running).
"""
import subprocess
import sys
import os

# Run the test with explicit port configuration
print("Running test_concept_mesh_3 with API port 8002...")
print("-" * 60)

# Use the fixed version if it exists, otherwise use original with port override
test_script = "test_concept_mesh_3_fixed.py" if os.path.exists("test_concept_mesh_3_fixed.py") else "test_concept_mesh_3"

# Run with port 8002
result = subprocess.run([
    sys.executable, 
    test_script,
    "--api-port", "8002"
], capture_output=False)

sys.exit(result.returncode)
