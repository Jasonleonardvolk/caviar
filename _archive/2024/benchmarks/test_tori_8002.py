#!/usr/bin/env python3
"""
TORI Test Launcher for Port 8002
Automatically uses the correct port configuration
"""

import subprocess
import sys
import json
from pathlib import Path

# Load port configuration
port_config_file = Path("tori_ports.json")
if port_config_file.exists():
    with open(port_config_file) as f:
        ports = json.load(f)
    API_URL = ports.get("api_url", "http://localhost:8002")
else:
    API_URL = "http://localhost:8002"

print(f"Running TORI tests with API URL: {API_URL}")
print("=" * 60)

# Run the test with the correct API URL
cmd = [sys.executable, "test_concept_mesh_e2e.py", f"--api-url={API_URL}"]
subprocess.run(cmd)
