#!/usr/bin/env python3
"""
Start API and Capture Startup Logs
==================================
This will help us see what's happening during startup
"""

import subprocess
import sys
import time
import threading
import requests

def capture_output(process, output_file):
    """Capture process output to file"""
    with open(output_file, "w") as f:
        for line in iter(process.stdout.readline, b''):
            if line:
                decoded = line.decode('utf-8', errors='replace')
                print(decoded.strip())
                f.write(decoded)
                f.flush()

print("Starting API server and capturing logs...")

# Start the API server
cmd = [sys.executable, "enhanced_launcher.py"]
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=False
)

# Start output capture in background
output_thread = threading.Thread(
    target=capture_output,
    args=(process, "startup_logs.txt"),
    daemon=True
)
output_thread.start()

# Wait for startup
print("\nWaiting for API to start...")
time.sleep(10)

# Test the endpoints
print("\nTesting endpoints:")
try:
    # Test health
    response = requests.get("http://localhost:8002/api/health")
    print(f"Health check: {response.status_code}")
    
    # Test soliton init
    response = requests.post(
        "http://localhost:8002/api/soliton/init",
        json={"user_id": "test_user"}
    )
    print(f"Soliton init: {response.status_code}")
    if response.status_code != 200:
        print(f"Response: {response.text}")
        
    # Check OpenAPI
    response = requests.get("http://localhost:8002/openapi.json")
    if response.status_code == 200:
        openapi = response.json()
        paths = openapi.get("paths", {})
        soliton_paths = [p for p in paths if "/soliton" in p]
        print(f"Soliton paths in OpenAPI: {len(soliton_paths)}")
        for path in soliton_paths:
            print(f"  - {path}")
            
except Exception as e:
    print(f"Error testing endpoints: {e}")

print("\nPress Ctrl+C to stop...")
try:
    process.wait()
except KeyboardInterrupt:
    process.terminate()
    print("\nStopped.")
