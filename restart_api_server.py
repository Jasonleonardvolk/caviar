#!/usr/bin/env python3
"""
Quick script to restart the API server and show initial logs (Windows compatible)
"""

import subprocess
import sys
import time
import os
import signal

print("🔄 Restarting TORI API server...")
print("-" * 50)

# Kill any existing uvicorn processes (Windows compatible)
print("Stopping existing servers...")
if sys.platform == "win32":
    # Windows
    try:
        subprocess.run(["taskkill", "/F", "/IM", "uvicorn.exe"], capture_output=True)
    except:
        pass
    # Also try to kill python processes running uvicorn
    try:
        result = subprocess.run(["wmic", "process", "where", "commandline like '%uvicorn%'", "get", "processid"], 
                               capture_output=True, text=True)
        for line in result.stdout.splitlines():
            line = line.strip()
            if line and line.isdigit():
                try:
                    subprocess.run(["taskkill", "/F", "/PID", line], capture_output=True)
                except:
                    pass
    except:
        pass
else:
    # Unix/Linux/Mac
    subprocess.run(["pkill", "-f", "uvicorn"], capture_output=True)

time.sleep(2)

print("\nStarting API server on port 8002...")
print("Press Ctrl+C to stop")
print("-" * 50)

# Start the server
try:
    # Change to the directory containing prajna_api.py
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "prajna_api:app",
        "--host", "0.0.0.0",
        "--port", "8002",
        "--reload",
        "--log-level", "info"
    ])
except KeyboardInterrupt:
    print("\n\nServer stopped.")
