#!/usr/bin/env python3
"""
Restart TORI using the canonical enhanced_launcher.py
"""

import subprocess
import sys
import time
import os
import signal

print("🔄 Restarting TORI using enhanced_launcher.py...")
print("-" * 50)

# Kill any existing TORI processes
print("Stopping existing TORI processes...")

if sys.platform == "win32":
    # Windows - kill processes on TORI ports
    ports = [8000, 8001, 8002, 5173]
    for port in ports:
        try:
            # Find process using the port
            result = subprocess.run(
                f'netstat -ano | findstr :{port}', 
                shell=True, 
                capture_output=True, 
                text=True
            )
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) > 4 and parts[4].isdigit():
                    pid = parts[4]
                    try:
                        subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
                        print(f"  Killed process on port {port} (PID: {pid})")
                    except:
                        pass
        except:
            pass
else:
    # Unix/Linux/Mac
    subprocess.run(["pkill", "-f", "enhanced_launcher"], capture_output=True)
    subprocess.run(["pkill", "-f", "prajna_api"], capture_output=True)
    subprocess.run(["pkill", "-f", "uvicorn"], capture_output=True)

print("Waiting for ports to clear...")
time.sleep(3)

print("\nStarting TORI with enhanced_launcher.py...")
print("Press Ctrl+C to stop")
print("-" * 50)

# Change to the TORI directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Start using poetry
try:
    subprocess.run([sys.executable, "-m", "poetry", "run", "python", "enhanced_launcher.py"])
except KeyboardInterrupt:
    print("\n\nTORI stopped.")
except Exception as e:
    print(f"\nError: {e}")
    print("\nTrying direct Python execution...")
    try:
        subprocess.run([sys.executable, "enhanced_launcher.py"])
    except KeyboardInterrupt:
        print("\n\nTORI stopped.")
