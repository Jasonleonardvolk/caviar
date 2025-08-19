from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
﻿#!/usr/bin/env python3
"""
TORI Minimal Launcher - Start core components only
"""

import subprocess
import time
import sys
import requests
import os

def is_port_free(port):
    """Check if port is available"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("localhost", port))
    sock.close()
    return result != 0

def start_service(name, command, port, health_check_url=None):
    """Start a service and wait for it to be healthy"""
    print(f"Starting {name}...")
    
    if not is_port_free(port):
        print(f"  WARNING: Port {port} is already in use")
        return None
    
    # Start the process
    if os.name == 'nt':
        proc = subprocess.Popen(command, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        proc = subprocess.Popen(command, shell=True)
    
    # Wait for health check
    if health_check_url:
        for i in range(30):
            try:
                response = requests.get(health_check_url, timeout=1)
                if response.status_code == 200:
                    print(f"  SUCCESS: {name} is healthy!")
                    return proc
            except:
                pass
            time.sleep(1)
        
        print(f"  ERROR: {name} failed health check")
        proc.terminate()
        return None
    else:
        time.sleep(3)  # Just wait a bit for services without health check
        print(f"  SUCCESS: {name} started")
        return proc

def main():
    print("TORI Minimal Launcher")
    print("=" * 60)
    
    os.chdir(r"{PROJECT_ROOT}")
    
    processes = []
    
    # Start API server
    api_proc = start_service(
        "API Server",
        "python -m uvicorn enhanced_launcher:app --port 8002 --reload",
        8002,
        "http://localhost:8002/api/health"
    )
    if api_proc:
        processes.append(api_proc)
    
    # Start frontend
    frontend_proc = start_service(
        "Frontend",
        "cd tori_ui_svelte && npm run dev",
        5173,
        "http://localhost:5173"
    )
    if frontend_proc:
        processes.append(frontend_proc)
    
    if processes:
        print(f"\nSUCCESS: Started {len(processes)} services successfully!")
        print("\nAccess TORI at: http://localhost:5173")
        print("API docs at: http://localhost:8002/docs")
        print("\nPress Ctrl+C to stop all services")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down services...")
            for proc in processes:
                proc.terminate()
            print("All services stopped")
    else:
        print("\nERROR: Failed to start any services")
        sys.exit(1)

if __name__ == "__main__":
    main()
