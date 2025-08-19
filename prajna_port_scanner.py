#!/usr/bin/env python3
"""
Prajna Port Scanner - Find what port Prajna actually binds to
"""

import subprocess
import sys
import os
from pathlib import Path
import time
import socket

def scan_prajna_ports():
    """Start Prajna and scan to see what port it actually binds to"""
    
    script_dir = Path(__file__).parent
    prajna_dir = script_dir / "prajna"
    start_script = prajna_dir / "start_prajna.py"
    
    print("🔍 Starting Prajna and scanning for bound ports...")
    
    # Start Prajna process
    prajna_cmd = [
        sys.executable,
        str(start_script),
        "--port", "8001",
        "--host", "0.0.0.0",
        "--log-level", "DEBUG"  # More verbose
    ]
    
    env = os.environ.copy()
    parent_dir = prajna_dir.parent
    env['PYTHONPATH'] = str(parent_dir)
    
    print(f"🚀 Starting: {' '.join(prajna_cmd)}")
    
    try:
        process = subprocess.Popen(
            prajna_cmd,
            cwd=str(prajna_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine outputs
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        print(f"📊 Process PID: {process.pid}")
        print("📄 Prajna Output:")
        print("-" * 50)
        
        # Monitor output for 10 seconds
        start_time = time.time()
        while time.time() - start_time < 10:
            output = process.stdout.readline()
            if output:
                print(f"PRAJNA: {output.strip()}")
            elif process.poll() is not None:
                print(f"📊 Process exited with code: {process.poll()}")
                break
            time.sleep(0.1)
        
        print("-" * 50)
        
        # Scan common ports to see what's bound
        common_ports = [8000, 8001, 8002, 8080, 3000, 5000, 5173]
        print("🔍 Scanning for bound ports...")
        
        bound_ports = []
        for port in common_ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    result = s.connect_ex(('127.0.0.1', port))
                    if result == 0:
                        bound_ports.append(port)
                        print(f"✅ Port {port} is bound!")
            except:
                pass
        
        if not bound_ports:
            print("❌ No ports found bound!")
        
        # Cleanup
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        
        return bound_ports
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

if __name__ == "__main__":
    ports = scan_prajna_ports()
    print(f"\n🎯 Summary: Found {len(ports)} bound ports: {ports}")
