#!/usr/bin/env python3
"""
MCP Service Diagnostic and Manual Startup Script
Helps diagnose and start the MCP service on port 8100
"""

import subprocess
import sys
import time
import socket
import requests
import os
from pathlib import Path

def check_port(port):
    """Check if a port is in use"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('127.0.0.1', port))
            return result == 0
    except:
        return False

def find_mcp_process():
    """Find any running MCP processes"""
    try:
        result = subprocess.run(
            ['netstat', '-ano'], 
            capture_output=True, 
            text=True
        )
        lines = result.stdout.splitlines()
        for line in lines:
            if ':8100' in line and 'LISTENING' in line:
                parts = line.split()
                if parts:
                    pid = parts[-1]
                    print(f"Found process on port 8100: PID {pid}")
                    # Get process name
                    try:
                        proc_result = subprocess.run(
                            ['tasklist', '/FI', f'PID eq {pid}'],
                            capture_output=True,
                            text=True
                        )
                        print(proc_result.stdout)
                    except:
                        pass
    except Exception as e:
        print(f"Error finding MCP process: {e}")

def test_mcp_endpoints():
    """Test MCP endpoints"""
    endpoints = [
        'http://localhost:8100/',
        'http://localhost:8100/sse',
        'http://localhost:8100/api/system/status',
        'http://localhost:8100/tools',
        'http://localhost:8100/consciousness'
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=2)
            print(f"✓ {endpoint} - Status: {response.status_code}")
        except Exception as e:
            print(f"✗ {endpoint} - Error: {type(e).__name__}")

def start_mcp_simple():
    """Start MCP service with simple approach"""
    script_dir = Path(__file__).parent
    mcp_dir = script_dir / "mcp_metacognitive"
    
    if not mcp_dir.exists():
        print(f"Error: MCP directory not found: {mcp_dir}")
        return
    
    # Set environment
    env = os.environ.copy()
    env.update({
        'TRANSPORT_TYPE': 'sse',
        'SERVER_PORT': '8100',
        'SERVER_HOST': '0.0.0.0',
        'PYTHONIOENCODING': 'utf-8',
        'TORI_INTEGRATION': 'true'
    })
    
    # Try different startup approaches
    print("\nTrying direct Python module approach...")
    try:
        cmd = [sys.executable, '-m', 'mcp_metacognitive.server_simple']
        print(f"Command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            cwd=str(script_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"Started MCP with PID: {process.pid}")
        
        # Wait a bit and check
        time.sleep(3)
        
        if process.poll() is None:
            print("✓ MCP process is running!")
        else:
            stdout, stderr = process.communicate()
            print(f"✗ MCP process exited with code: {process.returncode}")
            print(f"STDOUT:\n{stdout}")
            print(f"STDERR:\n{stderr}")
            
            # Try alternative approach
            print("\nTrying server_fixed.py directly...")
            server_fixed = mcp_dir / "server_fixed.py"
            if server_fixed.exists():
                cmd = [sys.executable, str(server_fixed)]
                process = subprocess.Popen(
                    cmd,
                    cwd=str(script_dir),
                    env=env
                )
                print(f"Started with PID: {process.pid}")
                time.sleep(3)
                if process.poll() is None:
                    print("✓ MCP process is running!")
                    
    except Exception as e:
        print(f"Error starting MCP: {e}")

def main():
    print("=== MCP Service Diagnostic ===\n")
    
    print("1. Checking port 8100...")
    if check_port(8100):
        print("✓ Port 8100 is in use")
        find_mcp_process()
        
        print("\n2. Testing MCP endpoints...")
        test_mcp_endpoints()
    else:
        print("✗ Port 8100 is not in use")
        
        print("\n2. Attempting to start MCP service...")
        start_mcp_simple()
        
        print("\n3. Rechecking port 8100...")
        time.sleep(2)
        if check_port(8100):
            print("✓ Port 8100 is now in use")
            test_mcp_endpoints()
        else:
            print("✗ Port 8100 is still not in use")
            print("\nPossible solutions:")
            print("1. Check if mcp_metacognitive directory exists")
            print("2. Install required packages: pip install fastmcp mcp")
            print("3. Check for asyncio conflicts in server_fixed.py")
            print("4. Try running: python -m mcp_metacognitive.server_simple")

if __name__ == "__main__":
    main()
