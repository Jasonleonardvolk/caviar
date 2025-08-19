#!/usr/bin/env python3
"""
Prajna Port Diagnostic Script
============================

This script checks for port conflicts and processes that might be blocking Prajna startup.
"""

import socket
import subprocess
import sys
import requests
from pathlib import Path

def check_port_usage(port=8001):
    """Check if port is in use and by what process"""
    print(f"🔍 Checking port {port} usage...")
    
    # Test if port is available
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', port))
            print(f"✅ Port {port} is available")
            return True, None
    except OSError as e:
        print(f"❌ Port {port} is in use: {e}")
        
        # Try to find what's using the port
        try:
            result = subprocess.run(
                ['netstat', '-ano', '|', 'findstr', f':{port}'],
                shell=True,
                capture_output=True,
                text=True
            )
            if result.stdout:
                print(f"📊 Port usage details:\n{result.stdout}")
            
            # Try alternative command
            result2 = subprocess.run(
                ['netstat', '-aon'],
                capture_output=True,
                text=True
            )
            for line in result2.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    print(f"🔍 Found: {line.strip()}")
                    
        except Exception as e:
            print(f"⚠️ Could not check port usage details: {e}")
        
        return False, "Port in use"

def check_prajna_process():
    """Check for existing Prajna processes"""
    print(f"\n🔍 Checking for existing Prajna processes...")
    
    try:
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe'],
            capture_output=True,
            text=True
        )
        
        python_processes = []
        for line in result.stdout.split('\n'):
            if 'python.exe' in line:
                python_processes.append(line.strip())
        
        if python_processes:
            print(f"📊 Found {len(python_processes)} Python processes:")
            for proc in python_processes:
                print(f"  {proc}")
        else:
            print("✅ No Python processes found")
            
        return python_processes
        
    except Exception as e:
        print(f"⚠️ Could not check processes: {e}")
        return []

def test_prajna_health(port=8001):
    """Test if Prajna is already running and responding"""
    print(f"\n🏥 Testing Prajna health on port {port}...")
    
    try:
        response = requests.get(f'http://localhost:{port}/api/health', timeout=5)
        if response.status_code == 200:
            print(f"✅ Prajna is already running and healthy on port {port}")
            print(f"📊 Response: {response.json()}")
            return True
        else:
            print(f"⚠️ Prajna responding but unhealthy: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ No Prajna service responding on port {port}")
        return False
    except Exception as e:
        print(f"⚠️ Error checking Prajna health: {e}")
        return False

def check_start_script():
    """Verify the start script exists and has correct path modifications"""
    print(f"\n📄 Checking start_prajna.py script...")
    
    script_path = Path("prajna/start_prajna.py")
    if not script_path.exists():
        print(f"❌ Start script not found: {script_path}")
        return False
    
    # Check line 35 fix
    with open(script_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if len(lines) >= 35:
        line_35 = lines[34].strip()  # Line 35 is index 34
        if "parent.parent" in line_35:
            print(f"✅ Line 35 path fix is present: {line_35}")
        else:
            print(f"⚠️ Line 35 might need fix: {line_35}")
    
    print(f"✅ Start script exists: {script_path}")
    return True

def main():
    """Run comprehensive diagnostic"""
    print("🚀 Prajna Startup Diagnostic")
    print("=" * 50)
    
    # Check port availability
    port_available, port_error = check_port_usage(8001)
    
    # Check for existing processes  
    processes = check_prajna_process()
    
    # Test if Prajna is already running
    already_running = test_prajna_health(8001)
    
    # Check start script
    script_ok = check_start_script()
    
    # Summary and recommendations
    print(f"\n📋 DIAGNOSTIC SUMMARY")
    print("=" * 30)
    print(f"Port 8001 available: {'✅' if port_available else '❌'}")
    print(f"Prajna already running: {'✅' if already_running else '❌'}")
    print(f"Python processes: {len(processes)}")
    print(f"Start script OK: {'✅' if script_ok else '❌'}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    
    if already_running:
        print("🎯 Prajna is already running successfully - no action needed")
    elif not port_available:
        print("🔧 Kill processes using port 8001 or restart system")
        print("📋 Use: netstat -ano | findstr :8001 to find PID, then taskkill /PID <pid>")
    elif not script_ok:
        print("🔧 Fix start_prajna.py script path issues")
    elif len(processes) > 0:
        print("🔧 Consider stopping other Python processes that might conflict")
        print("🔧 Or check if one of them is a stuck Prajna process")
    else:
        print("✅ System looks clear - try starting Prajna normally")

if __name__ == "__main__":
    main()
