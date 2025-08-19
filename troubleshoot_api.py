#!/usr/bin/env python3
"""
TORI API Troubleshooting Script
Diagnoses why the API server isn't starting properly
"""

import os
import sys
import socket
import subprocess
import time
import psutil
import requests
from pathlib import Path

def check_port_availability(port):
    """Check if a port is available or in use"""
    try:
        # Check who's using the port
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == 'LISTEN':
                try:
                    proc = psutil.Process(conn.pid)
                    print(f"❌ Port {port} is in use by: {proc.name()} (PID: {conn.pid})")
                    print(f"   Command: {' '.join(proc.cmdline())}")
                    print(f"   Started: {time.ctime(proc.create_time())}")
                    return False
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    print(f"❌ Port {port} is in use by PID: {conn.pid} (access denied)")
                    return False
        
        # Try to bind to the port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
            print(f"✅ Port {port} is available")
            return True
    except OSError:
        print(f"❌ Port {port} is not available (bind failed)")
        return False
    except Exception as e:
        print(f"❌ Error checking port {port}: {e}")
        return False

def kill_processes_on_port(port):
    """Kill all processes using a port"""
    killed = []
    try:
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                try:
                    proc = psutil.Process(conn.pid)
                    print(f"Killing {proc.name()} (PID: {conn.pid}) on port {port}...")
                    proc.terminate()
                    proc.wait(timeout=5)
                    killed.append(conn.pid)
                except psutil.TimeoutExpired:
                    proc.kill()
                    killed.append(conn.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    print(f"   Could not kill PID {conn.pid}: {e}")
    except Exception as e:
        print(f"Error killing processes: {e}")
    
    if killed:
        print(f"✅ Killed {len(killed)} process(es)")
        time.sleep(2)  # Wait for ports to be released
    return len(killed) > 0

def test_api_health(port, timeout=5):
    """Test if API health endpoint is responding"""
    url = f"http://localhost:{port}/api/health"
    print(f"\nTesting API health at {url}...")
    
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✅ API is healthy! Response: {response.text}")
            return True
        else:
            print(f"❌ API returned status {response.status_code}: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print(f"❌ API health check timed out after {timeout} seconds")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API (connection refused)")
        return False
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False

def find_python_processes():
    """Find all Python processes"""
    print("\n🐍 Python processes running:")
    found = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'tori' in cmdline.lower() or 'enhanced_launcher' in cmdline or 'prajna' in cmdline:
                    print(f"   PID {proc.info['pid']}: {cmdline[:100]}...")
                    print(f"   Started: {time.ctime(proc.info['create_time'])}")
                    found = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if not found:
        print("   No TORI-related Python processes found")

def check_firewall_rules():
    """Check Windows firewall rules for port 8002"""
    print("\n🔥 Checking firewall rules for port 8002...")
    
    try:
        # Check if port 8002 is blocked
        result = subprocess.run(
            ['netsh', 'advfirewall', 'firewall', 'show', 'rule', 'name=all'],
            capture_output=True,
            text=True
        )
        
        if '8002' in result.stdout:
            print("⚠️  Found firewall rules mentioning port 8002")
            print("   Run as admin: netsh advfirewall firewall show rule name=all | findstr 8002")
        else:
            print("✅ No specific firewall rules blocking port 8002")
            
        # Suggest adding allow rule
        print("\n💡 To add firewall exception (run as admin):")
        print('   netsh advfirewall firewall add rule name="TORI API" dir=in action=allow protocol=TCP localport=8002')
        
    except Exception as e:
        print(f"⚠️  Could not check firewall rules: {e}")

def test_enhanced_launcher():
    """Test if enhanced_launcher.py can be imported and run"""
    print("\n🚀 Testing enhanced_launcher.py...")
    
    launcher_path = Path.cwd() / "enhanced_launcher.py"
    if not launcher_path.exists():
        print(f"❌ enhanced_launcher.py not found at {launcher_path}")
        return
    
    # Try to import it
    try:
        # Test import
        subprocess.run(
            [sys.executable, '-c', 'import enhanced_launcher'],
            check=True,
            capture_output=True,
            text=True
        )
        print("✅ enhanced_launcher.py imports successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ enhanced_launcher.py import failed:")
        print(f"   {e.stderr}")
        return
    
    # Check for required environment variables
    print("\n📋 Environment variables:")
    important_vars = ['PYTHONPATH', 'PYTHONIOENCODING', 'PRAJNA_MODEL_TYPE']
    for var in important_vars:
        value = os.environ.get(var, '<not set>')
        print(f"   {var}: {value}")

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n📦 Checking Python dependencies...")
    
    required = ['uvicorn', 'fastapi', 'requests', 'psutil']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n💡 Install missing packages:")
        print(f"   pip install {' '.join(missing)}")

def main():
    print("=" * 60)
    print("🔍 TORI API TROUBLESHOOTING")
    print("=" * 60)
    
    # 1. Check port 8002
    print("\n1️⃣ Checking port 8002...")
    port_available = check_port_availability(8002)
    
    if not port_available:
        print("\n   Attempting to free port 8002...")
        if kill_processes_on_port(8002):
            port_available = check_port_availability(8002)
    
    # 2. Find Python processes
    find_python_processes()
    
    # 3. Test API health
    if not port_available:
        test_api_health(8002)
    
    # 4. Check firewall
    check_firewall_rules()
    
    # 5. Test enhanced launcher
    test_enhanced_launcher()
    
    # 6. Check dependencies
    check_dependencies()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    if port_available:
        print("✅ Port 8002 is available for use")
        print("\n🚀 Try starting the API manually:")
        print("   python enhanced_launcher.py")
        print("\n   Or if you have a specific API script:")
        print("   python -m uvicorn prajna.api.prajna_api:app --host 0.0.0.0 --port 8002")
    else:
        print("❌ Port 8002 is still blocked")
        print("\n💡 Try:")
        print("   1. Restart your computer to clear all ports")
        print("   2. Run as administrator: netstat -aon | findstr :8002")
        print("   3. Kill the process manually: taskkill /PID <pid> /F")

if __name__ == "__main__":
    main()
