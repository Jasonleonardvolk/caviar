#!/usr/bin/env python3
"""
TORI Quick Status Check
Shows current system status and provides recommendations
"""

import os
import sys
import json
import socket
import subprocess
from pathlib import Path
from datetime import datetime
import requests

def check_port(port):
    """Quick port check"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(('127.0.0.1', port)) == 0
    except:
        return False

def main():
    script_dir = Path(__file__).parent
    
    print("🔍 TORI QUICK STATUS CHECK")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check key services
    services = {
        'API Server': (8002, 'http://localhost:8002/api/health'),
        'Frontend': (5173, 'http://localhost:5173'),
        'Audio Bridge': (8765, None),
        'Concept Bridge': (8766, None),
        'MCP Server': (8100, None)
    }
    
    all_healthy = True
    
    for service, (port, url) in services.items():
        if check_port(port):
            # Try health endpoint if available
            if url and 'health' in url:
                try:
                    resp = requests.get(url, timeout=2)
                    if resp.status_code == 200:
                        print(f"✅ {service}: RUNNING (port {port}) - Health OK")
                    else:
                        print(f"⚠️ {service}: RUNNING (port {port}) - Health check failed")
                        all_healthy = False
                except:
                    print(f"⚠️ {service}: RUNNING (port {port}) - Health endpoint not responding")
                    all_healthy = False
            else:
                print(f"✅ {service}: RUNNING (port {port})")
        else:
            print(f"❌ {service}: NOT RUNNING (port {port})")
            all_healthy = False
    
    print()
    
    # Check for status file
    status_file = script_dir / 'tori_status.json'
    if status_file.exists():
        try:
            with open(status_file) as f:
                status = json.load(f)
            
            print("📊 Last Status Update:")
            print(f"   Session: {status.get('session_id', 'Unknown')}")
            print(f"   Stage: {status.get('stage', 'Unknown')}")
            print(f"   Status: {status.get('status', 'Unknown')}")
            
            # Calculate uptime
            if 'timestamp' in status:
                start_time = datetime.fromisoformat(status['timestamp'])
                uptime = datetime.now() - start_time
                print(f"   Uptime: {uptime}")
        except Exception as e:
            print(f"⚠️ Could not read status file: {e}")
    else:
        print("❌ No status file found - system may not be running")
    
    print()
    print("=" * 50)
    
    if all_healthy:
        print("✅ All services are healthy!")
        print("\n📋 Quick Links:")
        print("   Frontend: http://localhost:5173")
        print("   API Docs: http://localhost:8002/docs")
        print("   Health: http://localhost:8002/api/health")
    else:
        print("⚠️ Some services are not healthy!")
        print("\n💡 To start TORI:")
        print("   Option 1: Double-click START_TORI.bat")
        print("   Option 2: powershell .\\start_tori_hardened.ps1")
        print("   Option 3: python enhanced_launcher.py")
        print("\n🔧 To fix issues:")
        print("   python fix_frontend_and_bridges.py")
        print("\n🔍 To monitor:")
        print("   python tori_system_monitor.py")
    
    return 0 if all_healthy else 1

if __name__ == "__main__":
    sys.exit(main())
