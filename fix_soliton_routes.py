"""
Quick Fix: Enable Real Soliton Routes
=====================================

This script restarts TORI with the real soliton routes enabled.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def main():
    print("🔧 TORI Soliton Routes Fix")
    print("=" * 50)
    
    # Kill existing TORI processes
    print("🛑 Stopping existing TORI processes...")
    try:
        subprocess.run(["taskkill", "/F", "/IM", "python.exe", "/FI", "WINDOWTITLE eq TORI*"], 
                      capture_output=True, shell=True)
        time.sleep(2)
    except:
        pass
    
    # Clear any port locks
    print("🔓 Clearing port locks...")
    ports = [8002, 5173, 8100, 8765, 8766]
    for port in ports:
        try:
            subprocess.run(f"netstat -ano | findstr :{port} | findstr LISTENING", 
                          shell=True, capture_output=True)
        except:
            pass
    
    print("⏳ Waiting for ports to clear...")
    time.sleep(3)
    
    # Start enhanced launcher
    print("🚀 Starting TORI with real soliton routes...")
    launcher_path = Path(__file__).parent / "enhanced_launcher.py"
    
    if launcher_path.exists():
        subprocess.Popen([sys.executable, str(launcher_path)], 
                        creationflags=subprocess.CREATE_NEW_CONSOLE)
        print("✅ TORI launcher started!")
        print("\n📝 Next steps:")
        print("1. Wait for the system to fully start (~10-15 seconds)")
        print("2. Check http://localhost:8002/docs")
        print("3. Look for the real soliton endpoints:")
        print("   - POST /api/soliton/init")
        print("   - POST /api/soliton/store")
        print("   - GET /api/soliton/stats/{user_id}")
        print("   - POST /api/soliton/embed")
        print("\n4. The frontend at http://localhost:5173 should now work!")
    else:
        print(f"❌ Launcher not found at: {launcher_path}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
