#!/usr/bin/env python3
"""
TORI Startup Sequence Script
Ensures proper startup order to prevent proxy errors
"""

import subprocess
import time
import requests
import sys
from pathlib import Path

def check_backend_health():
    """Check if backend is ready"""
    try:
        response = requests.get("http://localhost:8002/api/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def check_soliton_endpoints():
    """Check if soliton endpoints are working"""
    try:
        response = requests.get("http://localhost:8002/api/soliton/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def main():
    print("🚀 Starting TORI with Proper Sequencing...")
    print("=" * 50)
    
    # Start backend first
    print("1️⃣ Starting Backend (Python API)...")
    backend_proc = subprocess.Popen([
        sys.executable, "enhanced_launcher_improved.py"
    ], cwd=Path(__file__).parent)
    
    # Wait for backend to be ready
    print("2️⃣ Waiting for backend to be ready...")
    max_wait = 30  # 30 seconds max
    wait_time = 0
    
    while wait_time < max_wait:
        if check_backend_health():
            print("   ✅ Backend health check passed!")
            break
        print(f"   ⏳ Waiting for backend... ({wait_time}s)")
        time.sleep(2)
        wait_time += 2
    else:
        print("   ❌ Backend failed to start within 30 seconds")
        backend_proc.terminate()
        return 1
    
    # Check soliton endpoints
    print("3️⃣ Verifying soliton endpoints...")
    if check_soliton_endpoints():
        print("   ✅ Soliton endpoints ready!")
    else:
        print("   ⚠️  Soliton endpoints not ready (but continuing)")
    
    # Start frontend
    print("4️⃣ Starting Frontend (Svelte)...")
    frontend_proc = subprocess.Popen([
        "npm", "run", "dev"
    ], cwd=Path(__file__).parent / "tori_ui_svelte")
    
    print("\n🎉 TORI Started Successfully!")
    print("📊 Backend: http://localhost:8002")
    print("🌐 Frontend: http://localhost:5173")
    print("\n💡 This prevents proxy errors by ensuring backend is ready first!")
    print("🛑 Press Ctrl+C to stop both services")
    
    try:
        # Keep both processes running
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down TORI...")
        frontend_proc.terminate()
        backend_proc.terminate()
        print("✅ Clean shutdown complete!")

if __name__ == "__main__":
    main()
