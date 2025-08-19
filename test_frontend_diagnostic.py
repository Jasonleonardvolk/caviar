#!/usr/bin/env python3
"""
Frontend Diagnostic Tool - Test SvelteKit frontend independently
"""
import subprocess
import time
import requests
import sys
import os
from pathlib import Path

def test_frontend():
    """Test frontend startup and health independently"""
    print("🔍 Frontend Diagnostic Tool")
    print("=" * 50)
    
    # Check if we're in the right directory
    frontend_dir = Path("C:/Users/jason/Desktop/tori/kha/tori_ui_svelte")
    if not frontend_dir.exists():
        print(f"❌ Frontend directory not found: {frontend_dir}")
        return False
    
    os.chdir(frontend_dir)
    print(f"📂 Working directory: {frontend_dir}")
    
    # Check package.json
    if not (frontend_dir / "package.json").exists():
        print("❌ package.json not found")
        return False
    
    print("✅ package.json found")
    
    # Check node_modules
    if not (frontend_dir / "node_modules").exists():
        print("⚠️ node_modules not found - running npm install...")
        subprocess.run("npm install", shell=True)
    else:
        print("✅ node_modules exists")
    
    # Set environment variables
    env = os.environ.copy()
    env.update({
        'PORT': '5173',
        'HOST': '0.0.0.0',
        'NODE_OPTIONS': '--max-old-space-size=4096',
        'FORCE_COLOR': '0',
        'VITE_ENABLE_CONCEPT_MESH': 'true'
    })
    
    print("\n🚀 Starting frontend server...")
    print("Command: npm run dev")
    
    # Start frontend process
    proc = subprocess.Popen(
        'npm run dev',
        shell=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    print("\n📋 Frontend output:")
    print("-" * 50)
    
    # Monitor output and wait for ready
    start_time = time.time()
    ready = False
    
    while time.time() - start_time < 60:  # 60 second timeout
        if proc.poll() is not None:
            print(f"\n❌ Process exited with code: {proc.returncode}")
            break
        
        # Read output
        line = proc.stdout.readline()
        if line:
            print(line.strip())
            if "ready in" in line.lower() or "local:" in line.lower():
                ready = True
        
        # Check health endpoint
        if ready and time.time() - start_time > 5:  # Give it 5 seconds after "ready"
            try:
                response = requests.get('http://localhost:5173/health', timeout=2)
                if response.status_code == 200:
                    print(f"\n✅ Health check passed! Response: {response.text}")
                    print(f"⏱️ Started in {time.time() - start_time:.1f} seconds")
                    
                    # Test main page too
                    try:
                        main_response = requests.get('http://localhost:5173/', timeout=2)
                        print(f"✅ Main page responded with status: {main_response.status_code}")
                    except Exception as e:
                        print(f"⚠️ Main page error: {e}")
                    
                    return True
                else:
                    print(f"⚠️ Health check returned {response.status_code}")
            except requests.exceptions.ConnectionError:
                pass  # Server not ready yet
            except Exception as e:
                print(f"⚠️ Health check error: {e}")
        
        time.sleep(0.1)
    
    print(f"\n❌ Frontend failed to become healthy within 60 seconds")
    
    # Kill the process
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except:
        proc.kill()
    
    return False

if __name__ == "__main__":
    success = test_frontend()
    sys.exit(0 if success else 1)
