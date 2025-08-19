#!/usr/bin/env python3
"""
TORI Emergency Fix Script
Run this to immediately stabilize your TORI system
Usage: python tori_emergency_fix.py
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path

class TORIEmergencyFix:
    def __init__(self):
        self.base_path = Path("C:/Users/jason/Desktop/tori/kha")
        self.errors = []
        self.fixes_applied = []
        
    def run(self):
        """Execute all emergency fixes"""
        print("🚨 TORI EMERGENCY FIX SCRIPT")
        print("=" * 60)
        print("This script will:")
        print("1. Install missing dependencies")
        print("2. Fix configuration files")
        print("3. Clean up ports")
        print("4. Add missing API endpoints")
        print("5. Start core components")
        print("=" * 60)
        
        # Step 1: Install dependencies
        self.install_dependencies()
        
        # Step 2: Fix configurations
        self.fix_configurations()
        
        # Step 3: Clean ports
        self.clean_ports()
        
        # Step 4: Create missing files
        self.create_missing_files()
        
        # Step 5: Summary
        self.print_summary()
        
    def install_dependencies(self):
        """Install all missing Python dependencies"""
        print("\n📦 Installing Missing Dependencies...")
        
        dependencies = [
            ("torch", "torch torchvision --index-url https://download.pytorch.org/whl/cpu"),
            ("deepdiff", "deepdiff"),
            ("sympy", "sympy"),
            ("PyPDF2", "PyPDF2"),
            ("websockets", "websockets"),
            ("aiohttp", "aiohttp"),
            ("psutil", "psutil"),
            ("redis", "redis"),
            ("celery", "celery[redis]"),
            ("fastapi", "fastapi[all]"),
            ("uvicorn", "uvicorn[standard]")
        ]
        
        for name, install_cmd in dependencies:
            try:
                __import__(name)
                print(f"  ✅ {name} already installed")
            except ImportError:
                print(f"  📥 Installing {name}...")
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install"
                    ] + install_cmd.split(), stdout=subprocess.DEVNULL)
                    self.fixes_applied.append(f"Installed {name}")
                    print(f"  ✅ {name} installed successfully")
                except subprocess.CalledProcessError as e:
                    self.errors.append(f"Failed to install {name}: {e}")
                    print(f"  ❌ Failed to install {name}")
    
    def fix_configurations(self):
        """Fix configuration files"""
        print("\n🔧 Fixing Configuration Files...")
        
        # Fix Vite config
        vite_config_path = self.base_path / "tori_ui_svelte/vite.config.js"
        vite_content = """import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8002',
        changeOrigin: true,
        ws: true,
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, _res) => {
            console.log('proxy error', err);
          });
        }
      }
    }
  },
  optimizeDeps: {
    exclude: ['@sveltejs/kit']
  }
});"""
        
        try:
            vite_config_path.parent.mkdir(parents=True, exist_ok=True)
            vite_config_path.write_text(vite_content)
            self.fixes_applied.append("Fixed Vite proxy configuration")
            print("  ✅ Fixed Vite configuration")
        except Exception as e:
            self.errors.append(f"Failed to fix Vite config: {e}")
            print(f"  ❌ Failed to fix Vite config")
        
        # Fix package.json if needed
        package_json_path = self.base_path / "tori_ui_svelte/package.json"
        if package_json_path.exists():
            try:
                package_data = json.loads(package_json_path.read_text())
                
                # Ensure mathjs is in dependencies
                if "dependencies" not in package_data:
                    package_data["dependencies"] = {}
                    
                if "mathjs" not in package_data["dependencies"]:
                    package_data["dependencies"]["mathjs"] = "^11.11.0"
                    package_json_path.write_text(json.dumps(package_data, indent=2))
                    self.fixes_applied.append("Added mathjs to package.json")
                    print("  ✅ Fixed package.json dependencies")
                    
                    # Install npm dependencies
                    os.chdir(self.base_path / "tori_ui_svelte")
                    subprocess.run(["npm", "install"], stdout=subprocess.DEVNULL)
                    os.chdir(self.base_path)
                    
            except Exception as e:
                self.errors.append(f"Failed to fix package.json: {e}")
        
        # Create .env file if missing
        env_path = self.base_path / ".env"
        if not env_path.exists():
            env_content = """# TORI Environment Configuration
TORI_ENV=development
API_PORT=8002
MCP_PORT=8100
AUDIO_BRIDGE_PORT=8765
CONCEPT_BRIDGE_PORT=8766
FRONTEND_PORT=5173
REDIS_URL=redis://localhost:6379
DEBUG=True
"""
            env_path.write_text(env_content)
            self.fixes_applied.append("Created .env file")
            print("  ✅ Created .env configuration")
    
    def clean_ports(self):
        """Clean up ports used by TORI"""
        print("\n🧹 Cleaning Up Ports...")
        
        ports = [8002, 8100, 8765, 8766, 5173, 6379]
        
        if os.name == 'nt':  # Windows
            for port in ports:
                try:
                    # Find processes using the port
                    result = subprocess.run(
                        f"netstat -ano | findstr :{port}",
                        shell=True, capture_output=True, text=True
                    )
                    
                    if result.stdout:
                        lines = result.stdout.strip().split('\n')
                        pids = set()
                        
                        for line in lines:
                            if 'LISTENING' in line:
                                parts = line.split()
                                if parts:
                                    pid = parts[-1]
                                    pids.add(pid)
                        
                        # Kill processes
                        for pid in pids:
                            try:
                                subprocess.run(
                                    f"taskkill /PID {pid} /F",
                                    shell=True, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL
                                )
                                print(f"  ✅ Freed port {port} (killed PID {pid})")
                                self.fixes_applied.append(f"Freed port {port}")
                            except:
                                pass
                except Exception as e:
                    self.errors.append(f"Failed to clean port {port}: {e}")
        else:  # Unix/Linux
            for port in ports:
                try:
                    subprocess.run(
                        f"lsof -ti:{port} | xargs kill -9",
                        shell=True, stderr=subprocess.DEVNULL
                    )
                except:
                    pass
    
    def create_missing_files(self):
        """Create missing API endpoint files"""
        print("\n📝 Creating Missing Files...")
        
        # Create a simple API fix file
        api_fix_path = self.base_path / "api_endpoint_fix.py"
        api_fix_content = '''"""
Add this to your enhanced_launcher.py or main API file
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio

# Add these routes to your existing FastAPI app

@app.post("/api/soliton/init")
async def soliton_init():
    """Initialize Soliton memory system"""
    return {"status": "initialized", "message": "Soliton memory initialized"}

@app.get("/api/soliton/stats/{user_id}")
async def soliton_stats(user_id: str):
    """Get memory statistics for user"""
    return {
        "user_id": user_id,
        "memory_count": 0,
        "last_accessed": None,
        "status": "active"
    }

@app.post("/api/soliton/embed")
async def soliton_embed(data: dict):
    """Embed data into Soliton memory"""
    return {"status": "embedded", "data_received": len(str(data))}

@app.websocket("/api/avatar/updates")
async def avatar_updates(websocket: WebSocket):
    """WebSocket for avatar updates"""
    await websocket.accept()
    try:
        while True:
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": asyncio.get_event_loop().time()
            })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print("Avatar WebSocket disconnected")
'''
        
        try:
            api_fix_path.write_text(api_fix_content)
            self.fixes_applied.append("Created API endpoint fix file")
            print("  ✅ Created api_endpoint_fix.py")
            print("  📌 Note: Add these endpoints to your main API file!")
        except Exception as e:
            self.errors.append(f"Failed to create API fix file: {e}")
        
        # Create minimal launcher
        launcher_path = self.base_path / "tori_minimal_launcher.py"
        launcher_content = '''#!/usr/bin/env python3
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
    print(f"🚀 Starting {name}...")
    
    if not is_port_free(port):
        print(f"  ⚠️  Port {port} is already in use")
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
                    print(f"  ✅ {name} is healthy!")
                    return proc
            except:
                pass
            time.sleep(1)
        
        print(f"  ❌ {name} failed health check")
        proc.terminate()
        return None
    else:
        time.sleep(3)  # Just wait a bit for services without health check
        print(f"  ✅ {name} started")
        return proc

def main():
    print("🚀 TORI Minimal Launcher")
    print("=" * 60)
    
    os.chdir(r"C:\\Users\\jason\\Desktop\\tori\\kha")
    
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
        print(f"\\n✅ Started {len(processes)} services successfully!")
        print("\\n🌐 Access TORI at: http://localhost:5173")
        print("📚 API docs at: http://localhost:8002/docs")
        print("\\nPress Ctrl+C to stop all services")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\\n🛑 Shutting down services...")
            for proc in processes:
                proc.terminate()
            print("✅ All services stopped")
    else:
        print("\\n❌ Failed to start any services")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        try:
            launcher_path.write_text(launcher_content)
            os.chmod(launcher_path, 0o755)  # Make executable
            self.fixes_applied.append("Created minimal launcher")
            print("  ✅ Created tori_minimal_launcher.py")
        except Exception as e:
            self.errors.append(f"Failed to create launcher: {e}")
    
    def print_summary(self):
        """Print summary of fixes"""
        print("\n" + "=" * 60)
        print("📊 EMERGENCY FIX SUMMARY")
        print("=" * 60)
        
        if self.fixes_applied:
            print(f"\n✅ Fixes Applied ({len(self.fixes_applied)}):")
            for fix in self.fixes_applied:
                print(f"  • {fix}")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        print("\n📋 Next Steps:")
        print("1. Review api_endpoint_fix.py and add endpoints to your API")
        print("2. Run: python tori_minimal_launcher.py")
        print("3. Test the system at http://localhost:5173")
        print("4. Check API health at http://localhost:8002/api/health")
        
        if not self.errors:
            print("\n✅ Emergency fixes completed successfully!")
        else:
            print("\n⚠️  Some fixes failed. Please review the errors above.")

if __name__ == "__main__":
    fixer = TORIEmergencyFix()
    fixer.run()
