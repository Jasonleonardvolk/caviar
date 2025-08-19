#!/usr/bin/env python3
"""
TORI Complete System Launcher
Starts all backend services for the holographic system:
- Audio to Hologram Bridge
- Concept Mesh to Hologram Bridge
- Frontend Development Server
"""

import subprocess
import time
import sys
import os
import signal
from pathlib import Path
import asyncio
from typing import List

class TORICompleteLauncher:
    def __init__(self):
        self.processes = []
        self.root_dir = Path(__file__).parent
        
    def print_banner(self):
        print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🌟 TORI COMPLETE HOLOGRAPHIC SYSTEM LAUNCHER 🌟           ║
║                                                              ║
║         Audio + Concepts + Holograms = Magic!               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
    def check_requirements(self):
        """Check if all requirements are met"""
        print("🔍 Checking requirements...")
        
        # Check Python packages
        required_packages = [
            'websockets', 'torch', 'transformers', 'torchaudio',
            'numpy', 'networkx', 'dataclasses', 'json'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            print(f"❌ Missing Python packages: {', '.join(missing)}")
            print(f"   Run: pip install {' '.join(missing)}")
            return False
        
        print("✅ All Python packages installed")
        
        # Check if scripts exist
        scripts = [
            "audio_hologram_bridge.py",
            "concept_mesh_hologram_bridge.py"
        ]
        
        for script in scripts:
            script_path = self.root_dir / script
            if not script_path.exists():
                print(f"❌ Script not found: {script}")
                return False
                
        print("✅ All backend scripts found")
        
        # Check frontend
        frontend_path = self.root_dir / "tori_ui_svelte"
        if not frontend_path.exists():
            print(f"❌ Frontend not found at {frontend_path}")
            return False
            
        print("✅ Frontend directory found")
        
        return True
        
    def start_audio_backend(self):
        """Start the Python audio processing backend"""
        print("\n🎵 Starting Audio-Hologram Bridge...")
        
        cmd = [sys.executable, "audio_hologram_bridge.py"]
        process = subprocess.Popen(
            cmd,
            cwd=self.root_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes.append(('Audio Bridge', process))
        
        # Wait for startup
        time.sleep(2)
        
        if process.poll() is None:
            print("✅ Audio backend started on ws://localhost:8765")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Audio backend failed to start:")
            print(stderr)
            return False
            
    def start_concept_mesh_backend(self):
        """Start the Concept Mesh backend"""
        print("\n🧠 Starting Concept Mesh-Hologram Bridge...")
        
        cmd = [sys.executable, "concept_mesh_hologram_bridge.py"]
        process = subprocess.Popen(
            cmd,
            cwd=self.root_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes.append(('Concept Mesh Bridge', process))
        
        # Wait for startup
        time.sleep(2)
        
        if process.poll() is None:
            print("✅ Concept Mesh backend started on ws://localhost:8766")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Concept Mesh backend failed to start:")
            print(stderr)
            return False
            
    def start_frontend(self):
        """Start the Svelte frontend"""
        print("\n🎨 Starting Frontend...")
        
        frontend_dir = self.root_dir / "tori_ui_svelte"
        
        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
            
        # Update the import to use v2
        self.update_frontend_imports()
        
        # Start dev server
        cmd = ["npm", "run", "dev", "--", "--port", "5173", "--host"]
        process = subprocess.Popen(
            cmd,
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes.append(('Frontend', process))
        
        # Wait for startup
        time.sleep(5)
        
        if process.poll() is None:
            print("✅ Frontend started on http://localhost:5173")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Frontend failed to start:")
            print(stderr)
            return False
            
    def update_frontend_imports(self):
        """Update frontend to use the new engine with Concept Mesh"""
        print("📝 Updating frontend imports for Concept Mesh...")
        
        bridge_file = self.root_dir / "tori_ui_svelte/src/lib/holographicBridge.js"
        
        try:
            content = f"""// holographicBridge.js - Import bridge with Concept Mesh support
import {{ RealGhostEngine }} from './realGhostEngine_v2.js';

// For backward compatibility - alias RealGhostEngine as GhostEngine
export class GhostEngine extends RealGhostEngine {{
    constructor() {{
        super();
        console.log('🎉 Using REAL Ghost Engine with Concept Mesh support!');
    }}
}}

// Also export the real engine directly
export {{ RealGhostEngine }};

// Export a singleton instance for components that expect it
export const ghostEngine = new GhostEngine();

// Auto-initialize if we're in a browser with WebGPU
if (typeof window !== 'undefined' && navigator.gpu) {{
    window.TORI_GHOST_ENGINE = ghostEngine;
    console.log('✨ TORI Ghost Engine with Concept Mesh available globally');
}}

// Helper to check system status
export async function checkHolographicSystem() {{
    const status = {{
        webgpu: !!navigator.gpu,
        audioContext: !!(window.AudioContext || window.webkitAudioContext),
        websocket: typeof WebSocket !== 'undefined',
        conceptMesh: true,
        components: {{
            ghostEngine: true,
            shaders: await checkShaders(),
            audioBackend: await checkBackend('ws://localhost:8765/audio_stream'),
            conceptBackend: await checkBackend('ws://localhost:8766/concepts')
        }}
    }};
    
    console.log('🔍 Holographic System Status:', status);
    return status;
}}

async function checkShaders() {{
    try {{
        const response = await fetch('/shaders/propagation.wgsl');
        return response.ok;
    }} catch {{
        return false;
    }}
}}

async function checkBackend(wsUrl) {{
    try {{
        const ws = new WebSocket(wsUrl);
        return new Promise((resolve) => {{
            ws.onopen = () => {{
                ws.close();
                resolve(true);
            }};
            ws.onerror = () => resolve(false);
            setTimeout(() => resolve(false), 1000);
        }});
    }} catch {{
        return false;
    }}
}}

// Export all the holographic components
export * from './holographicEngine.js';
export * from './holographicRenderer.js';
export * from './conceptHologramRenderer.js';
export * from './webgpu/fftCompute.js';
export * from './webgpu/hologramPropagation.js';
export * from './webgpu/quiltGenerator.js';

console.log('🌉 Holographic Bridge with Concept Mesh loaded!');
"""
            
            with open(bridge_file, 'w') as f:
                f.write(content)
                
            print("✅ Frontend imports updated")
            
        except Exception as e:
            print(f"⚠️  Could not update imports: {e}")
            
    def open_browser(self):
        """Open the browser to the hologram interface"""
        import webbrowser
        
        print("\n🌐 Opening browser...")
        time.sleep(2)
        
        # Open with concept mesh enabled
        url = "http://localhost:5173?hologram=auto&display=webgpu_only&conceptMesh=true"
        webbrowser.open(url)
        
    def monitor_processes(self):
        """Monitor running processes"""
        print("\n✨ TORI Complete Holographic System is running!")
        print("\nAll services started:")
        print("  🎵 Audio Backend: ws://localhost:8765")
        print("  🧠 Concept Mesh: ws://localhost:8766")
        print("  🎨 Frontend: http://localhost:5173")
        print("\nPress Ctrl+C to stop all services\n")
        
        print("📊 Quick Start Guide:")
        print("1. Click 'Start Hologram' in the browser")
        print("2. Allow microphone access when prompted")
        print("3. Make some noise - concepts will appear!")
        print("4. Click on concepts to explore relationships")
        print("5. Press 'H' for holographic mode")
        print("6. Press Space to capture holographic moments")
        print("\n" + "="*60 + "\n")
        
        try:
            while True:
                # Check if processes are still running
                for name, proc in self.processes:
                    if proc.poll() is not None:
                        print(f"⚠️  {name} stopped unexpectedly")
                        
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down TORI Complete System...")
            
    def cleanup(self):
        """Clean up all processes"""
        for name, proc in self.processes:
            if proc.poll() is None:
                print(f"  Stopping {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    
        print("✅ All services stopped")
        
    def run(self):
        """Main launcher sequence"""
        self.print_banner()
        
        if not self.check_requirements():
            print("\n❌ Requirements check failed. Please install missing components.")
            return 1
            
        try:
            # Start services in order
            if not self.start_audio_backend():
                return 1
                
            if not self.start_concept_mesh_backend():
                return 1
                
            if not self.start_frontend():
                return 1
                
            # Open browser
            self.open_browser()
            
            # Monitor
            self.monitor_processes()
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return 1
            
        finally:
            self.cleanup()
            
        return 0

if __name__ == "__main__":
    launcher = TORICompleteLauncher()
    sys.exit(launcher.run())
