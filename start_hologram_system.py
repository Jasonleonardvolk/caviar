#!/usr/bin/env python3
"""
TORI Holographic System Startup Script
Automatically starts all components in the correct order
"""

import subprocess
import time
import sys
import os
import signal
import asyncio
from pathlib import Path

class TORIHolographicLauncher:
    def __init__(self):
        self.processes = []
        self.root_dir = Path(__file__).parent
        
    def print_banner(self):
        print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         🌟 TORI HOLOGRAPHIC SYSTEM LAUNCHER 🌟                ║
║                                                              ║
║         Connecting all the amazing pieces you built!         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
    def check_requirements(self):
        """Check if all requirements are met"""
        print("🔍 Checking requirements...")
        
        # Check Python packages
        try:
            import websockets
            import torch
            import transformers
            import torchaudio
            print("✅ Python packages installed")
        except ImportError as e:
            print(f"❌ Missing Python package: {e}")
            print("   Run: pip install websockets torch transformers torchaudio")
            return False
            
        # Check if audio_hologram_bridge.py exists
        bridge_path = self.root_dir / "audio_hologram_bridge.py"
        if not bridge_path.exists():
            print(f"❌ audio_hologram_bridge.py not found at {bridge_path}")
            return False
        print("✅ Audio bridge script found")
        
        # Check if frontend exists
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
        
        self.processes.append(process)
        
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
            
    def start_frontend(self):
        """Start the Svelte frontend"""
        print("\n🎨 Starting Frontend...")
        
        frontend_dir = self.root_dir / "tori_ui_svelte"
        
        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
            
        # Start dev server
        cmd = ["npm", "run", "dev", "--", "--port", "5173", "--host"]
        process = subprocess.Popen(
            cmd,
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes.append(process)
        
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
            
    def open_browser(self):
        """Open the browser to the hologram interface"""
        import webbrowser
        
        print("\n🌐 Opening browser...")
        time.sleep(2)
        
        # Open with test parameters
        url = "http://localhost:5173?hologram=auto&display=webgpu_only"
        webbrowser.open(url)
        
    def monitor_processes(self):
        """Monitor running processes"""
        print("\n✨ TORI Holographic System is running!")
        print("\nPress Ctrl+C to stop all services\n")
        
        print("📊 Quick Start Guide:")
        print("1. Click 'Start Hologram' in the browser")
        print("2. Allow microphone access when prompted")
        print("3. Make some noise and watch the hologram respond!")
        print("4. Press 'H' for holographic mode")
        print("5. Press Space to capture holographic moments")
        print("\n" + "="*60 + "\n")
        
        try:
            while True:
                # Check if processes are still running
                for i, proc in enumerate(self.processes):
                    if proc.poll() is not None:
                        print(f"⚠️  Process {i} stopped unexpectedly")
                        
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down TORI Holographic System...")
            
    def cleanup(self):
        """Clean up all processes"""
        for proc in self.processes:
            if proc.poll() is None:
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
            # Start services
            if not self.start_audio_backend():
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
    launcher = TORIHolographicLauncher()
    sys.exit(launcher.run())
