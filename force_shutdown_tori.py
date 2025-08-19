#!/usr/bin/env python3
"""
Force shutdown script for TORI when Ctrl+C doesn't work
"""

import psutil
import subprocess
import sys
import time
from pathlib import Path

def force_shutdown_tori():
    """Force shutdown all TORI processes"""
    
    print("🛑 Force shutting down TORI processes...")
    
    killed_count = 0
    
    # Find all Python processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if it's a Python process
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'] or [])
                
                # Check if it's running TORI components
                if any(component in cmdline for component in [
                    'enhanced_launcher.py',
                    'prajna_api',
                    'mcp_metacognitive',
                    'lattice_evolution',
                    'uvicorn'
                ]):
                    print(f"  Killing PID {proc.info['pid']}: {proc.info['name']}")
                    proc.terminate()
                    killed_count += 1
                    
                    # Give it a moment
                    time.sleep(0.1)
                    
                    # Force kill if still alive
                    if proc.is_running():
                        proc.kill()
                        
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Also kill Node processes (frontend)
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'node' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'vite' in cmdline or '5173' in cmdline:
                    print(f"  Killing Node PID {proc.info['pid']}")
                    proc.terminate()
                    killed_count += 1
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Windows specific - kill by window title
    if sys.platform == "win32":
        subprocess.run('taskkill /FI "WINDOWTITLE eq TORI*" /F', shell=True, capture_output=True)
        subprocess.run('taskkill /FI "WINDOWTITLE eq Administrator*enhanced_launcher*" /F', shell=True, capture_output=True)
    
    print(f"\n✅ Killed {killed_count} processes")
    
    # Check what's still running
    time.sleep(1)
    still_running = []
    
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.cmdline())
                if 'tori' in cmdline.lower() or 'enhanced_launcher' in cmdline:
                    still_running.append(f"PID {proc.info['pid']}: {proc.info['name']}")
        except:
            pass
    
    if still_running:
        print("\n⚠️  Still running:")
        for proc in still_running:
            print(f"  {proc}")
    else:
        print("✅ All TORI processes stopped")

if __name__ == "__main__":
    force_shutdown_tori()
