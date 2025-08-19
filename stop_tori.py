#!/usr/bin/env python3
"""
Shutdown helper to stop all TORI processes gracefully
"""

import os
import psutil
import time

def stop_tori_processes():
    """Stop all TORI-related processes"""
    print("🛑 Stopping TORI processes...")
    
    # Process names to look for
    process_names = [
        "python.exe",
        "python3.exe",
        "uvicorn",
        "node.exe"
    ]
    
    # Keywords to identify TORI processes
    keywords = [
        "enhanced_launcher",
        "main.py",
        "lattice_evolution",
        "mcp_metacognitive",
        "prajna",
        "tori"
    ]
    
    stopped = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if it's a relevant process
            if proc.info['name'] in process_names:
                cmdline = ' '.join(proc.info.get('cmdline', []))
                
                # Check if it's a TORI process
                if any(keyword in cmdline.lower() for keyword in keywords):
                    print(f"  Stopping {proc.info['name']} (PID: {proc.info['pid']})")
                    proc.terminate()
                    stopped += 1
                    
                    # Give it a moment to terminate
                    time.sleep(0.5)
                    
                    # Force kill if still running
                    if proc.is_running():
                        proc.kill()
                        print(f"  Force killed PID: {proc.info['pid']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    print(f"✅ Stopped {stopped} TORI processes")

if __name__ == "__main__":
    stop_tori_processes()
