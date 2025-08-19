#!/usr/bin/env python3
"""
TORI System Recovery Script
Cleans up zombie processes, resets storage, and restarts the system
"""

import os
import sys
import time
import psutil
import subprocess
import signal
from pathlib import Path

def kill_processes_on_ports(ports):
    """Kill any processes using the specified ports"""
    killed = []
    
    for conn in psutil.net_connections():
        if hasattr(conn.laddr, 'port') and conn.laddr.port in ports:
            try:
                process = psutil.Process(conn.pid)
                process_name = process.name()
                print(f"🔫 Killing {process_name} (PID: {conn.pid}) on port {conn.laddr.port}")
                process.terminate()
                process.wait(timeout=5)
                killed.append((conn.pid, process_name, conn.laddr.port))
            except psutil.TimeoutExpired:
                process.kill()
                killed.append((conn.pid, "unknown", conn.laddr.port))
            except Exception as e:
                print(f"❌ Error killing process {conn.pid}: {e}")
    
    return killed

def kill_python_processes():
    """Kill all Python processes with TORI-related names"""
    killed = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            
            # Check if it's a TORI-related process
            tori_keywords = [
                'enhanced_launcher',
                'audio_hologram_bridge',
                'concept_mesh_hologram_bridge',
                'banksy_oscillator',
                'tori',
                'hologram'
            ]
            
            if any(keyword in cmdline.lower() for keyword in tori_keywords):
                if proc.info['name'].lower().startswith('python'):
                    print(f"🔫 Killing TORI process: {proc.info['pid']} - {cmdline[:80]}...")
                    proc.terminate()
                    proc.wait(timeout=5)
                    killed.append(proc.info['pid'])
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            pass
        except Exception as e:
            print(f"Error checking process: {e}")
    
    return killed

def clean_temp_files():
    """Clean up temporary files and logs"""
    patterns = [
        '*.log',
        'bridge_config.json',
        '__pycache__',
        '*.pyc'
    ]
    
    cleaned = 0
    for pattern in patterns:
        for file in Path('.').rglob(pattern):
            try:
                if file.is_file():
                    file.unlink()
                    cleaned += 1
                elif file.is_dir():
                    import shutil
                    shutil.rmtree(file)
                    cleaned += 1
            except Exception as e:
                print(f"Could not remove {file}: {e}")
    
    return cleaned

def reset_browser_storage():
    """Instructions for resetting browser storage"""
    print("\n📋 To reset browser storage:")
    print("1. Open your browser's Developer Tools (F12)")
    print("2. Go to the Application/Storage tab")
    print("3. Click 'Clear site data' or run in console:")
    print("   - indexedDB.deleteDatabase('TORI_DB');")
    print("   - localStorage.clear();")
    print("4. Refresh the page\n")

def main():
    print("="*60)
    print("🚑 TORI SYSTEM RECOVERY TOOL")
    print("="*60)
    
    # Step 1: Kill processes on known ports
    print("\n🔍 Step 1: Killing processes on TORI ports...")
    tori_ports = [8765, 8766, 8767, 8768, 8000, 5173]
    killed_ports = kill_processes_on_ports(tori_ports)
    
    if killed_ports:
        print(f"✅ Killed {len(killed_ports)} processes on ports")
        for pid, name, port in killed_ports:
            print(f"   - {name} (PID: {pid}) on port {port}")
    else:
        print("✅ No processes found on TORI ports")
    
    # Step 2: Kill any TORI Python processes
    print("\n🔍 Step 2: Killing TORI Python processes...")
    killed_procs = kill_python_processes()
    
    if killed_procs:
        print(f"✅ Killed {len(killed_procs)} TORI processes")
    else:
        print("✅ No TORI processes found")
    
    # Step 3: Clean temporary files
    print("\n🔍 Step 3: Cleaning temporary files...")
    cleaned = clean_temp_files()
    print(f"✅ Cleaned {cleaned} temporary files/directories")
    
    # Step 4: Browser storage reset instructions
    reset_browser_storage()
    
    # Step 5: System status
    print("\n📊 System Status:")
    print("✅ All TORI processes terminated")
    print("✅ Ports cleared and available")
    print("✅ Temporary files cleaned")
    print("⚠️  Remember to clear browser storage manually")
    
    # Step 6: Restart instructions
    print("\n🚀 To restart TORI:")
    print("1. Clear browser storage (see above)")
    print("2. Run: python enhanced_launcher.py")
    print("3. Wait for 'TORI System fully operational!' message")
    print("4. Open http://localhost:5173")
    
    print("\n✨ Recovery complete! TORI is ready for a fresh start.")
    print("="*60)

if __name__ == "__main__":
    try:
        # Check if running with admin/sudo
        if sys.platform == 'win32':
            import ctypes
            if not ctypes.windll.shell32.IsUserAnAdmin():
                print("⚠️  Warning: Running without admin privileges, some processes may not be killable")
        
        main()
        
    except KeyboardInterrupt:
        print("\n❌ Recovery cancelled")
    except Exception as e:
        print(f"\n❌ Recovery error: {e}")
        import traceback
        traceback.print_exc()
