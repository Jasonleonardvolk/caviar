#!/usr/bin/env python3
"""
TORI System Shutdown Script
Gracefully stops all TORI components
"""

import psutil
import os
import time
import subprocess

def kill_process_by_name(name):
    """Kill all processes matching name"""
    killed = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            
            # Check if process matches
            if name.lower() in proc.info['name'].lower() or name.lower() in cmdline.lower():
                print(f"  Terminating {proc.info['name']} (PID: {proc.info['pid']})")
                proc.terminate()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    return killed

def kill_port(port):
    """Kill process using specific port"""
    if os.name == 'nt':  # Windows
        try:
            # Get process using port
            result = subprocess.run(
                f'netstat -ano | findstr :{port}',
                shell=True,
                capture_output=True,
                text=True
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
                
                # Kill each PID
                for pid in pids:
                    try:
                        subprocess.run(f'taskkill /PID {pid} /F', shell=True, capture_output=True)
                        print(f"  Killed process on port {port} (PID: {pid})")
                    except:
                        pass
        except:
            pass

def main():
    print("🛑 TORI System Shutdown")
    print("=" * 60)
    
    # Step 1: Stop TORI-specific processes
    print("\n📋 Stopping TORI processes...")
    
    processes_to_kill = [
        "uvicorn",           # API server
        "enhanced_launcher", # API server
        "npm",               # Frontend dev server
        "node",              # Frontend
        "redis-server",      # Redis
        "python audio_hologram_bridge",
        "python concept_mesh_hologram_bridge",
        "python mcp_metacognitive"
    ]
    
    total_killed = 0
    for process in processes_to_kill:
        killed = kill_process_by_name(process)
        total_killed += killed
    
    print(f"\n  Total processes terminated: {total_killed}")
    
    # Step 2: Free up ports
    print("\n🔌 Freeing up ports...")
    
    ports = [8002, 8100, 8765, 8766, 5173, 6379]
    for port in ports:
        kill_port(port)
    
    # Step 3: Clean up any Python scripts
    print("\n🐍 Stopping any remaining Python scripts...")
    
    # Kill specific Python scripts
    python_scripts = [
        "isolated_startup.py",
        "health_monitor.py",
        "auto_recovery.py",
        "enhanced_launcher.py",
        "master_launcher.py"
    ]
    
    for script in python_scripts:
        kill_process_by_name(script)
    
    # Step 4: Wait for processes to terminate
    print("\n⏳ Waiting for graceful shutdown...")
    time.sleep(3)
    
    # Step 5: Force kill any remaining
    print("\n🔨 Force killing any remaining processes...")
    
    if os.name == 'nt':
        # Windows specific cleanup
        subprocess.run('taskkill /F /IM python.exe 2>nul', shell=True, capture_output=True)
        subprocess.run('taskkill /F /IM node.exe 2>nul', shell=True, capture_output=True)
        subprocess.run('taskkill /F /IM redis-server.exe 2>nul', shell=True, capture_output=True)
    
    # Step 6: Verify ports are free
    print("\n✅ Verifying shutdown...")
    
    import socket
    all_free = True
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"  ⚠️  Port {port} is still in use")
            all_free = False
        else:
            print(f"  ✅ Port {port} is free")
    
    print("\n" + "="*60)
    
    if all_free:
        print("✅ TORI System shutdown complete!")
        print("All components stopped and ports freed.")
    else:
        print("⚠️  Some ports are still in use.")
        print("You may need to manually check Task Manager.")
    
    print("\nTo restart TORI, run: python master_launcher.py")

if __name__ == "__main__":
    main()
