#!/usr/bin/env python3
"""
Kill processes on TORI ports
"""
import psutil
import sys

# TORI ports
PORTS = [5173, 8002, 8003, 8100, 8765, 8766]

def kill_port(port):
    """Kill all processes using a specific port"""
    killed = 0
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections():
                if conn.laddr.port == port:
                    print(f"Killing {proc.info['name']} (PID: {proc.info['pid']}) on port {port}")
                    proc.terminate()
                    killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return killed

def main():
    print("🧹 Cleaning TORI ports...")
    total_killed = 0
    
    for port in PORTS:
        killed = kill_port(port)
        if killed > 0:
            print(f"✅ Killed {killed} process(es) on port {port}")
        else:
            print(f"✅ Port {port} already free")
        total_killed += killed
    
    print(f"\n🎉 Cleaned {total_killed} processes total")

if __name__ == "__main__":
    main()
