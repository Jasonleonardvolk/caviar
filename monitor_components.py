#!/usr/bin/env python3
"""
Monitor TORI component initialization in real-time
"""
import requests
import time
import sys
from datetime import datetime

def monitor_components(api_port=8002):
    """Monitor component readiness"""
    print("🔍 TORI Component Registry Monitor")
    print("=" * 50)
    
    start_time = time.time()
    all_ready = False
    
    while True:
        try:
            # Get component status
            response = requests.get(f'http://localhost:{api_port}/api/system/components', timeout=1)
            if response.status_code == 200:
                components = response.json()
                
                # Clear screen (works on Windows and Unix)
                print('\033[2J\033[H')
                
                # Print header
                elapsed = time.time() - start_time
                print(f"🔍 TORI Component Status | Elapsed: {elapsed:.1f}s")
                print("=" * 50)
                
                # Check overall readiness
                ready_response = requests.get(f'http://localhost:{api_port}/api/system/ready', timeout=1)
                system_ready = ready_response.status_code == 200
                
                if system_ready and not all_ready:
                    all_ready = True
                    print("🎉 SYSTEM READY! All components initialized!")
                    print("=" * 50)
                elif not system_ready:
                    print("⏳ System initializing...")
                    print("=" * 50)
                
                # Print component status
                if components:
                    ready_count = sum(1 for ready in components.values() if ready)
                    total_count = len(components)
                    
                    print(f"\nComponents: {ready_count}/{total_count} ready\n")
                    
                    for name, ready in sorted(components.items()):
                        status = "✅" if ready else "⏳"
                        print(f"  {status} {name}")
                else:
                    print("\nNo components registered yet...")
                
                print("\n[Press Ctrl+C to exit]")
                
            else:
                print(f"⚠️ API not responding (status: {response.status_code})")
                
        except requests.exceptions.ConnectionError:
            print("⏳ Waiting for API to start...")
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(1)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
    monitor_components(port)
