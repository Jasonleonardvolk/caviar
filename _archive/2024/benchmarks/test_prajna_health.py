#!/usr/bin/env python3
"""
Quick Prajna Fix - Replace the problematic health check with a more robust one
"""

def check_prajna_really_started(port, max_attempts=10):
    """More robust check for Prajna startup"""
    import socket
    import time
    import requests
    
    print(f"🔍 Checking if port {port} is bound...")
    
    # First, wait for port to be bound
    for i in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('127.0.0.1', port))
                if result == 0:
                    print(f"✅ Port {port} is bound!")
                    break
        except:
            pass
        
        print(f"⏳ Waiting for port binding... ({i+1}/{max_attempts})")
        time.sleep(1)
    else:
        print(f"❌ Port {port} never became bound")
        return False
    
    # Now try health check
    print(f"🏥 Testing health endpoint...")
    for i in range(5):
        try:
            response = requests.get(f'http://127.0.0.1:{port}/api/health', timeout=3)
            if response.status_code == 200:
                print(f"✅ Health check passed!")
                return True
        except Exception as e:
            print(f"⏳ Health check attempt {i+1}: {e}")
        time.sleep(1)
    
    print(f"❌ Health check failed")
    return False

if __name__ == "__main__":
    # Test on port 8001
    result = check_prajna_really_started(8001)
    print(f"Result: {result}")
