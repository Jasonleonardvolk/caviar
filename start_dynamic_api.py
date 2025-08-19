#!/usr/bin/env python3
"""
🚀 DYNAMIC PORT MANAGER - Smart port allocation for TORI API
Automatically finds available port and saves it for SvelteKit to discover
"""

import socket
import json
import os
import time
from pathlib import Path

class PortManager:
    def __init__(self, base_port=8002, max_attempts=10):
        self.base_port = base_port
        self.max_attempts = max_attempts
        self.config_file = Path(__file__).parent / "api_port.json"
    
    def is_port_available(self, port):
        """Check if a port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return True
        except OSError:
            return False
    
    def find_available_port(self):
        """Find the first available port starting from base_port"""
        for i in range(self.max_attempts):
            port = self.base_port + i
            if self.is_port_available(port):
                print(f"✅ Found available port: {port}")
                return port
            else:
                print(f"❌ Port {port} is busy")
        
        raise Exception(f"No available ports found in range {self.base_port}-{self.base_port + self.max_attempts}")
    
    def save_port_config(self, port):
        """Save the active port to config file for SvelteKit to read"""
        config = {
            "api_port": port,
            "api_url": f"http://localhost:{port}",
            "timestamp": time.time(),
            "status": "active"
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"📝 Saved port config: {self.config_file}")
        return config
    
    def get_active_port(self):
        """Get the currently active API port"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                return config.get("api_port")
            except:
                pass
        return None
    
    def kill_existing_process(self, port):
        """Kill any existing process on the port (Windows)"""
        try:
            import subprocess
            # Find process using the port
            result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        print(f"🔫 Killing existing process {pid} on port {port}")
                        subprocess.run(['taskkill', '/PID', pid, '/F'], capture_output=True)
                        time.sleep(1)  # Give it time to die
                        return True
        except Exception as e:
            print(f"⚠️ Failed to kill existing process: {e}")
        return False

def start_dynamic_api():
    """Start the API server on the first available port"""
    pm = PortManager()
    
    print("🚀 DYNAMIC API PORT MANAGER")
    print("=" * 40)
    
    # Check if we should kill existing process
    existing_port = pm.get_active_port()
    if existing_port:
        print(f"🔍 Found existing config for port {existing_port}")
        if not pm.is_port_available(existing_port):
            print(f"🔫 Port {existing_port} is busy, attempting to kill existing process...")
            pm.kill_existing_process(existing_port)
    
    # Find available port
    try:
        port = pm.find_available_port()
        config = pm.save_port_config(port)
        
        print(f"\n🎯 STARTING API SERVER:")
        print(f"   Port: {port}")
        print(f"   URL: http://localhost:{port}")
        print(f"   Health: http://localhost:{port}/health")
        print(f"   Docs: http://localhost:{port}/docs")
        
        # Start the server
        import uvicorn
        uvicorn.run(
            "ingest_pdf.main:app",  
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        # Clean up config file on failure
        if pm.config_file.exists():
            pm.config_file.unlink()

if __name__ == "__main__":
    start_dynamic_api()
