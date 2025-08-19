from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
#!/usr/bin/env python3
"""
Start Tori backend with debug logging
"""

import os
import sys
import subprocess

def start_backend():
    project_root = r"{PROJECT_ROOT}"
    os.chdir(project_root)
    
    # Set environment variables
    env = os.environ.copy()
    env['LOG_LEVEL'] = 'DEBUG'
    env['TORI_DISABLE_MESH_CHECK'] = '1'
    
    print("Starting Tori backend with debug logging...")
    print(f"Working directory: {os.getcwd()}")
    print(f"TORI_DISABLE_MESH_CHECK: {env.get('TORI_DISABLE_MESH_CHECK')}")
    print("-" * 60)
    
    # Start uvicorn
    cmd = [
        sys.executable, '-m', 'uvicorn',
        'api.main:app',
        '--host', '127.0.0.1',
        '--port', '5173',
        '--reload',
        '--log-level', 'debug'
    ]
    
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\nBackend stopped.")

if __name__ == "__main__":
    start_backend()
