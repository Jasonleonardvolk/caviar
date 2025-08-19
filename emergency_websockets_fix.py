#!/usr/bin/env python3
"""
EMERGENCY WEBSOCKETS + UNICODE FIX
Fixes websockets installation AND Unicode crashes on Windows
"""

import subprocess
import sys
import os
import time


def force_install_websockets():
    """Force install websockets in the CURRENT Python environment"""
    print("=== EMERGENCY WEBSOCKETS INSTALLER ===")
    print("Fixing environment mismatch and installing in current Python...")
    
    # Get the exact Python executable being used
    python_exe = sys.executable
    print(f"Using Python: {python_exe}")
    
    success = False
    
    # Method 1: Direct pip install in current environment
    print("Method 1: Direct pip install...")
    try:
        result = subprocess.run([
            python_exe, '-m', 'pip', 'install', 'websockets'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("SUCCESS: Websockets installed via direct pip")
            success = True
        else:
            print(f"FAILED: {result.stderr}")
    except Exception as e:
        print(f"FAILED: {e}")
    
    # Method 2: Force upgrade
    if not success:
        print("Method 2: Force upgrade...")
        try:
            result = subprocess.run([
                python_exe, '-m', 'pip', 'install', '--upgrade', '--force-reinstall', 'websockets'
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("SUCCESS: Websockets force-installed")
                success = True
            else:
                print(f"FAILED: {result.stderr}")
        except Exception as e:
            print(f"FAILED: {e}")
    
    # Method 3: Poetry in current environment
    if not success:
        print("Method 3: Poetry activate + install...")
        try:
            # Try to get poetry env info
            env_result = subprocess.run(['poetry', 'env', 'info', '--path'], 
                                      capture_output=True, text=True, timeout=30)
            if env_result.returncode == 0:
                poetry_env = env_result.stdout.strip()
                print(f"Poetry environment: {poetry_env}")
                
                # Install in poetry env
                install_result = subprocess.run([
                    'poetry', 'run', 'pip', 'install', 'websockets'
                ], capture_output=True, text=True, timeout=120)
                
                if install_result.returncode == 0:
                    print("SUCCESS: Websockets installed in poetry environment")
                    success = True
        except Exception as e:
            print(f"FAILED: {e}")
    
    # Verify installation
    print("\\nVerifying installation...")
    try:
        result = subprocess.run([
            python_exe, '-c', 'import websockets; print(f"SUCCESS: {websockets.__version__}")'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"VERIFIED: {result.stdout.strip()}")
            return True
        else:
            print(f"VERIFICATION FAILED: {result.stderr}")
            return False
    except Exception as e:
        print(f"VERIFICATION ERROR: {e}")
        return False


if __name__ == "__main__":
    # Set console encoding for Windows
    if sys.platform.startswith('win'):
        os.system('chcp 65001 >nul 2>&1')  # Set to UTF-8
    
    success = force_install_websockets()
    
    if success:
        print("\\n=== WEBSOCKETS INSTALLATION COMPLETE ===")
        print("Ready to run TORI with working hologram bridges!")
    else:
        print("\\n=== INSTALLATION FAILED ===")
        print("Bridges will run in mock mode (still functional)")
