#!/usr/bin/env python3
"""
Direct Prajna Test - Bypass the launcher
"""

import subprocess
import sys
import os
from pathlib import Path

def test_prajna_direct():
    """Test starting Prajna directly to see actual errors"""
    print("🧪 Testing Prajna startup directly...")
    
    # Set up paths
    script_dir = Path(__file__).parent
    prajna_dir = script_dir / "prajna"
    start_script = prajna_dir / "start_prajna.py"
    
    print(f"📂 Prajna directory: {prajna_dir}")
    print(f"📄 Start script: {start_script}")
    
    if not start_script.exists():
        print("❌ Start script not found!")
        return
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = str(script_dir)  # Parent directory
    
    print(f"🔧 PYTHONPATH: {env['PYTHONPATH']}")
    print(f"🏠 Working directory: {prajna_dir}")
    
    # Command to run
    cmd = [
        sys.executable,
        str(start_script),
        "--port", "8001",
        "--host", "0.0.0.0", 
        "--log-level", "DEBUG"  # More verbose
    ]
    
    print(f"💻 Command: {' '.join(cmd)}")
    print("🚀 Starting Prajna directly (will show all output)...")
    print("-" * 50)
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            cwd=str(prajna_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        print(f"📊 Process started with PID: {process.pid}")
        
        # Monitor output in real-time
        timeout_counter = 0
        max_timeout = 30  # 30 seconds
        
        while True:
            output = process.stdout.readline()
            if output:
                print(f"PRAJNA: {output.strip()}")
                timeout_counter = 0  # Reset timeout if we get output
            elif process.poll() is not None:
                # Process has ended
                exit_code = process.poll()
                print(f"\n📊 Process exited with code: {exit_code}")
                break
            else:
                # No output, increment timeout
                timeout_counter += 1
                if timeout_counter >= max_timeout:
                    print(f"\n⏰ Timeout after {max_timeout} seconds with no output")
                    process.terminate()
                    break
                
                import time
                time.sleep(1)
        
        # Get any remaining output
        remaining_output, _ = process.communicate()
        if remaining_output:
            print(f"REMAINING: {remaining_output}")
            
    except Exception as e:
        print(f"❌ Error running direct test: {e}")

if __name__ == "__main__":
    test_prajna_direct()
