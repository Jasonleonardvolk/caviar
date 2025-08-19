#!/usr/bin/env python3
"""
Simple Prajna Diagnostic Script
Captures and displays the actual error output from Prajna startup
"""

import subprocess
import sys
import os
from pathlib import Path

def diagnose_prajna_startup():
    """Simple diagnostic test for Prajna startup"""
    print("🔍 Prajna Startup Diagnostic")
    print("=" * 50)
    
    # Set up paths
    script_dir = Path(__file__).parent
    prajna_dir = script_dir / "prajna"
    start_script = prajna_dir / "start_prajna.py"
    
    print(f"📂 Prajna directory: {prajna_dir}")
    print(f"📄 Start script: {start_script}")
    print(f"✅ Start script exists: {start_script.exists()}")
    
    if not start_script.exists():
        print("❌ Start script not found!")
        return
    
    # Set up environment
    env = os.environ.copy()
    # PYTHONPATH should point to parent directory so 'prajna' package can be imported
    parent_dir = prajna_dir.parent
    env['PYTHONPATH'] = str(parent_dir)
    
    print(f"🔧 PYTHONPATH: {env['PYTHONPATH']}")
    print(f"🏠 Working directory: {prajna_dir}")
    
    # Prepare command
    cmd = [
        sys.executable,
        str(start_script),
        "--port", "8001",
        "--host", "0.0.0.0",
        "--log-level", "DEBUG"  # Use DEBUG for more info
    ]
    
    print(f"💻 Command: {' '.join(cmd)}")
    print("\n🚀 Starting Prajna process...")
    print("-" * 50)
    
    try:
        # Run the process and capture output
        result = subprocess.run(
            cmd,
            cwd=str(prajna_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        print(f"📊 Return code: {result.returncode}")
        print("\n📄 STDOUT:")
        print("-" * 30)
        if result.stdout:
            print(result.stdout)
        else:
            print("(No stdout output)")
        
        print("\n⚠️ STDERR:")
        print("-" * 30)
        if result.stderr:
            print(result.stderr)
        else:
            print("(No stderr output)")
            
    except subprocess.TimeoutExpired:
        print("⏰ Process timed out after 30 seconds")
    except Exception as e:
        print(f"❌ Error running process: {e}")
    
    print("\n" + "=" * 50)
    print("🔍 Diagnostic complete")

if __name__ == "__main__":
    diagnose_prajna_startup()
