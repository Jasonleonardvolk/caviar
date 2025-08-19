#!/usr/bin/env python3
"""
Python wrapper to properly execute the PowerShell startup script
This allows you to run: python launch_tori_wrapper.py
"""

import subprocess
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Launch TORI system via PowerShell script')
    parser.add_argument('-f', '--force', action='store_true', 
                        help='Force kill processes on required ports')
    parser.add_argument('--skip-browser', action='store_true',
                        help='Skip browser/frontend startup')
    parser.add_argument('--skip-hologram', action='store_true',
                        help='Skip hologram services')
    
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ps1_script = os.path.join(script_dir, 'start_tori_hardened.ps1')
    
    # Build PowerShell command
    ps_args = ['powershell.exe', '-ExecutionPolicy', 'Bypass', '-File', ps1_script]
    
    # Add optional arguments
    if args.force:
        ps_args.append('-Force')
    if args.skip_browser:
        ps_args.append('-SkipBrowser')
    if args.skip_hologram:
        ps_args.append('-SkipHologram')
    
    print("Starting TORI system via PowerShell...")
    print(f"Executing: {' '.join(ps_args)}")
    print("-" * 60)
    
    try:
        # Run the PowerShell script
        result = subprocess.run(ps_args, check=False)
        
        # Exit with the same code as PowerShell
        sys.exit(result.returncode)
        
    except FileNotFoundError:
        print("ERROR: PowerShell not found. Please ensure PowerShell is installed and in PATH.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"ERROR: Failed to launch PowerShell script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
