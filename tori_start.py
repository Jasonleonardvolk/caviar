#!/usr/bin/env python3
"""
TORI Startup with AV Fix
This script ensures the mock av module is loaded before starting any TORI components
"""

import sys
import os

# Add current directory to the FRONT of sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import our mock av FIRST
print("Loading AV compatibility fix...")
import av
print(f"Mock av loaded from: {av.__file__}")

# Now import and run the desired component
import argparse

parser = argparse.ArgumentParser(description='TORI Startup with AV Fix')
parser.add_argument('component', choices=['api', 'launcher', 'isolated', 'test'],
                    help='Component to start: api, launcher, isolated, or test')
args = parser.parse_args()

print(f"\nStarting TORI {args.component}...")

try:
    if args.component == 'api':
        print("Starting API server...")
        import enhanced_launcher
        enhanced_launcher.main()
    elif args.component == 'launcher':
        print("Starting enhanced launcher...")
        import enhanced_launcher
        enhanced_launcher.main()
    elif args.component == 'isolated':
        print("Starting isolated startup...")
        import isolated_startup
        isolated_startup.main()
    elif args.component == 'test':
        print("Running entropy test...")
        import test_entropy_with_spec
except Exception as e:
    print(f"Error starting {args.component}: {e}")
    import traceback
    traceback.print_exc()
