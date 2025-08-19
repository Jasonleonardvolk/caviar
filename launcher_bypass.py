#!/usr/bin/env python3
"""
Quick fix to temporarily bypass the config validation error
"""

import os
import sys

# Remove problematic environment variables temporarily
env_vars_to_remove = ['SKIP_PREFLIGHT_CHECK', 'skip_preflight_check', 'PORT', 'port']

print("🔧 Removing problematic environment variables temporarily...")
for var in env_vars_to_remove:
    if var in os.environ:
        print(f"  Removing: {var} = {os.environ[var]}")
        del os.environ[var]

# Now run the enhanced launcher
print("\n🚀 Starting enhanced launcher...")
os.system("python enhanced_launcher.py")
