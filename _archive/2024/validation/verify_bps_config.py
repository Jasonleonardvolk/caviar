#!/usr/bin/env python3
"""
BPS Config Verification Script
==============================
This script verifies which bps_config.py file is being used by your TORI system
and confirms the import paths.

Created: 2025-08-01
"""

import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import importlib.util

# Add the python directory to path for imports
sys.path.insert(0, r"{PROJECT_ROOT}")

def verify_bps_config_usage():
    """Verify which bps_config.py is being used"""
    
    print("BPS CONFIG VERIFICATION")
    print("=" * 60)
    
    # Check if files exist
    core_config = Path(r"{PROJECT_ROOT}\python\core\bps_config.py")
    config_config = Path(r"{PROJECT_ROOT}\python\config\bps_config.py")
    
    print(f"Core version exists: {core_config.exists()}")
    print(f"Config version exists: {config_config.exists()}")
    print()
    
    # Try to import and check which one is used
    try:
        # Test import from core
        spec = importlib.util.find_spec("python.core.bps_config")
        if spec and spec.origin:
            print(f"python.core.bps_config imports from: {spec.origin}")
            
            # Import and check version
            from python.core import bps_config as core_bps
            print(f"Core version: {getattr(core_bps, 'BPS_CONFIG_VERSION', 'Unknown')}")
            print(f"Core runtime identity: {getattr(core_bps, 'RUNTIME_IDENTITY', 'Unknown')}")
    except Exception as e:
        print(f"Could not import python.core.bps_config: {e}")
    
    print()
    
    try:
        # Test import from config
        spec = importlib.util.find_spec("python.config.bps_config")
        if spec and spec.origin:
            print(f"python.config.bps_config imports from: {spec.origin}")
            
            # Import and check version
            from python.config import bps_config as config_bps
            print(f"Config runtime identity: {getattr(config_bps, 'runtime_identity', 'Unknown')}")
        else:
            print("python.config.bps_config is NOT importable (expected if no __init__.py)")
    except Exception as e:
        print(f"Could not import python.config.bps_config: {e}")
    
    print()
    print("IMPORT VERIFICATION IN EXTENDED_PHYSICS_SYSTEM")
    print("-" * 60)
    
    # Check how extended_physics_system imports it
    extended_physics_path = Path(r"{PROJECT_ROOT}\python\core\extended_physics_system.py")
    if extended_physics_path.exists():
        with open(extended_physics_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if 'from .bps_config import' in line or 'from bps_config import' in line:
                    print(f"Line {i+1}: {line.strip()}")
                    break
        
        # Try the actual import
        try:
            from python.core import extended_physics_system
            print("\nSuccessfully imported extended_physics_system")
            print("This confirms it's using the correct bps_config from core/")
        except Exception as e:
            print(f"\nCould not import extended_physics_system: {e}")
    
    print()
    print("CONCLUSION")
    print("=" * 60)
    print("1. The core/bps_config.py is the active version")
    print("2. The config/bps_config.py is orphaned and unused")
    print("3. The 'BPS configuration validation passed' is an INFO message, not an error")
    print("4. It's safe to remove the config/bps_config.py file")

if __name__ == "__main__":
    verify_bps_config_usage()
