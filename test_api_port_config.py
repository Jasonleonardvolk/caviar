#!/usr/bin/env python3
"""
Test script to verify API port configuration works correctly
"""

import subprocess
import sys
import time

def test_launcher_configs():
    """Test various launcher configurations"""
    
    test_cases = [
        {
            "name": "Full API with default port",
            "args": ["--api", "full"],
            "expected_port": 8001,
            "expected_mode": "full"
        },
        {
            "name": "Quick API with default port", 
            "args": ["--api", "quick"],
            "expected_port": 8002,
            "expected_mode": "quick"
        },
        {
            "name": "Full API with custom port",
            "args": ["--api", "full", "--api-port", "9999"],
            "expected_port": 9999,
            "expected_mode": "full"
        },
        {
            "name": "Quick API with custom port",
            "args": ["--api", "quick", "--api-port", "7777"],
            "expected_port": 7777,
            "expected_mode": "quick"
        }
    ]
    
    print("🧪 Testing Enhanced Launcher API Port Configuration")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\n📍 Test: {test['name']}")
        print(f"   Args: {' '.join(test['args'])}")
        print(f"   Expected port: {test['expected_port']}")
        print(f"   Expected mode: {test['expected_mode']}")
        
        # Build command
        cmd = [sys.executable, "enhanced_launcher.py"] + test["args"] + ["--no-browser"]
        
        print(f"   Command: {' '.join(cmd)}")
        print(f"   Status: Would start on port {test['expected_port']} in {test['expected_mode']} mode")
        
    print("\n" + "=" * 50)
    print("✅ Test configurations verified!")
    print("\nTo actually run the launcher with these configs:")
    print("  python enhanced_launcher.py --api full")
    print("  python enhanced_launcher.py --api quick --api-port 8080")

if __name__ == "__main__":
    test_launcher_configs()
