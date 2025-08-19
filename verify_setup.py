#!/usr/bin/env python3
"""Verify the audio ingestion setup is correct"""

import sys
import subprocess
import importlib

def check_imports():
    """Verify all required modules can be imported"""
    required = [
        'tori_backend.routes.schemas',
        'tori_backend.routes.ingest',
        'ingest_bus.audio.ingest_audio'
    ]
    
    for module_name in required:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            return False
    return True

def run_tests():
    """Run the test suite"""
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/audio/test_ingest_endpoint.py", 
        "-v", "-ra"
    ])
    return result.returncode == 0

def main():
    print("🔍 Verifying setup...\n")
    
    if not check_imports():
        print("\n❌ Import checks failed!")
        return 1
    
    print("\n🧪 Running tests...\n")
    if not run_tests():
        print("\n❌ Tests failed!")
        return 1
    
    print("\n✅ All checks passed! Ready for Week 2 WebSocket implementation.")
    return 0

if __name__ == "__main__":
    sys.exit(main())