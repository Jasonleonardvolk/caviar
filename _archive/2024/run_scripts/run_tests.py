#!/usr/bin/env python3
"""Run audio ingestion tests"""

import subprocess
import sys

def run_tests():
    """Run the audio ingestion test suite"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/audio/test_ingest_endpoint.py",
        "-v",  # verbose
        "--tb=short",  # short traceback
        "--no-header"  # no pytest header
    ]
    
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == "__main__":
    print("Running audio ingestion tests...")
    exit_code = run_tests()
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    sys.exit(exit_code)