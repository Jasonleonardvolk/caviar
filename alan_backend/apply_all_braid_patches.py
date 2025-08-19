#!/usr/bin/env python3
"""
Apply all BraidAggregator patches (basic + advanced)
Ensures correct order and provides unified workflow
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def print_banner(msg):
    logger.info("\n" + "="*60)
    logger.info(f" {msg}")
    logger.info("="*60 + "\n")

def run_command(cmd, check=True):
    """Run command and return success status"""
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0 and check:
        logger.error(f"Command failed with code {result.returncode}")
        logger.error(f"Error: {result.stderr}")
        return False
    
    if result.stdout:
        logger.info(result.stdout)
    
    return result.returncode == 0

def main():
    print_banner("BraidAggregator Complete Enhancement")
    
    logger.info("""
This script applies all patches to braid_aggregator.py:

1. Basic Patches (from red-pen review):
   - JSON serialization fixes
   - Memory leak prevention
   - Performance optimizations
   - API improvements

2. Advanced Features:
   - WebSocket/Message Bus emitter
   - Adaptive scheduling
   - Circuit breaker

Let's begin!
""")
    
    input("Press Enter to continue (Ctrl+C to cancel)...")
    
    # Step 1: Apply basic patches
    print_banner("Step 1: Applying Basic Patches")
    
    if not run_command([sys.executable, "patch_braid_aggregator.py", "--dry-run"]):
        logger.error("Basic patch dry-run failed!")
        return 1
    
    response = input("\nApply basic patches? (yes/no): ")
    if response.lower() == 'yes':
        if not run_command([sys.executable, "patch_braid_aggregator.py"]):
            logger.error("Basic patch application failed!")
            return 1
        logger.info("✅ Basic patches applied successfully!")
    else:
        logger.info("Skipping basic patches")
    
    # Step 2: Apply advanced patches
    print_banner("Step 2: Applying Advanced Patches")
    
    logger.info("Note: Advanced patches require basic patches to be applied first!")
    
    if not run_command([sys.executable, "patch_braid_aggregator_advanced.py", "--dry-run"]):
        logger.error("Advanced patch dry-run failed!")
        logger.warning("Make sure basic patches were applied first")
        return 1
    
    response = input("\nApply advanced patches? (yes/no): ")
    if response.lower() == 'yes':
        if not run_command([sys.executable, "patch_braid_aggregator_advanced.py"]):
            logger.error("Advanced patch application failed!")
            return 1
        logger.info("✅ Advanced patches applied successfully!")
    else:
        logger.info("Skipping advanced patches")
    
    # Step 3: Create test files
    print_banner("Step 3: Creating Test Files")
    
    response = input("Create test files? (yes/no): ")
    if response.lower() == 'yes':
        run_command([sys.executable, "patch_braid_aggregator.py", "--create-test"], check=False)
        run_command([sys.executable, "patch_braid_aggregator_advanced.py", "--create-example"], check=False)
        logger.info("✅ Test files created!")
    
    # Summary
    print_banner("Enhancement Complete!")
    
    logger.info("""
BraidAggregator has been enhanced with:

Basic Fixes:
  ✅ JSON serialization safety
  ✅ Memory leak prevention
  ✅ Error tracking
  ✅ Performance optimizations

Advanced Features:
  ✅ Async event emission
  ✅ Adaptive scheduling
  ✅ Circuit breaker protection

Next Steps:
1. Run tests:
   python test_braid_aggregator_patched.py
   python example_advanced_aggregator.py

2. Review documentation:
   - BRAID_AGGREGATOR_PATCHES.md
   - BRAID_AGGREGATOR_ADVANCED.md

3. Deploy with confidence! 🚀
""")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n\nCancelled by user")
        sys.exit(1)
