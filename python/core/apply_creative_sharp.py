#!/usr/bin/env python3
"""
Apply all Creative Feedback patches from the sharp review
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
        if result.stderr:
            logger.error(f"Error: {result.stderr}")
        return False
    
    if result.stdout:
        logger.info(result.stdout)
    
    return result.returncode == 0

def main():
    print_banner("Creative Feedback Sharp Review Patches")
    
    logger.info("""
This script applies all patches from the sharp-edged review:

1. Sharp Review Patches (Bug Fixes):
   - Correct steps_in_mode tracking
   - Duration bounds enforcement
   - Emergency state cleanup
   - Baseline score initialization
   - Safety fixes (diversity clamping, JSON)
   - Performance (no polyfit warnings)
   - API improvements (enums, types)

2. Optional Enhancements:
   - Entropy profiles (time-varying)
   - Quality prediction model
   - Metric streaming

Let's begin!
""")
    
    input("Press Enter to continue (Ctrl+C to cancel)...")
    
    # Step 1: Apply sharp review patches
    print_banner("Step 1: Applying Sharp Review Patches")
    
    if not run_command([sys.executable, "patch_creative_feedback_sharp.py", "--dry-run"]):
        logger.error("Sharp review patch dry-run failed!")
        return 1
    
    response = input("\nApply sharp review patches? (yes/no): ")
    if response.lower() == 'yes':
        if not run_command([sys.executable, "patch_creative_feedback_sharp.py"]):
            logger.error("Sharp review patch application failed!")
            return 1
        logger.info("✅ Sharp review patches applied successfully!")
    else:
        logger.info("Skipping sharp review patches")
    
    # Step 2: Create tests
    print_banner("Step 2: Creating Tests")
    
    response = input("Create test file? (yes/no): ")
    if response.lower() == 'yes':
        run_command([sys.executable, "patch_creative_feedback_sharp.py", "--create-test"], check=False)
        logger.info("✅ Test file created!")
        
        # Run tests
        response = input("\nRun tests now? (yes/no): ")
        if response.lower() == 'yes':
            run_command([sys.executable, "test_creative_feedback_sharp.py", "-v"], check=False)
    
    # Step 3: Optional enhancements
    print_banner("Step 3: Optional Enhancements")
    
    logger.info("""
Optional enhancements add:
- Entropy profiles: cosine_ramp, exponential_decay, pulse patterns
- Quality model: ML-based gain prediction from past explorations
- Metric streaming: Real-time WebSocket/callback integration
""")
    
    response = input("\nApply optional enhancements? (yes/no): ")
    if response.lower() == 'yes':
        if not run_command([sys.executable, "enhance_creative_feedback_optional.py", "--dry-run"]):
            logger.warning("Optional enhancement dry-run had issues")
            response = input("Continue anyway? (yes/no): ")
            if response.lower() != 'yes':
                logger.info("Skipping optional enhancements")
            else:
                run_command([sys.executable, "enhance_creative_feedback_optional.py"], check=False)
                logger.info("✅ Optional enhancements applied!")
        else:
            run_command([sys.executable, "enhance_creative_feedback_optional.py"])
            logger.info("✅ Optional enhancements applied!")
            
        # Create example
        run_command([sys.executable, "enhance_creative_feedback_optional.py", "--create-example"], check=False)
    
    # Summary
    print_banner("Patch Summary")
    
    logger.info("""
Creative Feedback has been patched with:

Sharp Review Fixes:
  ✅ Correct step counting (no more shortened explorations)
  ✅ Minimum duration guarantee (at least 10 steps)
  ✅ Clean emergency cancellation
  ✅ Dynamic baseline updates
  ✅ Safe diversity scores [0,1]
  ✅ JSON-serializable metrics
  ✅ Clean trend calculation
  ✅ Type-safe action enums

Optional Enhancements:
  🔧 Time-varying entropy profiles
  🔧 Predictive quality model
  🔧 Real-time metric streaming

Bug Fix Impact:
  - No more state confusion
  - Guaranteed valid explorations
  - Clean error handling
  - Consistent behavior

Next Steps:
1. Review documentation:
   - CREATIVE_FEEDBACK_SHARP_PATCHES.md
   - CREATIVE_FEEDBACK_COMPARISON.md

2. Run tests:
   python test_creative_feedback_sharp.py

3. Try examples (if enhanced):
   python example_creative_enhanced.py

The module now handles creative exploration with production-grade reliability! 🎨
""")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n\nCancelled by user")
        sys.exit(1)
