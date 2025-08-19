"""
Clean up patch files and re-run test
"""

import os
import subprocess

# Clean up the patch file
if os.path.exists("quality_patch.py"):
    os.remove("quality_patch.py")
    print("✅ Cleaned up patch file")

# Run the test
print("\n🚀 Running semantic extraction test on soliton PDF...")
subprocess.run(["poetry", "run", "python", "test_soliton_pdf.py"])
