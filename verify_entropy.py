#!/usr/bin/env python3
"""
Quick verification that entropy pruning is working
"""

import sys
from pathlib import Path

# Ensure we're in the right directory
sys.path.insert(0, str(Path(__file__).parent))

print("🔍 Verifying Entropy Pruning Setup...\n")

# Check 1: File exists
entropy_file = Path("ingest_pdf/entropy_prune.py")
if entropy_file.exists():
    print("✅ entropy_prune.py exists")
else:
    print("❌ entropy_prune.py NOT FOUND")
    sys.exit(1)

# Check 2: Can import directly
try:
    from ingest_pdf.entropy_prune import entropy_prune, entropy_prune_with_categories
    print("✅ Direct import successful")
except ImportError as e:
    print(f"❌ Direct import failed: {e}")

# Check 3: Can import from pruning
try:
    from ingest_pdf.pipeline.pruning import apply_entropy_pruning
    print("✅ Pruning module import successful")
except ImportError as e:
    print(f"❌ Pruning module import failed: {e}")

# Check 4: Quick function test
try:
    test_concepts = [
        {"name": "test1", "score": 0.9},
        {"name": "test2", "score": 0.8}
    ]
    result, stats = entropy_prune(test_concepts, top_k=1)
    print(f"✅ Function test successful - selected {len(result)} concepts")
except Exception as e:
    print(f"❌ Function test failed: {e}")

print("\n✨ Verification complete!")
