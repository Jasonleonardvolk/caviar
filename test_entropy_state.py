#!/usr/bin/env python3
"""
Quick test to see current state of entropy pruning
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

print("🔍 Checking Current State of Entropy Pruning")
print("=" * 50)

# Check 1: Import basic dependencies
print("\n1️⃣ Checking basic imports...")
issues = []

try:
    import numpy
    print(f"✅ numpy {numpy.__version__}")
except ImportError as e:
    print(f"❌ numpy: {e}")
    issues.append("numpy")

try:
    import scipy
    print(f"✅ scipy {scipy.__version__}")
except ImportError as e:
    print(f"❌ scipy: {e}")
    issues.append("scipy")

try:
    import sklearn
    print(f"✅ scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"❌ scikit-learn: {e}")
    issues.append("scikit-learn")

try:
    import transformers
    print(f"✅ transformers {transformers.__version__}")
    # Test specific import that was failing
    from transformers import PreTrainedModel
    print("✅ Can import PreTrainedModel")
except ImportError as e:
    print(f"❌ transformers: {e}")
    issues.append("transformers")

try:
    import sentence_transformers
    print(f"✅ sentence-transformers {sentence_transformers.__version__}")
    from sentence_transformers import SentenceTransformer
    print("✅ Can import SentenceTransformer")
except ImportError as e:
    print(f"❌ sentence-transformers: {e}")
    issues.append("sentence-transformers")

try:
    import spacy
    print(f"✅ spacy {spacy.__version__}")
except ImportError as e:
    print(f"❌ spacy: {e}")
    issues.append("spacy")

# Check 2: Import entropy pruning
print("\n2️⃣ Checking entropy pruning import...")
try:
    from ingest_pdf.entropy_prune import entropy_prune, entropy_prune_with_categories
    print("✅ entropy_prune imported successfully")
except ImportError as e:
    print(f"❌ entropy_prune import failed: {e}")
    issues.append("entropy_prune")

# Check 3: Test entropy pruning functionality
if "entropy_prune" not in issues:
    print("\n3️⃣ Testing entropy pruning functionality...")
    try:
        test_concepts = [
            {"name": "test concept 1", "score": 0.9},
            {"name": "test concept 2", "score": 0.8},
            {"name": "test concept 3", "score": 0.7},
        ]
        
        result, stats = entropy_prune(test_concepts, top_k=2)
        print(f"✅ Entropy pruning works! Selected {len(result)} from {len(test_concepts)} concepts")
        print(f"   Stats: {stats}")
    except Exception as e:
        print(f"❌ Entropy pruning failed: {e}")
        issues.append("entropy_prune_function")

# Summary
print("\n" + "=" * 50)
if not issues:
    print("✅ Everything is working! Entropy pruning is functional.")
else:
    print(f"❌ Found {len(issues)} issues:")
    for issue in issues:
        print(f"   - {issue}")
    print("\nTo fix, run one of these:")
    print("  1. .\\emergency_dependency_fix.ps1")
    print("  2. python comprehensive_dependency_fix.py")
