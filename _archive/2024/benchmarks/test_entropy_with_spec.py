#!/usr/bin/env python3
"""
Test entropy pruning with proper av mock
"""

# CRITICAL: Import our mock av FIRST before anything else
import sys
import os

# Add current directory to the FRONT of sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import our mock av
import av
print(f"Loaded av from: {av.__file__}")
print(f"av.__spec__ = {av.__spec__}")

# Now we can safely import everything else
print("\nTesting imports...")

try:
    import transformers
    print(f"SUCCESS: transformers {transformers.__version__}")
except Exception as e:
    print(f"FAILED: transformers - {e}")

try:
    import sentence_transformers
    print(f"SUCCESS: sentence-transformers {sentence_transformers.__version__}")
except Exception as e:
    print(f"FAILED: sentence-transformers - {e}")

print("\nTesting entropy pruning...")
try:
    from ingest_pdf.entropy_prune import entropy_prune
    print("SUCCESS: entropy_prune imported")
    
    # Test it
    test_concepts = [
        {"name": "machine learning", "score": 0.9},
        {"name": "deep learning", "score": 0.8},
        {"name": "AI", "score": 0.7},
    ]
    
    result, stats = entropy_prune(test_concepts, top_k=2)
    print(f"SUCCESS: Entropy pruning works! Selected {len(result)} concepts")
    print(f"Selected: {[c['name'] for c in result]}")
    
except Exception as e:
    print(f"FAILED: entropy_prune - {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test complete!")
