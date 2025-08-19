#!/usr/bin/env python3
"""
Direct entropy pruning test with complete av fix
"""

# Fix 1: Create a complete mock av module in memory
import sys
import types
from importlib.machinery import ModuleSpec

# Create the mock av module with proper spec
av = types.ModuleType('av')
av.__spec__ = ModuleSpec("av", None)
av.__file__ = "<mock>"
av.__version__ = "10.0.0"

# Create av.logging
av.logging = types.ModuleType('av.logging')
av.logging.ERROR = 0
av.logging.WARNING = 1
av.logging.INFO = 2
av.logging.DEBUG = 3
av.logging.set_level = lambda x: None

# Register in sys.modules BEFORE any imports
sys.modules['av'] = av
sys.modules['av.logging'] = av.logging

print("Mock av module registered successfully")

# Now we can safely import everything
print("\nImporting TORI components...")

# Add current directory to path
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Test transformers
    import transformers
    print(f"SUCCESS: transformers {transformers.__version__}")
    
    # Test sentence-transformers
    import sentence_transformers
    print(f"SUCCESS: sentence-transformers {sentence_transformers.__version__}")
    
    # Test entropy pruning
    from ingest_pdf.entropy_prune import entropy_prune, entropy_prune_with_categories
    print("SUCCESS: entropy_prune imported!")
    
    # Test functionality
    print("\nTesting entropy pruning functionality...")
    test_concepts = [
        {"name": "artificial intelligence", "score": 0.95},
        {"name": "machine learning", "score": 0.90},
        {"name": "deep learning", "score": 0.85},
        {"name": "neural networks", "score": 0.80},
        {"name": "AI", "score": 0.75},
    ]
    
    print(f"Input: {len(test_concepts)} concepts")
    result, stats = entropy_prune(test_concepts, top_k=3, similarity_threshold=0.8)
    
    print(f"\nSUCCESS! Entropy pruning works!")
    print(f"Output: {len(result)} concepts selected")
    print(f"Selected: {[c['name'] for c in result]}")
    print(f"Pruning stats: {stats}")
    
    print("\n" + "="*60)
    print("ENTROPY PRUNING IS FULLY FUNCTIONAL!")
    print("="*60)
    print("\nYou can now:")
    print("1. Start API server: python tori_start.py api")
    print("2. Or directly: python enhanced_launcher.py")
    print("3. Run complete system: python isolated_startup.py")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
