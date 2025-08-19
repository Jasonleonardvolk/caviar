#!/usr/bin/env python3
"""
Direct test for entropy pruning - no Unicode characters
"""

# Fix the av issue before any imports
import sys
import types

# Create a mock av module
av = types.ModuleType('av')
av.logging = types.ModuleType('av.logging')
av.logging.set_level = lambda x: None
av.logging.ERROR = 0
av.logging.WARNING = 1
av.logging.INFO = 2
av.logging.DEBUG = 3
sys.modules['av'] = av
sys.modules['av.logging'] = av.logging

print("Created mock av module")

# Now do the actual test
print("\nTesting Entropy Pruning")
print("=" * 50)

# Add current directory to path
sys.path.insert(0, '.')

try:
    from ingest_pdf.entropy_prune import entropy_prune, entropy_prune_with_categories
    print("SUCCESS: entropy_prune imported successfully!")
    
    # Test basic functionality
    test_concepts = [
        {"name": "artificial intelligence", "score": 0.95},
        {"name": "machine learning", "score": 0.90},
        {"name": "deep learning", "score": 0.85},
        {"name": "neural networks", "score": 0.80},
        {"name": "AI", "score": 0.75},
    ]
    
    print("\nTesting entropy_prune function...")
    print(f"Input: {len(test_concepts)} concepts")
    
    # Run entropy pruning
    result, stats = entropy_prune(
        test_concepts, 
        top_k=3,
        similarity_threshold=0.8,
        verbose=True
    )
    
    print(f"\nSUCCESS! Entropy pruning works!")
    print(f"Output: {len(result)} concepts selected")
    print(f"Selected: {[c['name'] for c in result]}")
    print(f"Stats: {stats}")
    
    print("\nENTROPY PRUNING IS WORKING!")
    print("\nYou can now:")
    print("1. Start the API server: python enhanced_launcher.py")
    print("2. Run the complete system: python isolated_startup.py")
    print("3. Process PDFs with quality scoring")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
