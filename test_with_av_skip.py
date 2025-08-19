#!/usr/bin/env python3
"""
Alternative fix - Skip video processing entirely
"""

import os
import sys

# Add this before importing sentence_transformers
os.environ["SENTENCE_TRANSFORMERS_SKIP_AV"] = "1"

print("🔧 Testing Entropy Pruning with AV Skip")
print("=" * 60)

# Add current directory to path
sys.path.insert(0, '.')

print("\n1️⃣ Testing with environment variable set...")

try:
    # Try importing with the skip flag
    import sentence_transformers
    print('✅ sentence_transformers imported successfully')
    
    from sentence_transformers import SentenceTransformer
    print('✅ SentenceTransformer class imported')
    
    # Test creating a model (use a small one)
    print('\n2️⃣ Testing model loading...')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('✅ Model loaded successfully')
    
except Exception as e:
    print(f'❌ sentence_transformers: {e}')
    print('\nTrying alternative approach...')
    
    # Alternative: Mock the problematic av module
    import sys
    import types
    
    # Create a mock av module
    av = types.ModuleType('av')
    av.logging = types.ModuleType('av.logging')
    av.logging.set_level = lambda x: None
    av.logging.ERROR = 0
    sys.modules['av'] = av
    sys.modules['av.logging'] = av.logging
    
    print('Created mock av module')
    
    try:
        import sentence_transformers
        print('✅ sentence_transformers imported with mock')
    except Exception as e2:
        print(f'❌ Still failed: {e2}')

# Test entropy pruning
print('\n3️⃣ Testing entropy pruning...')
try:
    from ingest_pdf.entropy_prune import entropy_prune, entropy_prune_with_categories
    print('✅ entropy_prune imported successfully')
    
    # Test functionality
    test_concepts = [
        {"name": "machine learning", "score": 0.9},
        {"name": "deep learning", "score": 0.8},
        {"name": "AI", "score": 0.7},
    ]
    
    result, stats = entropy_prune(test_concepts, top_k=2, verbose=True)
    print(f'✅ entropy_prune works! Selected {len(result)} from {len(test_concepts)} concepts')
    print(f'   Selected: {[c["name"] for c in result]}')
    
    print('\n🎉 Everything is working!')
    
except Exception as e:
    print(f'❌ entropy_prune: {e}')
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("If everything shows ✅ above, entropy pruning is ready to use!")
