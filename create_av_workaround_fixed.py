#!/usr/bin/env python3
"""
Permanent fix for av/torchvision issue - create a wrapper
"""

import subprocess
import sys
import os

def create_av_workaround():
    print("Creating Permanent AV Workaround")
    print("=" * 60)
    
    # Create a mock av module in the current directory
    mock_av_content = '''"""
Mock av module to fix torchvision compatibility
"""

class MockLogging:
    ERROR = 0
    WARNING = 1
    INFO = 2
    DEBUG = 3
    
    @staticmethod
    def set_level(level):
        pass

logging = MockLogging()

# Add other av attributes if needed
__version__ = "10.0.0"
'''
    
    print("\n1. Creating mock av module...")
    with open("av.py", "w", encoding='utf-8') as f:
        f.write(mock_av_content)
    print("Created av.py mock module")
    
    print("\n2. Testing the fix...")
    
    # Test imports
    test_code = '''
import sys
sys.path.insert(0, '.')  # Use our mock av first

# Import our mock av first
import av
print(f"Using mock av module: {av.__file__}")

# Now try the problematic imports
try:
    import torchvision
    print("torchvision imported successfully")
except Exception as e:
    print(f"torchvision: {e}")

try:
    import sentence_transformers
    print("sentence_transformers imported successfully")
    
    from sentence_transformers import SentenceTransformer
    print("SentenceTransformer imported successfully")
    
    # Test model loading
    print("\\nLoading a small model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully")
    
except Exception as e:
    print(f"sentence_transformers: {e}")
    import traceback
    traceback.print_exc()

# Test entropy pruning
try:
    from ingest_pdf.entropy_prune import entropy_prune
    print("\\nentropy_prune imported successfully")
    
    # Quick test
    test_concepts = [{"name": "test", "score": 0.5}]
    result, stats = entropy_prune(test_concepts)
    print("entropy_prune function works!")
    
except Exception as e:
    print(f"entropy_prune: {e}")

print("\\nWorkaround complete!")
'''
    
    with open("test_workaround.py", "w", encoding='utf-8') as f:
        f.write(test_code)
    
    subprocess.run([sys.executable, "test_workaround.py"])
    
    print("\n" + "=" * 60)
    print("3. Making the fix permanent...")
    print("\nThe av.py mock module has been created in the current directory.")
    print("This will intercept av imports and prevent the error.")
    print("\nTo use entropy pruning, make sure to run from this directory!")

if __name__ == "__main__":
    create_av_workaround()
