#!/usr/bin/env python3
"""
Fix the av/torchvision compatibility issue
"""

import subprocess
import sys

def fix_av_issue():
    print("🔧 Fixing av/torchvision compatibility issue")
    print("=" * 60)
    
    print("\n1️⃣ Uninstalling problematic av package...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "av"])
    
    print("\n2️⃣ Installing compatible av version...")
    # Install a compatible version of av
    subprocess.run([sys.executable, "-m", "pip", "install", "av==10.0.0"])
    
    print("\n3️⃣ Testing imports...")
    
    # Test if it works now
    test_code = """
import sys
sys.path.insert(0, '.')

try:
    import sentence_transformers
    print('✅ sentence_transformers imported successfully')
    from sentence_transformers import SentenceTransformer
    print('✅ SentenceTransformer class imported')
except Exception as e:
    print(f'❌ sentence_transformers: {e}')
    
try:
    from ingest_pdf.entropy_prune import entropy_prune
    print('✅ entropy_prune imported successfully')
    
    # Test functionality
    test_concepts = [{"name": "test", "score": 0.5}]
    result, stats = entropy_prune(test_concepts)
    print('✅ entropy_prune function works!')
except Exception as e:
    print(f'❌ entropy_prune: {e}')
"""
    
    with open("test_imports.py", "w") as f:
        f.write(test_code)
    
    subprocess.run([sys.executable, "test_imports.py"])
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    fix_av_issue()
