#!/usr/bin/env python3
"""
Minimal fix - Just get entropy pruning working
"""

import subprocess
import sys

def minimal_fix():
    print("🔧 Minimal Fix for Entropy Pruning\n")
    
    # Step 1: Fix the broken scipy
    print("1️⃣ Fixing broken scipy...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "scipy"], capture_output=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "scipy==1.10.1"])
    
    # Step 2: Downgrade numpy to compatible version
    print("\n2️⃣ Installing compatible numpy...")
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.24.3", "--force-reinstall"])
    
    # Step 3: Reinstall sentence-transformers
    print("\n3️⃣ Reinstalling sentence-transformers...")
    subprocess.run([sys.executable, "-m", "pip", "install", "sentence-transformers==2.2.2"])
    
    # Step 4: Test
    print("\n4️⃣ Testing imports...")
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers works!")
        
        # Test entropy pruning
        import sys
        sys.path.insert(0, '.')
        from ingest_pdf.entropy_prune import entropy_prune
        print("✅ entropy_prune imports successfully!")
        
        print("\n✨ Fix complete! Entropy pruning should work now.")
        return True
    except Exception as e:
        print(f"❌ Still having issues: {e}")
        return False

if __name__ == "__main__":
    minimal_fix()
