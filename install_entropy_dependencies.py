#!/usr/bin/env python3
"""
Install all dependencies required for entropy pruning
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    print(f"📦 Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    print("🔧 Installing Entropy Pruning Dependencies")
    print("=" * 50)
    
    # List of required packages
    packages = [
        "sentence-transformers",  # For text embeddings
        "scikit-learn",          # For clustering and similarity
        "numpy",                 # For numerical operations
    ]
    
    # Check current environment
    print(f"\n📍 Python: {sys.executable}")
    print(f"📍 Version: {sys.version}")
    
    # Install packages
    print("\n📦 Installing required packages...\n")
    
    failed = []
    for package in packages:
        if not install_package(package):
            failed.append(package)
    
    # Summary
    print("\n" + "=" * 50)
    if not failed:
        print("✅ All dependencies installed successfully!")
        print("\n🧪 Testing imports...")
        
        # Test imports
        try:
            from sentence_transformers import SentenceTransformer
            print("✅ sentence_transformers imported successfully")
        except ImportError as e:
            print(f"❌ sentence_transformers import failed: {e}")
        
        try:
            from sklearn.cluster import AgglomerativeClustering
            print("✅ sklearn imported successfully")
        except ImportError as e:
            print(f"❌ sklearn import failed: {e}")
        
        try:
            import numpy as np
            print("✅ numpy imported successfully")
        except ImportError as e:
            print(f"❌ numpy import failed: {e}")
        
        print("\n✨ Installation complete! Try running verify_entropy.py again.")
    else:
        print(f"❌ Failed to install: {', '.join(failed)}")
        print("\nTry installing manually:")
        for package in failed:
            print(f"  pip install {package}")

if __name__ == "__main__":
    main()
