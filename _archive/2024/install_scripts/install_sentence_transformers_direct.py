#!/usr/bin/env python3
"""
Alternative solution: Skip scipy downgrade and just install sentence-transformers
"""

import subprocess
import sys

def install_directly():
    """Install sentence-transformers directly without Poetry"""
    print("🔧 Installing sentence-transformers directly...\n")
    
    packages = [
        "sentence-transformers==5.0.0",  # Match the version in pyproject.toml
        "scikit-learn==1.7.0",  # Already in pyproject.toml
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}\n")
    
    print("\n✨ Installation complete!")
    print("\nNow test with:")
    print("  python verify_entropy.py")

if __name__ == "__main__":
    install_directly()
