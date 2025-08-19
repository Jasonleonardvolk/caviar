#!/usr/bin/env python3
"""
Download required NLTK data packages for TORI sentence-level processing.

This script downloads:
- punkt: Sentence tokenizer
- stopwords: Common words to filter out
"""

import nltk
import sys
import os

def download_nltk_data():
    """Download required NLTK data packages."""
    print("=== Downloading NLTK Data for TORI ===")
    print("This will download required data for sentence-level text processing.")
    print()
    
    # Set NLTK data directory if needed
    nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    
    packages = [
        ('punkt', 'Punkt tokenizer for sentence splitting'),
        ('stopwords', 'Common stopwords for filtering'),
    ]
    
    success = True
    
    for package_name, description in packages:
        print(f"[*] Downloading {package_name}: {description}")
        try:
            nltk.download(package_name, quiet=False)
            print(f"[OK] Successfully downloaded {package_name}")
        except Exception as e:
            print(f"[ERROR] Failed to download {package_name}: {e}")
            success = False
        print()
    
    if success:
        print("=== NLTK Data Download Complete ===")
        print("TORI is ready for sentence-level text processing!")
        return 0
    else:
        print("=== NLTK Data Download Failed ===")
        print("Some packages failed to download. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(download_nltk_data())
