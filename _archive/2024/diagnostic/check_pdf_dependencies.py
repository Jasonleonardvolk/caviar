#!/usr/bin/env python3
"""
Simple dependency checker for Prajna PDF system
"""

def check_dependencies():
    print("Checking PDF processing dependencies...")
    
    # Check PyPDF2
    try:
        import PyPDF2
        print("✅ PyPDF2 available")
        pypdf2_available = True
    except ImportError:
        print("❌ PyPDF2 not installed - run: pip install PyPDF2")
        pypdf2_available = False
    
    # Check PyMuPDF
    try:
        import fitz
        print("✅ PyMuPDF available (recommended)")
        pymupdf_available = True
    except ImportError:
        print("💡 PyMuPDF not installed - run: pip install PyMuPDF")
        pymupdf_available = False
    
    # Check other dependencies
    try:
        import json
        print("✅ JSON support available")
    except ImportError:
        print("❌ JSON support missing")
    
    try:
        import asyncio
        print("✅ Asyncio support available")
    except ImportError:
        print("❌ Asyncio support missing")
    
    print("\nSummary:")
    if pypdf2_available or pymupdf_available:
        print("✅ PDF processing is possible")
        if pymupdf_available:
            print("✅ Using PyMuPDF (best performance)")
        else:
            print("✅ Using PyPDF2 (basic functionality)")
    else:
        print("❌ No PDF processing libraries available")
        print("   Install at least one: pip install PyPDF2 or pip install PyMuPDF")
    
    return pypdf2_available or pymupdf_available

if __name__ == "__main__":
    check_dependencies()
