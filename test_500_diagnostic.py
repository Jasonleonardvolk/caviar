#!/usr/bin/env python3
"""
MINIMAL DIAGNOSTIC TEST - Find the exact error causing 500
"""

import sys
import os
from pathlib import Path
import traceback

# Add paths
current_dir = Path(__file__).parent
ingest_pdf_dir = current_dir / "ingest_pdf"
sys.path.insert(0, str(ingest_pdf_dir))
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test each import individually to find the problem"""
    print("🔍 TESTING IMPORTS ONE BY ONE...")
    
    try:
        print("  ✅ Testing basic imports...")
        from pathlib import Path
        from typing import List, Dict, Any, Tuple, Optional, Set
        import numpy as np
        import json
        import os
        import hashlib
        import PyPDF2
        import logging
        import time
        from datetime import datetime
        print("    ✅ Basic imports OK")
        
        print("  🔍 Testing pipeline import...")
        from ingest_pdf.pipeline import ingest_pdf_clean
        print("    ✅ Pipeline import OK")
        
        print("  🔍 Testing extractConceptsFromDocument...")
        from ingest_pdf.extractConceptsFromDocument import extractConceptsFromDocument
        print("    ✅ extractConceptsFromDocument OK")
        
        print("  🔍 Testing extract_chunks...")
        from ingest_pdf.extract_blocks import extract_chunks
        print("    ✅ extract_chunks OK")
        
        print("  🔍 Testing entropy_prune...")
        from ingest_pdf.entropy_prune import entropy_prune
        print("    ✅ entropy_prune OK")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Import failed: {e}")
        print(f"    Traceback: {traceback.format_exc()}")
        return False

def test_simple_extraction():
    """Test a simple extraction with minimal content"""
    print("\n🧪 TESTING SIMPLE EXTRACTION...")
    
    try:
        from ingest_pdf.pipeline import ingest_pdf_clean
        
        # Create a minimal test file
        test_file = current_dir / "minimal_test.txt"
        with open(test_file, "w") as f:
            f.write("Machine learning is a subset of artificial intelligence.")
        
        print(f"📄 Created minimal test: {test_file}")
        
        # Try the extraction
        print("🚀 Running ingest_pdf_clean...")
        result = ingest_pdf_clean(str(test_file), admin_mode=True)
        
        print(f"✅ Result status: {result.get('status')}")
        print(f"✅ Concept count: {result.get('concept_count', 0)}")
        print(f"✅ No crashes!")
        
        # Clean up
        os.remove(test_file)
        return True
        
    except Exception as e:
        print(f"❌ Simple extraction failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_specific_dependencies():
    """Test specific dependencies that might be missing"""
    print("\n🔍 TESTING SPECIFIC DEPENDENCIES...")
    
    dependencies = [
        "sentence_transformers",
        "sklearn",
        "yake",
        "spacy",
        "transformers"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  ✅ {dep}: Available")
        except ImportError as e:
            print(f"  ❌ {dep}: Missing - {e}")

def test_main_api_import():
    """Test if main.py can import the pipeline"""
    print("\n🌐 TESTING MAIN API IMPORT...")
    
    try:
        # Test the same import that main.py uses
        sys.path.insert(0, str(current_dir / "ingest_pdf"))
        from pipeline import ingest_pdf_clean
        print("  ✅ main.py style import works")
        
        # Test a quick call
        test_result = ingest_pdf_clean.__doc__
        print(f"  ✅ Function accessible: {test_result[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"  ❌ main.py style import failed: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🔍 MINIMAL DIAGNOSTIC TEST FOR 500 ERROR")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Dependency Test", test_specific_dependencies),
        ("Main API Import", test_main_api_import),
        ("Simple Extraction", test_simple_extraction)
    ]
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            success = test_func()
            print(f"{'✅ PASS' if success else '❌ FAIL'}: {test_name}")
        except Exception as e:
            print(f"❌ CRASH: {test_name} - {e}")
    
    print(f"\n" + "=" * 50)
    print("🎯 If any test failed above, that's likely causing your 500 error")
    print("🔧 Fix the failed imports/dependencies and try again")
