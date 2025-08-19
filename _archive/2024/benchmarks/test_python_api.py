#!/usr/bin/env python3
"""
🔍 QUICK API TEST - Check if Python API can start
"""

import sys
from pathlib import Path

print("🔍 TESTING PYTHON API DEPENDENCIES")
print("=" * 50)

# Test 1: Check Python version
print(f"🐍 Python version: {sys.version}")

# Test 2: Check if we can import FastAPI
try:
    from fastapi import FastAPI
    print("✅ FastAPI import: SUCCESS")
except ImportError as e:
    print(f"❌ FastAPI import: FAILED - {e}")
    print("   Fix: pip install fastapi uvicorn")

# Test 3: Check extraction module
try:
    sys.path.append(str(Path(__file__).parent / "ingest_pdf"))
    from extractConceptsFromDocument import extractConceptsFromDocument
    print("✅ Extraction module import: SUCCESS")
    
    # Test extraction
    test_text = "This is a test of machine learning and artificial intelligence."
    concepts = extractConceptsFromDocument(test_text, threshold=0.0)
    print(f"✅ Test extraction: {len(concepts)} concepts found")
    
except ImportError as e:
    print(f"❌ Extraction module import: FAILED - {e}")
    print("   Fix: Make sure ingest_pdf/ directory exists with extraction modules")
except Exception as e:
    print(f"❌ Extraction test: FAILED - {e}")

# Test 4: Check specific dependencies
dependencies = [
    ("yake", "yake"),
    ("keybert", "KeyBERT"), 
    ("sentence_transformers", "SentenceTransformer"),
    ("spacy", "spaCy")
]

print("\n🔍 CHECKING EXTRACTION DEPENDENCIES:")
for package, name in dependencies:
    try:
        __import__(package)
        print(f"✅ {name}: SUCCESS")
    except ImportError:
        print(f"❌ {name}: MISSING")
        print(f"   Fix: pip install {package}")

# Test 5: Check spaCy model
try:
    import spacy
    nlp = spacy.load("en_core_web_lg")
    print("✅ spaCy model (en_core_web_lg): SUCCESS")
except OSError:
    print("❌ spaCy model (en_core_web_lg): MISSING")
    print("   Fix: python -m spacy download en_core_web_lg")
except Exception as e:
    print(f"❌ spaCy model test: FAILED - {e}")

print("\n🎯 SUMMARY:")
print("If all items show ✅, the API should start successfully")
print("If any show ❌, install the missing dependencies first")
