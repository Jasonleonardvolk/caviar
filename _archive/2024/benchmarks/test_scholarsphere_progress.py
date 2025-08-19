#!/usr/bin/env python3
"""
🚀 SCHOLARSPHERE EXTRACTION WITH PROGRESS
Test ScholarSphere with progress indicators during ML model loading
"""

import requests
import time
import json
from pathlib import Path

def test_scholarsphere_with_progress():
    """Test ScholarSphere extraction with real-time progress monitoring"""
    
    # Find a test PDF
    script_dir = Path(__file__).parent
    test_pdfs = list(script_dir.glob("*.pdf"))
    
    if not test_pdfs:
        print("❌ No PDF files found for testing")
        return
    
    test_pdf = test_pdfs[0]
    print(f"📄 Testing with: {test_pdf.name}")
    
    # ScholarSphere endpoints
    upload_url = "http://localhost:8002/upload"
    extract_url = "http://localhost:8002/extract"
    
    try:
        print("🚀 Step 1: Uploading PDF...")
        
        # Upload file
        with open(test_pdf, 'rb') as f:
            files = {'file': (test_pdf.name, f, 'application/pdf')}
            upload_response = requests.post(upload_url, files=files, timeout=30)
        
        if upload_response.status_code != 200:
            print(f"❌ Upload failed: {upload_response.status_code}")
            return
        
        upload_data = upload_response.json()
        file_path = upload_data['file_path']
        print(f"✅ Upload complete: {upload_data['size_mb']:.1f}MB")
        
        print("🧬 Step 2: Starting extraction (this may take 10+ seconds for ML model loading)...")
        print("⏳ Loading YAKE extractor...")
        print("⏳ Loading KeyBERT with sentence-transformers...")
        print("⏳ Loading spaCy universal NER...")
        print("⏳ Please wait while models initialize...")
        
        # Start extraction
        extract_data = {
            "file_path": file_path,
            "filename": test_pdf.name,
            "content_type": "application/pdf"
        }
        
        start_time = time.time()
        extract_response = requests.post(extract_url, json=extract_data, timeout=120)
        total_time = time.time() - start_time
        
        if extract_response.status_code != 200:
            print(f"❌ Extraction failed: {extract_response.status_code}")
            print(f"Error: {extract_response.text}")
            return
        
        result = extract_response.json()
        print(f"✅ Extraction complete in {total_time:.1f}s!")
        print(f"📊 Found {result.get('concept_count', 0)} concepts")
        print(f"🧬 Methods used: {result.get('universal_methods', [])}")
        print(f"🏆 Purity efficiency: {result.get('purity_analysis', {}).get('purity_efficiency', 'N/A')}")
        
        # Show top concepts
        top_concepts = result.get('purity_analysis', {}).get('top_pure_concepts', [])[:5]
        if top_concepts:
            print(f"\n🏆 Top 5 concepts:")
            for i, concept in enumerate(top_concepts, 1):
                print(f"  {i}. {concept['name']} (score: {concept['score']:.3f})")
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out - ScholarSphere may be hanging during ML model loading")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("🔬 SCHOLARSPHERE EXTRACTION TEST WITH PROGRESS")
    print("=" * 60)
    test_scholarsphere_with_progress()
