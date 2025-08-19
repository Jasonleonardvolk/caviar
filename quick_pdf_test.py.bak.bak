#!/usr/bin/env python3
"""
Quick single PDF test - minimal version
"""
import os
import json
from pathlib import Path

# Simple PDF test
data_dir = Path("C:\\Users\\jason\\Desktop\\tori\\kha\\data")

print("🔍 Looking for PDFs...")
pdf_files = list(data_dir.rglob("*.pdf"))
print(f"📚 Found {len(pdf_files)} PDF files total")

if pdf_files:
    # Just test the first one that's not in USB Drive (to avoid corrupted files)
    for pdf_path in pdf_files:
        if "USB Drive" not in str(pdf_path):
            print(f"📄 Testing with: {pdf_path.name}")
            print(f"📏 Size: {pdf_path.stat().st_size} bytes")
            
            # Try to process it
            try:
                import PyPDF2
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    pages = len(reader.pages)
                    print(f"✅ Successfully read PDF: {pages} pages")
                    
                    # Extract some text
                    if pages > 0:
                        text = reader.pages[0].extract_text()[:200]
                        print(f"📝 Sample text: {text[:100]}...")
                    
                    # Create simple result
                    result = {
                        "pdf_documents": [{
                            "file_path": str(pdf_path),
                            "title": pdf_path.stem,
                            "pages": pages,
                            "size": pdf_path.stat().st_size,
                            "status": "success"
                        }],
                        "metadata": {
                            "processing_date": "2025-06-06",
                            "total_documents": 1
                        }
                    }
                    
                    # Save result
                    with open("prajna_pdf_knowledge.json", "w") as f:
                        json.dump(result, f, indent=2)
                    
                    print("✅ SUCCESS! Created prajna_pdf_knowledge.json")
                    break
                    
            except Exception as e:
                print(f"❌ Error processing {pdf_path.name}: {e}")
                continue
else:
    print("❌ No PDF files found")
                
print("🎉 Quick test complete!")
