# Save as: C:\\Users\\jason\\Desktop\\tori\\kha\simple_pdf_processor.py

import os
import sys
import json
from pathlib import Path

# Add paths
sys.path.insert(0, "C:/Users/jason/Desktop/tori/kha")

def process_pdfs_simple():
    """Process PDFs without transformer dependencies"""
    print("📚 SIMPLE PDF PROCESSOR FOR TONKA")
    print("=" * 60)
    
    # Set environment to disable transformers
    os.environ['DISABLE_TRANSFORMERS'] = '1'
    os.environ['EXTRACTION_MODE'] = 'basic'
    
    pdf_folder = Path("C:/Users/jason/Desktop/tori/kha/docs/material/dataset/06_27")
    output_dir = Path("C:/Users/jason/Desktop/tori/kha/concept_mesh")
    output_dir.mkdir(exist_ok=True)
    
    if not pdf_folder.exists():
        print(f"❌ PDF folder not found: {pdf_folder}")
        return
    
    # Try to import with basic mode
    try:
        # Import the basic extractor
        from ingest_pdf.extract_blocks import extract_chunks
        from ingest_pdf.pipeline.io import extract_pdf_metadata, preprocess_with_ocr
        
        all_concepts = []
        pdf_files = list(pdf_folder.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDFs")
        
        for pdf_path in pdf_files:
            print(f"\n📄 Processing: {pdf_path.name}")
            try:
                # Get metadata
                metadata = extract_pdf_metadata(str(pdf_path))
                
                # Extract chunks
                chunks = extract_chunks(str(pdf_path))
                print(f"   Extracted {len(chunks)} chunks")
                
                # Convert chunks to concepts
                for i, chunk in enumerate(chunks):
                    if isinstance(chunk, dict):
                        text = chunk.get('text', '')
                    else:
                        text = str(chunk)
                    
                    if text.strip():
                        concept = {
                            "name": f"concept_{pdf_path.stem}_{i}",
                            "text": text[:500],  # First 500 chars
                            "source": pdf_path.name,
                            "chunk_index": i,
                            "score": 0.5  # Default score
                        }
                        all_concepts.append(concept)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                # Try basic text extraction
                try:
                    import PyPDF2
                    with open(pdf_path, 'rb') as f:
                        pdf = PyPDF2.PdfReader(f)
                        for page_num in range(min(10, len(pdf.pages))):  # First 10 pages
                            text = pdf.pages[page_num].extract_text()
                            if text.strip():
                                concept = {
                                    "name": f"page_{pdf_path.stem}_{page_num}",
                                    "text": text[:500],
                                    "source": pdf_path.name,
                                    "page": page_num,
                                    "score": 0.5
                                }
                                all_concepts.append(concept)
                        print(f"   ✅ Extracted {len(pdf.pages)} pages using PyPDF2")
                except Exception as e2:
                    print(f"   ❌ PyPDF2 also failed: {e2}")
        
        # Save concepts
        output_file = output_dir / "pdf_concepts_simple.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_concepts": len(all_concepts),
                "concepts": all_concepts
            }, f, indent=2)
        
        print(f"\n✅ Total concepts extracted: {len(all_concepts)}")
        print(f"💾 Saved to: {output_file}")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nTrying ultra-simple PDF extraction...")
        ultra_simple_extraction(pdf_folder, output_dir)

def ultra_simple_extraction(pdf_folder, output_dir):
    """Ultra simple PDF text extraction"""
    try:
        import PyPDF2
    except ImportError:
        print("❌ PyPDF2 not installed. Install with:")
        print("   C:\\ALANPY311\\python.exe -m pip install PyPDF2")
        return
    
    all_text = []
    pdf_files = list(pdf_folder.glob("*.pdf"))
    
    for pdf_path in pdf_files:
        print(f"📄 Extracting: {pdf_path.name}")
        try:
            with open(pdf_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                pdf_text = ""
                for page in pdf.pages[:10]:  # First 10 pages
                    pdf_text += page.extract_text() + "\n"
                
                if pdf_text.strip():
                    all_text.append({
                        "source": pdf_path.name,
                        "text": pdf_text,
                        "pages": len(pdf.pages)
                    })
                    print(f"   ✅ Extracted {len(pdf.pages)} pages")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Save
    output_file = output_dir / "pdf_text_ultra_simple.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_text, f, indent=2)
    
    print(f"\n✅ Extracted text from {len(all_text)} PDFs")
    print(f"💾 Saved to: {output_file}")

if __name__ == "__main__":
    process_pdfs_simple()