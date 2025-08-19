# Save this as: C:\\Users\\jason\\Desktop\\tori\\kha\use_existing_pdfs.py

import os
import sys
from pathlib import Path
import asyncio

# Add pigpen to path
sys.path.insert(0, "C:/Users/jason/Desktop/tori/kha")

from ingest_pdf.pipeline.pipeline import ingest_pdf_clean

class UseExistingPDFs:
    """Use your PDFs in 06_27 folder"""
    
    def __init__(self):
        self.pdf_folder = Path("C:/Users/jason/Desktop/tori/kha/docs/material/dataset/06_27")
        self.output_dir = Path("C:/Users/jason/Desktop/tori/kha/concept_mesh")
        self.output_dir.mkdir(exist_ok=True)
        
    async def process_all_pdfs(self):
        """Process all PDFs in your folder"""
        print(f"📚 Processing PDFs from: {self.pdf_folder}")
        print("=" * 60)
        
        # Check if folder exists
        if not self.pdf_folder.exists():
            print(f"❌ Folder not found: {self.pdf_folder}")
            return
            
        all_concepts = []
        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        
        print(f"Found {len(pdf_files)} PDFs")
        
        for pdf in pdf_files:
            print(f"\n📄 Processing: {pdf.name}")
            try:
                result = ingest_pdf_clean(
                    str(pdf),
                    admin_mode=True  # Get all concepts
                )
                concepts = result.get("concepts", [])
                all_concepts.extend(concepts)
                print(f"   ✅ Extracted {len(concepts)} concepts")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Save to concept mesh
        import json
        output_file = self.output_dir / "pdf_concepts.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_concepts": len(all_concepts),
                "concepts": all_concepts
            }, f, indent=2)
        
        print(f"\n✅ Total concepts extracted: {len(all_concepts)}")
        print(f"💾 Saved to: {output_file}")

async def main():
    processor = UseExistingPDFs()
    await processor.process_all_pdfs()

if __name__ == "__main__":
    asyncio.run(main())