#!/usr/bin/env python3
"""
Test script for semantic PDF ingestion with relationship extraction
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_semantic_pdf_ingestion(pdf_path: str):
    """Test semantic PDF ingestion with relationships"""
    try:
        # Import the semantic extraction
        from ingest_pdf.extraction.concept_extraction import extract_semantic_concepts
        
        logger.info(f"📄 Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        text = ""
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    text += page_text
                    logger.debug(f"  Page {page_num + 1}: {len(page_text)} chars")
        except Exception as e:
            logger.warning(f"⚠️ PyPDF2 failed: {e}, trying pdfplumber")
            try:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        page_text = page.extract_text() or ""
                        text += page_text
                        logger.debug(f"  Page {page_num + 1}: {len(page_text)} chars")
            except Exception as e2:
                logger.error(f"❌ PDF parsing failed: {e2}")
                return
        
        if not text.strip():
            logger.warning(f"⚠️ No text extracted from {pdf_path}")
            return
        
        logger.info(f"📊 Extracted {len(text)} characters from PDF")
        
        # Extract semantic concepts with relations
        logger.info("🧠 Running semantic concept extraction with NLP relations enabled")
        concepts = extract_semantic_concepts(text, use_nlp=True)
        
        # Analyze results
        total_concepts = len(concepts)
        total_relations = 0
        concepts_with_relations = 0
        
        for concept in concepts:
            # Handle both object and dict formats
            if hasattr(concept, 'relationships'):
                relations = concept.relationships
            else:
                relations = concept.get('relationships', [])
            
            if relations:
                concepts_with_relations += 1
                total_relations += len(relations)
        
        # Log results
        logger.info(f"✅ Extraction complete:")
        logger.info(f"  📊 Total concepts: {total_concepts}")
        logger.info(f"  🔗 Total relationships: {total_relations}")
        logger.info(f"  🎯 Concepts with relations: {concepts_with_relations}")
        
        # Show sample concepts with relationships
        if total_relations > 0:
            logger.info("\n📋 Sample concepts with relationships:")
            shown = 0
            for concept in concepts:
                if hasattr(concept, 'relationships'):
                    name = concept.name
                    relations = concept.relationships
                else:
                    name = concept.get('name', '')
                    relations = concept.get('relationships', [])
                
                if relations and shown < 5:
                    logger.info(f"\n  🔹 Concept: {name}")
                    for rel in relations[:3]:  # Show first 3 relations
                        logger.info(f"     → {rel}")
                    if len(relations) > 3:
                        logger.info(f"     ... and {len(relations) - 3} more")
                    shown += 1
        
        # Store to memory vault if available
        try:
            from python.core.memory_vault import UnifiedMemoryVault
            
            vault = UnifiedMemoryVault.get_instance()
            if not vault:
                vault = UnifiedMemoryVault({'storage_path': 'data/memory_vault'})
            
            stored_count = 0
            for concept in concepts[:50]:  # Store first 50 concepts
                if hasattr(concept, 'name'):
                    concept_name = concept.name
                    concept_data = concept.__dict__ if hasattr(concept, '__dict__') else concept
                else:
                    concept_name = concept.get('name', '')
                    concept_data = concept
                
                if concept_name:
                    memory_id = f"pdf_concept_{concept_name.replace(' ', '_')}"
                    vault.store(memory_id, {
                        "type": "concept",
                        "content": concept_data,
                        "metadata": {
                            "source": "pdf_ingestion",
                            "file": os.path.basename(pdf_path)
                        }
                    })
                    stored_count += 1
            
            # Save to disk
            vault.save_all()
            logger.info(f"\n💾 Stored {stored_count} concepts to memory vault")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not store to memory vault: {e}")
        
        return concepts
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_test_pdf():
    """Create a simple test PDF with content that should have relationships"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        pdf_path = "test_semantic_extraction.pdf"
        c = canvas.Canvas(pdf_path, pagesize=letter)
        
        # Add test content with clear relationships
        c.drawString(100, 750, "Test Document for Semantic Extraction")
        c.drawString(100, 700, "")
        c.drawString(100, 650, "Albert Einstein developed the theory of relativity.")
        c.drawString(100, 630, "The theory of relativity revolutionized physics.")
        c.drawString(100, 610, "Quantum mechanics describes subatomic particles.")
        c.drawString(100, 590, "Einstein contributed to quantum theory.")
        c.drawString(100, 570, "Black holes were predicted by general relativity.")
        c.drawString(100, 550, "Stephen Hawking studied black hole radiation.")
        c.drawString(100, 530, "Machine learning uses neural networks.")
        c.drawString(100, 510, "Deep learning is a subset of machine learning.")
        c.drawString(100, 490, "Neural networks mimic brain structure.")
        c.drawString(100, 470, "Artificial intelligence includes machine learning.")
        
        c.save()
        logger.info(f"✅ Created test PDF: {pdf_path}")
        return pdf_path
        
    except ImportError:
        logger.warning("⚠️ reportlab not available, using existing PDF")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test semantic PDF ingestion")
    parser.add_argument("--pdf", help="Path to PDF file to test")
    parser.add_argument("--create-test", action="store_true", help="Create a test PDF")
    args = parser.parse_args()
    
    if args.create_test:
        pdf_path = create_test_pdf()
        if pdf_path:
            test_semantic_pdf_ingestion(pdf_path)
    elif args.pdf:
        if not os.path.exists(args.pdf):
            logger.error(f"❌ PDF not found: {args.pdf}")
        else:
            test_semantic_pdf_ingestion(args.pdf)
    else:
        # Look for PDFs in test directory
        test_dir = Path("data/test_pdfs")
        if test_dir.exists():
            pdfs = list(test_dir.glob("*.pdf"))
            if pdfs:
                logger.info(f"🔍 Found {len(pdfs)} PDFs in test directory")
                for pdf in pdfs[:2]:  # Test first 2
                    logger.info(f"\n{'='*60}")
                    test_semantic_pdf_ingestion(str(pdf))
            else:
                logger.warning("⚠️ No PDFs found in data/test_pdfs/")
                logger.info("💡 Add PDFs to data/test_pdfs/ or use --pdf option")
        else:
            logger.warning("⚠️ Test directory not found: data/test_pdfs/")
            logger.info("💡 Use --pdf option to specify a PDF file")
