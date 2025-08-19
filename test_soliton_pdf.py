#!/usr/bin/env python3
"""
Test semantic extraction on the Living Soliton Memory Systems PDF
This script tests relationship extraction from the specific PDF document
"""

import os
import sys
import logging
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_soliton_pdf():
    """Test the Living Soliton Memory Systems PDF"""
    pdf_path = r"{PROJECT_ROOT}\docs\LivingSolitonMemorySystemsCopy.pdf"
    
    if not os.path.exists(pdf_path):
        logger.error(f"❌ PDF not found: {pdf_path}")
        return
    
    logger.info(f"📄 Testing semantic extraction on: {os.path.basename(pdf_path)}")
    
    # Import and run the test
    from test_semantic_pdf_ingestion import test_semantic_pdf_ingestion
    
    # Run the extraction
    concepts = test_semantic_pdf_ingestion(pdf_path)
    
    if concepts:
        logger.info(f"\n✅ Extraction complete! Found {len(concepts)} concepts")
        
        # Show some statistics
        concepts_with_relations = sum(1 for c in concepts if hasattr(c, 'relationships') and c.relationships)
        total_relations = sum(len(c.relationships) if hasattr(c, 'relationships') else 0 for c in concepts)
        
        logger.info(f"📊 Summary:")
        logger.info(f"  - Total concepts: {len(concepts)}")
        logger.info(f"  - Concepts with relationships: {concepts_with_relations}")
        logger.info(f"  - Total relationships: {total_relations}")
        
        if total_relations == 0:
            logger.warning("\n⚠️ No relationships extracted!")
            logger.info("This could be because:")
            logger.info("  1. The spaCy model lacks dependency parsing")
            logger.info("  2. The text doesn't contain clear subject-verb-object patterns")
            logger.info("  3. The semantic extraction isn't being used")
            logger.info("\nTo fix, ensure you have:")
            logger.info("  - spacy with en_core_web_sm or en_core_web_trf model")
            logger.info("  - The parser component enabled")
            logger.info("  - The semantic extraction path in quality.py")

if __name__ == "__main__":
    test_soliton_pdf()
