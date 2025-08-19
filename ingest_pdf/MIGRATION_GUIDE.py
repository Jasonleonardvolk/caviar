"""
Migration Guide: From pipeline.py to modular pipeline package

This script helps verify that the new modular structure is working correctly
and provides guidance for updating imports in your code.
"""

import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_backward_compatibility():
    """Test that old imports still work"""
    logger.info("=" * 60)
    logger.info("TESTING BACKWARD COMPATIBILITY")
    logger.info("=" * 60)
    
    try:
        # Old way - direct import from pipeline.py
        logger.info("\n1. Testing direct import (old way):")
        logger.info("   from pipeline import ingest_pdf_clean")
        
        # This should now import from the package
        from pipeline import ingest_pdf_clean
        logger.info("   ✅ SUCCESS - ingest_pdf_clean imported")
        
    except ImportError as e:
        logger.error(f"   ❌ FAILED - {e}")
        return False
    
    try:
        # Test configuration imports
        logger.info("\n2. Testing configuration imports:")
        from pipeline import ENABLE_OCR_FALLBACK, OCR_MAX_PAGES
        logger.info(f"   ✅ ENABLE_OCR_FALLBACK = {ENABLE_OCR_FALLBACK}")
        logger.info(f"   ✅ OCR_MAX_PAGES = {OCR_MAX_PAGES}")
        
    except ImportError as e:
        logger.error(f"   ❌ FAILED - {e}")
        return False
    
    try:
        # Test utility imports
        logger.info("\n3. Testing utility imports:")
        from pipeline import safe_divide, safe_get
        result = safe_divide(10, 2)
        logger.info(f"   ✅ safe_divide(10, 2) = {result}")
        
    except ImportError as e:
        logger.error(f"   ❌ FAILED - {e}")
        return False
    
    logger.info("\n✅ All backward compatibility tests passed!")
    return True


def show_new_import_structure():
    """Show the new recommended import structure"""
    logger.info("\n" + "=" * 60)
    logger.info("NEW MODULAR IMPORT STRUCTURE")
    logger.info("=" * 60)
    
    logger.info("""
The pipeline is now organized into logical modules:

📁 pipeline/
   ├── __init__.py      # Backward compatibility layer
   ├── config.py        # All configuration and constants
   ├── utils.py         # Safe math and utility functions
   ├── io.py            # Input/output operations
   ├── quality.py       # Concept quality and analysis
   ├── pruning.py       # Entropy pruning and clustering
   ├── storage.py       # Soliton Memory integration
   └── pipeline.py      # Main orchestration

You can now import more specifically:

# Import just what you need
from pipeline.config import ENABLE_OCR_FALLBACK, OCR_MAX_PAGES
from pipeline.utils import safe_divide, safe_get
from pipeline.quality import calculate_concept_quality
from pipeline.pruning import cluster_similar_concepts

# Or continue using the old way (still works!)
from pipeline import ingest_pdf_clean
""")


def check_dependencies():
    """Check if all dependencies are available"""
    logger.info("\n" + "=" * 60)
    logger.info("CHECKING DEPENDENCIES")
    logger.info("=" * 60)
    
    dependencies = {
        'PyPDF2': 'PDF processing',
        'pytesseract': 'OCR support (optional)',
        'pdf2image': 'OCR support (optional)',
        'PIL': 'Image processing (optional)',
        'nltk': 'NLP enhancement (optional)'
    }
    
    for module, purpose in dependencies.items():
        try:
            if module == 'PIL':
                __import__('PIL')
            else:
                __import__(module)
            logger.info(f"✅ {module:<15} - {purpose}")
        except ImportError:
            logger.warning(f"⚠️  {module:<15} - {purpose} (NOT INSTALLED)")


def migration_checklist():
    """Show migration checklist for users"""
    logger.info("\n" + "=" * 60)
    logger.info("MIGRATION CHECKLIST")
    logger.info("=" * 60)
    
    logger.info("""
For existing code using pipeline.py:

1. ✅ No immediate changes required - backward compatibility maintained

2. 🔧 Optional optimizations:
   - Update imports to use specific modules (faster imports)
   - Example: from pipeline.utils import safe_divide
   
3. 📝 Update your imports gradually:
   OLD: from pipeline import *
   NEW: from pipeline import ingest_pdf_clean, safe_divide, safe_get

4. 🧪 Test your integration:
   - Run your existing PDF processing code
   - Verify results match previous version
   - Check logs for any warnings

5. 📚 For new projects:
   - Use the modular imports from the start
   - Refer to specific modules for functionality
   - Better IDE autocomplete and type hints
""")


if __name__ == "__main__":
    logger.info("🚀 TORI Pipeline Migration Helper\n")
    
    # Run tests
    if test_backward_compatibility():
        show_new_import_structure()
        check_dependencies()
        migration_checklist()
        
        logger.info("\n" + "=" * 60)
        logger.info("✨ Migration check complete! Your code should work as before.")
        logger.info("=" * 60)
    else:
        logger.error("\n❌ Some compatibility issues detected. Please check the errors above.")
        sys.exit(1)
