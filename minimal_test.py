"""
Minimal test script that directly checks the concept_extraction module
without loading the entire application.
"""

import os
import sys
import importlib.util
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("minimal_test")

def test_direct_import():
    """Try to directly import the module from its file path"""
    logger.info("Testing direct import of concept_extraction module...")
    
    # Get the absolute path to the module
    module_path = os.path.join(os.getcwd(), "ingest_pdf", "extraction", "concept_extraction.py")
    
    if not os.path.exists(module_path):
        logger.error(f"Module file not found at: {module_path}")
        return False
    
    logger.info(f"Found module file at: {module_path}")
    
    try:
        # Load the module directly from file
        spec = importlib.util.spec_from_file_location("concept_extraction", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["concept_extraction"] = module
        spec.loader.exec_module(module)
        
        logger.info("Successfully loaded the module")
        
        # Check if key functions/classes exist
        if hasattr(module, "extract_concepts_from_text"):
            logger.info("✅ Module has extract_concepts_from_text function")
        else:
            logger.warning("❌ Module does not have extract_concepts_from_text function")
            return False
        
        if hasattr(module, "Concept"):
            logger.info("✅ Module has Concept class")
        else:
            logger.warning("❌ Module does not have Concept class")
            return False
        
        if hasattr(module, "concepts_to_dicts"):
            logger.info("✅ Module has concepts_to_dicts function")
        else:
            logger.warning("❌ Module does not have concepts_to_dicts function")
            return False
        
        # List all functions in the module to help diagnosis
        logger.info("Available functions and classes in the module:")
        for name in dir(module):
            if not name.startswith("_"):  # Skip private attributes
                attr = getattr(module, name)
                if callable(attr):
                    logger.info(f"  - {name} ({'class' if isinstance(attr, type) else 'function'})")
        
        return True
    
    except Exception as e:
        logger.error(f"Error importing module: {e}")
        return False

def main():
    """Main function"""
    logger.info("Starting minimal test script")
    
    # Test direct import
    success = test_direct_import()
    
    # Print summary
    logger.info("\n" + "="*50)
    if success:
        logger.info("✅ TEST SUMMARY: concept_extraction module structure looks correct")
        logger.info("""
DIAGNOSIS:
The concept_extraction.py file contains the correct functions and classes.
The issue is with the import chain when trying to use it through normal imports.
This is most likely caused by:
1. Import loops in your application
2. Virtual environment package conflicts
3. PyTorch/transformers dependency issues
        """)
    else:
        logger.info("❌ TEST SUMMARY: concept_extraction module has structural issues")
    logger.info("="*50)

if __name__ == "__main__":
    main()
