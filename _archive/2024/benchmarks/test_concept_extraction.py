#!/usr/bin/env python
"""
Test script for concept_extraction module.
This script attempts to import the concept_extraction module and use its functions
to verify that the module is working correctly.
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("concept_extraction_test")

def test_imports():
    """Test importing the concept_extraction module"""
    logger.info("Testing imports...")
    
    # Add current directory to Python path to ensure we can import from the project
    sys.path.insert(0, os.getcwd())
    
    try:
        # Try the canonical import path
        logger.info("Attempting to import from ingest_pdf.extraction.concept_extraction...")
        from ingest_pdf.extraction.concept_extraction import (
            extract_concepts_from_text,
            Concept,
            concepts_to_dicts
        )
        
        logger.info("✅ Successfully imported from ingest_pdf.extraction.concept_extraction")
        
        # Get the module file path
        import ingest_pdf.extraction.concept_extraction as ce
        logger.info(f"Module file path: {ce.__file__}")
        
        return extract_concepts_from_text, Concept, concepts_to_dicts
    
    except ImportError as e:
        logger.error(f"❌ Failed to import from ingest_pdf.extraction: {e}")
        
        try:
            # Try direct import as fallback
            logger.info("Attempting fallback direct import from concept_extraction...")
            from concept_extraction import (
                extract_concepts_from_text,
                Concept,
                concepts_to_dicts
            )
            
            logger.info("✅ Successfully imported directly from concept_extraction")
            
            # Get the module file path
            import concept_extraction as ce
            logger.info(f"Module file path: {ce.__file__}")
            
            return extract_concepts_from_text, Concept, concepts_to_dicts
        
        except ImportError as e2:
            logger.error(f"❌ Fallback import also failed: {e2}")
            return None, None, None

def test_functionality(extract_fn, concept_cls, convert_fn):
    """Test the functionality of the concept_extraction module"""
    if not extract_fn:
        logger.error("Cannot test functionality: import failed")
        return False
    
    logger.info("Testing functionality...")
    
    # Test text for concept extraction
    test_text = """
    Artificial intelligence is transforming industries around the world. 
    Machine learning algorithms can analyze data at scale. 
    Natural language processing helps computers understand human language.
    """
    
    try:
        # Extract concepts
        logger.info("Extracting concepts from test text...")
        concepts = extract_fn(test_text)
        
        # Log results
        if concepts:
            logger.info(f"✅ Successfully extracted {len(concepts)} concepts")
            
            # Check if we got Concept objects or dictionaries
            if isinstance(concepts[0], concept_cls):
                logger.info("✅ Received Concept objects as expected")
                
                # Convert to dictionaries for display
                concept_dicts = convert_fn(concepts)
                for i, concept in enumerate(concept_dicts[:3]):  # Show first 3 concepts
                    logger.info(f"  Concept {i+1}: {concept['name']} (type: {concept['type']})")
            else:
                logger.warning("⚠️ Did not receive Concept objects as expected")
                for i, concept in enumerate(concepts[:3]):  # Show first 3 concepts
                    logger.info(f"  Concept {i+1}: {concept}")
        else:
            logger.warning("⚠️ No concepts extracted (this might be normal for this text)")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Error testing functionality: {e}")
        return False

def check_environment():
    """Check Python environment information"""
    logger.info("Checking Python environment...")
    
    # Python version
    logger.info(f"Python version: {sys.version}")
    
    # Python path
    logger.info("Python path:")
    for i, path in enumerate(sys.path):
        logger.info(f"  {i}: {path}")
    
    # Current working directory
    logger.info(f"Current working directory: {os.getcwd()}")

def main():
    """Main function"""
    logger.info("Starting concept_extraction test script")
    
    # Check environment
    check_environment()
    
    # Test imports
    extract_fn, concept_cls, convert_fn = test_imports()
    
    # Test functionality
    success = test_functionality(extract_fn, concept_cls, convert_fn)
    
    # Print summary
    logger.info("\n" + "="*50)
    if success:
        logger.info("✅ TEST SUMMARY: concept_extraction module appears to be working correctly")
    else:
        logger.info("❌ TEST SUMMARY: concept_extraction module test failed")
    logger.info("="*50)

if __name__ == "__main__":
    main()
