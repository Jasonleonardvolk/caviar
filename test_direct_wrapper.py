"""
Test the direct import wrapper for concept extraction.
"""

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("direct_test")

def test_concept_extraction():
    """Test the direct concept extraction wrapper"""
    logger.info("Testing concept extraction with direct import wrapper...")
    
    try:
        # Import from our direct wrapper
        from concept_extraction_direct import (
            extract_concepts_from_text,
            Concept,
            concepts_to_dicts
        )
        
        logger.info("✅ Successfully imported from concept_extraction_direct")
        
        # Test with sample text
        test_text = """
        Artificial intelligence and machine learning are transforming industries worldwide.
        Natural language processing helps computers understand human communication.
        Data science techniques enable better decision making.
        """
        
        # Extract concepts
        logger.info("Extracting concepts from test text...")
        concepts = extract_concepts_from_text(test_text)
        
        # Log results
        logger.info(f"✅ Successfully extracted {len(concepts)} concepts")
        
        # Convert to dictionaries for display
        concept_dicts = concepts_to_dicts(concepts)
        for i, concept in enumerate(concept_dicts[:5]):  # Show first 5 concepts
            logger.info(f"  Concept {i+1}: {concept['name']} (type: {concept['type']})")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return False

def main():
    """Main function"""
    logger.info("Starting direct wrapper test")
    
    # Run the test
    success = test_concept_extraction()
    
    # Print summary
    logger.info("\n" + "="*50)
    if success:
        logger.info("✅ SUCCESS: Direct import wrapper is working correctly")
        logger.info("""
USAGE INSTRUCTIONS:
1. Import concept extraction functions from the wrapper:
   from concept_extraction_direct import extract_concepts_from_text, Concept

2. Use these functions in your code as normal:
   concepts = extract_concepts_from_text(text)

This bypasses the problematic import chain while still giving you
access to all the functionality of the original module.
        """)
    else:
        logger.info("❌ FAILURE: Direct import wrapper test failed")
    logger.info("="*50)

if __name__ == "__main__":
    main()
