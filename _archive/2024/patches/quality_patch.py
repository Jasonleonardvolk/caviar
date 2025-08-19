"""
Patch for quality.py to add semantic extraction function
"""

def extract_concepts_with_spacy(text: str, doc_context: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Enhanced concept extraction with semantic relationships using spaCy.
    
    Args:
        text: Text to extract concepts from
        doc_context: Optional document context for better extraction
        
    Returns:
        List of concepts with metadata including relationships
    """
    if not CONCEPT_EXTRACTION_AVAILABLE or not extract_semantic_concepts:
        logger.warning("⚠️ Semantic extraction not available, falling back to basic extraction")
        return []
    
    try:
        # Try semantic extraction first (includes relationships)
        logger.info("🧠 Attempting semantic concept extraction with relationships...")
        concepts = extract_semantic_concepts(text)
        
        # Log extraction statistics
        total_concepts = len(concepts)
        total_relations = sum(len(c.get('metadata', {}).get('relationships', [])) for c in concepts)
        concepts_with_relations = sum(1 for c in concepts if c.get('metadata', {}).get('relationships'))
        
        logger.info(f"📊 Extracted {total_concepts} concepts with {total_relations} relationships")
        logger.info(f"🔗 {concepts_with_relations}/{total_concepts} concepts have relationships")
        
        # If we got good results, return them
        if concepts:
            # Log some sample concepts with relationships
            for i, concept in enumerate(concepts[:3]):
                relations = concept.get('metadata', {}).get('relationships', [])
                if relations:
                    logger.debug(f"   → {concept['name']}: {len(relations)} relationships")
                    for rel in relations[:2]:
                        logger.debug(f"      • {rel['type']}: {rel['target']}")
            
            return concepts
            
    except Exception as e:
        logger.error(f"❌ Semantic extraction failed: {e}")
        
    # Fallback to basic extraction without relationships
    try:
        logger.info("⚠️ Falling back to basic entity extraction (no relationships)")
        if hasattr(extract_concepts_from_text, '__call__'):
            concepts = extract_concepts_from_text(text)
            # Ensure all concepts have empty relationships field
            for concept in concepts:
                if 'metadata' not in concept:
                    concept['metadata'] = {}
                if 'relationships' not in concept['metadata']:
                    concept['metadata']['relationships'] = []
            return concepts
    except Exception as e:
        logger.error(f"❌ Basic extraction also failed: {e}")
        
    return []


# Add this function to quality.py right after the imports section
