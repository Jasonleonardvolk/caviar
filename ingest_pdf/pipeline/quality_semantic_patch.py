"""
Patch for semantic concept extraction with relations
This patches the quality.py file to use semantic extraction
"""

# Find and replace the extract_concepts_with_spacy function with this enhanced version:

def extract_concepts_with_spacy(chunk: str, chunk_section: str = "body") -> List[Dict[str, Any]]:
    """
    Extract concepts using spaCy with entity linking AND semantic relations.
    
    Args:
        chunk: Text chunk to process
        chunk_section: Academic section type
        
    Returns:
        List of entity-linked concepts with relationships
    """
    # First try to use the semantic extraction if available
    if CONCEPT_EXTRACTION_AVAILABLE and extract_semantic_concepts:
        logger.info("🧠 Using semantic concept extraction with relationship parsing")
        try:
            # Extract semantic concepts with relations
            semantic_concepts = extract_semantic_concepts(chunk, use_nlp=True)
            
            # Convert to our expected format
            formatted_concepts = []
            for concept in semantic_concepts:
                # Handle both object and dict formats
                if hasattr(concept, '__dict__'):
                    concept_dict = concept.__dict__
                    name = getattr(concept, 'name', '')
                    score = getattr(concept, 'score', 0.7)
                    relationships = getattr(concept, 'relationships', [])
                else:
                    concept_dict = concept
                    name = concept.get('name', '')
                    score = concept.get('score', 0.7)
                    relationships = concept.get('relationships', [])
                
                if name:
                    formatted_concept = {
                        "name": name,
                        "score": score,
                        "method": "semantic_extraction",
                        "metadata": {
                            "section": chunk_section,
                            "relationships": relationships,
                            "extraction_type": "semantic_with_relations"
                        }
                    }
                    
                    # Copy any additional metadata
                    if isinstance(concept, dict) and 'metadata' in concept:
                        formatted_concept['metadata'].update(concept['metadata'])
                    
                    formatted_concepts.append(formatted_concept)
            
            # Log extraction results
            total_relations = sum(len(c['metadata'].get('relationships', [])) for c in formatted_concepts)
            logger.info(f"📊 Extracted {len(formatted_concepts)} concepts with {total_relations} relationships")
            
            return formatted_concepts
            
        except Exception as e:
            logger.warning(f"⚠️ Semantic extraction failed, falling back to entity linking: {e}")
    
    # Fallback to original spaCy entity linking if semantic extraction not available
    nlp = get_spacy_linker()
    if not nlp:
        logger.debug("🚫 spaCy not available for concept extraction")
        return []
    
    semantic_hits = []
    
    try:
        # Process with spaCy
        doc = nlp(chunk)
        
        # Extract named entities with KB links
        for ent in doc.ents:
            # Check if entity has KB candidates
            if hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                # Get top KB candidate
                kb_id, confidence = ent._.kb_ents[0]
                
                semantic_hits.append({
                    "name": ent.text,
                    "score": confidence,
                    "method": "spacy_entity_linker",
                    "metadata": {
                        "wikidata_id": kb_id,
                        "entity_type": ent.label_,
                        "confidence": confidence,
                        "section": chunk_section,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "relationships": []  # Empty for entity linking only
                    }
                })
            else:
                # Regular entity without KB link
                semantic_hits.append({
                    "name": ent.text,
                    "score": 0.8,
                    "method": "spacy_ner",
                    "metadata": {
                        "entity_type": ent.label_,
                        "section": chunk_section,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "relationships": []  # Empty for entity linking only
                    }
                })
        
        # Also extract noun phrases for additional concepts
        for np in doc.noun_chunks:
            if len(np.text) >= MIN_CONCEPT_LENGTH and np.text.lower() not in GENERIC_TERMS:
                semantic_hits.append({
                    "name": np.text,
                    "score": 0.6,
                    "method": "spacy_noun_phrase",
                    "metadata": {
                        "section": chunk_section,
                        "root": np.root.text,
                        "relationships": []  # Empty for entity linking only
                    }
                })
        
    except Exception as e:
        logger.error(f"❌ spaCy processing error: {e}")
    
    return semantic_hits