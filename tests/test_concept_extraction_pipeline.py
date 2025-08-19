"""
Comprehensive test suite for semantic concept extraction pipeline
Tests keyword extraction, entity recognition, and relationship extraction
"""

import pytest
import json
from pathlib import Path
from typing import List, Dict, Any

# Import the extraction module
from ingest_pdf.extraction.concept_extraction import (
    extract_semantic_concepts,
    extract_concepts_from_text,
    extract_keywords_yake,
    extract_svo_relationships,
    load_spacy_model,
    Concept,
    Relationship
)


# -----------------------------------------------------------------------------
# 🧪 TEST DATA
# -----------------------------------------------------------------------------
SIMPLE_TEXT = "Alan Turing invented the modern computer. He changed history."

COMPLEX_TEXT = """
In 1953, Watson and Crick discovered the structure of DNA.
The double helix became a milestone in molecular biology.
DNA encodes genetic information in living organisms.
"""

TECHNICAL_TEXT = """
The transformer architecture revolutionized natural language processing.
BERT uses bidirectional training of transformers.
GPT implements autoregressive language modeling.
Attention mechanisms enable parallel processing of sequences.
"""

SOLITON_TEXT = """
Solitons are self-reinforcing wave packets that maintain their shape.
The soliton propagates through the medium without dispersion.
Optical solitons enable long-distance fiber communication.
Soliton dynamics exhibit remarkable stability properties.
"""

EMPTY_TEXT = "   "
SHORT_TEXT = "Hi"


# -----------------------------------------------------------------------------
# 🧪 FIXTURES
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def nlp_model():
    """Load spaCy model once for all tests"""
    model = load_spacy_model(auto_install=True)
    if model is None:
        pytest.skip("spaCy model not available")
    return model


# -----------------------------------------------------------------------------
# 🧪 BASIC EXTRACTION TESTS
# -----------------------------------------------------------------------------
def test_extract_semantic_concepts_simple():
    """Test basic extraction with simple text"""
    results = extract_semantic_concepts(SIMPLE_TEXT)
    
    assert isinstance(results, list)
    assert len(results) > 0
    
    # Check structure
    for concept in results:
        assert "name" in concept
        assert "score" in concept
        assert "method" in concept
        assert "metadata" in concept
        assert "relationships" in concept["metadata"]


def test_extract_semantic_concepts_relationships():
    """Test relationship extraction"""
    results = extract_semantic_concepts(SIMPLE_TEXT, use_nlp=True)
    
    # Find relationships
    all_relationships = []
    for concept in results:
        all_relationships.extend(concept["metadata"].get("relationships", []))
    
    assert len(all_relationships) > 0, "Should extract at least one relationship"
    
    # Check relationship structure
    for rel in all_relationships:
        assert "type" in rel
        assert "target" in rel
        assert rel["type"] in ["subject_of", "object_of", "related_to"]


def test_extract_semantic_concepts_complex():
    """Test extraction with complex scientific text"""
    results = extract_semantic_concepts(COMPLEX_TEXT)
    
    # Should find key entities
    concept_names = [c["name"].lower() for c in results]
    assert any("dna" in name for name in concept_names)
    assert any("watson" in name or "crick" in name for name in concept_names)
    
    # Should have relationships
    relationships_count = sum(
        len(c["metadata"].get("relationships", [])) 
        for c in results
    )
    assert relationships_count > 0


def test_extract_semantic_concepts_technical():
    """Test extraction with technical AI/ML text"""
    results = extract_semantic_concepts(TECHNICAL_TEXT)
    
    concept_names = [c["name"].lower() for c in results]
    
    # Should identify key technical terms
    assert any("transformer" in name for name in concept_names)
    assert any("bert" in name or "gpt" in name for name in concept_names)
    
    # Should identify relationships between concepts
    transformer_concepts = [c for c in results if "transformer" in c["name"].lower()]
    if transformer_concepts:
        assert len(transformer_concepts[0]["metadata"].get("relationships", [])) > 0


def test_extract_semantic_concepts_soliton():
    """Test extraction with soliton physics text"""
    results = extract_semantic_concepts(SOLITON_TEXT)
    
    # Should identify soliton-related concepts
    concept_names = [c["name"] for c in results]
    soliton_concepts = [name for name in concept_names if "soliton" in name.lower()]
    assert len(soliton_concepts) > 0
    
    # Should find relationships about solitons
    soliton_results = [c for c in results if "soliton" in c["name"].lower()]
    if soliton_results:
        relationships = soliton_results[0]["metadata"].get("relationships", [])
        assert len(relationships) > 0, "Should find relationships for soliton concepts"


# -----------------------------------------------------------------------------
# 🧪 EDGE CASE TESTS
# -----------------------------------------------------------------------------
def test_extract_semantic_concepts_empty():
    """Test with empty text"""
    results = extract_semantic_concepts(EMPTY_TEXT)
    assert results == []


def test_extract_semantic_concepts_short():
    """Test with very short text"""
    results = extract_semantic_concepts(SHORT_TEXT)
    assert results == []


def test_extract_semantic_concepts_none():
    """Test with None input"""
    results = extract_semantic_concepts(None)
    assert results == []


# -----------------------------------------------------------------------------
# 🧪 COMPONENT TESTS
# -----------------------------------------------------------------------------
def test_extract_keywords_yake():
    """Test YAKE keyword extraction"""
    concepts = extract_keywords_yake(TECHNICAL_TEXT)
    
    if concepts:  # Only test if YAKE is available
        assert all(isinstance(c, Concept) for c in concepts)
        assert all(0 <= c.score <= 1 for c in concepts)
        assert all(c.method == "yake" for c in concepts)


def test_svo_extraction_with_spacy(nlp_model):
    """Test SVO relationship extraction"""
    doc = nlp_model("The researcher discovered a new algorithm.")
    relationships = extract_svo_relationships(doc)
    
    assert len(relationships) > 0
    rel = relationships[0]
    assert rel["subject"] == "researcher"
    assert rel["verb"] == "discovered"
    assert rel["object"] == "algorithm"


def test_backward_compatibility():
    """Test backward compatibility with extract_concepts_from_text"""
    results = extract_concepts_from_text(SIMPLE_TEXT)
    
    assert isinstance(results, list)
    if results:
        # Should have basic structure but no relationships
        concept = results[0]
        assert "name" in concept
        assert "score" in concept
        # Relationships should be empty for backward compatibility
        if "metadata" in concept and "relationships" in concept["metadata"]:
            assert concept["metadata"]["relationships"] == []


# -----------------------------------------------------------------------------
# 🧪 INTEGRATION TESTS
# -----------------------------------------------------------------------------
def test_concept_deduplication():
    """Test that concepts are properly deduplicated"""
    text = "Python is great. Python programming is powerful. I love Python."
    results = extract_semantic_concepts(text)
    
    # Count unique concept names
    unique_names = set(c["name"].lower() for c in results)
    # Should not have excessive duplicates
    python_concepts = [c for c in results if "python" in c["name"].lower()]
    assert len(python_concepts) <= 2  # "Python" and maybe "Python programming"


def test_method_combination():
    """Test that multiple extraction methods are combined"""
    results = extract_semantic_concepts(TECHNICAL_TEXT)
    
    # Should have concepts from different methods
    methods = [c["method"] for c in results]
    method_types = set()
    for method in methods:
        if "+" in method:
            method_types.update(method.split("+"))
        else:
            method_types.add(method)
    
    # Should use multiple extraction methods
    assert len(method_types) >= 2


def test_score_ordering():
    """Test that concepts are ordered by score"""
    results = extract_semantic_concepts(COMPLEX_TEXT)
    
    if len(results) > 1:
        scores = [c["score"] for c in results]
        assert scores == sorted(scores, reverse=True)


# -----------------------------------------------------------------------------
# 🧪 PERFORMANCE TESTS
# -----------------------------------------------------------------------------
def test_large_text_performance():
    """Test performance with larger text"""
    large_text = TECHNICAL_TEXT * 10  # Repeat text 10 times
    
    import time
    start = time.time()
    results = extract_semantic_concepts(large_text)
    duration = time.time() - start
    
    assert duration < 5.0  # Should complete within 5 seconds
    assert len(results) > 0


# -----------------------------------------------------------------------------
# 🧪 ERROR HANDLING TESTS
# -----------------------------------------------------------------------------
def test_unicode_handling():
    """Test handling of unicode characters"""
    unicode_text = "Schrödinger's cat demonstrates quantum superposition. É fundamental à física."
    results = extract_semantic_concepts(unicode_text)
    
    assert len(results) > 0
    # Should handle unicode without errors


def test_malformed_text():
    """Test handling of malformed text"""
    malformed = "This is... um... you know... like... whatever..."
    results = extract_semantic_concepts(malformed)
    
    # Should still extract something
    assert isinstance(results, list)


# -----------------------------------------------------------------------------
# 🧪 OUTPUT FORMAT TESTS
# -----------------------------------------------------------------------------
def test_output_json_serializable():
    """Test that output can be JSON serialized"""
    results = extract_semantic_concepts(COMPLEX_TEXT)
    
    # Should be JSON serializable
    try:
        json_str = json.dumps(results, indent=2)
        # Should be able to parse it back
        parsed = json.loads(json_str)
        assert parsed == results
    except Exception as e:
        pytest.fail(f"Output not JSON serializable: {e}")


def test_relationship_format():
    """Test relationship data structure"""
    results = extract_semantic_concepts(SIMPLE_TEXT, use_nlp=True)
    
    for concept in results:
        for rel in concept["metadata"].get("relationships", []):
            # Check required fields
            assert "type" in rel
            assert "target" in rel
            
            # Check optional fields have correct types
            if "source" in rel:
                assert isinstance(rel["source"], str)
            if "verb" in rel:
                assert isinstance(rel["verb"], str)
            if "confidence" in rel:
                assert isinstance(rel["confidence"], (int, float))


# -----------------------------------------------------------------------------
# 🧪 MAIN TEST RUNNER
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
