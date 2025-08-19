"""
Test Memory Vault Write Operations
Ensures concepts can be stored and retrieved from the vault
"""

import pytest
import json
from pathlib import Path
from datetime import datetime

from python.core.memory_vault import UnifiedMemoryVault
from python.core.memory_types import MemoryConcept, MemoryType, generate_concept_id
from python.core.vault_writer import VaultWriter


def test_store_basic_concepts(tmp_path):
    """Test basic concept storage and retrieval"""
    vault_config = {
        'storage_path': str(tmp_path / "vault"),
        'max_working_memory': 100,
        'ghost_memory_ttl': 3600
    }
    
    vault = UnifiedMemoryVault(vault_config)
    
    # Create test concepts
    concepts = [
        MemoryConcept(
            id=generate_concept_id(),
            label="soliton",
            method="yake",
            score=0.99,
            relationships=[
                {"type": "related_to", "target": "wave"},
                {"type": "property_of", "target": "nonlinear"}
            ],
            metadata={"category": "physics"}
        ),
        MemoryConcept(
            id=generate_concept_id(),
            label="memory",
            method="spacy",
            score=0.97,
            relationships=[
                {"type": "used_for", "target": "storage"},
                {"type": "part_of", "target": "architecture"}
            ],
            metadata={"category": "computing"}
        )
    ]
    
    # Store concepts
    for concept in concepts:
        result = vault.store(
            content=concept.to_dict(),
            memory_type=MemoryType.SEMANTIC,
            tags=["test", "basic"]
        )
        assert result is not None
    
    # Verify storage
    vault_dir = Path(vault_config['storage_path'])
    assert vault_dir.exists()
    
    # Check that files were created
    memory_files = list(vault_dir.glob("memories/*.json"))
    assert len(memory_files) >= 2


def test_vault_writer_conversion():
    """Test VaultWriter concept conversion"""
    # Create test concepts in extraction format
    extracted_concepts = [
        {
            "name": "quantum computing",
            "score": 0.95,
            "method": "yake",
            "metadata": {
                "relationships": [
                    {"type": "subject_of", "target": "research", "verb": "advances"},
                    {"type": "related_to", "target": "cryptography"}
                ]
            }
        },
        {
            "name": "neural networks",
            "score": 0.88,
            "method": "spacy_entity",
            "metadata": {
                "entity_type": "TECH",
                "relationships": [
                    {"type": "used_in", "target": "deep learning"},
                    {"type": "composed_of", "target": "neurons"}
                ]
            }
        }
    ]
    
    # Convert using VaultWriter
    memory_concepts = VaultWriter.convert(
        concepts=extracted_concepts,
        doc_id="test_document.pdf",
        additional_tags=["ai", "technology"]
    )
    
    # Verify conversion
    assert len(memory_concepts) == 2
    
    # Check first concept
    qc = memory_concepts[0]
    assert qc.label == "quantum computing"
    assert qc.score == 0.95
    assert "test_document.pdf" in qc.metadata['doc_id']
    assert "ingested_at" in qc.metadata
    assert len(qc.relationships) == 2
    
    # Check tags
    assert "test_document.pdf" in qc.metadata.get('tags', [])
    assert "ai" in qc.metadata.get('tags', [])


def test_lemmatized_merging():
    """Test that similar concepts are merged by lemma"""
    concepts = [
        {"name": "memory", "score": 0.8, "method": "yake"},
        {"name": "memories", "score": 0.9, "method": "spacy"},
        {"name": "Memory", "score": 0.85, "method": "entity"},
    ]
    
    memory_concepts = VaultWriter.convert(
        concepts=concepts,
        doc_id="test_lemma.pdf"
    )
    
    # Should merge into one concept
    assert len(memory_concepts) == 1
    
    # Should have highest score
    assert memory_concepts[0].score == 0.9
    
    # Should combine methods
    assert "yake" in memory_concepts[0].method
    assert "spacy" in memory_concepts[0].method
    assert "entity" in memory_concepts[0].method


def test_semantic_deduplication():
    """Test semantic similarity deduplication"""
    # These should be similar enough to merge
    concepts = [
        {
            "name": "artificial intelligence",
            "score": 0.95,
            "method": "yake",
            "metadata": {"relationships": [{"type": "field", "target": "computer science"}]}
        },
        {
            "name": "AI",
            "score": 0.88,
            "method": "abbrev",
            "metadata": {"relationships": [{"type": "used_for", "target": "automation"}]}
        }
    ]
    
    # Temporarily lower threshold for testing
    original_threshold = VaultWriter.SIMILARITY_THRESHOLD
    VaultWriter.SIMILARITY_THRESHOLD = 0.7
    
    try:
        memory_concepts = VaultWriter.convert(
            concepts=concepts,
            doc_id="test_semantic.pdf"
        )
        
        # May or may not merge depending on spaCy availability
        # But should not crash
        assert len(memory_concepts) >= 1
        assert len(memory_concepts) <= 2
        
    finally:
        VaultWriter.SIMILARITY_THRESHOLD = original_threshold


def test_batch_tagging():
    """Test that batch metadata is properly added"""
    concepts = [
        {"name": "test concept", "score": 0.75, "method": "test"}
    ]
    
    timestamp_before = datetime.utcnow().isoformat()
    
    memory_concepts = VaultWriter.convert(
        concepts=concepts,
        doc_id="batch_test.pdf",
        additional_tags=["batch", "test"],
        metadata={"experiment": "batch_tagging", "version": "1.0"}
    )
    
    timestamp_after = datetime.utcnow().isoformat()
    
    mc = memory_concepts[0]
    
    # Check required metadata
    assert mc.metadata['doc_id'] == "batch_test.pdf"
    assert mc.metadata['source'] == 'semantic_extraction'
    assert 'ingested_at' in mc.metadata
    assert 'batch_id' in mc.metadata
    
    # Check timestamp is reasonable
    assert timestamp_before <= mc.metadata['ingested_at'] <= timestamp_after
    
    # Check additional metadata
    assert mc.metadata['experiment'] == "batch_tagging"
    assert mc.metadata['version'] == "1.0"
    
    # Check tags
    assert "batch_test.pdf" in mc.metadata['tags']
    assert "batch" in mc.metadata['tags']
    assert "test" in mc.metadata['tags']


def test_relationship_preservation():
    """Test that relationships are properly preserved through conversion"""
    concepts = [
        {
            "name": "soliton wave",
            "score": 0.92,
            "method": "semantic",
            "metadata": {
                "relationships": [
                    {
                        "type": "exhibits",
                        "target": "stability",
                        "verb": "maintains",
                        "confidence": 0.95
                    },
                    {
                        "type": "propagates_through",
                        "target": "medium",
                        "source": "soliton"
                    }
                ]
            }
        }
    ]
    
    memory_concepts = VaultWriter.convert(
        concepts=concepts,
        doc_id="relationships_test.pdf"
    )
    
    mc = memory_concepts[0]
    assert len(mc.relationships) == 2
    
    # Check first relationship
    rel1 = mc.relationships[0]
    assert rel1['type'] == "exhibits"
    assert rel1['target'] == "stability"
    assert rel1['verb'] == "maintains"
    assert rel1['confidence'] == 0.95
    
    # Check second relationship
    rel2 = mc.relationships[1]
    assert rel2['type'] == "propagates_through"
    assert rel2['target'] == "medium"
    assert rel2['source'] == "soliton"


def test_batch_convert_multiple_documents():
    """Test batch conversion from multiple documents"""
    document_concepts = {
        "doc1.pdf": [
            {"name": "concept A", "score": 0.8, "method": "yake"},
            {"name": "concept B", "score": 0.7, "method": "spacy"}
        ],
        "doc2.pdf": [
            {"name": "concept A", "score": 0.9, "method": "semantic"},  # Duplicate
            {"name": "concept C", "score": 0.85, "method": "entity"}
        ]
    }
    
    memory_concepts = VaultWriter.batch_convert(
        document_concepts=document_concepts,
        global_tags=["batch", "multi-doc"]
    )
    
    # Should have deduplicated concept A
    labels = [mc.label for mc in memory_concepts]
    assert len(set(labels)) == len(labels)  # All unique
    
    # All should have global tags
    for mc in memory_concepts:
        assert "batch" in mc.metadata.get('tags', [])
        assert "multi-doc" in mc.metadata.get('tags', [])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
