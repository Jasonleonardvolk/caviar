"""
Snapshot Test for JSON-Serialized Vault
Verifies the complete data structure and integrity
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
import asyncio

from python.core.memory_vault import UnifiedMemoryVault
from python.core.memory_types import MemoryConcept, MemoryType, generate_concept_id
from python.core.vault_writer import VaultWriter


async def test_vault_serialization_snapshot(tmp_path):
    """Test complete vault serialization to JSON"""
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    
    # Create test concepts with full metadata
    extracted_concepts = [
        {
            "name": "soliton",
            "score": 0.99,
            "method": "yake+spacy",
            "metadata": {
                "relationships": [
                    {"type": "exhibits", "target": "wave behavior", "verb": "shows"},
                    {"type": "maintains", "target": "shape", "confidence": 0.95}
                ],
                "entity_type": "PHYSICS_CONCEPT",
                "section": "introduction"
            }
        },
        {
            "name": "memory architecture",
            "score": 0.97,
            "method": "semantic",
            "metadata": {
                "relationships": [
                    {"type": "contains", "target": "storage layers"},
                    {"type": "implements", "target": "caching", "verb": "uses"}
                ],
                "importance": "high"
            }
        },
        {
            "name": "memories",  # Should merge with "memory architecture"
            "score": 0.85,
            "method": "entity",
            "metadata": {
                "relationships": [
                    {"type": "stored_in", "target": "vault"}
                ]
            }
        }
    ]
    
    # Convert using VaultWriter
    memory_concepts = VaultWriter.convert(
        concepts=extracted_concepts,
        doc_id="LivingSolitonMemorySystemsCopy.pdf",
        additional_tags=["physics", "computing", "test"],
        metadata={
            "extraction_version": "2.0",
            "pipeline": "semantic_enhanced"
        }
    )
    
    # Initialize vault
    vault_config = {
        'storage_path': str(vault_path),
        'max_working_memory': 100,
        'ghost_memory_ttl': 3600
    }
    vault = UnifiedMemoryVault(vault_config)
    
    # Store concepts
    for mc in memory_concepts:
        await vault.store(
            content=mc.to_dict(),
            memory_type=MemoryType.SEMANTIC,
            tags=["snapshot_test"]
        )
    
    # Force save
    await vault.save_all()
    
    # Find the saved memory files
    memory_files = list((vault_path / "memories").glob("*.json"))
    assert len(memory_files) > 0
    
    # Read and validate JSON structure
    for memory_file in memory_files:
        with open(memory_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate top-level structure
        assert "id" in data
        assert "type" in data
        assert "content" in data
        assert "metadata" in data
        assert "timestamp" in data
        
        # Validate content structure (the MemoryConcept)
        content = data["content"]
        assert "id" in content
        assert "label" in content
        assert "method" in content
        assert "score" in content
        assert "relationships" in content
        assert "metadata" in content
        assert "created_at" in content
        assert "updated_at" in content
        
        # Validate metadata enrichment
        meta = content["metadata"]
        assert "doc_id" in meta
        assert "ingested_at" in meta
        assert "batch_id" in meta
        assert "source" in meta
        assert meta["source"] == "semantic_extraction"
        assert "extraction_version" in meta
        assert meta["extraction_version"] == "2.0"
        
        # Validate relationships structure
        for rel in content["relationships"]:
            assert "type" in rel
            assert "target" in rel
            # Optional fields
            if "confidence" in rel:
                assert isinstance(rel["confidence"], (int, float))
            if "verb" in rel:
                assert isinstance(rel["verb"], str)
        
        # Validate tags
        assert "tags" in meta
        assert "LivingSolitonMemorySystemsCopy.pdf" in meta["tags"]
        assert "physics" in meta["tags"] or "computing" in meta["tags"]
    
    # Create a snapshot summary
    snapshot_path = vault_path / "snapshot_summary.json"
    summary = {
        "total_concepts": len(memory_concepts),
        "concepts": [
            {
                "label": mc.label,
                "score": mc.score,
                "method": mc.method,
                "relationships_count": len(mc.relationships),
                "tags": mc.metadata.get("tags", [])
            }
            for mc in memory_concepts
        ],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Verify summary is valid JSON
    with open(snapshot_path, 'r', encoding='utf-8') as f:
        loaded_summary = json.load(f)
    
    assert loaded_summary["total_concepts"] == len(memory_concepts)
    assert len(loaded_summary["concepts"]) == len(memory_concepts)


def test_vault_deduplication_strategy():
    """Test that vault properly handles duplicate concepts"""
    # This test would verify the vault's internal deduplication
    # Would need to mock or implement the actual vault dedup logic
    pass


def test_full_pipeline_snapshot():
    """Test the complete pipeline from PDF extraction to vault storage"""
    # This would be an integration test combining:
    # 1. PDF text extraction
    # 2. Semantic concept extraction
    # 3. VaultWriter conversion
    # 4. Memory vault storage
    # 5. JSON snapshot validation
    pass


if __name__ == "__main__":
    # Run async test
    asyncio.run(test_vault_serialization_snapshot(Path("./test_output")))
