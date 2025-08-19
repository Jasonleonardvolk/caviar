import numpy as np
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

try:
    from alan_backend.concept_index.concept_store import ConceptStore, Concept
    CONCEPT_STORE_AVAILABLE = True
except ImportError:
    CONCEPT_STORE_AVAILABLE = False

try:
    # Try absolute import first
    from models import ConceptTuple, ConceptExtractionResult
except ImportError:
    # Fallback to relative import
    try:
        # Try absolute import first
        from models import ConceptTuple, ConceptExtractionResult
    except ImportError:
        # Fallback to relative import
        from .models import ConceptTuple, ConceptExtractionResult

def save_concepts(
    tuples: List[ConceptTuple], 
    index_path: str, 
    save_full_metadata: bool = True,
    create_backup: bool = True
) -> bool:
    """
    Save concepts to index with enhanced metadata preservation (addresses Issue #4).
    
    Args:
        tuples: List of ConceptTuple objects to save
        index_path: Path to the concept index
        save_full_metadata: Whether to save complete metadata
        create_backup: Whether to create backup of existing index
        
    Returns:
        True if save was successful, False otherwise
    """
    try:
        # Create backup if requested and index exists
        if create_backup and Path(index_path).exists():
            backup_path = f"{index_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            import shutil
            shutil.copy2(index_path, backup_path)
            print(f"Created backup: {backup_path}")
        
        # Save to ConceptStore if available (basic compatibility)
        if CONCEPT_STORE_AVAILABLE:
            store = ConceptStore(Path(index_path))
            
            # Convert ConceptTuple to tuple for ConceptStore (basic fields only)
            raw_tuples = [(c.name, c.embedding, c.context, c.passage_embedding) for c in tuples]
            store.add_batch(raw_tuples)
            print(f"Saved {len(raw_tuples)} concepts to ConceptStore")
        
        # Save enhanced metadata as separate JSON files
        if save_full_metadata:
            # Full metadata file
            meta_path = str(Path(index_path).with_suffix(".meta.json"))
            save_full_metadata_file(tuples, meta_path)
            
            # Minimal metadata file for quick access
            minimal_path = str(Path(index_path).with_suffix(".minimal.json"))
            save_minimal_metadata_file(tuples, minimal_path)
            
            # Schema-compliant concepts file (for diagnostic tools)
            concepts_path = str(Path(index_path).with_suffix(".concepts.json"))
            save_schema_compliant_file(tuples, concepts_path)
        
        return True
        
    except Exception as e:
        print(f"Error saving concepts: {e}")
        return False

def save_full_metadata_file(tuples: List[ConceptTuple], meta_path: str) -> bool:
    """Save complete concept metadata to JSON file."""
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            metadata = []
            for c in tuples:
                concept_data = {
                    # Core identification
                    "name": c.name,
                    "eigenfunction_id": c.eigenfunction_id,
                    "extraction_timestamp": c.extraction_timestamp,
                    
                    # Enhanced metadata (addresses Issue #4)
                    "confidence": float(c.confidence),
                    "method": c.method,
                    "source_reference": c.source_reference,
                    
                    # Quality metrics
                    "resonance_score": float(c.resonance_score),
                    "narrative_centrality": float(c.narrative_centrality),
                    "predictability_score": float(c.predictability_score),
                    "cluster_coherence": float(c.cluster_coherence),
                    "overall_quality_score": c.get_quality_score(),
                    
                    # Provenance and lineage
                    "source_provenance": c.source_provenance,
                    "spectral_lineage": [(float(real), float(mag)) for real, mag in c.spectral_lineage],
                    
                    # Processing information
                    "processing_metadata": c.processing_metadata,
                    "quality_metrics": c.quality_metrics,
                    
                    # Cluster information
                    "cluster_size": len(c.cluster_members),
                    "cluster_members": c.cluster_members,
                    
                    # Context and embeddings info (without actual embeddings for file size)
                    "context": c.context,
                    "context_length": len(c.context),
                    "embedding_dim": c.embedding.shape[0] if c.embedding is not None else 0,
                    "passage_embedding_dim": c.passage_embedding.shape[0] if c.passage_embedding is not None else 0
                }
                metadata.append(concept_data)
            
            json.dump({
                "metadata_version": "1.0",
                "saved_timestamp": datetime.now().isoformat(),
                "concept_count": len(metadata),
                "concepts": metadata
            }, f, indent=2)
        
        print(f"Saved full metadata for {len(tuples)} concepts to {meta_path}")
        return True
        
    except Exception as e:
        print(f"Error saving full metadata: {e}")
        return False

def save_minimal_metadata_file(tuples: List[ConceptTuple], minimal_path: str) -> bool:
    """Save minimal concept metadata for quick access."""
    try:
        with open(minimal_path, "w", encoding="utf-8") as f:
            minimal_data = []
            for c in tuples:
                minimal_concept = {
                    "name": c.name,
                    "confidence": float(c.confidence),
                    "method": c.method,
                    "source": c.source_reference,
                    "eigenfunction_id": c.eigenfunction_id,
                    "quality_score": c.get_quality_score()
                }
                minimal_data.append(minimal_concept)
            
            json.dump({
                "format": "minimal_concepts",
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "concepts": minimal_data
            }, f, indent=2)
        
        print(f"Saved minimal metadata for {len(tuples)} concepts to {minimal_path}")
        return True
        
    except Exception as e:
        print(f"Error saving minimal metadata: {e}")
        return False

def save_schema_compliant_file(tuples: List[ConceptTuple], concepts_path: str) -> bool:
    """Save concepts in schema-compliant format for diagnostic tools (addresses Issue #5)."""
    try:
        with open(concepts_path, "w", encoding="utf-8") as f:
            # Create schema-compliant concept list for diagnostic tools
            concepts_data = []
            for c in tuples:
                concept = {
                    # Required fields for validation
                    "name": c.name,
                    "confidence": float(c.confidence),
                    "method": c.method,
                    "source": c.source_reference,
                    
                    # Context for traceability
                    "context": c.context[:150] + "..." if len(c.context) > 150 else c.context,
                    
                    # Additional metadata
                    "eigenfunction_id": c.eigenfunction_id,
                    "extraction_timestamp": c.extraction_timestamp,
                    "quality_score": c.get_quality_score(),
                    
                    # Embeddings as list for JSON compatibility
                    "embedding": c.embedding.tolist() if c.embedding is not None else [],
                    
                    # Quality metrics
                    "resonance_score": float(c.resonance_score),
                    "narrative_centrality": float(c.narrative_centrality),
                    "cluster_coherence": float(c.cluster_coherence)
                }
                concepts_data.append(concept)
            
            json.dump(concepts_data, f, indent=2)
        
        print(f"Saved schema-compliant concepts file: {concepts_path}")
        return True
        
    except Exception as e:
        print(f"Error saving schema-compliant file: {e}")
        return False

def save_extraction_result(
    result: ConceptExtractionResult, 
    output_dir: str, 
    filename_prefix: str = None
) -> Dict[str, str]:
    """
    Save complete extraction result with all metadata.
    
    Args:
        result: ConceptExtractionResult to save
        output_dir: Directory to save files
        filename_prefix: Prefix for output files
        
    Returns:
        Dictionary mapping file type to saved path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not filename_prefix:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_prefix = f"extraction_{timestamp}"
    
    saved_files = {}
    
    try:
        # Save complete extraction result
        full_result_path = output_path / f"{filename_prefix}_complete.json"
        with open(full_result_path, "w", encoding="utf-8") as f:
            # Create serializable version
            result_dict = result.to_dict()
            
            # Convert numpy arrays to lists for JSON serialization
            for concept in result_dict["concepts"]:
                if "embedding" in concept and hasattr(concept["embedding"], "tolist"):
                    concept["embedding"] = concept["embedding"].tolist()
                if "passage_embedding" in concept and hasattr(concept["passage_embedding"], "tolist"):
                    concept["passage_embedding"] = concept["passage_embedding"].tolist()
            
            json.dump(result_dict, f, indent=2)
        
        saved_files["complete"] = str(full_result_path)
        
        # Save minimal version
        minimal_result_path = output_path / f"{filename_prefix}_minimal.json"
        with open(minimal_result_path, "w", encoding="utf-8") as f:
            json.dump(result.to_minimal_dict(), f, indent=2)
        
        saved_files["minimal"] = str(minimal_result_path)
        
        # Save diagnostic-friendly format
        diagnostic_path = output_path / f"{filename_prefix}_semantic_concepts.json"
        concepts_for_diagnostics = []
        for concept in result.concepts:
            concepts_for_diagnostics.append({
                "name": concept.name,
                "confidence": concept.confidence,
                "method": concept.method,
                "source": concept.source_reference,
                "context": concept.context[:100] + "..." if len(concept.context) > 100 else concept.context,
                "eigenfunction_id": concept.eigenfunction_id
            })
        
        with open(diagnostic_path, "w", encoding="utf-8") as f:
            json.dump(concepts_for_diagnostics, f, indent=2)
        
        saved_files["diagnostic"] = str(diagnostic_path)
        
        print(f"Saved extraction result to {len(saved_files)} files in {output_dir}")
        return saved_files
        
    except Exception as e:
        print(f"Error saving extraction result: {e}")
        return {}

def load_concepts_from_metadata(meta_path: str) -> List[ConceptTuple]:
    """Load concepts from enhanced metadata file."""
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        concepts = []
        concept_data_list = data.get("concepts", [])
        
        for concept_data in concept_data_list:
            # Create minimal embedding arrays (actual embeddings not stored in metadata)
            embedding_dim = concept_data.get("embedding_dim", 0)
            passage_dim = concept_data.get("passage_embedding_dim", 0)
            
            dummy_embedding = np.zeros(embedding_dim, dtype=np.float32) if embedding_dim > 0 else np.array([])
            dummy_passage = np.zeros(passage_dim, dtype=np.float32) if passage_dim > 0 else np.array([])
            
            concept = ConceptTuple(
                name=concept_data.get("name", ""),
                embedding=dummy_embedding,
                context=concept_data.get("context", ""),
                passage_embedding=dummy_passage,
                cluster_members=concept_data.get("cluster_members", []),
                resonance_score=concept_data.get("resonance_score", 0.0),
                narrative_centrality=concept_data.get("narrative_centrality", 0.0),
                predictability_score=concept_data.get("predictability_score", 0.5),
                confidence=concept_data.get("confidence", 0.0),
                method=concept_data.get("method", "unknown"),
                source_reference=concept_data.get("source_reference", {}),
                eigenfunction_id=concept_data.get("eigenfunction_id", ""),
                source_provenance=concept_data.get("source_provenance", {}),
                spectral_lineage=concept_data.get("spectral_lineage", []),
                cluster_coherence=concept_data.get("cluster_coherence", 0.0),
                extraction_timestamp=concept_data.get("extraction_timestamp"),
                processing_metadata=concept_data.get("processing_metadata", {}),
                quality_metrics=concept_data.get("quality_metrics", {})
            )
            
            concepts.append(concept)
        
        print(f"Loaded {len(concepts)} concepts from {meta_path}")
        return concepts
        
    except Exception as e:
        print(f"Error loading concepts from metadata: {e}")
        return []

def create_concept_index_summary(index_path: str) -> Dict[str, Any]:
    """Create summary of concept index for monitoring."""
    summary = {
        "index_path": index_path,
        "timestamp": datetime.now().isoformat(),
        "files_found": [],
        "total_concepts": 0,
        "metadata_available": False,
        "schema_compliant": False
    }
    
    base_path = Path(index_path)
    
    # Check for different file types
    file_types = {
        "meta": base_path.with_suffix(".meta.json"),
        "minimal": base_path.with_suffix(".minimal.json"),
        "concepts": base_path.with_suffix(".concepts.json"),
        "store": base_path if base_path.suffix == "" else None
    }
    
    for file_type, file_path in file_types.items():
        if file_path and file_path.exists():
            summary["files_found"].append(file_type)
            
            if file_type == "meta":
                summary["metadata_available"] = True
                try:
                    with open(file_path, "r") as f:
                        meta_data = json.load(f)
                    summary["total_concepts"] = meta_data.get("concept_count", 0)
                except:
                    pass
            elif file_type == "concepts":
                summary["schema_compliant"] = True
    
    return summary

# Backward compatibility functions
def save_concepts_legacy(tuples: List[ConceptTuple], index_path: str):
    """Legacy save function for backward compatibility."""
    return save_concepts(tuples, index_path, save_full_metadata=True, create_backup=False)
