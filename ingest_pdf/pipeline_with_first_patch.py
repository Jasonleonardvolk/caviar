from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import json
import os
import hashlib
import PyPDF2
import logging
from datetime import datetime
try:
    # Try absolute import first
    from extract_blocks import extract_concept_blocks
except ImportError:
    # Fallback to relative import
    from .extract_blocks import extract_concept_blocks
try:
    # Try absolute import first
    from features import build_feature_matrix
except ImportError:
    # Fallback to relative import
    from .features import build_feature_matrix
try:
    # Try absolute import first
    from spectral import spectral_embed
except ImportError:
    # Fallback to relative import
    from .spectral import spectral_embed
try:
    # Try absolute import first
    from clustering import run_oscillator_clustering, cluster_cohesion
except ImportError:
    # Fallback to relative import
    from .clustering import run_oscillator_clustering, cluster_cohesion
try:
    # Try absolute import first
    from scoring import score_clusters, resonance_score, narrative_centrality, build_cluster_adjacency, filter_concepts
except ImportError:
    # Fallback to relative import
    from .scoring import score_clusters, resonance_score, narrative_centrality, build_cluster_adjacency, filter_concepts
try:
    # Try absolute import first
    from keywords import extract_keywords
except ImportError:
    # Fallback to relative import
    from .keywords import extract_keywords
try:
    # Try absolute import first
    from models import ConceptTuple
except ImportError:
    # Fallback to relative import
    from .models import ConceptTuple
try:
    # Try absolute import first
    from persistence import save_concepts
except ImportError:
    # Fallback to relative import
    from .persistence import save_concepts
try:
    # Try absolute import first
    from lyapunov import concept_predictability, document_chaos_profile
except ImportError:
    # Fallback to relative import
    from .lyapunov import concept_predictability, document_chaos_profile
try:
    # Try absolute import first
    from source_validator import validate_source, SourceValidationResult
except ImportError:
    # Fallback to relative import
    from .source_validator import validate_source, SourceValidationResult
try:
    # Try absolute import first
    from memory_gating import apply_memory_gating
except ImportError:
    # Fallback to relative import
    from .memory_gating import apply_memory_gating
try:
    # Try absolute import first
    from phase_walk import PhaseCoherentWalk
except ImportError:
    # Fallback to relative import
    from .phase_walk import PhaseCoherentWalk
try:
    # Try absolute import first
    from pipeline_validator import validate_concepts
except ImportError:
    # Fallback to relative import
    from .pipeline_validator import validate_concepts
try:
    # Try absolute import first
    from concept_logger import default_concept_logger as concept_logger, log_loop_record, log_concept_summary, warn_empty_segment
except ImportError:
    # Fallback to relative import
    from .concept_logger import default_concept_logger as concept_logger, log_loop_record, log_concept_summary, warn_empty_segment
try:
    # Try absolute import first
    from threshold_config import MIN_CONFIDENCE, FALLBACK_MIN_COUNT
except ImportError:
    # Fallback to relative import
    from .threshold_config import MIN_CONFIDENCE, FALLBACK_MIN_COUNT
try:
    # Try absolute import first
    from cognitive_interface import add_concept_diff
except ImportError:
    # Fallback to relative import
    from .cognitive_interface import add_concept_diff

# Configure logging
logger = logging.getLogger("pdf_ingestion")
logger.setLevel(logging.INFO)

def extract_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    metadata = {
        "filename": Path(pdf_path).name,
        "file_path": pdf_path,
        "extraction_timestamp": datetime.now().isoformat(),
    }
    try:
        with open(pdf_path, "rb") as f:
            metadata["sha256"] = hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        logger.warning(f"Could not calculate file hash: {e}")

    try:
        with open(pdf_path, "rb") as f:
            pdf = PyPDF2.PdfReader(f)
            if pdf.metadata:
                metadata["pdf_metadata"] = {
                    k.lower().replace('/', ''): str(v)
                    for k, v in pdf.metadata.items() if k and v
                }
                for key in ("author", "title", "subject", "producer"):
                    if key in metadata["pdf_metadata"]:
                        metadata[key] = metadata["pdf_metadata"][key]
            metadata["page_count"] = len(pdf.pages)
    except Exception as e:
        logger.warning(f"Could not extract PDF metadata: {e}")
    return metadata

def extract_concepts_from_blocks(blocks: List[str], doc_id: str) -> List[Dict[str, Any]]:
    """
    YOUR FIRST PATCH INTEGRATED HERE:
    Extract concepts from blocks with LoopRecord logging, threshold filtering, and fallback logic
    """
    all_concepts = []
    
    for page_num, page_text in enumerate(blocks):
        # Segment ID for logs - YOUR EXACT PATTERN
        segment_id = f"{doc_id}_page_{page_num + 1}"
        
        # Extract raw concepts from the page - YOUR PLACEHOLDER LOGIC
        raw_concepts = extract_concepts(page_text, page_num)
        
        if not raw_concepts:
            warn_empty_segment(segment_id)
            continue
        
        # Filter + fallback logic - YOUR EXACT PATTERN
        filtered = filter_concepts(raw_concepts, threshold=MIN_CONFIDENCE)
        if len(filtered) < FALLBACK_MIN_COUNT:
            filtered = sorted(raw_concepts, key=lambda c: c['confidence'], reverse=True)[:FALLBACK_MIN_COUNT]
            logger.info(f"[{segment_id}] Confidence threshold too strict; falling back to top {len(filtered)} concepts.")
        
        # Log loop record - YOUR EXACT CALL
        log_loop_record(segment_id, filtered)
        
        # Accumulate - YOUR EXACT PATTERN
        all_concepts.extend(filtered)
    
    if not all_concepts:
        logger.warning(f"No concepts extracted for {doc_id}. PDF ingestion yielded empty result set.")
        return []
    
    # Save to output JSON - YOUR EXACT PATTERN
    os.makedirs("./semantic_outputs/", exist_ok=True)
    output_path = f"./semantic_outputs/{doc_id}_semantic_concepts.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_concepts, f, indent=2)
    
    # Final summary - YOUR EXACT CALL
    log_concept_summary(doc_id, all_concepts, output_path)
    return all_concepts

def extract_concepts(text: str, page_num: int) -> List[Dict[str, Any]]:
    """
    YOUR EXACT PLACEHOLDER CONCEPT EXTRACTION LOGIC
    """
    hash_id = hashlib.md5(text.encode()).hexdigest()[:6]
    return [
        {
            "name": f"Concept-{hash_id}",
            "confidence": 0.73,
            "method": "embedding_cluster",
            "source": {"page": page_num + 1},
            "context": text[:150] + "...",
            "embedding": [0.1, 0.2, 0.3]  # Dummy
        }
    ]

def ingest_pdf_and_update_index(
    pdf_path: str,
    index_path: str,
    max_concepts: int = 12,
    dim: int = 16,
    json_out: str = None,
    min_quality_score: float = 0.6,
    apply_gating: bool = True,
    coherence_threshold: float = 0.7
) -> dict:
    """
    INTEGRATED YOUR FIRST PATCH INTO EXISTING PIPELINE STRUCTURE
    """
    validation = validate_source(pdf_path, min_quality_score)
    if not validation.is_valid:
        logger.warning(f"Source rejected: {pdf_path} (Score: {validation.quality_score:.2f})")
        return {"filename": Path(pdf_path).name, "concept_count": 0, "status": "rejected"}

    logger.info(f"Source validated: {pdf_path} (Score: {validation.quality_score:.2f})")
    
    # Extract blocks - this gives us text segments per page
    blocks = extract_concept_blocks(pdf_path)
    if not blocks:
        return {"filename": Path(pdf_path).name, "concept_count": 0, "status": "empty"}

    doc_metadata = extract_pdf_metadata(pdf_path)
    doc_metadata["source_validation"] = validation.to_dict()

    # 🚀 YOUR FIRST PATCH INTEGRATION - Extract concepts with LoopRecord logging
    doc_id = Path(pdf_path).stem
    extracted_concepts = extract_concepts_from_blocks(blocks, doc_id)
    
    if not extracted_concepts:
        warn_empty_segment(f"{doc_id}_final", "No concepts survived extraction pipeline")
        return {"filename": Path(pdf_path).name, "concept_count": 0, "status": "empty"}

    # Continue with spectral analysis for advanced processing
    feats, vocab = build_feature_matrix(blocks)
    tfidf = feats
    emb = spectral_embed(feats, k=dim)

    eigenvalues = np.ones(dim) * 0.8
    eigenvectors = np.eye(dim)

    labels = run_oscillator_clustering(emb)
    top_cids = score_clusters(labels, emb)[:max_concepts]
    predictability_scores = concept_predictability(labels, emb, list(range(len(blocks))))
    chaos_profile = document_chaos_profile(labels, emb, list(range(len(blocks))))

    # Convert extracted concepts to ConceptTuple format
    tuples: List[ConceptTuple] = []
    adj = build_cluster_adjacency(labels, emb)

    for i, concept_data in enumerate(extracted_concepts[:max_concepts]):  # Respect max_concepts limit
        # Create ConceptTuple with YOUR PATCH METADATA
        concept = ConceptTuple(
            name=concept_data["name"],
            embedding=np.array(concept_data.get("embedding", [0.0] * dim), dtype=np.float32),
            context=concept_data.get("context", ""),
            passage_embedding=np.array([0.0] * dim, dtype=np.float32),  # Placeholder
            cluster_members=[i],
            resonance_score=concept_data.get("confidence", 0.0),
            narrative_centrality=0.5,  # Default
            predictability_score=0.5,  # Default
            
            # YOUR PATCH METADATA INTEGRATION
            confidence=concept_data.get("confidence", 0.0),
            method=concept_data.get("method", "embedding_cluster"),
            source_reference=concept_data.get("source", {}),
            
            eigenfunction_id="",  # Will be auto-generated
            source_provenance=doc_metadata,
            spectral_lineage=[(0.8, 0.5)],  # Placeholder
            cluster_coherence=0.5
        )

        # Validate with pipeline validator
        validate_concepts([concept.to_minimal_dict()], segment_id=f"concept_{i}")
        concept_logger.log_concept_birth(concept, source=pdf_path)

        tuples.append(concept)

    if apply_gating and len(tuples) > 1:
        tuples = apply_memory_gating(tuples)

    concept_graph = PhaseCoherentWalk()
    concept_graph.add_concepts_from_tuples(tuples)
    concept_graph.run_dynamics(steps=30, dt=0.1, noise=0.01)

    save_concepts(tuples, index_path)

    # JSON output with YOUR EXTRACTED CONCEPTS
    if json_out:
        json_data = [t.to_dict() for t in tuples]
        for i, t in enumerate(tuples):
            json_data[i]["embedding"] = t.embedding.tolist()
            if hasattr(t, "passage_embedding") and t.passage_embedding is not None:
                json_data[i]["passage_embedding"] = t.passage_embedding.tolist()
        json_data.append({"type": "document_chaos_profile", "values": chaos_profile})
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

    # 🧠 YOUR ConceptMesh integration
    if tuples:
        add_concept_diff({
            "type": "document",
            "title": Path(pdf_path).stem,
            "concepts": [t.to_dict() for t in tuples],
            "summary": f"Ingested {len(tuples)} concepts from {Path(pdf_path).name}",
            "metadata": {
                "source_file": pdf_path,
                "page_count": doc_metadata.get("page_count", 0),
                "quality_score": validation.quality_score,
                "eigen_ids": [t.eigenfunction_id for t in tuples],
                "gated": apply_gating,
                "timestamp": datetime.now().isoformat()
            }
        })

    return {
        "filename": Path(pdf_path).name,
        "concept_count": len(tuples),
        "concept_names": [t.name for t in tuples],
        "quality_score": validation.quality_score,
        "source_type": validation.source_type,
        "status": "success"
    }
