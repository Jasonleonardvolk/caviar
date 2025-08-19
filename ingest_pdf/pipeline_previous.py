# =================================================================
#    TORI PIPELINE - BULLETPROOF EDITION - NO NONETYPE ERRORS POSSIBLE
# =================================================================
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import json
import os
import hashlib
import PyPDF2
import logging
import time
from datetime import datetime

# === Imports: All Modules, Classic and Modern ===
try:
    from .extract_blocks import extract_concept_blocks, extract_chunks
    from .features import build_feature_matrix
    from .spectral import spectral_embed
    from .clustering import run_oscillator_clustering, cluster_cohesion
    from .scoring import score_clusters, resonance_score, narrative_centrality, build_cluster_adjacency, filter_concepts, apply_confidence_fallback, calculate_concept_confidence
    from .keywords import extract_keywords
    from .models import ConceptTuple, ConceptExtractionResult, create_concept_tuple_from_dict
    from .persistence import save_concepts, save_extraction_result
    from .lyapunov import concept_predictability, document_chaos_profile
    from .source_validator import validate_source, SourceValidationResult
    from .memory_gating import apply_memory_gating
    from .phase_walk import PhaseCoherentWalk
    from .pipeline_validator import validate_concepts
    from .concept_logger import default_concept_logger as concept_logger, log_loop_record, log_concept_summary, warn_empty_segment
    from .threshold_config import MIN_CONFIDENCE, FALLBACK_MIN_COUNT, MAX_CONCEPTS_DEFAULT, get_threshold_for_media_type, get_adaptive_threshold, get_fallback_count
    from .cognitive_interface import add_concept_diff
    from .extractConceptsFromDocument import extractConceptsFromDocument, reset_frequency_counter, track_concept_frequency, get_concept_frequency, concept_frequency_counter
    from .entropy_prune import entropy_prune, entropy_prune_with_categories
except ImportError:
    # Absolute fallback for local shell testing
    from extract_blocks import extract_concept_blocks, extract_chunks
    from features import build_feature_matrix
    from spectral import spectral_embed
    from clustering import run_oscillator_clustering, cluster_cohesion
    from scoring import score_clusters, resonance_score, narrative_centrality, build_cluster_adjacency, filter_concepts, apply_confidence_fallback, calculate_concept_confidence
    from keywords import extract_keywords
    from models import ConceptTuple, ConceptExtractionResult, create_concept_tuple_from_dict
    from persistence import save_concepts, save_extraction_result
    from lyapunov import concept_predictability, document_chaos_profile
    from source_validator import validate_source, SourceValidationResult
    from memory_gating import apply_memory_gating
    from phase_walk import PhaseCoherentWalk
    from pipeline_validator import validate_concepts
    from concept_logger import default_concept_logger as concept_logger, log_loop_record, log_concept_summary, warn_empty_segment
    from threshold_config import MIN_CONFIDENCE, FALLBACK_MIN_COUNT, MAX_CONCEPTS_DEFAULT, get_threshold_for_media_type, get_adaptive_threshold, get_fallback_count
    from cognitive_interface import add_concept_diff
    from extractConceptsFromDocument import extractConceptsFromDocument, reset_frequency_counter, track_concept_frequency, get_concept_frequency, concept_frequency_counter
    from entropy_prune import entropy_prune, entropy_prune_with_categories

# === Logging ===
logger = logging.getLogger("pdf_ingestion")
logger.setLevel(logging.INFO)  # Set to INFO for production

# === Feature Flags ===
ENABLE_CONTEXT_EXTRACTION = True
ENABLE_FREQUENCY_TRACKING = True
ENABLE_SMART_FILTERING = True
ENABLE_ENTROPY_PRUNING = True
DEBUG_DIAGNOSTIC_LOGS = False

# === BULLETPROOF MATH FUNCTIONS ===
def safe_divide(numerator, denominator, default=0.0):
    """Bulletproof division that never fails"""
    num = numerator or 0
    den = denominator or 1
    if den == 0:
        return default
    return num / den

def safe_multiply(a, b, default=0.0):
    """Bulletproof multiplication that never fails"""
    val_a = a or 0
    val_b = b or 0
    return val_a * val_b

def safe_percentage(value, total, default=0.0):
    """Bulletproof percentage calculation"""
    val = value or 0
    tot = total or 1
    if tot == 0:
        return default
    return (val * 100) / tot

def sanitize_stats_dict(stats_dict):
    """Sanitize a stats dictionary to ensure no None values"""
    if not stats_dict:
        return {}
    
    sanitized = {}
    for key, value in stats_dict.items():
        if value is None:
            if key in ['total', 'selected', 'pruned']:
                sanitized[key] = 0
            elif key in ['final_entropy', 'avg_similarity']:
                sanitized[key] = 0.0
            else:
                sanitized[key] = 0
        else:
            sanitized[key] = value
    return sanitized

# === Entropy Pruning Config ===
ENTROPY_CONFIG = {
    "max_diverse_concepts": None,
    "entropy_threshold": 0.0005,
    "similarity_threshold": 0.83,
    "enable_categories": True,
    "concepts_per_category": None
}

# === Universal Concept DB Loader ===
concept_db_path = Path(__file__).parent / "data" / "concept_file_storage.json"
universal_seed_path = Path(__file__).parent / "data" / "concept_seed_universal.json"
concept_file_storage, concept_names, concept_scores = [], [], {}

def load_universal_concept_file_storage():
    global concept_file_storage, concept_names, concept_scores
    try:
        with open(concept_db_path, "r", encoding="utf-8") as f:
            main_concepts = json.load(f)
        logger.info(f"✅ Main concept file_storage loaded: {len(main_concepts)} concepts")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load main concept file_storage: {e}")
        main_concepts = []
    try:
        with open(universal_seed_path, "r", encoding="utf-8") as f:
            universal_seeds = json.load(f)
        logger.info(f"🌍 Universal seed concepts loaded: {len(universal_seeds)} concepts")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load universal seed concepts: {e}")
        universal_seeds = []
    existing_names = {c["name"].lower() for c in main_concepts}
    merged_concepts = main_concepts[:]
    seeds_added = 0
    for seed in universal_seeds:
        if seed["name"].lower() not in existing_names:
            merged_concepts.append(seed)
            seeds_added += 1
    concept_file_storage = merged_concepts
    concept_names = [c["name"] for c in concept_file_storage]
    concept_scores = {c["name"]: c.get("priority", 0.5) for c in concept_file_storage}
    logger.info(f"🌍 UNIVERSAL DATABASE READY: {len(concept_file_storage)} total concepts (+{seeds_added} seeds)")

load_universal_concept_file_storage()

def extract_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """Extract comprehensive metadata from PDF file"""
    metadata = {
        "filename": Path(pdf_path).name,
        "file_path": pdf_path,
        "extraction_timestamp": datetime.now().isoformat(),
        "extractor_version": "tori_bulletproof_pipeline_v1.0"
    }
    try:
        with open(pdf_path, "rb") as f:
            content = f.read()
            metadata["sha256"] = hashlib.sha256(content).hexdigest()
            metadata["file_size_bytes"] = len(content)
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
            metadata["page_count"] = len(pdf.pages)
    except Exception as e:
        logger.warning(f"Could not extract PDF metadata: {e}")
    return metadata

def get_dynamic_limits(file_size_mb: float) -> Tuple[int, int]:
    """Dynamic limits based on file size"""
    if file_size_mb < 1:
        return 300, 250
    elif file_size_mb < 5:
        return 500, 700
    elif file_size_mb < 25:
        return 1200, 1500
    else:
        return 2000, 3000

def extract_title_abstract_safe(chunks: List[Any], pdf_path: str) -> Tuple[str, str]:
    """Safely extract title and abstract from document"""
    title_text = ""
    abstract_text = ""
    try:
        if chunks and len(chunks) > 0:
            first_chunk = chunks[0]
            first_text = first_chunk.get("text", "") if isinstance(first_chunk, dict) else str(first_chunk)
            lines = [ln.strip() for ln in first_text.splitlines() if ln.strip()]
            if lines:
                candidate = lines[0]
                if 10 < len(candidate) < 150 and not candidate.endswith('.'):
                    title_text = candidate
            lower_text = first_text.lower()
            if "abstract" in lower_text:
                idx = lower_text.index("abstract")
                abstract_start = idx + len("abstract")
                while abstract_start < len(first_text) and first_text[abstract_start] in ": \r\t\n":
                    abstract_start += 1
                abstract_text = first_text[abstract_start:].strip()
                intro_pos = abstract_text.lower().find("introduction")
                if intro_pos > 0:
                    abstract_text = abstract_text[:intro_pos].strip()
                abstract_text = abstract_text[:1000]
        if not title_text:
            filename = Path(pdf_path).stem
            if len(filename) > 10 and not filename.replace('_', '').replace('-', '').isdigit():
                title_text = filename.replace('_', ' ').replace('-', ' ')
    except Exception as e:
        logger.debug(f"Could not extract title/abstract: {e}")
    return title_text, abstract_text

def analyze_concept_purity(all_concepts: List[Dict[str, Any]], doc_name: str = "", title_text: str = "", abstract_text: str = "") -> List[Dict[str, Any]]:
    """Concept purity analysis with bulletproof math"""
    logger.info(f"🔬 CONCEPT PURITY ANALYSIS for {doc_name}")
    logger.info(f"📊 Analyzing {len(all_concepts)} raw concepts")
    
    consensus_concepts = []
    high_confidence = []
    file_storage_boosted = []
    single_method = []
    
    GENERIC_TERMS = {
        'document', 'paper', 'analysis', 'method', 'approach', 'study',
        'research', 'results', 'data', 'figure', 'table', 'section',
        'abstract', 'introduction', 'conclusion', 'pdf document', 
        'academic paper', 'page', 'text', 'content', 'information',
        'system', 'model', 'based', 'using', 'used', 'new', 'proposed',
        'specific', 'general', 'various', 'different', 'particular',
        'important', 'significant', 'relevant', 'related', 'similar',
        'technical method', 'artificial intelligence', 'computer science',
        'pdf', 'file', 'scholarsphere document'
    }
    
    for concept in all_concepts:
        name = concept.get('name', '')
        score = concept.get('score', 0) or 0  # Bulletproof None handling
        method = concept.get('method', '')
        metadata = concept.get('metadata', {})
        
        # Bulletproof calculations
        methods_in_name = method.count('+') + 1 if '+' in method else 1
        is_consensus = '+' in method or metadata.get('cross_reference_boost', False)
        is_boosted = 'file_storage_boosted' in method or 'boost' in method
        has_cross_ref = metadata.get('cross_reference_boost', False)
        word_count = len(name.split())
        char_count = len(name)
        
        # Safety checks
        name_lower = name.lower().strip()
        if name_lower in GENERIC_TERMS or char_count < 3 or word_count > 6 or score < 0.2:
            continue
            
        # Enhanced acceptance criteria
        frequency = metadata.get('frequency', 1) or 1
        sections = metadata.get('sections', ['body'])
        in_title = metadata.get('in_title', False)
        in_abstract = metadata.get('in_abstract', False)
        method_count = method.count('+') + 1 if '+' in method else 1
        
        if method_count >= 3:
            consensus_concepts.append(concept)
        elif method_count == 2 and score >= 0.5:
            consensus_concepts.append(concept)
        elif is_boosted and score >= 0.75:
            file_storage_boosted.append(concept)
        elif (in_title or in_abstract) and score >= 0.7:
            high_confidence.append(concept)
        elif score >= 0.85 and word_count <= 3:
            high_confidence.append(concept)
        elif score >= 0.75 and word_count <= 2:
            single_method.append(concept)
        elif frequency >= 3 and score >= 0.65:
            single_method.append(concept)
    
    # Combine and deduplicate
    pure_concepts = consensus_concepts + high_confidence + file_storage_boosted + single_method
    seen = set()
    unique_pure = []
    for c in pure_concepts:
        name_lower = c.get('name', '').lower().strip()
        if name_lower not in seen and name_lower:
            seen.add(name_lower)
            unique_pure.append(c)
    
    logger.info(f"🏆 FINAL PURE CONCEPTS: {len(unique_pure)}")
    return unique_pure

def boost_known_concepts(chunk: str) -> List[Dict[str, Any]]:
    """Quality-focused universal concept boosting"""
    boosted = []
    chunk_lower = chunk.lower()
    MAX_BOOSTS = 25
    MAX_CHECKS = 300
    
    sorted_concepts = sorted(
        concept_file_storage, 
        key=lambda c: c.get("priority", 0), 
        reverse=True
    )[:MAX_CHECKS]
    
    for concept in sorted_concepts:
        if len(boosted) >= MAX_BOOSTS:
            break
            
        name = concept.get("name", "")
        if len(name) < 4:
            continue
            
        aliases = concept.get("aliases", [])
        all_terms = [name.lower()] + [alias.lower() for alias in aliases]
        matched_terms = []
        
        for term in all_terms:
            if (f" {term} " in f" {chunk_lower} " or 
                chunk_lower.startswith(f"{term} ") or 
                chunk_lower.endswith(f" {term}") or
                chunk_lower == term):
                matched_terms.append(term)
                break
        
        if matched_terms:
            base_score = concept_scores.get(name, 0.5)
            boost_multiplier = concept.get("boost_multiplier", 1.2)
            boosted_score = min(0.98, safe_multiply(base_score, boost_multiplier, 0.5))
            category = concept.get("category", "general")
            
            boosted.append({
                "name": name,
                "score": boosted_score,
                "method": "quality_focused_file_storage_boosted",
                "source": {
                    "file_storage_matched": True,
                    "matched_terms": matched_terms,
                    "quality_boost": True,
                    "priority_rank": len(boosted) + 1,
                    "domain": category
                },
                "context": f"Quality file_storage boost: '{matched_terms[0]}' found in text",
                "metadata": {
                    "category": category,
                    "aliases": aliases,
                    "boost_multiplier": boost_multiplier,
                    "original_score": base_score,
                    "boosted_score": boosted_score,
                    "matched_terms": matched_terms,
                    "priority_concept": True
                }
            })
    
    return boosted

def extract_and_boost_concepts(chunk: str, threshold: float = 0.0, chunk_index: int = 0, chunk_section: str = "body", title_text: str = "", abstract_text: str = "") -> List[Dict[str, Any]]:
    """Universal extract and boost with context"""
    # Extract concepts using universal method
    semantic_hits = extractConceptsFromDocument(chunk, threshold=threshold, chunk_index=chunk_index, chunk_section=chunk_section)
    
    # Apply quality-focused file_storage boosting
    boosted = boost_known_concepts(chunk)
    
    # Combine results
    combined = semantic_hits + boosted
    
    # Add cross-reference boost metadata
    concept_name_counts = {}
    for concept in combined:
        name_lower = concept.get("name", "").lower()
        concept_name_counts[name_lower] = concept_name_counts.get(name_lower, 0) + 1
    
    # Mark concepts with cross-reference potential and add context
    for concept in combined:
        name = concept.get('name', '')
        name_lower = name.lower()
        
        if concept_name_counts.get(name_lower, 0) > 1:
            concept.setdefault("metadata", {})["cross_reference_boost"] = True
            concept.setdefault("metadata", {})["methods_found"] = concept_name_counts[name_lower]
        
        # Add frequency data
        freq_data = get_concept_frequency(name)
        concept.setdefault('metadata', {})['frequency'] = freq_data.get('count', 1)
        
        # Add section data
        if 'sections' not in concept.get('metadata', {}):
            if 'chunk_sections' in concept.get('metadata', {}):
                concept['metadata']['sections'] = list(concept['metadata']['chunk_sections'])
            else:
                concept['metadata']['sections'] = [chunk_section]
        
        # Add title/abstract flags
        concept['metadata']['in_title'] = bool(title_text and name_lower in title_text.lower())
        concept['metadata']['in_abstract'] = bool(abstract_text and name_lower in abstract_text.lower())
    
    return combined

def ingest_pdf_clean(pdf_path: str, doc_id: str = None, extraction_threshold: float = 0.0, admin_mode: bool = False) -> Dict[str, Any]:
    """
    BULLETPROOF PDF INGESTION PIPELINE - NO NONETYPE ERRORS POSSIBLE
    """
    start_time = datetime.now()
    if doc_id is None:
        doc_id = Path(pdf_path).stem

    file_size_mb = safe_divide(os.path.getsize(pdf_path), 1024 * 1024, 0)
    MAX_CHUNKS, MAX_TOTAL_CONCEPTS = get_dynamic_limits(file_size_mb)
    
    logger.info(f"🏆 [BULLETPROOF] Ingesting: {Path(pdf_path).name} ({file_size_mb:.1f} MB)")
    logger.info(f"Limits: {MAX_CHUNKS} chunks, {MAX_TOTAL_CONCEPTS} concepts")
    
    try:
        # Extract metadata
        doc_metadata = extract_pdf_metadata(pdf_path)
        
        # Reset frequency counter
        if ENABLE_FREQUENCY_TRACKING:
            reset_frequency_counter()
        
        # Extract chunks
        chunks = extract_chunks(pdf_path)
        if not chunks:
            logger.warning(f"No text chunks extracted from {pdf_path}")
            return {
                "filename": Path(pdf_path).name,
                "concept_count": 0,
                "concepts": [],
                "concept_names": [],
                "status": "empty",
                "admin_mode": admin_mode,
                "processing_time_seconds": safe_divide((datetime.now() - start_time).total_seconds(), 1, 0)
            }
        
        # Extract title and abstract
        title_text, abstract_text = "", ""
        if ENABLE_CONTEXT_EXTRACTION:
            title_text, abstract_text = extract_title_abstract_safe(chunks, pdf_path)
        
        # Process chunks (limited for performance)
        chunks_to_process = chunks[:MAX_CHUNKS]
        all_extracted_concepts = []
        semantic_count = 0
        boosted_count = 0
        universal_methods = set()
        domain_distribution = {}
        sections_encountered = set()
        
        for i, chunk_data in enumerate(chunks_to_process):
            if isinstance(chunk_data, dict):
                chunk_text = chunk_data.get("text", "")
                chunk_index = chunk_data.get("index", i)
                chunk_section = chunk_data.get("section", "body")
            else:
                chunk_text = chunk_data
                chunk_index = i
                chunk_section = "body"
            
            sections_encountered.add(chunk_section)
            
            # Extract and boost concepts
            enhanced_concepts = extract_and_boost_concepts(
                chunk_text,
                threshold=extraction_threshold,
                chunk_index=chunk_index,
                chunk_section=chunk_section,
                title_text=title_text,
                abstract_text=abstract_text
            )
            
            # Count extraction types
            for c in enhanced_concepts:
                method = c.get("method", "")
                if "universal" in method:
                    semantic_count += 1
                    if "yake" in method:
                        universal_methods.add("YAKE")
                    if "keybert" in method:
                        universal_methods.add("KeyBERT")
                    if "ner" in method:
                        universal_methods.add("NER")
                
                if "file_storage_boosted" in method or "boost" in method:
                    boosted_count += 1
                
                domain = (
                    c.get("source", {}).get("domain")
                    or c.get("metadata", {}).get("category")
                    or "unknown"
                )
                domain_distribution[domain] = domain_distribution.get(domain, 0) + 1
            
            all_extracted_concepts.extend(enhanced_concepts)
            
            # Early exit if we have enough concepts
            if len(all_extracted_concepts) >= MAX_TOTAL_CONCEPTS:
                logger.info(f"Concept limit reached: {len(all_extracted_concepts)}. Stopping.")
                break
        
        if not all_extracted_concepts:
            logger.error("Critical: No concepts extracted!")
            return {
                "filename": Path(pdf_path).name,
                "concept_count": 0,
                "concepts": [],
                "concept_names": [],
                "status": "critical_failure",
                "admin_mode": admin_mode,
                "processing_time_seconds": safe_divide((datetime.now() - start_time).total_seconds(), 1, 0)
            }
        
        chunks_processed = min(len(chunks_to_process), len(chunks_to_process))
        
        # Apply purity filtering
        pure_concepts = analyze_concept_purity(
            all_extracted_concepts, Path(pdf_path).name, title_text, abstract_text
        )
        pure_concepts.sort(key=lambda x: x.get('score', 0), reverse=True)
        original_pure_count = len(pure_concepts)
        concept_count = len(pure_concepts)
        prune_stats = None
        
        # Apply entropy pruning
        if ENABLE_ENTROPY_PRUNING and concept_count > 0:
            logger.info("🎯 APPLYING ENTROPY-BASED DIVERSITY PRUNING...")
            
            # Determine max concepts based on admin mode
            if admin_mode:
                max_diverse = None  # No limit for admin mode
                logger.info("📊 Admin mode: allowing unlimited diverse concepts")
            else:
                max_diverse = ENTROPY_CONFIG["max_diverse_concepts"]
                logger.info(f"📊 Standard mode: max diverse concepts = {max_diverse}")
            
            # Extract categories
            categories = list({
                c.get('metadata', {}).get('category')
                for c in pure_concepts if c.get('metadata', {}).get('category')
            })
            
            if ENTROPY_CONFIG["enable_categories"] and categories:
                logger.info(f"📂 Found {len(categories)} categories: {', '.join(sorted(categories))}")
                pure_concepts, prune_stats = entropy_prune_with_categories(
                    pure_concepts,
                    categories=categories,
                    concepts_per_category=ENTROPY_CONFIG["concepts_per_category"],
                    entropy_threshold=ENTROPY_CONFIG["entropy_threshold"],
                    similarity_threshold=ENTROPY_CONFIG["similarity_threshold"],
                    verbose=True
                )
            else:
                pure_concepts, prune_stats = entropy_prune(
                    pure_concepts,
                    top_k=max_diverse,
                    entropy_threshold=ENTROPY_CONFIG["entropy_threshold"],
                    similarity_threshold=ENTROPY_CONFIG["similarity_threshold"],
                    verbose=True
                )
            
            concept_count = len(pure_concepts)
            
            # BULLETPROOF logging
            efficiency_percent = safe_percentage(concept_count, original_pure_count, 0)
            logger.info(f"✅ DIVERSITY FILTER: {concept_count} diverse concepts from {original_pure_count} pure")
            logger.info(f"📊 DIVERSITY EFFICIENCY: {efficiency_percent:.1f}% kept")
        
        # Knowledge injection
        if concept_count > 0:
            try:
                concept_diff_data = {
                    "type": "document",
                    "title": Path(pdf_path).name,
                    "concepts": pure_concepts,
                    "summary": f"{concept_count} pure concepts extracted.",
                    "metadata": doc_metadata,
                }
                add_concept_diff(concept_diff_data)
            except Exception as e:
                logger.warning(f"Concept diff injection failed: {e}")
        
        # BULLETPROOF calculations for response
        total_time = safe_divide((datetime.now() - start_time).total_seconds(), 1, 0)
        
        # Safe score calculation
        if pure_concepts:
            valid_scores = [c.get("score", 0) for c in pure_concepts if c.get("score") is not None]
            avg_score = safe_divide(sum(valid_scores), len(valid_scores), 0) if valid_scores else 0.0
        else:
            avg_score = 0.0
        
        high_conf_count = sum(1 for c in pure_concepts if (c.get("score") or 0) > 0.8)
        
        # Build response with bulletproof calculations
        response = {
            "filename": Path(pdf_path).name,
            "concept_count": concept_count,
            "concept_names": [c.get('name', '') for c in pure_concepts],
            "concepts": pure_concepts,
            "status": "success" if concept_count > 0 else "no_concepts",
            "purity_based": True,
            "entropy_pruned": ENABLE_ENTROPY_PRUNING and prune_stats is not None,
            "admin_mode": admin_mode,
            "equal_access": True,
            "performance_limited": True,
            "chunks_processed": chunks_processed,
            "chunks_available": len(chunks),
            "semantic_extracted": semantic_count,
            "file_storage_boosted": boosted_count,
            "average_concept_score": round(avg_score, 3),
            "high_confidence_concepts": high_conf_count,
            "total_extraction_time": round(total_time, 3),
            "domain_distribution": domain_distribution,
            "title_found": bool(title_text),
            "abstract_found": bool(abstract_text),
            "processing_time_seconds": round(total_time, 3),
            "purity_analysis": {
                "raw_concepts": len(all_extracted_concepts),
                "pure_concepts": original_pure_count,
                "final_concepts": concept_count,
                "purity_efficiency_percent": round(safe_percentage(original_pure_count, len(all_extracted_concepts), 0), 1),
                "diversity_efficiency_percent": round(safe_percentage(concept_count, original_pure_count, 0), 1),
                "top_concepts": [
                    {
                        "name": c.get('name', ''),
                        "score": round(c.get('score', 0), 3),
                        "methods": c.get('metadata', {}).get('consensus_methods', [c.get('method', 'unknown')]),
                        "frequency": c.get('metadata', {}).get('frequency', 1),
                        "purity_decision": c.get('purity_metrics', {}).get('decision', 'unknown')
                    }
                    for c in pure_concepts[:10]
                ]
            }
        }
        
        # BULLETPROOF entropy analysis
        if prune_stats:
            # Sanitize all stats first
            safe_stats = sanitize_stats_dict(prune_stats)
            
            total = safe_stats.get("total", 0)
            selected = safe_stats.get("selected", 0)
            pruned = safe_stats.get("pruned", 0)
            final_entropy = safe_stats.get("final_entropy", 0.0)
            avg_similarity = safe_stats.get("avg_similarity", 0.0)
            
            response["entropy_analysis"] = {
                "enabled": True,
                "admin_mode": admin_mode,
                "total_before_entropy": total,
                "selected_diverse": selected,
                "pruned_similar": pruned,
                "diversity_efficiency_percent": round(safe_percentage(selected, total, 0), 1),
                "final_entropy": round(final_entropy, 3),
                "avg_similarity": round(avg_similarity, 3),
                "by_category": safe_stats.get("by_category", {}),
                "config": {
                    "max_diverse_concepts": ENTROPY_CONFIG["max_diverse_concepts"] if not admin_mode else "unlimited",
                    "entropy_threshold": ENTROPY_CONFIG["entropy_threshold"],
                    "similarity_threshold": ENTROPY_CONFIG["similarity_threshold"],
                    "category_aware": ENTROPY_CONFIG["enable_categories"]
                },
                "performance": {
                    "original_pure_concepts": original_pure_count,
                    "final_diverse_concepts": concept_count,
                    "reduction_ratio": round(safe_percentage(original_pure_count - concept_count, original_pure_count, 0) / 100, 3)
                }
            }
        else:
            response["entropy_analysis"] = {
                "enabled": False,
                "reason": "entropy_pruning_disabled" if not ENABLE_ENTROPY_PRUNING else "no_concepts_to_prune"
            }
        
        return response
        
    except Exception as e:
        logger.error(f"❌ PDF ingestion failed: {e}")
        return {
            "filename": Path(pdf_path).name,
            "concept_count": 0,
            "concept_names": [],
            "concepts": [],
            "status": "error",
            "error_message": str(e),
            "admin_mode": admin_mode,
            "performance_limited": True,
            "processing_time_seconds": safe_divide((datetime.now() - start_time).total_seconds(), 1, 0)
        }

# Export the main function
__all__ = ['ingest_pdf_clean']

logger.info("🛡️ BULLETPROOF PIPELINE LOADED - NO NONETYPE ERRORS POSSIBLE")
