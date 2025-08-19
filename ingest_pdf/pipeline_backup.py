# =================================================================
#    TORI PIPELINE - ATOMIC, NO-COMPROMISE, FULL-DIAGNOSTIC EDITION
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
logger.setLevel(logging.DEBUG)  # FULL DIAGNOSTIC (set to INFO for prod)

# === Feature Flags ===
ENABLE_CONTEXT_EXTRACTION = True
ENABLE_FREQUENCY_TRACKING = True
ENABLE_SMART_FILTERING = True
ENABLE_ENTROPY_PRUNING = True
DEBUG_DIAGNOSTIC_LOGS = True
LEGACY_EXTRACTION_MODE = False  # For old code paths

# === Entropy Pruning Config (Ultra-Verbose) ===
ENTROPY_CONFIG = {
    "max_diverse_concepts": None,    # None = "give me everything, no artificial cap"
    "entropy_threshold": 0.0005,     # Even lower: retain more
    "similarity_threshold": 0.83,    # Stricter (allow more diversity)
    "enable_categories": True,
    "concepts_per_category": None
}

# === Universal Concept DB Loader - Ultra-Verbose ===
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
    if DEBUG_DIAGNOSTIC_LOGS:
        domain_counter = {}
        for c in concept_file_storage:
            domain = c.get("category", "general")
            domain_counter[domain] = domain_counter.get(domain, 0) + 1
        logger.debug(f"Domain stats: {json.dumps(domain_counter, indent=2)}")

load_universal_concept_file_storage()

# === Utility: Full Metadata Extraction with Deep Logging ===
def extract_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    metadata = {
        "filename": Path(pdf_path).name,
        "file_path": pdf_path,
        "extraction_timestamp": datetime.now().isoformat(),
        "extractor_version": "tori_atomic_pipeline_v9001"
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
    if DEBUG_DIAGNOSTIC_LOGS:
        logger.debug(f"PDF metadata for {pdf_path}:\n{json.dumps(metadata, indent=2)}")
    return metadata

def get_dynamic_limits(file_size_mb: float) -> Tuple[int, int]:
    # Ultra-dynamic scaling
    if file_size_mb < 1:
        return 300, 250
    elif file_size_mb < 5:
        return 500, 700
    elif file_size_mb < 25:
        return 1200, 1500
    else:
        return 2000, 3000
# === Deep Helper: Title and Abstract Extraction ===
def extract_title_abstract_safe(chunks: List[Any], pdf_path: str) -> Tuple[str, str]:
    title_text = ""
    abstract_text = ""
    try:
        if chunks and len(chunks) > 0:
            first_chunk = chunks[0]
            first_text = first_chunk.get("text", "") if isinstance(first_chunk, dict) else str(first_chunk)
            lines = [ln.strip() for ln in first_text.splitlines() if ln.strip()]
            if lines:
                candidate = lines[0]
                # Typical title heuristics
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

# === Deep Helper: "Rogue" Concept Detection (with Context) ===
def is_rogue_concept_contextual(name: str, concept: dict) -> tuple[bool, str]:
    name_lower = name.lower()
    frequency = concept.get('metadata', {}).get('frequency', 1)
    sections = concept.get('metadata', {}).get('sections', ['body'])
    score = concept.get('score', 0)
    PERIPHERAL_PATTERNS = {
        'puzzle', 'example', 'case study', 'illustration', 'exercise', 
        'problem set', 'quiz', 'test case', 'figure', 'table', 
        'equation', 'listing', 'algorithm', 'lemma', 'theorem'
    }
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
    for pattern in PERIPHERAL_PATTERNS:
        if pattern in name_lower:
            if frequency >= 3:
                return False, ""
            if any(sec in ['abstract', 'introduction', 'conclusion'] for sec in sections):
                return False, ""
            if concept.get('source', {}).get('file_storage_matched'):
                return False, ""
            return True, "peripheral_pattern"
    if sections == ['references'] and frequency <= 2:
        return True, "references_only"
    if frequency == 1 and not any(sec in ['abstract', 'introduction'] for sec in sections):
        if score < 0.7:
            return True, "single_peripheral_mention"
    if name_lower in GENERIC_TERMS and frequency < 3:
        return True, "generic_low_frequency"
    return False, ""

# === Deep Helper: Concept Purity Analysis (Full, Deterministic) ===
def analyze_concept_purity(all_concepts: List[Dict[str, Any]], doc_name: str = "", title_text: str = "", abstract_text: str = "") -> List[Dict[str, Any]]:
    logger.info(f"🔬 CONCEPT PURITY ANALYSIS for {doc_name}")
    logger.info(f"📊 Analyzing {len(all_concepts)} raw concepts")
    consensus_concepts = []
    high_confidence = []
    file_storage_boosted = []
    single_method = []
    rejected_concepts = []
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
        name = concept['name']
        score = concept.get('score', 0)
        method = concept.get('method', '')
        source = concept.get('source', {})
        metadata = concept.get('metadata', {})
        methods_in_name = method.count('+') + 1 if '+' in method else 1
        is_consensus = '+' in method or metadata.get('cross_reference_boost', False)
        is_boosted = 'file_storage_boosted' in method or 'boost' in method
        has_cross_ref = metadata.get('cross_reference_boost', False)
        word_count = len(name.split())
        char_count = len(name)
        consensus_indicators = []
        if '+' in method:
            consensus_indicators.append('multi_method')
        if has_cross_ref:
            consensus_indicators.append('cross_reference')
        if is_boosted and score > 0.7:
            consensus_indicators.append('file_storage_validated')
        purity_score = score
        purity_reasons = []
        if len(consensus_indicators) >= 2:
            purity_score *= 1.5
            purity_reasons.append(f"consensus({len(consensus_indicators)})")
            is_consensus = True
        if has_cross_ref:
            purity_score *= 1.3
            purity_reasons.append("cross-ref")
        if is_boosted and score > 0.8:
            purity_score *= 1.1
            purity_reasons.append("db-boost")
        if methods_in_name >= 2:
            purity_score *= 1.2
            purity_reasons.append(f"multi-method({methods_in_name})")
        decision = "REJECTED"
        rejection_reason = ""
        name_lower = name.lower().strip()
        if name_lower in GENERIC_TERMS:
            rejection_reason = "generic_term"
        elif char_count < 3:
            rejection_reason = "too_short"
        elif word_count > 6:
            rejection_reason = "too_specific"
        elif score < 0.2:
            rejection_reason = "very_low_score"
        elif any(bad in name_lower for bad in ['document', 'paper', 'file', 'pdf', 'text']):
            rejection_reason = "document_metadata"
        elif len(name_lower.replace(' ', '').replace('-', '')) < 4:
            rejection_reason = "insufficient_content"
        else:
            if ENABLE_SMART_FILTERING:
                is_rogue, rogue_reason = is_rogue_concept_contextual(name, concept)
                if is_rogue:
                    rejection_reason = rogue_reason
                    decision = "REJECTED"
            if rejection_reason == "":
                frequency = concept.get('metadata', {}).get('frequency', 1)
                sections = concept.get('metadata', {}).get('sections', ['body'])
                in_title = concept.get('metadata', {}).get('in_title', False)
                in_abstract = concept.get('metadata', {}).get('in_abstract', False)
                method_count = method.count('+') + 1 if '+' in method else 1
                if method_count >= 3:
                    decision = "TRIPLE_CONSENSUS"
                    consensus_concepts.append(concept)
                elif method_count == 2 and score >= 0.5:
                    decision = "DOUBLE_CONSENSUS"
                    consensus_concepts.append(concept)
                elif is_boosted and score >= 0.75:
                    decision = "DB_BOOST"
                    file_storage_boosted.append(concept)
                elif (in_title or in_abstract) and score >= 0.7:
                    decision = "TITLE_ABSTRACT_BOOST"
                    high_confidence.append(concept)
                elif score >= 0.85 and word_count <= 3:
                    decision = "HIGH_CONF"
                    high_confidence.append(concept)
                elif score >= 0.75 and word_count <= 2:
                    decision = "ACCEPTED"
                    single_method.append(concept)
                elif frequency >= 3 and score >= 0.65:
                    decision = "FREQUENCY_BOOST"
                    single_method.append(concept)
                else:
                    rejection_reason = "below_relaxed_thresholds"
        concept['purity_metrics'] = {
            'purity_score': round(purity_score, 3),
            'purity_reasons': purity_reasons,
            'decision': decision,
            'rejection_reason': rejection_reason,
            'methods_count': methods_in_name,
            'word_count': word_count,
            'char_count': char_count,
            'is_consensus': is_consensus,
            'is_boosted': is_boosted,
            'has_cross_ref': has_cross_ref,
            'consensus_indicators': consensus_indicators
        }
        if decision == "REJECTED":
            rejected_concepts.append((concept, rejection_reason))
    pure_concepts = consensus_concepts + high_confidence + file_storage_boosted + single_method
    seen = set()
    unique_pure = []
    for c in pure_concepts:
        name_lower = c['name'].lower().strip()
        if name_lower not in seen:
            seen.add(name_lower)
            unique_pure.append(c)
    logger.info(f"🏆 FINAL PURE CONCEPTS: {len(unique_pure)}")
    return unique_pure
# === Deep Helper: Quality-Focused Universal Concept Boosting ===
def boost_known_concepts(chunk: str) -> List[Dict[str, Any]]:
    boosted = []
    chunk_lower = chunk.lower()
    MAX_BOOSTS = 25
    MAX_CHECKS = 300
    logger.info(f"🚀 QUALITY BOOST: Checking top {MAX_CHECKS} priority concepts (max {MAX_BOOSTS} quality matches)")
    sorted_concepts = sorted(
        concept_file_storage, 
        key=lambda c: c.get("priority", 0), 
        reverse=True
    )[:MAX_CHECKS]
    domain_matches = {}
    for concept in sorted_concepts:
        if len(boosted) >= MAX_BOOSTS:
            logger.info(f"🛑 Quality boost limit reached ({MAX_BOOSTS} concepts)")
            break
        name = concept["name"]
        aliases = concept.get("aliases", [])
        if len(name) < 4:
            continue
        all_terms = [name.lower()] + [alias.lower() for alias in aliases]
        matched_terms = []
        for term in all_terms:
            # Precise word boundary matching
            if (f" {term} " in f" {chunk_lower} " or 
                chunk_lower.startswith(f"{term} ") or 
                chunk_lower.endswith(f" {term}") or
                chunk_lower == term):
                matched_terms.append(term)
                break
        if matched_terms:
            base_score = concept_scores.get(name, 0.5)
            boost_multiplier = concept.get("boost_multiplier", 1.2)
            boosted_score = min(0.98, base_score * boost_multiplier)
            category = concept.get("category", "general")
            domain_matches[category] = domain_matches.get(category, 0) + 1
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
    logger.info(f"🚀 Quality-focused boost complete: {len(boosted)} high-priority concepts found")
    return boosted

# === Deep Helper: Extract and Boost (Semantic + DB) ===
def extract_and_boost_concepts(chunk: str, threshold: float = 0.0, chunk_index: int = 0, chunk_section: str = "body", title_text: str = "", abstract_text: str = "") -> List[Dict[str, Any]]:
    logger.info(f"🔧 🌍 UNIVERSAL EXTRACT AND BOOST: threshold: {threshold}")
    logger.info(f"🔬 Chunk length: {len(chunk)} chars, index: {chunk_index}, section: {chunk_section}")
    logger.info("🔬 STEP 1: Calling universal extractConceptsFromDocument...")
    semantic_hits = extractConceptsFromDocument(chunk, threshold=threshold, chunk_index=chunk_index, chunk_section=chunk_section)
    logger.info(f"📊 UNIVERSAL SEMANTIC EXTRACTION RESULT: {len(semantic_hits)} concepts")
    logger.info("🔬 STEP 2: Calling quality-focused boost_known_concepts...")
    boosted = boost_known_concepts(chunk)
    logger.info(f"🚀 QUALITY-FOCUSED DATABASE BOOST RESULT: {len(boosted)} concepts")
    combined = semantic_hits + boosted
    logger.info(f"🔧 UNIVERSAL COMBINED RESULT: {len(combined)} total concepts (before purity analysis)")
    concept_name_counts = {}
    for concept in combined:
        name_lower = concept["name"].lower()
        concept_name_counts[name_lower] = concept_name_counts.get(name_lower, 0) + 1
    for concept in combined:
        name = concept['name']
        name_lower = name.lower()
        if concept_name_counts[name_lower] > 1:
            concept.setdefault("metadata", {})["cross_reference_boost"] = True
            concept.setdefault("metadata", {})["methods_found"] = concept_name_counts[name_lower]
        freq_data = get_concept_frequency(name)
        concept.setdefault('metadata', {})['frequency'] = freq_data['count']
        if 'sections' not in concept.get('metadata', {}):
            if 'chunk_sections' in concept.get('metadata', {}):
                concept['metadata']['sections'] = list(concept['metadata']['chunk_sections'])
            else:
                concept['metadata']['sections'] = [chunk_section]
        concept['metadata']['in_title'] = bool(title_text and name_lower in title_text.lower())
        concept['metadata']['in_abstract'] = bool(abstract_text and name_lower in abstract_text.lower())
    return combined
def ingest_pdf_clean(pdf_path: str, doc_id: str = None, extraction_threshold: float = 0.0, admin_mode: bool = False) -> Dict[str, Any]:
    """
    ATOMIC EQUAL ACCESS, PURITY-BASED PDF INGESTION PIPELINE (TORI STYLE)
    """
    start_time = datetime.now()
    if doc_id is None:
        doc_id = Path(pdf_path).stem

    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    MAX_CHUNKS, MAX_TOTAL_CONCEPTS = get_dynamic_limits(file_size_mb)
    logger.info(f"🏆 [TORI] Ingesting: {Path(pdf_path).name} ({file_size_mb:.1f} MB)  LIMITS: {MAX_CHUNKS} chunks, {MAX_TOTAL_CONCEPTS} concepts")
    logger.info(f"Entropy Pruning: {'ENABLED' if ENABLE_ENTROPY_PRUNING else 'DISABLED'}")
    try:
        doc_metadata = extract_pdf_metadata(pdf_path)
        if ENABLE_FREQUENCY_TRACKING:
            reset_frequency_counter()
        chunks = extract_chunks(pdf_path)
        if not chunks:
            logger.warning(f"No text chunks extracted from {pdf_path}")
            return {"filename": Path(pdf_path).name, "concept_count": 0, "status": "empty"}
        title_text, abstract_text = "", ""
        if ENABLE_CONTEXT_EXTRACTION:
            title_text, abstract_text = extract_title_abstract_safe(chunks, pdf_path)
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
            enhanced_concepts = extract_and_boost_concepts(
                chunk_text,
                threshold=extraction_threshold,
                chunk_index=chunk_index,
                chunk_section=chunk_section,
                title_text=title_text,
                abstract_text=abstract_text
            )
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
            if len(all_extracted_concepts) >= MAX_TOTAL_CONCEPTS:
                logger.info(f"Concept count limit reached: {len(all_extracted_concepts)}. Stopping.")
                break
        if not all_extracted_concepts:
            logger.error("Critical: No concepts extracted!")
            return {"filename": Path(pdf_path).name, "concept_count": 0, "status": "critical_failure"}
        chunks_processed = min(
            len(chunks_to_process),
            next((i+1 for i, _ in enumerate(chunks_to_process)
                  if len(all_extracted_concepts) >= MAX_TOTAL_CONCEPTS),
                 len(chunks_to_process))
        )
        # --- Purity Filtering ---
        pure_concepts = analyze_concept_purity(
            all_extracted_concepts, Path(pdf_path).name, title_text, abstract_text
        )
        pure_concepts.sort(key=lambda x: x['score'], reverse=True)
        concept_count = len(pure_concepts)
        prune_stats = None
        # --- Enhanced Entropy Pruning (after purity) ---
        original_pure_count = len(pure_concepts)
        prune_stats = None
        
        if ENABLE_ENTROPY_PRUNING and concept_count > 0:
            logger.info("\n" + "="*50)
            logger.info("🎯 APPLYING ENTROPY-BASED DIVERSITY PRUNING...")
            logger.info("="*50)
            
            # Determine max concepts based on admin mode
            if admin_mode:
                max_diverse = None  # No limit for admin mode
                logger.info(f"📊 Admin mode: allowing unlimited diverse concepts")
            else:
                max_diverse = ENTROPY_CONFIG["max_diverse_concepts"]
                logger.info(f"📊 Standard mode: max diverse concepts = {max_diverse}")
            
            # Extract categories from concepts
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
                # Regular entropy pruning
                pure_concepts, prune_stats = entropy_prune(
                    pure_concepts,
                    top_k=max_diverse,
                    entropy_threshold=ENTROPY_CONFIG["entropy_threshold"],
                    similarity_threshold=ENTROPY_CONFIG["similarity_threshold"],
                    verbose=True
                )
            
            concept_count = len(pure_concepts)
            
            # Log results with safe math
            efficiency_percent = 100.0 if original_pure_count == 0 else (concept_count / original_pure_count * 100)
            logger.info(f"✅ DIVERSITY FILTER: {concept_count} diverse concepts from {original_pure_count} pure")
            logger.info(f"📊 DIVERSITY EFFICIENCY: {efficiency_percent:.1f}% kept")
            if prune_stats:
                logger.info(f"📈 Final entropy: {prune_stats.get('final_entropy', 0):.3f}")
                logger.info(f"🔗 Avg similarity: {prune_stats.get('avg_similarity', 0):.3f}")
            
            # Log top diverse concepts
            logger.info("\n🌟 TOP DIVERSE CONCEPTS:")
            for i, concept in enumerate(pure_concepts[:10]):
                logger.info(f"  {i+1}. {concept['name']} (score={concept['score']:.3f})")
        else:
            logger.info("⏭️ Entropy pruning disabled or no concepts to prune")
        # --- Knowledge Injection ---
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
        # Calculate averages with safe math (avoid NoneType and division by zero)
        avg_score = 0.0
        if pure_concepts:
            scores = [c.get("score", 0) for c in pure_concepts if c.get("score") is not None]
            avg_score = sum(scores) / len(scores) if scores else 0.0
        
        high_conf_count = sum(1 for c in pure_concepts if c.get("score", 0) > 0.8)
        total_time = (datetime.now() - start_time).total_seconds()
        response = {
            "filename": Path(pdf_path).name,
            "concept_count": concept_count,
            "concept_names": [c['name'] for c in pure_concepts],
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
            "average_concept_score": avg_score,
            "high_confidence_concepts": high_conf_count,
            "total_extraction_time": total_time,
            "domain_distribution": domain_distribution,
            "title_found": bool(title_text),
            "abstract_found": bool(abstract_text),
            "purity_analysis": {
                "raw_concepts": len(all_extracted_concepts),
                "pure_concepts": original_pure_count,
                "final_concepts": concept_count,
                "purity_efficiency_percent": round((original_pure_count or 0) * 100 / max(len(all_extracted_concepts), 1), 1),
                "diversity_efficiency_percent": round((concept_count or 0) * 100 / max(original_pure_count, 1), 1),
                "top_concepts": [
                    {
                        "name": c['name'],
                        "score": round(c.get('score', 0), 3),
                        "methods": c.get('metadata', {}).get('consensus_methods', [c.get('method', 'unknown')]),
                        "frequency": c.get('metadata', {}).get('frequency', 1),
                        "purity_decision": c.get('purity_metrics', {}).get('decision', 'unknown')
                    }
                    for c in pure_concepts[:10]
                ]
            }
        }
        # Add entropy analysis to response with BULLETPROOF ATOMIC PROTECTION
        if prune_stats:
            # 🔧 ATOMIC PATCH: Sanitize ALL None values before ANY calculations
            # This prevents: unsupported operand type(s) for *: 'NoneType' and 'int'
            for key in ["total", "pruned", "selected", "final_entropy", "avg_similarity"]:
                if key in prune_stats and prune_stats[key] is None:
                    prune_stats[key] = 0
            
            # BULLETPROOF extraction with double-safe defaults
            total = prune_stats.get("total") or 0
            selected = prune_stats.get("selected") or 0
            pruned = prune_stats.get("pruned") or 0
            final_entropy = prune_stats.get("final_entropy") or 0.0
            avg_similarity = prune_stats.get("avg_similarity") or 0.0
            
            # ATOMIC SAFE percentage calculation - NO None multiplication possible
            def safe_percent(value, total_val):
                value = value or 0
                total_val = total_val or 1  # Never divide by zero
                return (value * 100) / total_val
            
            diversity_efficiency = safe_percent(selected, total)
            
            response["entropy_analysis"] = {
                "enabled": True,
                "admin_mode": admin_mode,
                "total_before_entropy": total,
                "selected_diverse": selected,
                "pruned_similar": pruned,
                "diversity_efficiency_percent": round(diversity_efficiency, 1),
                "final_entropy": round(final_entropy, 3),
                "avg_similarity": round(avg_similarity, 3),
                "by_category": prune_stats.get("by_category", {}),
                "config": {
                    "max_diverse_concepts": ENTROPY_CONFIG["max_diverse_concepts"] if not admin_mode else "unlimited",
                    "entropy_threshold": ENTROPY_CONFIG["entropy_threshold"],
                    "similarity_threshold": ENTROPY_CONFIG["similarity_threshold"],
                    "category_aware": ENTROPY_CONFIG["enable_categories"]
                },
                "performance": {
                    "original_pure_concepts": original_pure_count,
                    "final_diverse_concepts": concept_count,
                    "reduction_ratio": round(safe_percent(original_pure_count - concept_count, original_pure_count), 3)
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
            "performance_limited": True
        }

# ==============================================================
# 🚀 ATOMIC. UNAPOLOGETIC. TORI. LEGENDARY. BOOYAH. 🚀
# ==============================================================

# === Optional: Ultra-Verbose Diagnostic Snapshot (For Deep Debugging) ===
def save_ingestion_snapshot(
    pdf_path: str,
    concepts: List[Dict[str, Any]],
    step_label: str = "final",
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save an atomic JSON snapshot of the current state of concept extraction at a given pipeline step.
    Useful for deep audit trails and reproducing edge-case bugs in dev/CI.
    """
    try:
        snap_dir = Path(__file__).parent / "debug_snapshots"
        snap_dir.mkdir(exist_ok=True)
        filename = f"{Path(pdf_path).stem}_{step_label}_{int(time.time())}.json"
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "step": step_label,
            "filename": str(pdf_path),
            "concept_count": len(concepts),
            "concepts": concepts
        }
        if extra:
            snapshot.update(extra)
        snap_path = snap_dir / filename
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        logger.info(f"📝 Diagnostic snapshot saved: {snap_path}")
    except Exception as e:
        logger.warning(f"⚠️ Could not save diagnostic snapshot: {e}")

# === Optional: Experimental Legacy Extraction Logic (For Research) ===
def legacy_concept_extraction(pdf_path: str) -> List[Dict[str, Any]]:
    """
    (For backward compatibility or research. 
     Replace with your legacy concept extraction approach, if needed.)
    """
    logger.info(f"[LEGACY] Running legacy concept extraction for {pdf_path}")
    try:
        # Old logic goes here (stubbed)
        return []
    except Exception as e:
        logger.warning(f"Legacy extraction failed: {e}")
        return []

# === Optional: Cross-File, Cross-Chunk Concept Frequency Report ===
def report_global_concept_frequencies() -> None:
    """
    Print a sorted report of all tracked concept frequencies across entire ingestion session.
    Useful for system health diagnostics or "concept overfitting" debugging.
    """
    freq_table = get_concept_frequency(None)
    logger.info("=== GLOBAL CONCEPT FREQUENCY REPORT ===")
    for concept, data in sorted(freq_table.items(), key=lambda x: -x[1]["count"]):
        logger.info(f"{concept:40}  count={data['count']:4d}  last_seen={data['last_seen_chunk']:4d}")

# === (Optional) Utility: Manual Diagnostic Trigger (For Dev CLI or REST) ===
def debug_manual_extract(pdf_path: str, threshold: float = 0.0):
    """
    Quick CLI test for extraction, with full logging, snapshot, and detailed output.
    """
    logger.info("=== DEBUG MANUAL EXTRACT TRIGGERED ===")
    result = ingest_pdf_clean(pdf_path, extraction_threshold=threshold)
    save_ingestion_snapshot(pdf_path, result.get("concepts", []), step_label="manual_debug")
    print(json.dumps(result, indent=2))
    return result

# === End of pipeline.py: All hooks, all options, all diagnostics ===
# === Advanced Integration: RESTful Pipeline Invocation ===
import requests

def post_to_remote_pipeline(
    endpoint: str,
    pdf_path: str,
    extra_payload: Optional[Dict[str, Any]] = None,
    timeout: int = 180
) -> Dict[str, Any]:
    """
    Call a remote ingestion pipeline REST endpoint, streaming PDF bytes and optional extra fields.
    Used for distributed deployments, cloud microservices, or MCP scenarios.
    """
    logger.info(f"🌐 Posting {pdf_path} to remote pipeline: {endpoint}")
    with open(pdf_path, "rb") as f:
        files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
        data = extra_payload or {}
        try:
            resp = requests.post(endpoint, files=files, data=data, timeout=timeout)
            resp.raise_for_status()
            logger.info(f"✅ Remote pipeline success: {resp.status_code}")
            return resp.json()
        except Exception as e:
            logger.error(f"❌ Remote pipeline call failed: {e}")
            return {"status": "error", "error_message": str(e)}

# === Advanced: Hybrid Local + Remote Extraction Chain ===
def hybrid_ingestion(
    pdf_path: str,
    remote_endpoint: Optional[str] = None,
    run_local_first: bool = True,
    failover_to_local: bool = True
) -> Dict[str, Any]:
    """
    Orchestrate multi-stage pipeline: try local extraction, remote fallback, or remote-to-local chaining.
    Ideal for cloud/edge deployments, CI/CD, or redundancy.
    """
    if run_local_first:
        try:
            logger.info("🔗 [Hybrid] Running local pipeline first...")
            local_result = ingest_pdf_clean(pdf_path)
            if local_result.get("concept_count", 0) > 0:
                logger.info("🔗 [Hybrid] Local pipeline succeeded.")
                return local_result
            else:
                logger.warning("🔗 [Hybrid] Local pipeline returned no concepts; trying remote...")
        except Exception as e:
            logger.error(f"🔗 [Hybrid] Local pipeline exception: {e}")
    if remote_endpoint:
        remote_result = post_to_remote_pipeline(remote_endpoint, pdf_path)
        if remote_result.get("concept_count", 0) > 0:
            logger.info("🔗 [Hybrid] Remote pipeline succeeded.")
            return remote_result
        elif failover_to_local and not run_local_first:
            logger.warning("🔗 [Hybrid] Remote pipeline empty; fallback to local.")
            try:
                return ingest_pdf_clean(pdf_path)
            except Exception as e:
                logger.error(f"🔗 [Hybrid] Local fallback failed: {e}")
                return {"status": "error", "error_message": str(e)}
    return {"status": "error", "error_message": "All pipeline paths failed."}

# === MCP Orchestration Example: Batch Ingestion ===
def batch_ingest_folder(
    folder_path: str,
    output_dir: str = "./ingestion_results",
    remote_endpoint: Optional[str] = None,
    hybrid: bool = True
) -> None:
    """
    Batch-ingest every PDF in a folder, using local, remote, or hybrid logic.
    Writes atomic results for each file.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if f.lower().endswith(".pdf")]
    logger.info(f"📂 Batch ingesting {len(pdf_files)} PDFs from {folder_path}")
    for pdf_file in pdf_files:
        logger.info(f"--- Batch Ingest: {pdf_file} ---")
        if hybrid and remote_endpoint:
            result = hybrid_ingestion(pdf_file, remote_endpoint=remote_endpoint)
        elif remote_endpoint:
            result = post_to_remote_pipeline(remote_endpoint, pdf_file)
        else:
            result = ingest_pdf_clean(pdf_file)
        out_file = os.path.join(output_dir, f"{Path(pdf_file).stem}_result.json")
        with open(out_file, "w", encoding="utf-8") as out:
            json.dump(result, out, indent=2)
        logger.info(f"✔️ Ingest result saved: {out_file}")

# === Advanced Integration: Post-Processing & Webhook Sync ===
def post_ingestion_webhook(
    result: Dict[str, Any],
    webhook_url: str,
    extra_headers: Optional[Dict[str, str]] = None,
    retries: int = 3
) -> bool:
    """
    POST the result of ingestion to a webhook (e.g., MCP event bus, Slack, cloud queue, etc).
    Retries for reliability.
    """
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(webhook_url, headers=headers, data=json.dumps(result))
            resp.raise_for_status()
            logger.info(f"🟢 Webhook POST success: {webhook_url}")
            return True
        except Exception as e:
            logger.warning(f"🔴 Webhook POST failed (attempt {attempt}): {e}")
            time.sleep(2 * attempt)
    return False

# === Integration: Plug-in Hooks for Upstream/Downstream Processing ===
# Example: You can register a callback to be called after every successful pipeline run.
pipeline_post_hooks: List = []

def register_pipeline_hook(hook_fn):
    """
    Register a function to be called with (result, pdf_path) after every pipeline run.
    """
    pipeline_post_hooks.append(hook_fn)

def run_pipeline_with_hooks(pdf_path: str, *args, **kwargs) -> Dict[str, Any]:
    """
    Run the pipeline, then invoke all registered hooks with (result, pdf_path).
    """
    result = ingest_pdf_clean(pdf_path, *args, **kwargs)
    for hook_fn in pipeline_post_hooks:
        try:
            hook_fn(result, pdf_path)
        except Exception as e:
            logger.warning(f"⚠️ Pipeline post-hook failed: {e}")
    return result

# === CLI/Dev Entry Point (for advanced users and test harnesses) ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TORI ATOMIC PIPELINE - ADVANCED INTEGRATION CLI")
    parser.add_argument("pdf", nargs="?", help="PDF file to ingest")
    parser.add_argument("--folder", help="Ingest every PDF in this folder (batch mode)")
    parser.add_argument("--remote", help="Remote pipeline endpoint (REST)")
    parser.add_argument("--webhook", help="Webhook to POST results to after extraction")
    parser.add_argument("--threshold", type=float, default=0.0, help="Extraction threshold (default: 0)")
    parser.add_argument("--hybrid", action="store_true", help="Hybrid local/remote mode")
    parser.add_argument("--output", default="./ingestion_results", help="Output folder for batch mode")
    args = parser.parse_args()

    if args.folder:
        batch_ingest_folder(args.folder, output_dir=args.output, remote_endpoint=args.remote, hybrid=args.hybrid)
    elif args.pdf:
        if args.hybrid and args.remote:
            result = hybrid_ingestion(args.pdf, remote_endpoint=args.remote)
        elif args.remote:
            result = post_to_remote_pipeline(args.remote, args.pdf)
        else:
            result = ingest_pdf_clean(args.pdf, extraction_threshold=args.threshold)
        out_path = os.path.join(args.output, f"{Path(args.pdf).stem}_result.json")
        os.makedirs(args.output, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as out:
            json.dump(result, out, indent=2)
        logger.info(f"✔️ Output saved: {out_path}")
        if args.webhook:
            posted = post_ingestion_webhook(result, args.webhook)
            logger.info(f"Webhook POST {'succeeded' if posted else 'failed'}.")

# ==============================================================
# 🔥 ADVANCED. ORCHESTRATABLE. DISTRIBUTED. TORI-LEVEL. 🔥
# ==============================================================

