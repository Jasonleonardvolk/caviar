"""
pipeline/quality.py

Concept quality analysis, purity filtering, and concept boosting.
Handles scoring, theme relevance, and file_storage-based concept enhancement.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

# Local imports
from .config import (
    CONCEPT_DB_PATH, UNIVERSAL_SEED_PATH, GENERIC_TERMS,
    MIN_CONCEPT_SCORE, HIGH_QUALITY_THRESHOLD, MIN_CONCEPT_LENGTH,
    MAX_CONCEPT_WORDS, MAX_DATABASE_BOOSTS
)
from .utils import safe_get, safe_divide, safe_multiply, get_logger

# Setup logger
logger = get_logger(__name__)

# Import dependencies with fallback
try:
    from ..extractConceptsFromDocument import (
        extractConceptsFromDocument, get_concept_frequency,
        reset_frequency_counter
    )
except ImportError:
    try:
        from extractConceptsFromDocument import (
            extractConceptsFromDocument, get_concept_frequency,
            reset_frequency_counter
        )
    except ImportError:
        logger.error("❌ Failed to import concept extraction modules")
        raise

# === Universal Concept Database ===
concept_file_storage: List[Dict] = []
concept_names: List[str] = []
concept_scores: Dict[str, float] = {}


def load_universal_concept_file_storage() -> None:
    """Load the universal concept file_storage and seed concepts."""
    global concept_file_storage, concept_names, concept_scores
    
    try:
        with open(CONCEPT_DB_PATH, "r", encoding="utf-8") as f:
            main_concepts = json.load(f)
        logger.info(f"✅ Main concept file_storage loaded: {len(main_concepts)} concepts")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load main concept file_storage: {e}")
        main_concepts = []
    
    try:
        with open(UNIVERSAL_SEED_PATH, "r", encoding="utf-8") as f:
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


# Load file_storage on module import
load_universal_concept_file_storage()


def calculate_theme_relevance(concept_name: str, doc_context: Dict) -> float:
    """
    Calculate how relevant a concept is to the document's main theme.
    
    Args:
        concept_name: Name of the concept
        doc_context: Document context with title, abstract, etc.
        
    Returns:
        Relevance score between 0 and 1
    """
    title = safe_get(doc_context, 'title', '').lower()
    abstract = safe_get(doc_context, 'abstract', '').lower()
    concept_lower = concept_name.lower()
    
    relevance = 0.0
    
    # Direct mention in title/abstract
    if concept_lower in title:
        relevance += 0.5
    if concept_lower in abstract:
        relevance += 0.3
    
    # Partial matches (words from concept appear in title/abstract)
    concept_words = set(concept_lower.split())
    title_words = set(title.split())
    abstract_words = set(abstract.split())
    
    title_overlap = len(concept_words & title_words) / max(len(concept_words), 1)
    abstract_overlap = len(concept_words & abstract_words) / max(len(concept_words), 1)
    
    relevance += title_overlap * 0.3
    relevance += abstract_overlap * 0.2
    
    return min(relevance, 1.0)


def calculate_concept_quality(concept: Dict, doc_context: Dict) -> float:
    """
    Calculate comprehensive quality score for concept.
    
    Args:
        concept: Concept dictionary with score, metadata, etc.
        doc_context: Document context for theme relevance
        
    Returns:
        Quality score between 0 and 1
    """
    base_score = safe_get(concept, 'score', 0.5)
    
    # Existing factors
    frequency = safe_get(concept.get('metadata', {}), 'frequency', 1)
    in_title = safe_get(concept.get('metadata', {}), 'in_title', False)
    in_abstract = safe_get(concept.get('metadata', {}), 'in_abstract', False)
    
    # Section weights for academic papers
    section_weight = {
        'title': 2.0,
        'abstract': 1.5,
        'introduction': 1.2,
        'conclusion': 1.2,
        'methodology': 1.1,
        'results': 1.1,
        'discussion': 1.0,
        'body': 1.0,
        'references': 0.7
    }
    
    section = safe_get(concept.get('metadata', {}), 'section', 'body')
    
    # Theme relevance calculation
    theme_relevance = calculate_theme_relevance(concept['name'], doc_context)
    
    # Combine factors
    quality = base_score * section_weight.get(section, 1.0)
    quality *= (1 + min(frequency, 5) * 0.1)  # Frequency boost, capped
    quality *= (1.3 if in_title else 1.0)
    quality *= (1.2 if in_abstract else 1.0)
    quality *= (0.8 + theme_relevance * 0.4)  # Theme relevance factor
    
    # Boost for multi-method extraction
    method = safe_get(concept, 'method', '')
    if '+' in method:
        quality *= 1.1
    
    return min(quality, 1.0)


def boost_known_concepts(chunk: str) -> List[Dict[str, Any]]:
    """
    Boost concepts from the universal file_storage found in the chunk.
    
    Args:
        chunk: Text chunk to search for known concepts
        
    Returns:
        List of boosted concept dictionaries
    """
    boosted = []
    chunk_lower = chunk.lower()
    
    for concept in concept_file_storage[:300]:  # Limit for performance
        if len(boosted) >= MAX_DATABASE_BOOSTS:
            break
            
        name = safe_get(concept, "name", "")
        if len(name) < 4:
            continue
        
        if name.lower() in chunk_lower:
            base_score = safe_get(concept_scores, name, 0.5)
            boost_multiplier = safe_get(concept, "boost_multiplier", 1.2)
            boosted_score = min(0.98, safe_multiply(base_score, boost_multiplier, 0.5))
            
            boosted.append({
                "name": name,
                "score": boosted_score,
                "method": "file_storage_boosted",
                "source": {"file_storage_matched": True},
                "metadata": {"category": safe_get(concept, "category", "general")}
            })
    
    return boosted


def extract_and_boost_concepts(chunk: str, threshold: float = 0.0, 
                              chunk_index: int = 0, chunk_section: str = "body", 
                              title_text: str = "", abstract_text: str = "") -> List[Dict[str, Any]]:
    """
    Extract concepts from chunk and boost with file_storage matches.
    
    Args:
        chunk: Text chunk to process
        threshold: Minimum score threshold
        chunk_index: Index of this chunk
        chunk_section: Academic section type
        title_text: Document title for context
        abstract_text: Document abstract for context
        
    Returns:
        List of extracted and boosted concepts
    """
    try:
        # Extract concepts
        semantic_hits = extractConceptsFromDocument(
            chunk, threshold=threshold, 
            chunk_index=chunk_index, 
            chunk_section=chunk_section
        )
        boosted = boost_known_concepts(chunk)
        combined = semantic_hits + boosted
        
        # Add metadata safely
        for concept in combined:
            if not isinstance(concept, dict):
                continue
                
            name = safe_get(concept, 'name', '')
            name_lower = name.lower()
            
            # Ensure metadata exists
            if 'metadata' not in concept:
                concept['metadata'] = {}
            
            # Add safe frequency data
            freq_data = get_concept_frequency(name)
            concept['metadata']['frequency'] = safe_get(freq_data, 'count', 1)
            concept['metadata']['sections'] = [chunk_section]
            concept['metadata']['section'] = chunk_section
            concept['metadata']['in_title'] = bool(title_text and name_lower in title_text.lower())
            concept['metadata']['in_abstract'] = bool(abstract_text and name_lower in abstract_text.lower())
        
        return combined
        
    except Exception as e:
        logger.error(f"Error in extract_and_boost_concepts: {e}")
        return []


def analyze_concept_purity(all_concepts: List[Dict[str, Any]], 
                          doc_name: str = "", 
                          title_text: str = "", 
                          abstract_text: str = "", 
                          doc_context: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Analyze concepts for purity and quality, filtering out generic terms.
    
    Args:
        all_concepts: List of raw extracted concepts
        doc_name: Document name for logging
        title_text: Document title
        abstract_text: Document abstract
        doc_context: Additional document context
        
    Returns:
        List of pure, high-quality concepts
    """
    logger.info(f"🔬 CONCEPT PURITY ANALYSIS for {doc_name}")
    logger.info(f"📊 Analyzing {len(all_concepts)} raw concepts")
    
    pure_concepts = []
    
    # Prepare document context for quality calculation
    if doc_context is None:
        doc_context = {
            'title': title_text,
            'abstract': abstract_text,
            'filename': doc_name
        }
    
    for concept in all_concepts:
        if not concept or not isinstance(concept, dict):
            continue
            
        name = safe_get(concept, 'name', '')
        if not name or len(name) < MIN_CONCEPT_LENGTH:
            continue
            
        score = safe_get(concept, 'score', 0)
        if score < MIN_CONCEPT_SCORE:
            continue
            
        method = safe_get(concept, 'method', '')
        metadata = safe_get(concept, 'metadata', {})
        
        name_lower = name.lower().strip()
        if name_lower in GENERIC_TERMS:
            continue
        
        word_count = len(name.split())
        if word_count > MAX_CONCEPT_WORDS:
            continue
        
        # Calculate enhanced quality score
        quality_score = calculate_concept_quality(concept, doc_context)
        concept['quality_score'] = quality_score
        
        # Enhanced acceptance criteria with quality score
        frequency = safe_get(metadata, 'frequency', 1)
        in_title = safe_get(metadata, 'in_title', False)
        in_abstract = safe_get(metadata, 'in_abstract', False)
        method_count = method.count('+') + 1 if '+' in method else 1
        is_boosted = 'file_storage_boosted' in method or 'boost' in method
        
        # Accept based on various criteria including quality score
        if (quality_score >= 0.7 or
            method_count >= 2 or 
            is_boosted and score >= HIGH_QUALITY_THRESHOLD or
            (in_title or in_abstract) and score >= 0.7 or
            score >= 0.85 and word_count <= 3 or
            frequency >= 3 and score >= 0.65):
            pure_concepts.append(concept)
    
    # Deduplicate safely
    seen = set()
    unique_pure = []
    for c in pure_concepts:
        name_lower = safe_get(c, 'name', '').lower().strip()
        if name_lower and name_lower not in seen:
            seen.add(name_lower)
            unique_pure.append(c)
    
    # Sort by quality score
    unique_pure.sort(key=lambda x: safe_get(x, 'quality_score', 0), reverse=True)
    
    logger.info(f"🏆 FINAL PURE CONCEPTS: {len(unique_pure)}")
    return unique_pure


def is_rogue_concept_contextual(concept_name: str, doc_context: Optional[Dict] = None) -> bool:
    """
    Check if a concept is rogue/generic based on context.
    
    Args:
        concept_name: Name of the concept to check
        doc_context: Optional document context for better filtering
        
    Returns:
        True if the concept is rogue/generic and should be filtered
    """
    if not concept_name:
        return True
        
    name_lower = concept_name.lower().strip()
    
    # Check against generic terms
    if name_lower in GENERIC_TERMS:
        return True
    
    # Check for very short concepts
    if len(name_lower) < MIN_CONCEPT_LENGTH:
        return True
    
    # Check for single common words
    common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'}
    if name_lower in common_words:
        return True
    
    # Check for too many words
    word_count = len(concept_name.split())
    if word_count > MAX_CONCEPT_WORDS:
        return True
    
    # Context-aware filtering if context provided
    if doc_context:
        # Could add more sophisticated context-based filtering here
        pass
    
    return False
