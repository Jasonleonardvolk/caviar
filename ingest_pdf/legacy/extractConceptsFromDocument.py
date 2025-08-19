"""
🌍 UNIVERSAL CONCEPT EXTRACTION MODULE

This module provides universal concept extraction that works across ALL domains:
- Science, Mathematics, Philosophy, Arts, Literature, Humanities, etc.
- Uses YAKE + KeyBERT (universal embeddings) + spaCy NER + Wikidata linking
- Replaces the previous STEM-biased SciBERT/SciSpaCy approach
"""

import logging
import json
from typing import List, Dict, Any, Set

# Configure logging
logger = logging.getLogger(__name__)

# Global model instances (loaded once for performance)
_yake_extractor = None
_kb_model = None
_nlp = None
_entity_linking = False

# Global frequency counter for current document
concept_frequency_counter = {}

def reset_frequency_counter():
    """Reset counter between documents"""
    global concept_frequency_counter
    concept_frequency_counter = {}

def track_concept_frequency(concept_name: str, chunk_index: int):
    """Track concept frequency across chunks"""
    global concept_frequency_counter
    if concept_name not in concept_frequency_counter:
        concept_frequency_counter[concept_name] = {
            "count": 0,
            "chunks": set()
        }
    concept_frequency_counter[concept_name]["count"] += 1
    concept_frequency_counter[concept_name]["chunks"].add(chunk_index)

def get_concept_frequency(concept_name: str) -> Dict[str, Any]:
    """Get frequency data for a concept"""
    global concept_frequency_counter
    return concept_frequency_counter.get(concept_name, {"count": 1, "chunks": {0}})

def _initialize_models():
    """Initialize all models once at module load time for performance"""
    global _yake_extractor, _kb_model, _nlp, _entity_linking
    
    if _yake_extractor is not None:
        return  # Already initialized
    
    logger.info("🌍 INITIALIZING UNIVERSAL CONCEPT EXTRACTION MODELS...")
    
    try:
        # YAKE: Domain-agnostic statistical keyphrase extraction
        import yake
        _yake_extractor = yake.KeywordExtractor(
            lan="en", 
            n=3,           # Up to 3-word phrases
            dedupLim=0.9,  # High deduplication threshold
            top=20         # Top 20 candidates
        )
        logger.info("✅ YAKE extractor initialized")
        
        # KeyBERT: Universal semantic keyphrase extraction
        from keybert import KeyBERT
        # Using all-mpnet-base-v2: Universal model, NOT science-biased
        _kb_model = KeyBERT(model='sentence-transformers/all-mpnet-base-v2')
        logger.info("✅ KeyBERT with universal embeddings initialized")
        
        # spaCy: Universal English NER (not scientific-only)
        import spacy
        try:
            _nlp = spacy.load("en_core_web_lg")
            logger.info("✅ spaCy universal NER initialized")
        except OSError:
            logger.error("❌ spaCy model 'en_core_web_lg' not found. Please run: python -m spacy download en_core_web_lg")
            raise
        
        # Optional: Wikidata entity linking for universal knowledge base
        try:
            import spacy_entity_linker
            _nlp.add_pipe("entityLinker", last=True)
            _entity_linking = True
            logger.info("✅ Wikidata entity linker activated")
        except ImportError:
            _entity_linking = False
            logger.info("ℹ️ spacy-entity-linker not installed; proceeding without entity linking")
            
    except ImportError as e:
        logger.error(f"❌ Failed to import required libraries: {e}")
        logger.error("Please install: pip install yake keybert sentence-transformers spacy")
        raise

def extractConceptsUniversal(chunk: str, chunk_index: int = 0, chunk_section: str = "body") -> List[Dict[str, Any]]:
    """
    🌍 UNIVERSAL CONCEPT EXTRACTION
    
    Extract concepts from text using three complementary methods:
    - YAKE: Statistical keyphrase extraction (domain-agnostic)
    - KeyBERT: Semantic keyphrase extraction (universal embeddings) 
    - spaCy NER: Named entity recognition + Wikidata linking
    
    Args:
        chunk: Text to extract concepts from
        chunk_index: Index of this chunk in the document
        chunk_section: Section this chunk belongs to
        
    Returns:
        List of concept dictionaries with name, score, method, metadata
    """
    _initialize_models()
    
    if not chunk or not isinstance(chunk, str):
        return []
    
    text = chunk.strip()
    if not text:
        return []
    
    logger.info(f"🌍 UNIVERSAL EXTRACTION: Processing {len(text)} chars from chunk {chunk_index} ({chunk_section})")
    
    # Accumulate concepts: name_key -> concept_data
    concepts = {}
    
    # ===== YAKE EXTRACTION =====
    try:
        yake_keywords = _yake_extractor.extract_keywords(text)
        
        if yake_keywords:
            # YAKE scores: Lower = more relevant, so we invert them
            yake_scores = [score for _, score in yake_keywords]
            min_score = min(yake_scores) if yake_scores else 0
            max_score = max(yake_scores) if yake_scores else 0
            range_score = (max_score - min_score) if (max_score - min_score) != 0 else 1e-6
            
            for keyword, score in yake_keywords:
                # Invert and normalize YAKE score
                if max_score > min_score:
                    norm_score = 1.0 - ((score - min_score) / range_score)
                else:
                    norm_score = 1.0
                
                weighted_score = 0.3 * norm_score  # 30% weight for YAKE
                
                name = keyword.strip()
                if not name:
                    continue
                    
                name_key = name.lower()
                if name_key not in concepts:
                    concepts[name_key] = {
                        "name": name, 
                        "score": 0.0, 
                        "methods": set(), 
                        "metadata": {
                            "chunk_sections": set()
                        }
                    }
                
                concepts[name_key]["score"] += weighted_score
                concepts[name_key]["methods"].add("YAKE")
                concepts[name_key]["metadata"]["chunk_sections"].add(chunk_section)
                
                # Track frequency
                track_concept_frequency(name, chunk_index)
                
        logger.debug(f"🔤 YAKE found {len(yake_keywords)} candidates")
        
    except Exception as e:
        logger.warning(f"⚠️ YAKE extraction failed: {e}")
    
    # ===== KEYBERT EXTRACTION =====
    try:
        # Extract up to 3-word phrases, filter English stopwords
        kb_keywords = _kb_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 3), 
            stop_words="english", 
            top_n=20
        )
        
        if kb_keywords:
            # KeyBERT scores: Higher = more relevant
            kb_scores = [score for _, score in kb_keywords]
            max_kb = max(kb_scores) if kb_scores else 1
            
            for keyword, score in kb_keywords:
                norm_score = score / max_kb if max_kb > 0 else 0
                weighted_score = 0.4 * norm_score  # 40% weight for KeyBERT
                
                name = keyword.strip()
                if not name:
                    continue
                    
                name_key = name.lower()
                if name_key not in concepts:
                    concepts[name_key] = {
                        "name": name, 
                        "score": 0.0, 
                        "methods": set(), 
                        "metadata": {
                            "chunk_sections": set()
                        }
                    }
                
                concepts[name_key]["score"] += weighted_score
                concepts[name_key]["methods"].add("KeyBERT")
                concepts[name_key]["metadata"]["chunk_sections"].add(chunk_section)
                
                # Track frequency
                track_concept_frequency(name, chunk_index)
                
        logger.debug(f"🧠 KeyBERT found {len(kb_keywords)} candidates")
        
    except Exception as e:
        logger.warning(f"⚠️ KeyBERT extraction failed: {e}")
    
    # ===== SPACY NER EXTRACTION =====
    try:
        doc = _nlp(text)
        
        # Count entity frequencies
        ent_counts = {}
        for ent in doc.ents:
            if not ent.text.strip():
                continue
                
            ent_text = ent.text.strip()
            ent_key = ent_text.lower()
            
            if ent_key not in ent_counts:
                ent_counts[ent_key] = {
                    "text": ent_text,
                    "count": 0,
                    "label": ent.label_
                }
            ent_counts[ent_key]["count"] += 1
        
        if ent_counts:
            # Normalize by maximum frequency
            max_freq = max(val["count"] for val in ent_counts.values())
            
            for ent_key, val in ent_counts.items():
                ent_text = val["text"]
                count = val["count"]
                
                # Score: 0.5 base + 0.5 * relative_frequency
                raw_score = 0.5 + 0.5 * (count / max_freq)
                weighted_score = 0.3 * raw_score  # 30% weight for NER
                
                if ent_key not in concepts:
                    concepts[ent_key] = {
                        "name": ent_text, 
                        "score": 0.0, 
                        "methods": set(), 
                        "metadata": {
                            "chunk_sections": set()
                        }
                    }
                
                concepts[ent_key]["score"] += weighted_score
                concepts[ent_key]["methods"].add("NER")
                concepts[ent_key]["metadata"]["entity_type"] = val["label"]
                concepts[ent_key]["metadata"]["chunk_sections"].add(chunk_section)
                
                # Track frequency
                track_concept_frequency(ent_text, chunk_index)
                
                # Add Wikidata linking if available
                if _entity_linking:
                    for linked_ent in doc._.linkedEntities:
                        if linked_ent.get_span().text == ent_text:
                            concepts[ent_key]["metadata"]["wikidata_id"] = linked_ent.get_id()
                            concepts[ent_key]["metadata"]["wikidata_url"] = linked_ent.get_url()
                            break
        
        logger.debug(f"🏷️ spaCy NER found {len(ent_counts)} unique entities")
        
    except Exception as e:
        logger.warning(f"⚠️ spaCy NER extraction failed: {e}")
    
    # ===== CONSENSUS BOOST & NORMALIZATION =====
    
    # Boost concepts found by multiple methods
    for name_key, data in concepts.items():
        method_count = len(data["methods"])
        if method_count > 1:
            # +10% bonus per additional method
            bonus_factor = 1.0 + 0.1 * (method_count - 1)
            data["score"] *= bonus_factor
            logger.debug(f"🚀 Consensus boost: {data['name']} ({method_count} methods)")
    
    # Normalize all scores to [0,1] range
    if concepts:
        max_score = max(data["score"] for data in concepts.values())
        if max_score > 0:
            for data in concepts.values():
                data["score"] = round(data["score"] / max_score, 4)
        
        # Sort by score (descending)
        sorted_concepts = sorted(concepts.values(), key=lambda x: x["score"], reverse=True)
    else:
        sorted_concepts = []
    
    # Format output
    formatted_concepts = []
    for concept in sorted_concepts:
        methods_list = sorted(list(concept["methods"]))
        method_string = "+".join(methods_list)
        
        # Convert sets to lists for JSON serialization
        chunk_sections_list = list(concept["metadata"].get("chunk_sections", set()))
        
        formatted_concept = {
            "name": concept["name"],
            "score": concept["score"],
            "method": f"universal_{method_string.lower()}",
            "source": {
                "universal_extraction": True,
                "methods": method_string,
                "extraction_methods": methods_list
            },
            "context": f"Universal extraction via {method_string}",
            "metadata": {
                "extraction_method": "universal_pipeline",
                "universal": True,
                "method_count": len(methods_list),
                "chunk_sections": chunk_sections_list,
                **{k: v for k, v in concept.get("metadata", {}).items() if k != "chunk_sections"}
            }
        }
        
        # Remove empty metadata
        if not formatted_concept["metadata"] or len(formatted_concept["metadata"]) <= 3:
            formatted_concept["metadata"] = {
                "universal": True, 
                "extraction_method": "universal_pipeline",
                "chunk_sections": chunk_sections_list
            }
        
        formatted_concepts.append(formatted_concept)
    
    logger.info(f"🌍 UNIVERSAL EXTRACTION COMPLETE: {len(formatted_concepts)} concepts extracted")
    
    # Log top concepts for visibility
    if formatted_concepts:
        logger.info("🌍 TOP UNIVERSAL CONCEPTS:")
        for i, concept in enumerate(formatted_concepts[:5], 1):
            methods = concept["source"]["methods"]
            score = concept["score"]
            logger.info(f"  {i}. {concept['name']} (score: {score:.3f}, methods: {methods})")
    
    return formatted_concepts

def extractConceptsFromDocument(chunk: str, threshold: float = 0.0, chunk_index: int = 0, chunk_section: str = "body") -> List[Dict[str, Any]]:
    """
    🌍 MAIN EXTRACTION FUNCTION - Universal concept extraction
    
    This replaces the previous STEM-biased extraction with universal coverage.
    Compatible with existing pipeline architecture.
    
    Args:
        chunk: Text chunk to extract concepts from
        threshold: Minimum score threshold for filtering (0.0 = no filtering)
        chunk_index: Index of this chunk in the document
        chunk_section: Section this chunk belongs to
        
    Returns:
        List of concept dictionaries compatible with existing pipeline
    """
    logger.info("🌍 🧬 UNIVERSAL CONCEPT EXTRACTION START")
    logger.info(f"🔬 Text length: {len(chunk)} chars, Threshold: {threshold}")
    
    # Extract concepts using universal method
    universal_concepts = extractConceptsUniversal(chunk, chunk_index, chunk_section)
    
    # Apply threshold filtering if specified
    if threshold > 0:
        pre_filter_count = len(universal_concepts)
        universal_concepts = [c for c in universal_concepts if c["score"] >= threshold]
        logger.info(f"🔧 Threshold filtering: {pre_filter_count} → {len(universal_concepts)} concepts")
    
    # Enhanced logging for surgical debugging compatibility
    if universal_concepts:
        method_counts = {}
        for concept in universal_concepts:
            methods = concept["source"]["methods"]
            method_counts[methods] = method_counts.get(methods, 0) + 1
        
        logger.info("🌍 EXTRACTION METHOD BREAKDOWN:")
        for method, count in method_counts.items():
            logger.info(f"  📊 {method}: {count} concepts")
    
    logger.info(f"🌍 🧬 UNIVERSAL EXTRACTION END: {len(universal_concepts)} concepts ready")
    return universal_concepts

# Legacy compatibility functions
def extractConceptsWithEmbeddings(chunk: str, embeddings_model=None, threshold: float = 0.0) -> List[Dict[str, Any]]:
    """Legacy compatibility - forwards to universal extraction"""
    logger.info("🔄 Legacy embeddings function called - forwarding to universal extraction")
    return extractConceptsFromDocument(chunk, threshold)

def extractConceptsWithVectorSearch(chunk: str, vector_db=None, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
    """Legacy compatibility - forwards to universal extraction"""
    logger.info("🔄 Legacy vector search function called - forwarding to universal extraction")
    return extractConceptsFromDocument(chunk, similarity_threshold)

# Configuration for different extraction methods
EXTRACTION_METHODS = {
    "simple": extractConceptsFromDocument,
    "universal": extractConceptsFromDocument,
    "embeddings": extractConceptsWithEmbeddings,
    "vector_search": extractConceptsWithVectorSearch
}

def extract_concepts_adaptive(chunk: str, method: str = "universal", threshold: float = 0.0, **kwargs) -> List[Dict[str, Any]]:
    """
    🌍 Adaptive concept extraction with universal default
    """
    if method not in EXTRACTION_METHODS:
        logger.warning(f"Unknown extraction method '{method}', using 'universal'")
        method = "universal"
    
    extraction_func = EXTRACTION_METHODS[method]
    
    try:
        if 'threshold' in extraction_func.__code__.co_varnames:
            concepts = extraction_func(chunk, threshold=threshold, **kwargs)
        else:
            concepts = extraction_func(chunk, **kwargs)
        
        logger.debug(f"🌍 Extracted {len(concepts)} concepts using {method} method")
        return concepts
    except Exception as e:
        logger.error(f"Error in {method} extraction: {e}")
        if method != "universal":
            logger.info("🔄 Falling back to universal extraction method")
            return extractConceptsFromDocument(chunk, threshold)
        return []

# Initialize models on import
try:
    _initialize_models()
    logger.info("🌍 🧬 UNIVERSAL CONCEPT EXTRACTION MODULE LOADED")
    logger.info("✅ Ready for cross-domain concept extraction: Science, Humanities, Arts, Philosophy, Mathematics!")
except Exception as e:
    logger.error(f"❌ Failed to initialize universal extraction models: {e}")
    logger.error("Please install required dependencies and models")
