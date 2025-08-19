"""
Vault Writer Utility
Converts extracted concepts to memory concepts with intelligent deduplication
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
import uuid
import re

try:
    import spacy
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from python.core.memory_types import MemoryConcept, Relationship, generate_concept_id

logger = logging.getLogger(__name__)


class VaultWriter:
    """
    Intelligent concept-to-memory converter with:
    - Lemmatized merging
    - Semantic similarity deduplication
    - Batch metadata tagging
    - Relationship preservation
    """
    
    # Class-level spaCy model cache
    _nlp = None
    _nlp_loaded = False
    
    # Configuration
    SIMILARITY_THRESHOLD = 0.92  # Semantic similarity threshold for deduplication
    
    @classmethod
    def _get_nlp(cls):
        """Get or load spaCy model with caching"""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available for lemmatization")
            return None
            
        if cls._nlp is None and not cls._nlp_loaded:
            try:
                # Try to load medium model first (has word vectors)
                try:
                    cls._nlp = spacy.load("en_core_web_md", disable=["ner", "parser"])
                    logger.info("✅ spaCy model loaded for VaultWriter: en_core_web_md (with word vectors)")
                except OSError:
                    # Fallback to small model
                    logger.warning("⚠️ en_core_web_md not found, falling back to en_core_web_sm")
                    logger.info("💡 Install with: python -m spacy download en_core_web_md")
                    cls._nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
                    logger.info("✅ spaCy model loaded for VaultWriter: en_core_web_sm (no word vectors)")
                
                cls._nlp_loaded = True
            except Exception as e:
                logger.error(f"❌ Failed to load any spaCy model: {e}")
                cls._nlp_loaded = True  # Don't retry
        
        return cls._nlp
    
    @staticmethod
    def _simple_normalize(text: str) -> str:
        """Simple normalization fallback when spaCy unavailable"""
        # Remove punctuation and lowercase
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        return normalized
    
    @classmethod
    def _lemmatized_key(cls, text: str) -> str:
        """Generate lemmatized key for concept merging"""
        nlp = cls._get_nlp()
        
        if nlp is None:
            # Fallback to simple normalization
            return cls._simple_normalize(text)
        
        try:
            doc = nlp(text)
            # Extract lemmas, keeping only alphabetic tokens
            lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
            return ' '.join(lemmas) if lemmas else cls._simple_normalize(text)
        except Exception as e:
            logger.warning(f"Lemmatization failed: {e}")
            return cls._simple_normalize(text)
    
    @classmethod
    def _calculate_similarity(cls, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        nlp = cls._get_nlp()
        
        if nlp is None or not hasattr(nlp, 'vocab'):
            return 0.0
        
        try:
            doc1 = nlp(text1)
            doc2 = nlp(text2)
            
            # Check if documents have vectors
            if doc1.has_vector and doc2.has_vector:
                # Suppress the warning about missing word vectors
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*W007.*")
                    return doc1.similarity(doc2)
            else:
                # Fallback to token overlap (Jaccard similarity)
                tokens1 = set(token.lemma_.lower() for token in doc1 if token.is_alpha)
                tokens2 = set(token.lemma_.lower() for token in doc2 if token.is_alpha)
                
                if not tokens1 or not tokens2:
                    return 0.0
                
                intersection = len(tokens1 & tokens2)
                union = len(tokens1 | tokens2)
                jaccard = intersection / union if union > 0 else 0.0
                
                # Boost similarity for very short texts with high overlap
                if len(tokens1) <= 3 and len(tokens2) <= 3 and intersection >= 2:
                    jaccard = min(jaccard * 1.5, 1.0)
                
                return jaccard
                
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0
    
    @classmethod
    def _merge_concepts_by_lemma(cls, concepts: List[Dict[str, Any]]) -> Dict[str, MemoryConcept]:
        """First pass: merge concepts with same lemmatized form"""
        merged = {}
        
        for concept in concepts:
            # Handle both dict and object formats
            if hasattr(concept, 'to_dict'):
                concept_dict = concept.to_dict()
            else:
                concept_dict = concept
            
            name = concept_dict.get('name', concept_dict.get('label', ''))
            if not name:
                continue
            
            # Generate lemmatized key
            key = cls._lemmatized_key(name)
            
            if key in merged:
                # Merge into existing concept
                existing = merged[key]
                
                # Update score (take maximum)
                new_score = concept_dict.get('score', 0.5)
                if new_score > (existing.score or 0):
                    existing.score = new_score
                
                # Merge relationships
                new_rels = concept_dict.get('metadata', {}).get('relationships', [])
                if new_rels:
                    # Convert to tuples for deduplication
                    existing_rel_keys = {(r['type'], r['target']) for r in existing.relationships}
                    
                    for rel in new_rels:
                        rel_key = (rel.get('type', ''), rel.get('target', ''))
                        if rel_key not in existing_rel_keys and rel_key[0] and rel_key[1]:
                            existing.relationships.append(rel)
                            existing_rel_keys.add(rel_key)
                
                # Merge methods
                new_method = concept_dict.get('method', '')
                if new_method and new_method not in existing.method:
                    existing.method += f"+{new_method}"
                    
            else:
                # Create new memory concept
                concept_id = generate_concept_id()
                
                # Extract relationships
                relationships = concept_dict.get('metadata', {}).get('relationships', [])
                
                merged[key] = MemoryConcept(
                    id=concept_id,
                    label=name,
                    method=concept_dict.get('method', 'unknown'),
                    score=concept_dict.get('score', 0.5),
                    relationships=relationships,
                    metadata=concept_dict.get('metadata', {})
                )
        
        return merged
    
    @classmethod
    def _semantically_deduplicate(cls, concepts: Dict[str, MemoryConcept]) -> Dict[str, MemoryConcept]:
        """Second pass: merge semantically similar concepts"""
        if not concepts:
            return concepts
        
        # Convert to list for processing
        concept_list = list(concepts.items())
        deduped = {}
        merged_indices = set()
        
        for i, (key1, concept1) in enumerate(concept_list):
            if i in merged_indices:
                continue
            
            # This concept will be kept
            deduped[key1] = concept1
            
            # Look for similar concepts
            for j in range(i + 1, len(concept_list)):
                if j in merged_indices:
                    continue
                
                key2, concept2 = concept_list[j]
                
                # Calculate similarity
                similarity = cls._calculate_similarity(concept1.label, concept2.label)
                
                if similarity >= cls.SIMILARITY_THRESHOLD:
                    # Merge concept2 into concept1
                    logger.debug(f"Merging '{concept2.label}' into '{concept1.label}' (similarity: {similarity:.3f})")
                    
                    concept1.merge_with(concept2)
                    merged_indices.add(j)
                    
                    # Update method to show merge
                    if concept2.method not in concept1.method:
                        concept1.method += f"+{concept2.method}"
        
        logger.info(f"🧬 Semantic deduplication: {len(concepts)} → {len(deduped)} concepts")
        return deduped
    
    @classmethod
    def convert(cls, 
                concepts: List[Any], 
                doc_id: str,
                additional_tags: Optional[List[str]] = None,
                metadata: Optional[Dict[str, Any]] = None) -> List[MemoryConcept]:
        """
        Convert extracted concepts to memory concepts with intelligent processing
        
        Args:
            concepts: List of concept dictionaries or objects
            doc_id: Document identifier for batch tagging
            additional_tags: Extra tags to add to all concepts
            metadata: Additional metadata to merge into all concepts
            
        Returns:
            List of deduplicated, enriched MemoryConcept instances
        """
        if not concepts:
            return []
        
        # Timestamp for batch
        timestamp = datetime.utcnow().isoformat()
        
        # First pass: Lemmatized merging
        logger.info(f"📚 Processing {len(concepts)} concepts for lemmatized merging...")
        merged = cls._merge_concepts_by_lemma(concepts)
        
        # Second pass: Semantic deduplication
        if SPACY_AVAILABLE and cls._get_nlp() is not None:
            logger.info("🧬 Performing semantic deduplication...")
            deduped = cls._semantically_deduplicate(merged)
        else:
            logger.info("⚠️ Skipping semantic deduplication (spaCy unavailable)")
            deduped = merged
        
        # Convert to list and enrich with batch metadata
        memory_concepts = []
        
        for concept in deduped.values():
            # Add batch metadata
            concept.metadata.update({
                'source': 'semantic_extraction',
                'doc_id': doc_id,
                'ingested_at': timestamp,
                'batch_id': f"batch_{uuid.uuid4().hex[:8]}"
            })
            
            # Add any additional metadata
            if metadata:
                concept.metadata.update(metadata)
            
            # Add document ID to tags if not present
            if doc_id not in concept.metadata.get('tags', []):
                concept.metadata.setdefault('tags', []).append(doc_id)
            
            # Add additional tags
            if additional_tags:
                existing_tags = set(concept.metadata.get('tags', []))
                for tag in additional_tags:
                    if tag not in existing_tags:
                        concept.metadata.setdefault('tags', []).append(tag)
            
            memory_concepts.append(concept)
        
        # Sort by score (highest first)
        memory_concepts.sort(key=lambda c: c.score or 0, reverse=True)
        
        logger.info(f"✅ VaultWriter produced {len(memory_concepts)} memory concepts from {len(concepts)} inputs")
        return memory_concepts
    
    @classmethod
    def batch_convert(cls,
                      document_concepts: Dict[str, List[Any]],
                      global_tags: Optional[List[str]] = None) -> List[MemoryConcept]:
        """
        Convert concepts from multiple documents
        
        Args:
            document_concepts: Dict mapping doc_id to concept lists
            global_tags: Tags to apply to all concepts
            
        Returns:
            Combined list of memory concepts from all documents
        """
        all_concepts = []
        
        for doc_id, concepts in document_concepts.items():
            doc_concepts = cls.convert(
                concepts=concepts,
                doc_id=doc_id,
                additional_tags=global_tags
            )
            all_concepts.extend(doc_concepts)
        
        # Final cross-document deduplication
        if all_concepts:
            # Create temporary dict for dedup
            temp_dict = {c.id: c for c in all_concepts}
            merged = cls._merge_concepts_by_lemma([c.to_dict() for c in all_concepts])
            
            if SPACY_AVAILABLE:
                final_deduped = cls._semantically_deduplicate(merged)
                all_concepts = list(final_deduped.values())
            else:
                all_concepts = list(merged.values())
        
        return all_concepts
