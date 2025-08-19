"""
Dynamic High-Fidelity Concept Extraction Service for TORI

This service removes the 10-concept cap and implements a comprehensive, 
unlimited concept extraction pipeline that captures ALL meaningful concepts
while maintaining high quality through multi-pass filtering, semantic 
clustering, weighted scoring, and co-occurrence analysis.

Key Features:
- NO CONCEPT CAPS - Dynamic thresholds instead of hard limits
- Multi-pass quality filtering (confidence, context bleed, domain validation)
- Semantic clustering to prevent redundancy
- Weighted scoring based on document structure
- Co-occurrence and graph-based importance ranking
- Rich metadata for ψMesh integration
- Comprehensive monitoring and logging
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import re
from collections import defaultdict, Counter
from pathlib import Path
import math

# ML and NLP imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Import existing TORI modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ingest_pdf'))

try:
    from ingest_pdf.extract_blocks import extract_concept_blocks
    from ingest_pdf.features import build_feature_matrix, _tokenise
    from ingest_pdf.spectral import spectral_embed
    from ingest_pdf.clustering import run_oscillator_clustering, cluster_cohesion
    from ingest_pdf.scoring import score_clusters, resonance_score, narrative_centrality, build_cluster_adjacency
    from ingest_pdf.keywords import extract_keywords
    from ingest_pdf.models import ConceptTuple
    TORI_MODULES_AVAILABLE = True
except ImportError:
    TORI_MODULES_AVAILABLE = False
    logging.warning("TORI ingest_pdf modules not available, using fallback methods")

# Configure logging
logger = logging.getLogger("tori.dynamic_concept_extractor")

class ConceptSource(str, Enum):
    """Sources where concepts can be extracted from."""
    TITLE = "title"
    ABSTRACT = "abstract" 
    SECTION_HEADER = "section_header"
    BODY_TEXT = "body_text"
    CAPTION = "caption"
    EQUATION = "equation"
    METADATA = "metadata"
    CITATION = "citation"

class ConceptQuality(str, Enum):
    """Quality levels for extracted concepts."""
    EXCELLENT = "excellent"  # 0.9+
    GOOD = "good"           # 0.75-0.89
    FAIR = "fair"           # 0.6-0.74
    POOR = "poor"           # <0.6

@dataclass
class ProcessedConcept:
    """Fully processed concept with all metadata for ψMesh integration."""
    # Core concept data
    concept: str
    confidence: float
    quality: ConceptQuality
    
    # Source attribution
    sources: List[str]  # e.g., ["title", "abstract", "section: 2.3"]
    frequency: int
    document_positions: List[int]
    
    # Rich metadata for ψMesh
    document_id: str
    embedding: List[float]  # Vector embedding
    tags: List[str]  # Domain tags like ["quantum", "physics", "anyons"]
    
    # Importance metrics
    tfidf_score: float
    cooccurrence_score: float
    graph_centrality: float
    final_score: float
    
    # Context and relationships
    contexts: List[str]
    related_concepts: List[str]
    
    # Processing metadata
    extraction_timestamp: str
    extraction_method: str

@dataclass
class ExtractionStats:
    """Statistics for concept extraction monitoring."""
    total_concepts_found: int
    concepts_after_filtering: int
    concepts_after_clustering: int
    concepts_dropped_by_reason: Dict[str, int]
    confidence_distribution: Dict[str, int]
    processing_time: float
    document_info: Dict[str, Any]

class DynamicConceptExtractor:
    """
    Advanced concept extraction service with no hard caps and comprehensive
    quality filtering, clustering, weighted scoring, and co-occurrence analysis.
    """
    
    def __init__(self):
        """Initialize the dynamic concept extractor."""
        # Configuration - NO HARD CAPS!
        self.config = {
            # Dynamic thresholds instead of caps
            "min_confidence_score": 0.75,
            "min_quality_threshold": 0.6,
            "context_bleed_threshold": 0.8,
            "domain_relevance_threshold": 0.7,
            "semantic_similarity_threshold": 0.15,
            
            # Weighted scoring configuration
            "source_weights": {
                "title": 0.9,
                "abstract": 0.8,
                "section_header": 0.7,
                "body_text": 0.5,
                "caption": 0.6,
                "equation": 0.8,
                "metadata": 0.4,
                "citation": 0.6
            },
            
            # Co-occurrence analysis
            "min_cooccurrence_frequency": 2,
            "cooccurrence_window": 5,
            
            # Clustering parameters
            "dbscan_eps": 0.15,
            "dbscan_min_samples": 1,
            "merge_duplicate_threshold": 0.6
        }
        
        # Domain vocabulary for validation
        self.domain_terms = self._load_domain_vocabulary()
        
        # Context bleed detection patterns
        self.context_bleed_patterns = [
            r'\b(document|file|page|section|chapter|pdf|text)\b',
            r'\b(above|below|following|preceding|next|previous)\b',
            r'\b(figure|table|equation|reference|citation)\b',
            r'\b(click|select|choose|view|see|read)\b'
        ]
        
        logger.info("🚀 Dynamic Concept Extractor initialized - NO CONCEPT CAPS!")
    
    def _load_domain_vocabulary(self) -> Set[str]:
        """Load domain-specific vocabulary for concept validation."""
        # Comprehensive scientific/technical vocabulary
        return {
            # Physics & Math
            "algorithm", "analysis", "calculation", "computation", "dimension",
            "dynamic", "equation", "function", "geometry", "hypothesis",
            "matrix", "method", "model", "parameter", "probability",
            "quantum", "statistical", "system", "theory", "variable",
            "resonance", "oscillation", "frequency", "amplitude", "phase",
            "spectrum", "eigenvalue", "eigenvector", "fourier", "transform",
            
            # Computer Science & AI
            "architecture", "artificial", "automation", "clustering", "computer",
            "file_storage", "intelligence", "learning", "machine", "network",
            "optimization", "processing", "programming", "recognition", "software",
            "embedding", "vector", "neural", "deep", "training", "model",
            
            # General Scientific
            "abstract", "approach", "concept", "framework", "mechanism",
            "phenomenon", "principle", "process", "property", "relationship",
            "research", "solution", "structure", "technique", "understanding",
            "experiment", "observation", "measurement", "correlation", "causation"
        }
    
    async def extract_concepts_unlimited(self, payload, use_advanced_pipeline: bool = True) -> Tuple[List[ProcessedConcept], ExtractionStats]:
        """
        Extract ALL meaningful concepts with no caps using dynamic thresholds.
        
        Args:
            payload: ParsedPayload with extracted text and structure
            use_advanced_pipeline: Use TORI's advanced pipeline if available
            
        Returns:
            Tuple of (processed_concepts, extraction_statistics)
        """
        start_time = datetime.now()
        
        logger.info(f"🔍 Starting unlimited concept extraction (Advanced: {use_advanced_pipeline and TORI_MODULES_AVAILABLE})")
        
        try:
            if use_advanced_pipeline and TORI_MODULES_AVAILABLE:
                # Use TORI's advanced pipeline
                concepts, stats = await self._extract_with_tori_pipeline(payload)
            else:
                # Use enhanced fallback pipeline
                concepts, stats = await self._extract_with_enhanced_pipeline(payload)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            stats.processing_time = processing_time
            
            logger.info(f"🎉 Extraction complete: {len(concepts)} high-quality concepts in {processing_time:.2f}s")
            await self._log_extraction_results(concepts, stats)
            
            return concepts, stats
            
        except Exception as e:
            logger.error(f"❌ Concept extraction failed: {str(e)}")
            raise
    
    async def _extract_with_tori_pipeline(self, payload) -> Tuple[List[ProcessedConcept], ExtractionStats]:
        """Extract concepts using TORI's advanced pipeline."""
        try:
            # Split text into blocks for TORI processing
            text = payload.extracted_text
            if not text:
                return [], ExtractionStats(0, 0, 0, {}, {}, 0.0, {})
            
            # Create blocks from segments or split text
            if payload.raw_segments:
                blocks = [segment['text'] for segment in payload.raw_segments if segment.get('text')]
            else:
                # Fallback: split by paragraphs
                blocks = [block.strip() for block in text.split('\n\n') if block.strip()]
            
            if not blocks:
                return [], ExtractionStats(0, 0, 0, {}, {}, 0.0, {})
            
            # PART 1: Use TORI's feature extraction
            feats, vocab = build_feature_matrix(blocks, vocab_size=2000)
            emb = spectral_embed(feats, k=32)
            labels = run_oscillator_clustering(emb)
            
            # PART 2: Extract raw concepts from clusters - NO CAPS!
            raw_concepts = []
            adj = build_cluster_adjacency(labels, emb)
            
            for cluster_id in set(labels):
                cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
                if len(cluster_indices) == 0:
                    continue
                
                cluster_blocks = [blocks[i] for i in cluster_indices]
                other_blocks = [blocks[i] for i in range(len(blocks)) if i not in cluster_indices]
                
                # Extract keywords
                keywords = extract_keywords(cluster_blocks, other_blocks, n=5)
                if not keywords:
                    continue
                
                concept_name = " ".join(w.capitalize() for w in keywords)
                
                # Calculate scores
                res_score = resonance_score(cluster_indices, emb)
                cent_score = narrative_centrality(cluster_indices, adj)
                coherence = cluster_cohesion(emb, cluster_indices)
                
                # Calculate TF-IDF
                cluster_tfidf = feats[cluster_indices].mean(axis=0)
                tfidf_score = float(np.linalg.norm(cluster_tfidf))
                
                # Get context
                context = cluster_blocks[0] if cluster_blocks else ""
                
                raw_concepts.append({
                    'name': concept_name,
                    'keywords': keywords,
                    'confidence': min(1.0, (res_score + cent_score + coherence) / 3),
                    'tfidf_score': tfidf_score,
                    'resonance_score': res_score,
                    'centrality_score': cent_score,
                    'coherence_score': coherence,
                    'context': context,
                    'cluster_indices': cluster_indices,
                    'embedding': cluster_tfidf.tolist()
                })
            
            # PART 3: Apply multi-pass filtering - NO CAPS!
            filtered_concepts = await self._apply_advanced_filtering(raw_concepts, blocks)
            
            # PART 4: Semantic clustering to remove redundancy
            clustered_concepts = await self._apply_semantic_clustering(filtered_concepts)
            
            # PART 5: Enhanced scoring and ranking
            final_concepts = await self._apply_enhanced_scoring_and_metadata(
                clustered_concepts, payload
            )
            
            # Generate statistics
            stats = self._generate_stats(raw_concepts, filtered_concepts, clustered_concepts, final_concepts, payload)
            
            return final_concepts, stats
            
        except Exception as e:
            logger.error(f"❌ TORI pipeline extraction failed: {str(e)}")
            # Fallback to enhanced pipeline
            return await self._extract_with_enhanced_pipeline(payload)
    
    async def _extract_with_enhanced_pipeline(self, payload) -> Tuple[List[ProcessedConcept], ExtractionStats]:
        """Enhanced fallback pipeline when TORI modules unavailable."""
        try:
            text = payload.extracted_text
            if not text:
                return [], ExtractionStats(0, 0, 0, {}, {}, 0.0, {})
            
            # PART 1: Extract raw concepts using multiple methods
            raw_concepts = []
            
            # Method 1: TF-IDF based extraction
            tfidf_concepts = await self._extract_tfidf_concepts(text, payload)
            raw_concepts.extend(tfidf_concepts)
            
            # Method 2: Structure-based extraction
            structure_concepts = await self._extract_structure_concepts(payload)
            raw_concepts.extend(structure_concepts)
            
            # Method 3: Domain-specific extraction
            domain_concepts = await self._extract_domain_concepts(text)
            raw_concepts.extend(domain_concepts)
            
            # PART 2: Multi-pass filtering - NO CAPS!
            filtered_concepts = await self._apply_advanced_filtering(raw_concepts, [text])
            
            # PART 3: Semantic clustering
            clustered_concepts = await self._apply_semantic_clustering(filtered_concepts)
            
            # PART 4: Enhanced scoring and metadata
            final_concepts = await self._apply_enhanced_scoring_and_metadata(
                clustered_concepts, payload
            )
            
            # Generate statistics
            stats = self._generate_stats(raw_concepts, filtered_concepts, clustered_concepts, final_concepts, payload)
            
            return final_concepts, stats
            
        except Exception as e:
            logger.error(f"❌ Enhanced pipeline extraction failed: {str(e)}")
            return [], ExtractionStats(0, 0, 0, {"extraction_error": 1}, {}, 0.0, {})
    
    async def _extract_tfidf_concepts(self, text: str, payload) -> List[Dict[str, Any]]:
        """Extract concepts using TF-IDF analysis."""
        try:
            # Create documents from segments or paragraphs
            if payload.raw_segments:
                documents = [seg['text'] for seg in payload.raw_segments if seg.get('text')]
            else:
                documents = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            if len(documents) < 2:
                documents = [text]  # Single document fallback
            
            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 3),
                stop_words='english',
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract high-scoring terms
            concepts = []
            scores = tfidf_matrix.sum(axis=0).A1
            
            for idx, score in enumerate(scores):
                if score > 0.1:  # Dynamic threshold, not hard cap
                    term = feature_names[idx]
                    if self._is_valid_concept_term(term):
                        concepts.append({
                            'name': term.title(),
                            'keywords': term.split(),
                            'confidence': min(1.0, score * 2),
                            'tfidf_score': float(score),
                            'extraction_method': 'tfidf',
                            'context': self._find_context(term, documents[0])
                        })
            
            return concepts
            
        except Exception as e:
            logger.error(f"TF-IDF extraction error: {str(e)}")
            return []
    
    async def _extract_structure_concepts(self, payload) -> List[Dict[str, Any]]:
        """Extract concepts from document structure elements."""
        concepts = []
        
        try:
            for element in payload.structure_elements:
                if element.get('type') == 'heading' and element.get('text'):
                    heading = element['text'].strip()
                    if self._is_valid_concept_term(heading) and len(heading.split()) <= 5:
                        concepts.append({
                            'name': heading,
                            'keywords': heading.lower().split(),
                            'confidence': 0.8,
                            'extraction_method': 'heading_analysis',
                            'source_element': 'heading',
                            'context': f"Document heading: {heading}"
                        })
                
                elif element.get('type') == 'json_key' and element.get('key'):
                    key = element['key']
                    if self._is_valid_concept_term(key):
                        concepts.append({
                            'name': key.replace('_', ' ').title(),
                            'keywords': key.replace('_', ' ').lower().split(),
                            'confidence': 0.6,
                            'extraction_method': 'json_structure',
                            'source_element': 'json_key',
                            'context': f"JSON key: {key}"
                        })
            
            return concepts
            
        except Exception as e:
            logger.error(f"Structure extraction error: {str(e)}")
            return []
    
    async def _extract_domain_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract domain-specific concepts."""
        concepts = []
        text_lower = text.lower()
        
        try:
            # Extract domain terms that appear in text
            for term in self.domain_terms:
                if term in text_lower:
                    # Count frequency
                    frequency = text_lower.count(term)
                    if frequency >= 2:  # Appears multiple times
                        concepts.append({
                            'name': term.title(),
                            'keywords': term.split(),
                            'confidence': min(1.0, 0.6 + (frequency * 0.1)),
                            'frequency': frequency,
                            'extraction_method': 'domain_vocabulary',
                            'context': self._find_context(term, text)
                        })
            
            # Extract potential technical terms (capitalized multi-word phrases)
            tech_pattern = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3}\b'
            tech_terms = re.findall(tech_pattern, text)
            
            for term in set(tech_terms):
                if len(term.split()) >= 2 and len(term) > 5:
                    frequency = text.count(term)
                    if frequency >= 2:
                        concepts.append({
                            'name': term,
                            'keywords': term.lower().split(),
                            'confidence': min(1.0, 0.5 + (frequency * 0.1)),
                            'frequency': frequency,
                            'extraction_method': 'technical_terms',
                            'context': self._find_context(term, text)
                        })
            
            return concepts
            
        except Exception as e:
            logger.error(f"Domain extraction error: {str(e)}")
            return []
    
    async def _apply_advanced_filtering(self, raw_concepts: List[Dict[str, Any]], texts: List[str]) -> List[Dict[str, Any]]:
        """Apply multi-pass filtering to remove low-quality concepts."""
        if not raw_concepts:
            return []
        
        filtered = []
        dropped_reasons = defaultdict(int)
        
        for concept in raw_concepts:
            # Filter 1: Confidence threshold
            if concept.get('confidence', 0) < self.config['min_confidence_score']:
                dropped_reasons['low_confidence'] += 1
                continue
            
            # Filter 2: Context bleed detection
            if self._is_context_bleed(concept):
                dropped_reasons['context_bleed'] += 1
                continue
            
            # Filter 3: Domain relevance
            if not self._is_domain_relevant(concept):
                dropped_reasons['not_domain_relevant'] += 1
                continue
            
            # Filter 4: Term validity
            if not self._is_valid_concept_term(concept.get('name', '')):
                dropped_reasons['invalid_term'] += 1
                continue
            
            filtered.append(concept)
        
        logger.info(f"🔍 Filtering: {len(raw_concepts)} → {len(filtered)} concepts")
        for reason, count in dropped_reasons.items():
            logger.info(f"   Dropped {count} for {reason}")
        
        return filtered
    
    async def _apply_semantic_clustering(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply semantic clustering to merge similar concepts."""
        if len(concepts) <= 1:
            return concepts
        
        try:
            # Create embeddings for clustering
            concept_texts = [concept.get('name', '') + ' ' + ' '.join(concept.get('keywords', [])) 
                           for concept in concepts]
            
            # Simple similarity-based clustering
            clustered = []
            processed = set()
            
            for i, concept in enumerate(concepts):
                if i in processed:
                    continue
                
                # Find similar concepts
                similar_indices = [i]
                concept_text = concept_texts[i].lower()
                
                for j, other_concept in enumerate(concepts[i+1:], i+1):
                    if j in processed:
                        continue
                    
                    other_text = concept_texts[j].lower()
                    
                    # Check similarity
                    if self._are_concepts_similar(concept_text, other_text):
                        similar_indices.append(j)
                        processed.add(j)
                
                processed.update(similar_indices)
                
                # Merge similar concepts
                if len(similar_indices) > 1:
                    merged_concept = self._merge_concepts([concepts[idx] for idx in similar_indices])
                    clustered.append(merged_concept)
                else:
                    clustered.append(concept)
            
            logger.info(f"🔗 Clustering: {len(concepts)} → {len(clustered)} concepts after merging")
            return clustered
            
        except Exception as e:
            logger.error(f"Clustering error: {str(e)}")
            return concepts
    
    async def _apply_enhanced_scoring_and_metadata(self, concepts: List[Dict[str, Any]], payload) -> List[ProcessedConcept]:
        """Apply enhanced scoring and build rich metadata."""
        processed_concepts = []
        
        try:
            # Build co-occurrence graph for centrality analysis
            cooccurrence_graph = self._build_cooccurrence_graph(concepts, payload.extracted_text)
            
            for concept in concepts:
                # Calculate enhanced scores
                tfidf_score = concept.get('tfidf_score', 0.0)
                resonance_score = concept.get('resonance_score', 0.0)
                centrality_score = concept.get('centrality_score', 0.0)
                
                # Calculate graph centrality
                graph_centrality = self._calculate_graph_centrality(concept['name'], cooccurrence_graph)
                
                # Calculate co-occurrence score
                cooccurrence_score = self._calculate_cooccurrence_score(concept['name'], payload.extracted_text)
                
                # Final weighted score
                final_score = (
                    0.3 * tfidf_score +
                    0.2 * resonance_score +
                    0.2 * centrality_score +
                    0.15 * graph_centrality +
                    0.15 * cooccurrence_score
                )
                
                # Determine quality level
                if final_score >= 0.9:
                    quality = ConceptQuality.EXCELLENT
                elif final_score >= 0.75:
                    quality = ConceptQuality.GOOD
                elif final_score >= 0.6:
                    quality = ConceptQuality.FAIR
                else:
                    quality = ConceptQuality.POOR
                
                # Extract sources and positions
                sources = self._extract_concept_sources(concept, payload)
                positions = self._extract_concept_positions(concept['name'], payload.extracted_text)
                
                # Generate tags
                tags = self._generate_concept_tags(concept)
                
                # Build processed concept with rich metadata
                processed_concept = ProcessedConcept(
                    concept=concept['name'],
                    confidence=concept.get('confidence', 0.0),
                    quality=quality,
                    sources=sources,
                    frequency=concept.get('frequency', 1),
                    document_positions=positions,
                    document_id=payload.document_id,
                    embedding=concept.get('embedding', []),
                    tags=tags,
                    tfidf_score=tfidf_score,
                    cooccurrence_score=cooccurrence_score,
                    graph_centrality=graph_centrality,
                    final_score=final_score,
                    contexts=[concept.get('context', '')],
                    related_concepts=self._find_related_concepts(concept['name'], concepts),
                    extraction_timestamp=datetime.now(timezone.utc).isoformat(),
                    extraction_method=concept.get('extraction_method', 'dynamic_extraction')
                )
                
                processed_concepts.append(processed_concept)
            
            # Sort by final score (best first) - NO CAPS APPLIED!
            processed_concepts.sort(key=lambda x: x.final_score, reverse=True)
            
            return processed_concepts
            
        except Exception as e:
            logger.error(f"Enhanced scoring error: {str(e)}")
            return []
    
    def _is_context_bleed(self, concept: Dict[str, Any]) -> bool:
        """Check if concept is context bleed (UI/navigation terms)."""
        name = concept.get('name', '').lower()
        
        for pattern in self.context_bleed_patterns:
            if re.search(pattern, name):
                return True
        
        return False
    
    def _is_domain_relevant(self, concept: Dict[str, Any]) -> bool:
        """Check if concept is domain relevant."""
        name = concept.get('name', '').lower()
        keywords = concept.get('keywords', [])
        
        # Check if any keyword is in domain vocabulary
        for keyword in keywords:
            if keyword.lower() in self.domain_terms:
                return True
        
        # Check for technical indicators
        technical_indicators = ['system', 'method', 'analysis', 'process', 'model', 'algorithm']
        for indicator in technical_indicators:
            if indicator in name:
                return True
        
        # Check length and complexity
        if len(name.split()) >= 2 and len(name) > 8:
            return True
        
        return False
    
    def _is_valid_concept_term(self, term: str) -> bool:
        """Check if term is valid for concept extraction."""
        if not term or len(term) < 3:
            return False
        
        # Remove common stop words and short terms
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = term.lower().split()
        
        if len(words) == 1 and words[0] in stop_words:
            return False
        
        # Check for numeric-only terms
        if term.replace(' ', '').isdigit():
            return False
        
        # Check for single characters
        if len(term.replace(' ', '')) <= 2:
            return False
        
        return True
    
    def _are_concepts_similar(self, text1: str, text2: str) -> bool:
        """Check if two concepts are semantically similar."""
        # Simple similarity check based on word overlap
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard_similarity = overlap / union if union > 0 else 0
        
        return jaccard_similarity > self.config['merge_duplicate_threshold']
    
    def _merge_concepts(self, similar_concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge similar concepts into one."""
        # Use the concept with highest confidence as base
        base_concept = max(similar_concepts, key=lambda x: x.get('confidence', 0))
        
        # Merge data
        merged = base_concept.copy()
        merged['confidence'] = max(c.get('confidence', 0) for c in similar_concepts)
        merged['frequency'] = sum(c.get('frequency', 1) for c in similar_concepts)
        
        # Combine keywords
        all_keywords = []
        for concept in similar_concepts:
            all_keywords.extend(concept.get('keywords', []))
        merged['keywords'] = list(set(all_keywords))
        
        # Combine contexts
        all_contexts = [c.get('context', '') for c in similar_concepts if c.get('context')]
        merged['context'] = ' | '.join(all_contexts[:3])  # Limit context length
        
        return merged
    
    def _build_cooccurrence_graph(self, concepts: List[Dict[str, Any]], text: str) -> nx.Graph:
        """Build co-occurrence graph for centrality analysis."""
        graph = nx.Graph()
        
        # Add concept nodes
        for concept in concepts:
            graph.add_node(concept['name'])
        
        # Add edges based on co-occurrence
        window_size = self.config['cooccurrence_window']
        text_lower = text.lower()
        
        for i, concept1 in enumerate(concepts):
            name1 = concept1['name'].lower()
            for concept2 in concepts[i+1:]:
                name2 = concept2['name'].lower()
                
                # Check co-occurrence within window
                if self._check_cooccurrence(name1, name2, text_lower, window_size):
                    graph.add_edge(concept1['name'], concept2['name'])
        
        return graph
    
    def _check_cooccurrence(self, term1: str, term2: str, text: str, window_size: int) -> bool:
        """Check if two terms co-occur within a window."""
        words = text.split()
        term1_positions = [i for i, word in enumerate(words) if term1 in word]
        term2_positions = [i for i, word in enumerate(words) if term2 in word]
        
        for pos1 in term1_positions:
            for pos2 in term2_positions:
                if abs(pos1 - pos2) <= window_size:
                    return True
        
        return False
    
    def _calculate_graph_centrality(self, concept_name: str, graph: nx.Graph) -> float:
        """Calculate graph centrality for concept."""
        try:
            if concept_name in graph:
                return nx.degree_centrality(graph).get(concept_name, 0.0)
            return 0.0
        except:
            return 0.0
    
    def _calculate_cooccurrence_score(self, concept_name: str, text: str) -> float:
        """Calculate co-occurrence score for concept."""
        text_lower = text.lower()
        concept_lower = concept_name.lower()
        
        # Count occurrences
        occurrences = text_lower.count(concept_lower)
        
        # Normalize by text length
        text_length = len(text.split())
        if text_length == 0:
            return 0.0
        
        return min(1.0, occurrences / (text_length / 1000))  # Normalize per 1000 words
    
    def _extract_concept_sources(self, concept: Dict[str, Any], payload) -> List[str]:
        """Extract source attribution for concept."""
        sources = []
        concept_name = concept['name'].lower()
        
        # Check in different document sections
        if concept.get('source_element') == 'heading':
            sources.append('section_header')
        
        # Check in structure elements
        for element in payload.structure_elements:
            if element.get('text') and concept_name in element['text'].lower():
                if element.get('type') == 'heading':
                    sources.append('section_header')
                elif element.get('type') == 'paragraph':
                    sources.append('body_text')
        
        # Default source
        if not sources:
            sources.append('body_text')
        
        return list(set(sources))
    
    def _extract_concept_positions(self, concept_name: str, text: str) -> List[int]:
        """Extract positions where concept appears in text."""
        positions = []
        concept_lower = concept_name.lower()
        text_lower = text.lower()
        
        start = 0
        while True:
            pos = text_lower.find(concept_lower, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        return positions
    
    def _generate_concept_tags(self, concept: Dict[str, Any]) -> List[str]:
        """Generate domain tags for concept."""
        tags = []
        name = concept['name'].lower()
        keywords = concept.get('keywords', [])
        
        # Domain-specific tagging
        if any(word in name for word in ['quantum', 'physics', 'particle']):
            tags.append('physics')
        if any(word in name for word in ['neural', 'machine', 'learning', 'ai']):
            tags.append('artificial_intelligence')
        if any(word in name for word in ['algorithm', 'computation', 'software']):
            tags.append('computer_science')
        if any(word in name for word in ['analysis', 'statistical', 'data']):
            tags.append('data_science')
        if any(word in name for word in ['network', 'graph', 'system']):
            tags.append('systems')
        
        # Add extraction method as tag
        if concept.get('extraction_method'):
            tags.append(concept['extraction_method'])
        
        return list(set(tags))
    
    def _find_related_concepts(self, concept_name: str, all_concepts: List[Dict[str, Any]]) -> List[str]:
        """Find related concepts based on keyword overlap."""
        related = []
        concept_words = set(concept_name.lower().split())
        
        for other_concept in all_concepts:
            if other_concept['name'] == concept_name:
                continue
            
            other_words = set(other_concept['name'].lower().split())
            overlap = len(concept_words.intersection(other_words))
            
            if overlap > 0:
                related.append(other_concept['name'])
        
        return related[:5]  # Limit to 5 related concepts
    
    def _find_context(self, term: str, text: str) -> str:
        """Find context sentence for a term."""
        sentences = text.split('.')
        term_lower = term.lower()
        
        for sentence in sentences:
            if term_lower in sentence.lower():
                return sentence.strip()[:200]  # Limit context length
        
        return f"Term '{term}' found in document"
    
    def _generate_stats(self, raw_concepts, filtered_concepts, clustered_concepts, final_concepts, payload) -> ExtractionStats:
        """Generate comprehensive extraction statistics."""
        # Calculate dropped reasons
        dropped_reasons = {
            'low_confidence': len(raw_concepts) - len(filtered_concepts),
            'semantic_clustering': len(filtered_concepts) - len(clustered_concepts)
        }
        
        # Calculate confidence distribution
        confidence_dist = {}
        for concept in final_concepts:
            conf_bucket = f"{int(concept.confidence * 10) * 10}-{int(concept.confidence * 10) * 10 + 10}%"
            confidence_dist[conf_bucket] = confidence_dist.get(conf_bucket, 0) + 1
        
        return ExtractionStats(
            total_concepts_found=len(raw_concepts),
            concepts_after_filtering=len(filtered_concepts),
            concepts_after_clustering=len(clustered_concepts),
            concepts_dropped_by_reason=dropped_reasons,
            confidence_distribution=confidence_dist,
            processing_time=0.0,  # Will be set by caller
            document_info={
                'text_length': len(payload.extracted_text),
                'segments': len(payload.raw_segments),
                'structure_elements': len(payload.structure_elements)
            }
        )
    
    async def _log_extraction_results(self, concepts: List[ProcessedConcept], stats: ExtractionStats):
        """Log comprehensive extraction results."""
        logger.info("🎯 CONCEPT EXTRACTION COMPLETE - UNLIMITED RESULTS")
        logger.info(f"📊 Extracted {len(concepts)} concepts (NO CAPS APPLIED!)")
        logger.info(f"⏱️  Processing time: {stats.processing_time:.2f}s")
        
        # Log quality distribution
        quality_dist = {}
        for concept in concepts:
            quality_dist[concept.quality] = quality_dist.get(concept.quality, 0) + 1
        
        logger.info("🏆 Quality Distribution:")
        for quality, count in quality_dist.items():
            logger.info(f"   {quality}: {count}")
        
        # Log top concepts
        logger.info("🔥 Top 10 Concepts by Score:")
        for i, concept in enumerate(concepts[:10]):
            logger.info(f"   {i+1}. {concept.concept} (score: {concept.final_score:.3f}, confidence: {concept.confidence:.3f})")
        
        # Log dropped concepts
        logger.info("🗑️  Concepts Dropped:")
        for reason, count in stats.concepts_dropped_by_reason.items():
            logger.info(f"   {reason}: {count}")


# Global extractor instance
dynamic_extractor = DynamicConceptExtractor()
