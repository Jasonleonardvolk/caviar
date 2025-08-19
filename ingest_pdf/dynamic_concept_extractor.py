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

# ML and NLP imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Import existing TORI modules
try:
    # Try absolute import first
    from extract_blocks import extract_concept_blocks
except ImportError:
    # Fallback to relative import
    from .extract_blocks import extract_concept_blocks
try:
    # Try absolute import first
    from features import build_feature_matrix, _tokenise
except ImportError:
    # Fallback to relative import
    from .features import build_feature_matrix, _tokenise
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
    from scoring import score_clusters, resonance_score, narrative_centrality, build_cluster_adjacency
except ImportError:
    # Fallback to relative import
    from .scoring import score_clusters, resonance_score, narrative_centrality, build_cluster_adjacency
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
class TextChunk:
    """Text chunk with metadata for concept extraction."""
    text: str
    source: ConceptSource
    position: int
    length: int
    weight: float
    context: Dict[str, Any] = None

@dataclass 
class RawConcept:
    """Raw extracted concept before processing."""
    term: str
    confidence: float
    chunk_id: str
    position: int
    context: str
    tfidf_score: float
    embedding: Optional[np.ndarray] = None

@dataclass
class ProcessedConcept:
    """Fully processed concept with all metadata."""
    term: str
    confidence: float
    quality: ConceptQuality
    
    # Sources and frequency
    sources: List[ConceptSource]
    frequency: int
    document_positions: List[int]
    
    # Semantic information
    embedding: np.ndarray
    semantic_cluster: Optional[str]
    related_concepts: List[str]
    
    # Importance metrics
    tfidf_score: float
    cooccurrence_score: float
    graph_centrality: float
    weighted_score: float
    final_score: float
    
    # Context and metadata
    contexts: List[str]
    tags: List[str]
    domain_classification: str
    
    # ψMesh integration data
    concept_id: str
    document_id: str
    extraction_timestamp: datetime
    metadata: Dict[str, Any]

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
                ConceptSource.TITLE: 0.9,
                ConceptSource.ABSTRACT: 0.8,
                ConceptSource.SECTION_HEADER: 0.7,
                ConceptSource.BODY_TEXT: 0.5,
                ConceptSource.CAPTION: 0.6,
                ConceptSource.EQUATION: 0.8,
                ConceptSource.METADATA: 0.4,
                ConceptSource.CITATION: 0.6
            },
            
            # Co-occurrence analysis
            "min_cooccurrence_frequency": 2,
            "cooccurrence_window": 5,
            
            # Clustering parameters
            "dbscan_eps": 0.15,
            "dbscan_min_samples": 2,
            "merge_duplicate_threshold": 0.8
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
        
        # Extraction statistics
        self.extraction_stats = None
        
        logger.info("🚀 Dynamic Concept Extractor initialized - NO CONCEPT CAPS!")
    
    def _load_domain_vocabulary(self) -> Set[str]:
        """Load domain-specific vocabulary for concept validation."""
        # In production, this would load from a comprehensive domain vocabulary
        # For now, using a basic scientific/technical vocabulary
        basic_vocab = {
            # Physics & Math
            "algorithm", "analysis", "calculation", "computation", "dimension",
            "dynamic", "equation", "function", "geometry", "hypothesis",
            "matrix", "method", "model", "parameter", "probability",
            "quantum", "statistical", "system", "theory", "variable",
            
            # Computer Science
            "architecture", "artificial", "automation", "clustering", "computer",
            "file_storage", "intelligence", "learning", "machine", "network",
            "optimization", "processing", "programming", "recognition", "software",
            
            # General Scientific
            "abstract", "approach", "concept", "framework", "mechanism",
            "phenomenon", "principle", "process", "property", "relationship",
            "research", "solution", "structure", "technique", "understanding"
        }
        return basic_vocab
    
    async def extract_concepts_unlimited(
        self,
        pdf_path: str,
        document_id: Optional[str] = None,
        enable_monitoring: bool = True
    ) -> Tuple[List[ProcessedConcept], ExtractionStats]:
        """
        Extract ALL meaningful concepts from a document with no caps.
        
        Args:
            pdf_path: Path to the PDF document
            document_id: Optional document identifier
            enable_monitoring: Enable comprehensive monitoring and logging
            
        Returns:
            Tuple of (processed_concepts, extraction_statistics)
        """
        start_time = datetime.now()
        document_id = document_id or str(uuid.uuid4())
        
        logger.info(f"🔍 Starting unlimited concept extraction for: {pdf_path}")
        
        try:
            # PART 1: Extract text with source attribution
            text_chunks = await self._extract_structured_text(pdf_path)
            logger.info(f"📄 Extracted {len(text_chunks)} text chunks")
            
            # PART 2: Initial concept extraction (no caps applied)
            raw_concepts = await self._extract_raw_concepts(text_chunks)
            logger.info(f"🧠 Found {len(raw_concepts)} raw concepts")
            
            # PART 3: Multi-pass quality filtering
            filtered_concepts = await self._apply_multi_pass_filtering(raw_concepts)
            logger.info(f"✅ {len(filtered_concepts)} concepts passed quality filters")
            
            # PART 4: Semantic clustering to prevent redundancy
            clustered_concepts = await self._apply_semantic_clustering(filtered_concepts)
            logger.info(f"🔗 {len(clustered_concepts)} concepts after redundancy removal")
            
            # PART 5: Enhanced scoring with all metrics
            scored_concepts = await self._apply_enhanced_scoring(clustered_concepts, text_chunks)
            logger.info(f"📊 Scored {len(scored_concepts)} concepts")
            
            # PART 6: Build concept relationships and metadata
            final_concepts = await self._build_concept_metadata(
                scored_concepts, text_chunks, document_id, pdf_path
            )
            
            # PART 7: Generate extraction statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            extraction_stats = self._generate_extraction_stats(
                raw_concepts, filtered_concepts, clustered_concepts, 
                final_concepts, processing_time, pdf_path
            )
            
            # PART 8: Log comprehensive results
            if enable_monitoring:
                await self._log_extraction_results(final_concepts, extraction_stats)
            
            logger.info(f"🎉 Extraction complete: {len(final_concepts)} high-quality concepts extracted in {processing_time:.2f}s")
            
            return final_concepts, extraction_stats
            
        except Exception as e:
            logger.error(f"❌ Concept extraction failed: {str(e)}")
            raise
    
    async def _extract_structured_text(self, pdf_path: str) -> List[TextChunk]:
        """Extract text with source structure information."""
        try:
            # Use existing TORI block extraction with enhancements
            blocks = extract_concept_blocks(pdf_path)
            
            text_chunks = []
            for i, block in enumerate(blocks):
                # Enhanced source detection
                source = self._detect_source_type(block, i, len(blocks))
                
                # Calculate positional weight
                weight = self._calculate_positional_weight(i, len(blocks), source)
                
                chunk = TextChunk(
                    text=block,
                    source=source,
                    position=i,
                    length=len(block),
                    weight=weight,
                    context={"block_index": i, "total_blocks": len(blocks)}
                )
                text_chunks.append(chunk)
            
            return text_chunks
            
        except Exception as e:
            logger.error(f"❌ Text extraction failed: {str(e)}")
            return []
    
    def _detect_source_type(self, text: str, position: int, total_blocks: int) -> ConceptSource:
        """Detect the source type of a text block."""
        text_lower = text.lower().strip()
        
        # Title detection (first few blocks, short, capitalized)
        if position < 3 and len(text) < 200 and text.isupper():
            return ConceptSource.TITLE
        
        # Abstract detection
        if "abstract" in text_lower[:50] or (position < 5 and "summary" in text_lower[:50]):
            return ConceptSource.ABSTRACT
        
        # Section header detection
        if (len(text) < 100 and 
            (text.startswith(tuple('0123456789')) or 
             any(word in text_lower for word in ['introduction', 'conclusion', 'method', 'result']))):
            return ConceptSource.SECTION_HEADER
        
        # Equation detection
        if re.search(r'[=\+\-\*/\^]+.*[=\+\-\*/\^]', text) and len(text) < 500:
            return ConceptSource.EQUATION
        
        # Caption detection
        if any(word in text_lower[:50] for word in ['figure', 'table', 'fig.', 'tab.', 'image']):
            return ConceptSource.CAPTION
        
        # Citation detection
        if text.count('[') > 2 or text.count('(') > 3:
            return ConceptSource.CITATION
        
        # Default to body text
        return ConceptSource.BODY_TEXT
    
    def _calculate_positional_weight(self, position: int, total_blocks: int, source: ConceptSource) -> float:
        """Calculate weight based on position and source type."""
        # Base weight from source type
        base_weight = self.config["source_weights"][source]
        
        # Position-based modifier
        relative_position = position / max(total_blocks - 1, 1)
        
        # Higher weight for beginning and end (title, abstract, conclusion)
        if relative_position < 0.1:  # Beginning
            position_modifier = 1.2
        elif relative_position > 0.9:  # End
            position_modifier = 1.1
        elif 0.1 <= relative_position <= 0.3:  # Early content
            position_modifier = 1.05
        else:  # Middle content
            position_modifier = 1.0
        
        return min(1.0, base_weight * position_modifier)
    
    async def _extract_raw_concepts(self, text_chunks: List[TextChunk]) -> List[RawConcept]:
        """Extract raw concepts using enhanced TF-IDF and NLP."""
        try:
            raw_concepts = []
            
            # Build enhanced feature matrix
            blocks = [chunk.text for chunk in text_chunks]
            feats, vocab = build_feature_matrix(blocks, vocab_size=2000)  # Increased vocab
            
            # Apply spectral embedding for concept discovery
            emb = spectral_embed(feats, k=32)  # Increased dimensionality
            
            # Use oscillator clustering for initial grouping
            labels = run_oscillator_clustering(emb)
            
            # Extract concepts from each cluster
            for cluster_id in set(labels):
                cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
                
                if len(cluster_indices) == 0:
                    continue
                
                # Get representative text chunks for this cluster
                cluster_chunks = [text_chunks[i] for i in cluster_indices]
                cluster_blocks = [blocks[i] for i in cluster_indices]
                other_blocks = [blocks[i] for i in range(len(blocks)) if i not in cluster_indices]
                
                # Extract keywords using existing TORI method
                keywords = extract_keywords(cluster_blocks, other_blocks, n=5)  # More keywords