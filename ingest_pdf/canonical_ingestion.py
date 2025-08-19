"""canonical_ingestion.py - Implements ALAN's pure emergent cognition approach to ingestion.

This module serves as the primary entry point for ALAN's principled knowledge acquisition
system, implementing the five key commitments:

1. No Pretraining: Knowledge derived only from curated, transparent canonical sources
2. No Token Imitation: Using Koopman eigenflows for conceptual reasoning, not statistical imitation
3. No Memory Bloat: Entropy-gated memory integration to maintain only useful concepts
4. No Opaque Models: Tracking concept provenance and eigenfunction IDs for transparency
5. No Blind Inference: Using oscillator synchrony to reflect phase-coherent truth

This approach ensures ALAN builds a foundation from first principles rather than
relying on pre-trained weights derived from opaque internet-scale corpora.

References:
- Koopman phase graphs for active ingestion and resonant integration
- Canonical source curation principles
- Information-theoretic memory gating
"""

import os
import sys
import json
import hashlib
import re
import logging
import numpy as np
import uuid
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
from pathlib import Path
import concurrent.futures
from collections import defaultdict

try:
    # Try absolute import first
    from models import ConceptTuple
except ImportError:
    # Fallback to relative import
    from .models import ConceptTuple
try:
    # Try absolute import first
    from koopman_phase_graph import get_koopman_phase_graph, SourceDocument
except ImportError:
    # Fallback to relative import
    from .koopman_phase_graph import get_koopman_phase_graph, SourceDocument
try:
    # Try absolute import first
    from spectral_monitor import get_cognitive_spectral_monitor
except ImportError:
    # Fallback to relative import
    from .spectral_monitor import get_cognitive_spectral_monitor
try:
    # Try absolute import first
    from fractality import get_cognitive_fractal_analyzer
except ImportError:
    # Fallback to relative import
    from .fractality import get_cognitive_fractal_analyzer
try:
    # Try absolute import first
    from source_validator import validate_source
except ImportError:
    # Fallback to relative import
    from .source_validator import validate_source
try:
    # Try absolute import first
    from memory_gating import apply_memory_gate
except ImportError:
    # Fallback to relative import
    from .memory_gating import apply_memory_gate
try:
    # Try absolute import first
    from fft_privacy import get_fft_privacy_engine
except ImportError:
    # Fallback to relative import
    from .fft_privacy import get_fft_privacy_engine

# Configure logger
logger = logging.getLogger("alan_canonical_ingestion")

@dataclass
class CanonicalSource:
    """Represents a validated canonical source for knowledge acquisition."""
    id: str  # Unique identifier
    title: str  # Document title
    author: str  # Document author(s)
    source_type: str  # Type of source (paper, manual, specification, etc.)
    domain: str  # Knowledge domain (mathematics, physics, control_theory, etc.)
    content_hash: str  # Hash of original content
    file_path: Optional[str] = None  # Path to source file if available
    uri: Optional[str] = None  # URI for external reference
    quality_score: float = 0.0  # Quality assessment score (0-1)
    concepts_extracted: int = 0  # Number of concepts extracted
    ingestion_date: datetime = field(default_factory=datetime.now)  # When ingested
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "author": self.author,
            "source_type": self.source_type,
            "domain": self.domain,
            "content_hash": self.content_hash,
            "file_path": self.file_path,
            "uri": self.uri,
            "quality_score": float(self.quality_score),
            "concepts_extracted": self.concepts_extracted,
            "ingestion_date": self.ingestion_date.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class SourceVerdict:
    """Represents the verdict on a potential source document."""
    is_canonical: bool  # Whether the source meets canonical criteria
    quality_score: float  # Quality assessment score (0-1)
    reasons: List[str]  # Reasons for the verdict
    metadata: Dict[str, Any]  # Additional metadata about the assessment
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_canonical": self.is_canonical,
            "quality_score": float(self.quality_score),
            "reasons": self.reasons,
            "metadata": self.metadata
        }


class CanonicalSourceCurator:
    """
    Curates high-quality canonical sources for ALAN's knowledge acquisition.
    
    This class implements ALAN's first commitment: "No Pretraining" by ensuring
    only principled, transparent sources are used for knowledge acquisition.
    """
    
    def __init__(
        self,
        quality_threshold: float = 0.75,
        domains_of_interest: Optional[List[str]] = None,
        blacklisted_sources: Optional[List[str]] = None
    ):
        """
        Initialize the canonical source curator.
        
        Args:
            quality_threshold: Minimum quality score for canonical sources
            domains_of_interest: List of prioritized knowledge domains
            blacklisted_sources: List of source patterns to reject
        """
        self.quality_threshold = quality_threshold
        self.domains_of_interest = domains_of_interest or [
            "mathematics", "physics", "control_theory", "dynamical_systems",
            "neuroscience", "logic", "geometry", "computer_science",
            "systems_theory", "formal_methods"
        ]
        self.blacklisted_sources = blacklisted_sources or [
            "reddit.com", "twitter.com", "facebook.com", "instagram.com",
            "social_media", "forum", "blog", "news", "opinion", "commentary"
        ]
        
        # Maintain registry of approved canonical sources
        self.canonical_sources = {}  # id -> CanonicalSource
        
        # Track assessment history
        self.assessment_history = []
        
        # Ensure storage paths exist
        os.makedirs("data/canonical_sources", exist_ok=True)
        
        # Track failed sources
        self.rejected_sources = {}  # id -> reasons
        
        logger.info("Canonical source curator initialized")
        
    def assess_source(
        self,
        title: str,
        author: str,
        content: str,
        source_type: str,
        domain: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SourceVerdict:
        """
        Assess whether a source meets canonical criteria.
        
        Args:
            title: Document title
            author: Document author(s)
            content: Document content
            source_type: Type of source
            domain: Knowledge domain
            metadata: Additional source metadata
            
        Returns:
            SourceVerdict with assessment results
        """
        # Initialize with default scores
        scores = {
            "domain_relevance": 0.0,
            "format_quality": 0.0,
            "content_depth": 0.0,
            "formal_structure": 0.0,
            "source_credibility": 0.0
        }
        
        reasons = []
        
        # 1. Check domain relevance
        if domain in self.domains_of_interest:
            scores["domain_relevance"] = 1.0
        else:
            # Check for related domains
            related_domains = [d for d in self.domains_of_interest 
                             if d in domain or domain in d]
            if related_domains:
                # Partial match
                scores["domain_relevance"] = 0.5
                reasons.append(f"Domain '{domain}' partially matches interests")
            else:
                scores["domain_relevance"] = 0.0
                reasons.append(f"Domain '{domain}' not in primary interests")
                
        # 2. Check source type
        if source_type.lower() in ["paper", "scientific_paper", "arxiv", "journal"]:
            scores["format_quality"] = 0.9
        elif source_type.lower() in ["manual", "specification", "documentation", "textbook"]:
            scores["format_quality"] = 0.8
        elif source_type.lower() in ["dataset", "benchmark", "survey"]:
            scores["format_quality"] = 0.6
        elif any(blacklisted in source_type.lower() for blacklisted in self.blacklisted_sources):
            scores["format_quality"] = 0.0
            reasons.append(f"Source type '{source_type}' is blacklisted")
        else:
            scores["format_quality"] = 0.3
            reasons.append(f"Source type '{source_type}' not preferred")
            
        # 3. Assess content depth and quality
        # Check for mathematical content
        math_pattern = r'\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]'
        has_math = bool(re.search(math_pattern, content))
        
        # Check for structured sections
        sections_pattern = r'\b(abstract|introduction|method|conclusion|reference|bibliography)\b'
        section_matches = re.findall(sections_pattern, content.lower())
        has_structure = len(set(section_matches)) >= 3
        
        # Check for references/citations
        citation_pattern = r'\[[\d,\s-]+\]|\(\w+\s+et\s+al\.\s*,\s*\d{4}\)'
        has_citations = bool(re.search(citation_pattern, content))
        
        # Compute content depth score
        if has_math and has_structure and has_citations:
            scores["content_depth"] = 1.0
        elif has_structure and has_citations:
            scores["content_depth"] = 0.8
        elif has_structure or has_citations:
            scores["content_depth"] = 0.5
            reasons.append("Limited structure or citations")
        else:
            scores["content_depth"] = 0.3
            reasons.append("Lacks formal structure and citations")
            
        # 4. Assess formal structure (equations, definitions, theorems)
        formal_elements = []
        
        # Look for definitions
        definition_pattern = r'\b(definition|def\.|theorem|thm\.|lemma|corollary|proposition|axiom)\b'
        formal_matches = re.findall(definition_pattern, content, re.IGNORECASE)
        formal_elements.extend(formal_matches)
        
        # Calculate formal structure score
        if len(formal_elements) > 5:
            scores["formal_structure"] = 1.0
        elif len(formal_elements) > 2:
            scores["formal_structure"] = 0.7
        elif len(formal_elements) > 0:
            scores["formal_structure"] = 0.4
            reasons.append("Limited formal elements")
        else:
            scores["formal_structure"] = 0.2
            reasons.append("Lacks formal mathematical structure")
            
        # 5. Source credibility
        # Check for known credible authors or sources in metadata
        is_credible = False
        if metadata and "source_credibility" in metadata:
            is_credible = metadata["source_credibility"] > 0.7
            
        if is_credible:
            scores["source_credibility"] = 0.9
        elif "arxiv" in author.lower() or "journal" in metadata.get("publication", "").lower():
            scores["source_credibility"] = 0.8
        else:
            scores["source_credibility"] = 0.5
            reasons.append("Unknown source credibility")
            
        # Compute overall quality score (weighted average)
        weights = {
            "domain_relevance": 0.3,
            "format_quality": 0.2,
            "content_depth": 0.2,
            "formal_structure": 0.2,
            "source_credibility": 0.1
        }
        
        quality_score = sum(score * weights[key] for key, score in scores.items())
        
        # Determine if canonical
        is_canonical = quality_score >= self.quality_threshold
        
        if not is_canonical:
            reasons.insert(0, f"Quality score {quality_score:.2f} below threshold {self.quality_threshold:.2f}")
        
        # Create source verdict
        verdict = SourceVerdict(
            is_canonical=is_canonical,
            quality_score=quality_score,
            reasons=reasons,
            metadata={
                "scores": scores,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Record assessment
        self.assessment_history.append({
            "title": title,
            "author": author,
            "source_type": source_type,
            "domain": domain,
            "verdict": verdict.to_dict(),
            "timestamp": datetime.now().isoformat()
        })
        
        return verdict
        
    def register_canonical_source(
        self,
        title: str,
        author: str,
        content: str,
        source_type: str,
        domain: str,
        file_path: Optional[str] = None,
        uri: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[CanonicalSource]:
        """
        Register a source as canonical if it meets criteria.
        
        Args:
            title: Document title
            author: Document author(s)
            content: Document content
            source_type: Type of source
            domain: Knowledge domain
            file_path: Optional path to source file
            uri: Optional URI
            metadata: Additional source metadata
            
        Returns:
            CanonicalSource if registered, None if rejected
        """
        # Assess source first
        verdict = self.assess_source(
            title=title,
            author=author,
            content=content,
            source_type=source_type,
            domain=domain,
            metadata=metadata
        )
        
        if not verdict.is_canonical:
            # Create a unique ID for tracking rejected sources
            source_id = f"rejected_{hashlib.md5(f'{title}_{author}'.encode()).hexdigest()[:12]}"
            self.rejected_sources[source_id] = verdict.reasons
            logger.info(f"Source '{title}' rejected: {verdict.reasons[0]}")
            return None
            
        # Create source ID
        source_id = f"src_{hashlib.md5(f'{title}_{author}'.encode()).hexdigest()[:12]}"
        
        # Create content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Check if source already exists
        if source_id in self.canonical_sources:
            existing = self.canonical_sources[source_id]
            if existing.content_hash == content_hash:
                logger.info(f"Source '{title}' already registered with ID {source_id}")
                return existing
            else:
                # Update hash for existing source (new version)
                existing.content_hash = content_hash
                logger.info(f"Updated content hash for existing source '{title}' (ID: {source_id})")
                return existing
                
        # Create canonical source
        source = CanonicalSource(
            id=source_id,
            title=title,
            author=author,
            source_type=source_type,
            domain=domain,
            content_hash=content_hash,
            file_path=file_path,
            uri=uri,
            quality_score=verdict.quality_score,
            metadata=metadata or {}
        )
        
        # Store source
        self.canonical_sources[source_id] = source
        
        # Register source in Koopman phase graph
        kpg = get_koopman_phase_graph()
        kpg.register_source_document(
            title=title,
            author=author,
            content=content,
            source_type=source_type,
            domain=domain,
            uri=uri,
            metadata=metadata
        )
        
        logger.info(f"Registered canonical source: '{title}' (ID: {source_id})")
        
        return source
        
    def get_source_by_id(self, source_id: str) -> Optional[CanonicalSource]:
        """
        Get a canonical source by its ID.
        
        Args:
            source_id: Source ID
            
        Returns:
            CanonicalSource if found, None otherwise
        """
        return self.canonical_sources.get(source_id)
        
    def get_sources_by_domain(self, domain: str) -> List[CanonicalSource]:
        """
        Get all canonical sources in a specific domain.
        
        Args:
            domain: Knowledge domain
            
        Returns:
            List of canonical sources in the domain
        """
        return [
            source for source in self.canonical_sources.values()
            if source.domain == domain
        ]
        
    def save_registry(self, path: str = "data/canonical_sources/registry.json") -> Dict[str, Any]:
        """
        Save the canonical source registry to disk.
        
        Args:
            path: Output file path
            
        Returns:
            Dictionary with save results
        """
        try:
            registry_data = {
                "canonical_sources": [source.to_dict() for source in self.canonical_sources.values()],
                "rejected_count": len(self.rejected_sources),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(path, "w") as f:
                json.dump(registry_data, f, indent=2)
                
            return {
                "status": "success",
                "sources_saved": len(self.canonical_sources),
                "path": path
            }
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
            return {"status": "error", "message": str(e)}
            
    def load_registry(self, path: str = "data/canonical_sources/registry.json") -> Dict[str, Any]:
        """
        Load the canonical source registry from disk.
        
        Args:
            path: Input file path
            
        Returns:
            Dictionary with load results
        """
        try:
            if not os.path.exists(path):
                return {"status": "error", "message": "Registry file not found"}
                
            with open(path, "r") as f:
                registry_data = json.load(f)
                
            # Create canonical sources
            self.canonical_sources = {}
            
            for source_dict in registry_data["canonical_sources"]:
                source = CanonicalSource(
                    id=source_dict["id"],
                    title=source_dict["title"],
                    author=source_dict["author"],
                    source_type=source_dict["source_type"],
                    domain=source_dict["domain"],
                    content_hash=source_dict["content_hash"],
                    file_path=source_dict.get("file_path"),
                    uri=source_dict.get("uri"),
                    quality_score=source_dict["quality_score"],
                    concepts_extracted=source_dict.get("concepts_extracted", 0),
                    ingestion_date=datetime.fromisoformat(source_dict["ingestion_date"]),
                    metadata=source_dict.get("metadata", {})
                )
                
                self.canonical_sources[source.id] = source
                
            return {
                "status": "success",
                "sources_loaded": len(self.canonical_sources),
                "timestamp": registry_data.get("timestamp")
            }
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
            return {"status": "error", "message": str(e)}


class ActiveIngestionManager:
    """
    Manages the active ingestion process for canonical sources.
    
    This class implements ALAN's core knowledge acquisition strategy:
    1. Canonical Source Curation - Selects high-quality sources
    2. Active Ingestion with Koopman Phase Graphs - Structured knowledge extraction
    3. Entropy-Gated Memory Integration - Prevents memorizing noise or redundancy
    4. Auto-Sourcing with Lineage Tracking - Maintains provenance
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        max_concepts_per_source: int = 500,
        concept_extraction_rate: float = 0.1  # Concepts per token
    ):
        """
        Initialize the active ingestion manager.
        
        Args:
            embedding_dim: Dimension of concept embeddings
            max_concepts_per_source: Maximum concepts to extract per source
            concept_extraction_rate: Rate of concept extraction per token
        """
        self.embedding_dim = embedding_dim
        self.max_concepts_per_source = max_concepts_per_source
        self.concept_extraction_rate = concept_extraction_rate
        
        # Initialize components
        self.source_curator = CanonicalSourceCurator()
        self.koopman_graph = get_koopman_phase_graph(embedding_dim)
        self.spectral_monitor = get_cognitive_spectral_monitor()
        self.privacy_engine = get_fft_privacy_engine(embedding_dim)
        
        # Track ingestion history
        self.ingestion_history = []
        
        # Track active ingestion tasks
        self.active_tasks = {}  # task_id -> task_status
        
        # Initialize concept statistics
        self.concept_stats = {
            "total_ingested": 0,
            "total_rejected": 0,
            "by_domain": defaultdict(int),
            "by_source": defaultdict(int)
        }
        
        logger.info("Active ingestion manager initialized")
        
    def extract_concepts_from_text(
        self,
        text: str,
        source_id: str,
        domain: str
    ) -> List[Dict[str, Any]]:
        """
        Extract concepts from source text.
        
        This is a placeholder for a more sophisticated extraction method.
        In a real implementation, this would use semantic parsing, domain-specific
        extractors, or other methods to identify meaningful concepts.
        
        Args:
            text: Source text
            source_id: Source document ID
            domain: Knowledge domain
            
        Returns:
            List of extracted concept dictionaries
        """
        # Split into paragraphs and sentences
        paragraphs = text.split('\n\n')
        
        concepts = []
        
        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
                
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            for sent_idx, sentence in enumerate(sentences):
                if len(sentence.split()) < 5:  # Skip very short sentences
                    continue
                    
                # Simple heuristic: look for noun phrases, defined terms, etc.
                # This is a placeholder for more sophisticated extraction
                
                # Example: look for defined terms (e.g., "X is defined as...")
                definition_matches = re.finditer(
                    r'\b([A-Z][a-zA-Z]*(?:\s+[a-zA-Z]+){0,3})\s+(?:is|are)\s+(?:defined\s+as|called)\s+([^.!?]+)',
                    sentence
                )
                
                for match in definition_matches:
                    concept_name = match.group(1)
                    concept_context = match.group(0)
                    
                    # Create a pseudo-embedding for demonstration
                    # In a real system, this would use a proper embedding model
                    concept_embedding = np.random.randn(self.embedding_dim)
                    concept_embedding = concept_embedding / np.linalg.norm(concept_embedding)
                    
                    concept = {
                        "name": concept_name,
                        "embedding": concept_embedding,
                        "context": concept_context,
                        "source_id": source_id,
                        "domain": domain,
                        "location": {
                            "paragraph": para_idx,
                            "sentence": sent_idx,
                            "text": sentence
                        }
                    }
                    
                    concepts.append(concept)
                    
                # Also extract key terms (nouns with adjectives)
                term_matches = re.finditer(
                    r'\b((?:[A-Z][a-z]*\s+)?[a-zA-Z]+\s+[a-zA-Z]+)\b',
                    sentence
                )
                
                for match in term_matches:
                    term = match.group(1)
                    
                    # Skip common words, short terms, etc.
                    if len(term) < 5 or term.lower() in ["the", "and", "this", "that"]:
                        continue
                        
                    # Create concept
                    concept_embedding = np.random.randn(self.embedding_dim)
                    concept_embedding = concept_embedding / np.linalg.norm(concept_embedding)
                    
                    concept = {
                        "name": term,
                        "embedding": concept_embedding,
                        "context": sentence,
                        "source_id": source_id,
                        "domain": domain,
                        "location": {
                            "paragraph": para_idx,
                            "sentence": sent_idx,
                            "text": sentence
                        }
                    }
                    
                    concepts.append(concept)
        
        # Limit number of concepts
        max_concepts = min(self.max_concepts_per_source, int(len(text) * self.concept_extraction_rate))
        
        if len(concepts) > max_concepts:
            # Prioritize definition concepts and then sample randomly
            definitions = [c for c in concepts if "defined as" in c["context"] or "called" in c["context"]]
            others = [c for c in concepts if c not in definitions]
            
            if len(definitions) <= max_concepts:
                # Keep all definitions and sample from others
                remaining = max_concepts - len(definitions)
                if remaining > 0 and others:
                    # Randomly sample from others
                    sampled_others = np.random.choice(others, min(remaining, len(others)), replace=False)
                    concepts = definitions + list(sampled_others)
                else:
                    concepts = definitions
            else:
                # Too many definitions, sample from them
                concepts = list(np.random.choice(definitions, max_concepts, replace=False))
                
        return concepts
    
    def apply_penrose_similarity(self, concepts: List[Dict[str, Any]]) -> np.ndarray:
        """Apply Penrose O(n^2.32) similarity to extracted concepts"""
        try:
            from python.core.penrose_adapter import PenroseAdapter
            
            # Extract embeddings
            embeddings = np.array([c['embedding'] for c in concepts if 'embedding' in c])
            if len(embeddings) < 2:
                return np.eye(len(concepts))  # Identity if too few
            
            # Initialize Penrose with rank 32
            penrose = PenroseAdapter.get_instance(rank=32, embedding_dim=embeddings.shape[1])
            
            # Compute similarity matrix in O(n^2.32)
            start_time = time.time()
            similarity_matrix = penrose.similarity_matrix(embeddings)
            elapsed = time.time() - start_time
            
            logger.info(f"✅ Penrose similarity computed in {elapsed:.2f}s for {len(concepts)} concepts")
            logger.info(f"   Speedup vs O(n²): {(len(concepts)**2 / len(concepts)**2.32):.1f}x")
            
            # Log to PsiArchive if available
            try:
                from python.core.psi_archive import PSI_ARCHIVER
                PSI_ARCHIVER.log_event({
                    'event_type': 'PENROSE_SIM',
                    'concept_count': len(concepts),
                    'computation_time': elapsed,
                    'similarity_threshold': 0.7,
                    'rank': 32
                })
            except ImportError:
                pass
            
            return similarity_matrix
            
        except ImportError:
            logger.warning("⚠️ Penrose not available, using cosine similarity")
            # Fallback to standard cosine similarity
            embeddings = np.array([c['embedding'] for c in concepts if 'embedding' in c])
            if len(embeddings) < 2:
                return np.eye(len(concepts))
            
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_normalized = embeddings / (norms + 1e-8)
            return embeddings_normalized @ embeddings_normalized.T
        
    def ingest_concepts_from_source(
        self,
        source: CanonicalSource,
        content: str
    ) -> Dict[str, Any]:
        """
        Ingest concepts from a canonical source.
        
        Args:
            source: Canonical source
            content: Source content
            
        Returns:
            Dictionary with ingestion results
        """
        # Create tracking ID for this ingestion task
        task_id = f"ingest_{uuid.uuid4().hex[:8]}"
        
        # Update task status
        self.active_tasks[task_id] = {
            "status": "extracting_concepts",
            "source_id": source.id,
            "title": source.title,
            "start_time": datetime.now().isoformat()
        }
        
        # Extract concepts
        concepts = self.extract_concepts_from_text(
            text=content,
            source_id=source.id,
            domain=source.domain
        )
        
        # Update task status
        self.active_tasks[task_id]["status"] = "computing_similarity"
        self.active_tasks[task_id]["concepts_extracted"] = len(concepts)
        
        # Apply Penrose similarity computation
        if len(concepts) > 1:
            similarity_matrix = self.apply_penrose_similarity(concepts)
            
            # Use similarity matrix to filter redundant concepts
            # Keep concepts that are sufficiently distinct
            distinct_indices = []
            similarity_threshold = 0.85  # High similarity means redundant
            
            for i in range(len(concepts)):
                is_distinct = True
                for j in distinct_indices:
                    if similarity_matrix[i, j] > similarity_threshold:
                        is_distinct = False
                        break
                if is_distinct:
                    distinct_indices.append(i)
            
            # Filter to distinct concepts
            concepts = [concepts[i] for i in distinct_indices]
            logger.info(f"Filtered to {len(concepts)} distinct concepts using Penrose similarity")
        
        # Update task status
        self.active_tasks[task_id]["status"] = "ingesting_concepts"
        
        # Process each concept
        ingested_concepts = []
        rejected_concepts = []
        
        for concept_dict in concepts:
            # Create concept in Koopman graph
            concept = self.koopman_graph.create_concept_from_embedding(
                name=concept_dict["name"],
                embedding=concept_dict["embedding"],
                source_document_id=concept_dict["source_id"],
                source_location=concept_dict["location"]
            )
            
            # Apply privacy processing
            if hasattr(concept, 'embedding') and concept.embedding is not None:
                # Apply FFT-based privacy
                private_embedding, privacy_metadata = self.privacy_engine.privatize_embedding(
                    embedding=concept.embedding,
                    operation_type="concept_ingestion",
                    importance=0.8  # High importance for ingestion
                )
                
                # Update concept with privatized embedding
                concept.embedding = private_embedding
            
            # Ingest into Koopman graph (applies entropy gate)
            success, details = self.koopman_graph.ingest_concept(concept)
            
            if success:
                ingested_concepts.append(concept)
                
                # Track statistics
                self.concept_stats["total_ingested"] += 1
                self.concept_stats["by_domain"][source.domain] += 1
                self.concept_stats["by_source"][source.id] += 1
            else:
                rejected_concepts.append({
                    "name": concept.name,
                    "reason": details.get("reason", "unknown")
                })
                self.concept_stats["total_rejected"] += 1
                
        # Update phase oscillators in graph
        self.koopman_graph.update_concept_phase_oscillators()
        
        # Update source with ingestion statistics
        source.concepts_extracted = len(ingested_concepts)
        
        # Update task status
        self.active_tasks[task_id]["status"] = "completed"
        self.active_tasks[task_id]["ingested_count"] = len(ingested_concepts)
        self.active_tasks[task_id]["rejected_count"] = len(rejected_concepts)
        self.active_tasks[task_id]["end_time"] = datetime.now().isoformat()
        
        # Create result
        result = {
            "status": "success",
            "task_id": task_id,
            "source_id": source.id,
            "title": source.title,
            "concepts_extracted": len(concepts),
            "concepts_ingested": len(ingested_concepts),
            "concepts_rejected": len(rejected_concepts),
            "ingested_concepts": [c.id for c in ingested_concepts],
            "rejected_concepts": rejected_concepts
        }
        
        # Add to history
        self.ingestion_history.append({
            "timestamp": datetime.now().isoformat(),
            "source_id": source.id,
            "task_id": task_id,
            "concepts_ingested": len(ingested_concepts),
            "concepts_rejected": len(rejected_concepts)
        })
        
        logger.info(f"Ingested {len(ingested_concepts)} concepts from '{source.title}'")
        
        return result
        
    def process_pdf_file(
        self,
        file_path: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        source_type: str = "paper",
        domain: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a PDF file for canonical source ingestion.
        
        Args:
            file_path: Path to PDF file
            title: Optional title override
            author: Optional author override
            source_type: Source type
            domain: Knowledge domain
            metadata: Additional metadata
            
        Returns:
            Dictionary with processing results
        """
        # Basic path validation
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "message": f"File not found: {file_path}"
            }
            
        if not file_path.lower().endswith('.pdf'):
            return {
                "status": "error",
                "message": f"Not a PDF file: {file_path}"
            }
            
        # Extract PDF metadata and content
        try:
            # This would use a proper PDF extraction library in production
            # For now, we'll use a placeholder approach
            
            # If title/author not provided, try to extract from filename
            if title is None:
                # Extract from filename
                base_name = os.path.basename(file_path)
                title = os.path.splitext(base_name)[0].replace('_', ' ').title()
                
            if author is None:
                # Default author
                author = "Unknown"
                
            # Read PDF content (simplified)
            with open(file_path, 'rb') as f:
                # In production, use a proper PDF parser like PyPDF2, PDFMiner, etc.
                # For demonstration, we'll read it as binary and extract some text
                pdf_bytes = f.read()
                
                # Create hash for tracking
                content_hash = hashlib.sha256(pdf_bytes).hexdigest()
                
                # In a real system, extract text properly
                # For demonstration, we'll generate placeholder text
                content = f"Sample extracted content from {title}.\n\n"
                content += "This would be the actual extracted text from the PDF.\n\n"
                content += f"Author: {author}\nDomain: {domain}\nType: {source_type}\n\n"
                
                # Add placeholder sections and content
                content += "Abstract\nThis paper presents a novel approach to...\n\n"
                content += "Introduction\nRecent advances in the field have shown...\n\n"
                content += "Methods\nWe propose a new algorithm based on...\n\n"
                content += "Results\nOur approach achieved significant improvements...\n\n"
                content += "Conclusion\nWe have demonstrated the effectiveness of...\n\n"
                content += "References\n[1] Smith et al., 2022. A related study...\n"
                
            # Register as canonical source
            source = self.source_curator.register_canonical_source(
                title=title,
                author=author,
                content=content,
                source_type=source_type,
                domain=domain,
                file_path=file_path,
                metadata=metadata or {}
            )
            
            if source is None:
                return {
                    "status": "rejected",
                    "message": "File does not meet canonical source criteria",
                    "file_path": file_path
                }
                
            # Ingest concepts
            ingestion_result = self.ingest_concepts_from_source(source, content)
            
            # Combine results
            result = {
                "status": "success",
                "source_id": source.id,
                "title": source.title,
                "file_path": file_path,
                "domain": domain,
                "source_type": source_type,
                "quality_score": source.quality_score,
                "content_hash": content_hash,
                "ingestion_result": ingestion_result
            }
            
            return result
        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "file_path": file_path
            }
    
    def get_ingestion_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the ingestion process.
        
        Returns:
            Dictionary with ingestion statistics
        """
        return {
            "total_ingested": self.concept_stats["total_ingested"],
            "total_rejected": self.concept_stats["total_rejected"],
            "ingestion_rate": (
                self.concept_stats["total_ingested"] /
                (self.concept_stats["total_ingested"] + self.concept_stats["total_rejected"])
                if (self.concept_stats["total_ingested"] + self.concept_stats["total_rejected"]) > 0
                else 0.0
            ),
            "by_domain": dict(self.concept_stats["by_domain"]),
            "by_source": dict(self.concept_stats["by_source"]),
            "active_tasks": len(self.active_tasks),
            "sources_processed": len(self.source_curator.canonical_sources),
            "sources_rejected": len(self.source_curator.rejected_sources)
        }
    
    def save_state(self, base_path: str = "data") -> Dict[str, Any]:
        """
        Save the entire ingestion state to disk.
        
        Args:
            base_path: Base directory for saving
            
        Returns:
            Dictionary with save results
        """
        try:
            # Ensure directories exist
            os.makedirs(base_path, exist_ok=True)
            
            # Save components
            source_result = self.source_curator.save_registry(
                os.path.join(base_path, "canonical_sources/registry.json")
            )
            
            koopman_result = self.koopman_graph.save_to_disk(
                os.path.join(base_path, "koopman_phase_graph")
            )
            
            # Save ingestion history
            history_path = os.path.join(base_path, "ingestion_history.json")
            with open(history_path, "w") as f:
                json.dump({
                    "history": self.ingestion_history,
                    "stats": self.concept_stats
                }, f, indent=2)
                
            return {
                "status": "success",
                "source_result": source_result,
                "koopman_result": koopman_result,
                "history_path": history_path
            }
        except Exception as e:
            logger.error(f"Error saving ingestion state: {e}")
            return {"status": "error", "message": str(e)}
    
    def load_state(self, base_path: str = "data") -> Dict[str, Any]:
        """
        Load the entire ingestion state from disk.
        
        Args:
            base_path: Base directory for loading
            
        Returns:
            Dictionary with load results
        """
        try:
            # Load source registry
            source_result = self.source_curator.load_registry(
                os.path.join(base_path, "canonical_sources/registry.json")
            )
            
            # Load Koopman phase graph
            koopman_result = self.koopman_graph.load_from_disk(
                os.path.join(base_path, "koopman_phase_graph")
            )
            
            # Load ingestion history
            history_path = os.path.join(base_path, "ingestion_history.json")
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    history_data = json.load(f)
                    self.ingestion_history = history_data.get("history", [])
                    self.concept_stats = history_data.get("stats", self.concept_stats)
                    
            return {
                "status": "success",
                "source_result": source_result,
                "koopman_result": koopman_result
            }
        except Exception as e:
            logger.error(f"Error loading ingestion state: {e}")
            return {"status": "error", "message": str(e)}


# Singleton instance for easy access
_active_ingestion_manager = None

def get_active_ingestion_manager(embedding_dim: int = 768) -> ActiveIngestionManager:
    """
    Get or create the singleton active ingestion manager.
    
    Args:
        embedding_dim: Dimension of concept embeddings
        
    Returns:
        ActiveIngestionManager instance
    """
    global _active_ingestion_manager
    if _active_ingestion_manager is None:
        _active_ingestion_manager = ActiveIngestionManager(embedding_dim=embedding_dim)
    return _active_ingestion_manager


def ingest_pdf_file(
    file_path: str,
    title: Optional[str] = None,
    author: Optional[str] = None,
    source_type: str = "paper",
    domain: str = "general",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenient function to ingest a PDF file.
    
    Args:
        file_path: Path to PDF file
        title: Optional title override
        author: Optional author override
        source_type: Source type
        domain: Knowledge domain
        metadata: Additional metadata
        
    Returns:
        Dictionary with ingestion results
    """
    manager = get_active_ingestion_manager()
    return manager.process_pdf_file(
        file_path=file_path,
        title=title,
        author=author,
        source_type=source_type,
        domain=domain,
        metadata=metadata
    )


def batch_ingest_pdfs(
    directory_path: str,
    recursive: bool = False,
    domain: str = "general",
    source_type: str = "paper",
    max_workers: int = 4
) -> Dict[str, Any]:
    """
    Batch ingest multiple PDF files from a directory.
    
    Args:
        directory_path: Path to directory containing PDFs
        recursive: Whether to search subdirectories
        domain: Default domain for all PDFs
        source_type: Default source type for all PDFs
        max_workers: Maximum number of concurrent workers
        
    Returns:
        Dictionary with batch ingestion results
    """
    if not os.path.isdir(directory_path):
        return {
            "status": "error",
            "message": f"Not a directory: {directory_path}"
        }
        
    # Find PDF files
    pdf_files = []
    
    if recursive:
        # Walk directory recursively
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
    else:
        # List only top-level PDFs
        for file in os.listdir(directory_path):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(directory_path, file))
                
    if not pdf_files:
        return {
            "status": "warning",
            "message": f"No PDF files found in {directory_path}"
        }
        
    # Process files with thread pool
    results = {}
    manager = get_active_ingestion_manager()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all ingestion tasks
        future_to_file = {
            executor.submit(
                manager.process_pdf_file,
                file_path=pdf_path,
                domain=domain,
                source_type=source_type
            ): pdf_path
            for pdf_path in pdf_files
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            pdf_path = future_to_file[future]
            try:
                result = future.result()
                results[pdf_path] = result
                
                # Log progress
                status = result.get("status", "unknown")
                if status == "success":
                    logger.info(f"Successfully ingested {pdf_path}")
                else:
                    logger.warning(f"Failed to ingest {pdf_path}: {result.get('message', 'unknown error')}")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {str(e)}")
                results[pdf_path] = {
                    "status": "error",
                    "message": str(e),
                    "file_path": pdf_path
                }
                
    # Summarize results
    successful = [path for path, result in results.items() if result.get("status") == "success"]
    rejected = [path for path, result in results.items() if result.get("status") == "rejected"]
    errors = [path for path, result in results.items() if result.get("status") == "error"]
    
    return {
        "status": "completed",
        "total_files": len(pdf_files),
        "successful": len(successful),
        "rejected": len(rejected),
        "errors": len(errors),
        "successful_paths": successful,
        "rejected_paths": rejected,
        "error_paths": errors,
        "detailed_results": results
    }


def validate_and_ingest_pdf(
    pdf_path: str,
    validate_only: bool = False,
    domain: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate a PDF file against canonical criteria and optionally ingest it.
    
    Args:
        pdf_path: Path to PDF file
        validate_only: If True, only validate without ingesting
        domain: Optional domain classification override
        metadata: Additional metadata
        
    Returns:
        Dictionary with validation/ingestion results
    """
    if not os.path.exists(pdf_path):
        return {
            "status": "error",
            "message": f"File not found: {pdf_path}"
        }
        
    # Validate source using source_validator module
    validation_result = validate_source(pdf_path)
    
    if not validation_result.get("is_valid", False):
        return {
            "status": "invalid",
            "message": "Source does not meet validation criteria",
            "validation_result": validation_result
        }
        
    # Extract metadata from validation
    extracted_metadata = validation_result.get("metadata", {})
    
    # Override domain if provided
    if domain is not None:
        extracted_metadata["domain"] = domain
    
    # Merge with provided metadata
    if metadata is not None:
        extracted_metadata.update(metadata)
        
    # If validate only, return validation result
    if validate_only:
        return {
            "status": "valid",
            "message": "Source meets validation criteria",
            "validation_result": validation_result
        }
        
    # Otherwise, ingest the PDF
    ingestion_result = ingest_pdf_file(
        file_path=pdf_path,
        title=extracted_metadata.get("title"),
        author=extracted_metadata.get("author"),
        source_type=extracted_metadata.get("source_type", "paper"),
        domain=extracted_metadata.get("domain", "general"),
        metadata=extracted_metadata
    )
    
    # Return combined result
    return {
        "status": ingestion_result.get("status", "unknown"),
        "validation_result": validation_result,
        "ingestion_result": ingestion_result
    }
