"""
Concept Mesh - Improved Implementation (Python)
Handles concept extraction, relationships, and mesh operations
WITH ALL AUDIT FIXES APPLIED

Improvements:
1. UUID-based ID generation (no collisions)
2. Configurable paths via config dict and env vars
3. Specific exception handling
4. LRU cache for embeddings
5. Unsubscribe mechanism for events
6. Class-level stop words constant
7. Complete type annotations
8. Thread safety maintained
"""

import json
import logging
import os
import uuid
from typing import Dict, Any, List, Set, Tuple, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import numpy as np
from collections import defaultdict, deque
import networkx as nx
from pathlib import Path
import pickle
import gzip
from threading import RLock
import atexit
import tempfile
import shutil
from functools import lru_cache
import weakref
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases
ConceptID = str
CallbackID = str
EmbeddingVector = np.ndarray

# Class-level constants
STOP_WORDS: Set[str] = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
    'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
    'who', 'when', 'where', 'why', 'how', 'not', 'no', 'yes'
}

@dataclass
class Concept:
    """A single concept in the mesh"""
    id: ConceptID
    name: str
    description: str = ""
    category: str = "general"
    importance: float = 1.0
    embedding: Optional[EmbeddingVector] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

@dataclass
class ConceptRelation:
    """Relationship between concepts"""
    source_id: ConceptID
    target_id: ConceptID
    relation_type: str  # "is_a", "part_of", "related_to", "causes", "requires", etc.
    strength: float = 1.0
    bidirectional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ConceptDiff:
    """A difference/change in the concept mesh"""
    id: str
    diff_type: str  # "add", "remove", "modify", "relate", "unrelate"
    concepts: List[ConceptID]  # Concept IDs involved
    old_value: Any = None
    new_value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class ConceptMeshConfig:
    """Configuration for ConceptMesh with environment variable support"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
    def get_storage_path(self) -> Path:
        """Get storage path from config, env var, or default"""
        return Path(
            self.config.get('storage_path') or
            os.environ.get('CONCEPT_MESH_STORAGE_PATH') or
            'data/concept_mesh'
        )
    
    def get_seed_files(self) -> List[str]:
        """Get seed file paths from config, env var, or defaults"""
        if 'seed_files' in self.config:
            return self.config['seed_files']
        
        env_seeds = os.environ.get('CONCEPT_MESH_SEED_FILES')
        if env_seeds:
            return env_seeds.split(',')
        
        # Default seed files
        return [
            "data/concept_seed_universal.json",
            "ingest_pdf/data/concept_seed_universal.json",
            "data/seed_concepts.json",
            "data/concept_mesh/seed_concepts.json"
        ]
    
    def get_initial_data_file(self) -> Optional[Path]:
        """Get initial data file from config or env var"""
        path_str = (
            self.config.get('initial_data_file') or
            os.environ.get('CONCEPT_MESH_INITIAL_DATA')
        )
        if path_str:
            return Path(path_str)
        
        # Default location
        default = Path("concept_mesh/data.json")
        return default if default.exists() else None
    
    def get_max_diff_history(self) -> int:
        """Get max diff history size"""
        return int(
            self.config.get('max_diff_history') or
            os.environ.get('CONCEPT_MESH_MAX_DIFF_HISTORY') or
            1000
        )
    
    def get_embedding_cache_size(self) -> int:
        """Get embedding cache size"""
        return int(
            self.config.get('embedding_cache_size') or
            os.environ.get('CONCEPT_MESH_EMBEDDING_CACHE_SIZE') or
            10000
        )
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return int(
            self.config.get('embedding_dim') or
            os.environ.get('CONCEPT_MESH_EMBEDDING_DIM') or
            768
        )

class ConceptMesh:
    """
    Real implementation of concept mesh for knowledge representation
    WITH ALL AUDIT IMPROVEMENTS
    """
    
    # Singleton instance
    _instance: Optional['ConceptMesh'] = None
    # Thread safety lock
    _lock = RLock()
    
    @classmethod
    def instance(cls, config: Optional[Dict[str, Any]] = None) -> 'ConceptMesh':
        """Get or create the singleton instance"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config or {})
                # Register shutdown hook
                atexit.register(cls._instance._save_mesh_on_exit)
            return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (mainly for testing)"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._save_mesh()  # Save before reset
            cls._instance = None
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = ConceptMeshConfig(config)
        
        # Storage
        self.storage_path = self.config.get_storage_path()
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core data structures
        self.concepts: Dict[ConceptID, Concept] = {}
        self.relations: List[ConceptRelation] = []
        self.graph = nx.DiGraph()  # TODO: Consider migration to rust-graph for performance
        
        # Diff tracking
        self.diff_history: deque = deque(maxlen=self.config.get_max_diff_history())
        self.diff_subscribers: Dict[CallbackID, weakref.ref] = {}  # Weak references to prevent leaks
        
        # Indexing
        self.category_index: Dict[str, Set[ConceptID]] = defaultdict(set)
        self.name_index: Dict[str, ConceptID] = {}  # name -> id mapping
        self.kb_index: Dict[str, ConceptID] = {}  # kb_id -> concept_id mapping for deduplication
        
        # Performance - LRU cache for embeddings
        cache_size = self.config.get_embedding_cache_size()
        self._get_embedding_cached = lru_cache(maxsize=cache_size)(self._get_embedding)
        
        # Similarity engine configuration
        self.similarity_engine = config.get('similarity_engine', 'cosine')
        self.penrose_adapter = None
        if self.similarity_engine == 'penrose':
            self._init_penrose()
        
        # Load existing mesh FIRST
        logger.info("Loading persisted mesh data...")
        self._load_mesh()
        
        # THEN load initial data only if empty
        if self.count() == 0:
            logger.info("Mesh is empty, loading initial data...")
            self._load_initial_data()
        else:
            logger.info(f"Restored {self.count()} concepts from disk")
    
    def _save_mesh_on_exit(self) -> None:
        """Save mesh on exit (called by atexit)"""
        try:
            logger.info("Saving ConceptMesh on exit...")
            self._save_mesh()
        except Exception as e:
            logger.error(f"Failed to save mesh on exit: {e}")
    
    def _load_initial_data(self) -> None:
        """Load initial data from configured location if available"""
        # Try to load from the configured location
        data_file = self.config.get_initial_data_file()
        if data_file and data_file.exists() and len(self.concepts) == 0:
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and 'concepts' in data:
                    for concept in data['concepts']:
                        # Generate proper UUID
                        concept_id = str(uuid.uuid4())
                        
                        # Create concept with UUID
                        self.concepts[concept_id] = Concept(
                            id=concept_id,
                            name=concept.get('name', 'unnamed'),
                            description=concept.get('description', ''),
                            category=concept.get('category', 'general'),
                            importance=concept.get('strength', concept.get('importance', 1.0)),
                            embedding=np.array(concept.get('embedding')) if concept.get('embedding') else None,
                            metadata=concept.get('metadata', {})
                        )
                        
                        # Update indices
                        self.name_index[concept.get('name', 'unnamed')] = concept_id
                        self.category_index[concept.get('category', 'general')].add(concept_id)
                        
                    logger.info(f"Loaded {len(data['concepts'])} concepts from {data_file}")
            except FileNotFoundError:
                logger.warning(f"Initial data file not found: {data_file}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in initial data file: {e}")
            except KeyError as e:
                logger.error(f"Missing required field in initial data: {e}")
        
        # Ensure populated with seed concepts if still empty
        if self.count() == 0:
            self.ensure_populated()
        
        # Log final count after all loading is done
        logger.info(f"ConceptMesh ready with {len(self.concepts)} concepts")
    
    def _init_penrose(self) -> None:
        """Initialize Penrose adapter for O(n^2.32) similarity"""
        try:
            from python.core.penrose_adapter import PenroseAdapter
            self.penrose_adapter = PenroseAdapter.get_instance()
            logger.info("✅ Penrose acceleration enabled for ConceptMesh")
        except ImportError:
            logger.warning("⚠️ Penrose not available, falling back to cosine similarity")
            self.similarity_engine = 'cosine'
    
    def count(self) -> int:
        """Return the number of concepts in the mesh"""
        return len(self.concepts)
    
    def add_concept(
        self,
        name: str,
        description: str = "",
        category: str = "general",
        importance: float = 1.0,
        embedding: Optional[EmbeddingVector] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConceptID:
        """Add a new concept to the mesh (with KB deduplication)"""
        with self._lock:
            metadata = metadata or {}
            
            # Check for Wikidata ID in metadata for deduplication
            if 'wikidata_id' in metadata:
                kb_id = metadata['wikidata_id']
                
                # Check if this KB entity already exists
                if kb_id in self.kb_index:
                    existing_id = self.kb_index[kb_id]
                    logger.info(f"KB entity '{kb_id}' already exists as concept {existing_id}")
                    
                    # Update access info
                    self.concepts[existing_id].last_accessed = datetime.now()
                    self.concepts[existing_id].access_count += 1
                    
                    # Update phase metadata if provided
                    if 'entity_phase' in metadata:
                        self.concepts[existing_id].metadata['entity_phase'] = metadata['entity_phase']
                    if 'phase_locked' in metadata:
                        self.concepts[existing_id].metadata['phase_locked'] = metadata['phase_locked']
                    
                    return existing_id
                
                # Mark as canonical entity
                metadata['canonical'] = True
            
            # Check if concept already exists by name
            if name in self.name_index:
                existing_id = self.name_index[name]
                existing_concept = self.concepts[existing_id]
                
                # If existing concept doesn't have KB ID but new one does, update it
                if 'wikidata_id' in metadata and 'wikidata_id' not in existing_concept.metadata:
                    existing_concept.metadata.update({
                        'wikidata_id': metadata['wikidata_id'],
                        'canonical': True,
                        'entity_phase': metadata.get('entity_phase'),
                        'phase_locked': metadata.get('phase_locked', False)
                    })
                    # Add to KB index
                    self.kb_index[metadata['wikidata_id']] = existing_id
                    logger.info(f"Updated concept '{name}' with KB ID {metadata['wikidata_id']}")
                else:
                    logger.info(f"Concept '{name}' already exists with ID {existing_id}")
                
                # Update access info
                existing_concept.last_accessed = datetime.now()
                existing_concept.access_count += 1
                
                return existing_id
            
            # Generate UUID for new concept
            concept_id = str(uuid.uuid4())
            
            # Create concept
            concept = Concept(
                id=concept_id,
                name=name,
                description=description,
                category=category,
                importance=importance,
                embedding=embedding,
                metadata=metadata
            )
            
            # Add to storage
            self.concepts[concept_id] = concept
            self.graph.add_node(concept_id, **asdict(concept))
            
            # Update indices
            self.name_index[name] = concept_id
            self.category_index[category].add(concept_id)
            
            # Update KB index if applicable
            if 'wikidata_id' in metadata:
                self.kb_index[metadata['wikidata_id']] = concept_id
            
            # Cache embedding if provided
            if embedding is not None:
                # Clear cache for this ID to update it
                self._get_embedding_cached.cache_clear()
            
            # Create diff
            diff = ConceptDiff(
                id=str(uuid.uuid4()),
                diff_type="add",
                concepts=[concept_id],
                new_value=asdict(concept)
            )
            self._record_diff(diff)
            
            logger.info(f"Added concept: {name} (ID: {concept_id}){' [KB: ' + metadata.get('wikidata_id', '') + ']' if 'wikidata_id' in metadata else ''}")
            return concept_id
    
    def add_concept_from_kb(
        self,
        name: str,
        kb_id: str,
        entity_type: str = "ENTITY",
        confidence: float = 1.0,
        description: str = "",
        category: str = "entity",
        entity_phase: Optional[float] = None,
        phase_locked: bool = False,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> ConceptID:
        """Add a concept from knowledge base with canonical metadata"""
        metadata = {
            'wikidata_id': kb_id,
            'entity_type': entity_type,
            'confidence': confidence,
            'canonical': True,
            'source': 'entity_linker'
        }
        
        # Add phase information if provided
        if entity_phase is not None:
            metadata['entity_phase'] = entity_phase
            metadata['phase_locked'] = phase_locked
        
        # Merge additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return self.add_concept(
            name=name,
            description=description,
            category=category,
            importance=confidence,
            metadata=metadata
        )
    
    def find_concept_by_kb_id(self, kb_id: str) -> Optional[Concept]:
        """Find a concept by its knowledge base ID"""
        with self._lock:
            concept_id = self.kb_index.get(kb_id)
            if concept_id and concept_id in self.concepts:
                concept = self.concepts[concept_id]
                # Update access info
                concept.last_accessed = datetime.now()
                concept.access_count += 1
                return concept
            return None
    
    def get_canonical_concepts(self) -> List[Concept]:
        """Get all concepts that are linked to knowledge base entities"""
        with self._lock:
            return [
                concept for concept in self.concepts.values()
                if concept.metadata.get('canonical', False)
            ]
    
    def link_concept_to_kb(
        self, 
        concept_id: ConceptID, 
        kb_id: str,
        entity_phase: Optional[float] = None,
        phase_locked: bool = False
    ) -> bool:
        """Link an existing concept to a KB entity"""
        with self._lock:
            if concept_id not in self.concepts:
                logger.warning(f"Concept {concept_id} not found")
                return False
            
            # Check if KB ID already exists
            if kb_id in self.kb_index and self.kb_index[kb_id] != concept_id:
                logger.warning(f"KB ID {kb_id} already linked to different concept")
                return False
            
            concept = self.concepts[concept_id]
            
            # Update metadata
            concept.metadata['wikidata_id'] = kb_id
            concept.metadata['canonical'] = True
            
            if entity_phase is not None:
                concept.metadata['entity_phase'] = entity_phase
                concept.metadata['phase_locked'] = phase_locked
            
            # Update KB index
            self.kb_index[kb_id] = concept_id
            
            logger.info(f"Linked concept {concept_id} to KB entity {kb_id}")
            return True
    
    def remove_concept(self, concept_id: ConceptID) -> bool:
        """Remove a concept from the mesh"""
        with self._lock:
            if concept_id not in self.concepts:
                logger.warning(f"Concept {concept_id} not found")
                return False
            
            concept = self.concepts[concept_id]
            
            # Remove from storage
            del self.concepts[concept_id]
            if self.graph.has_node(concept_id):
                self.graph.remove_node(concept_id)
            
            # Update indices
            if concept.name in self.name_index:
                del self.name_index[concept.name]
            self.category_index[concept.category].discard(concept_id)
            
            # Remove from KB index if applicable
            if 'wikidata_id' in concept.metadata:
                kb_id = concept.metadata['wikidata_id']
                if kb_id in self.kb_index and self.kb_index[kb_id] == concept_id:
                    del self.kb_index[kb_id]
            
            # Clear embedding cache
            self._get_embedding_cached.cache_clear()
            
            # Remove relations
            self.relations = [
                r for r in self.relations 
                if r.source_id != concept_id and r.target_id != concept_id
            ]
            
            # Create diff
            diff = ConceptDiff(
                id=str(uuid.uuid4()),
                diff_type="remove",
                concepts=[concept_id],
                old_value=asdict(concept)
            )
            self._record_diff(diff)
            
            logger.info(f"Removed concept: {concept.name} (ID: {concept_id})")
            return True
    
    def add_relation(
        self,
        source_id: ConceptID,
        target_id: ConceptID,
        relation_type: str,
        strength: float = 1.0,
        bidirectional: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a relation between concepts"""
        with self._lock:
            # Validate concepts exist
            if source_id not in self.concepts or target_id not in self.concepts:
                logger.warning(f"One or both concepts not found: {source_id}, {target_id}")
                return False
            
            # Create relation
            relation = ConceptRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                strength=strength,
                bidirectional=bidirectional,
                metadata=metadata or {}
            )
            
            # Add to storage
            self.relations.append(relation)
            self.graph.add_edge(source_id, target_id, **asdict(relation))
            
            if bidirectional:
                self.graph.add_edge(target_id, source_id, **asdict(relation))
            
            # Create diff
            diff = ConceptDiff(
                id=str(uuid.uuid4()),
                diff_type="relate",
                concepts=[source_id, target_id],
                new_value=asdict(relation)
            )
            self._record_diff(diff)
            
            logger.info(f"Added relation: {source_id} -{relation_type}-> {target_id}")
            return True
    
    def find_concept(self, name: str) -> Optional[Concept]:
        """Find a concept by name"""
        with self._lock:
            concept_id = self.name_index.get(name)
            if concept_id:
                concept = self.concepts[concept_id]
                # Update access info
                concept.last_accessed = datetime.now()
                concept.access_count += 1
                return concept
            return None
    
    def get_related_concepts(
        self,
        concept_id: ConceptID,
        relation_type: Optional[str] = None,
        max_depth: int = 1
    ) -> List[Tuple[ConceptID, str, float]]:
        """Get concepts related to the given concept"""
        with self._lock:
            if concept_id not in self.concepts:
                return []
            
            related = []
            
            if max_depth == 1:
                # Direct relations only
                for relation in self.relations:
                    if relation.source_id == concept_id:
                        if relation_type is None or relation.relation_type == relation_type:
                            related.append((
                                relation.target_id,
                                relation.relation_type,
                                relation.strength
                            ))
                    elif relation.bidirectional and relation.target_id == concept_id:
                        if relation_type is None or relation.relation_type == relation_type:
                            related.append((
                                relation.source_id,
                                relation.relation_type,
                                relation.strength
                            ))
            else:
                # Use graph traversal for deeper relations
                try:
                    paths = nx.single_source_shortest_path(
                        self.graph, concept_id, cutoff=max_depth
                    )
                    for target_id, path in paths.items():
                        if target_id != concept_id:
                            # Calculate aggregate strength
                            strength = 1.0
                            for i in range(len(path) - 1):
                                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                                if edge_data:
                                    strength *= edge_data.get('strength', 1.0)
                            
                            related.append((target_id, 'path', strength))
                except nx.NetworkXError as e:
                    logger.warning(f"Graph traversal error: {e}")
            
            return related
    
    def _get_embedding(self, concept_id: ConceptID) -> Optional[EmbeddingVector]:
        """Get embedding for a concept (cached via LRU)"""
        concept = self.concepts.get(concept_id)
        if concept and concept.embedding is not None:
            return concept.embedding
        return None
    
    def find_similar_concepts(
        self,
        concept_id: ConceptID,
        threshold: float = 0.7,
        max_results: int = 10
    ) -> List[Tuple[ConceptID, float]]:
        """Find concepts similar to the given concept"""
        if concept_id not in self.concepts:
            return []
        
        concept = self.concepts[concept_id]
        
        # Get embedding (from cache if available)
        query_embedding = self._get_embedding_cached(concept_id)
        
        if query_embedding is None:
            # Fallback to name/category similarity
            return self._find_similar_by_metadata(concept, threshold, max_results)
        
        # Use Penrose if enabled
        if self.similarity_engine == 'penrose' and self.penrose_adapter:
            return self.find_similar_concepts_penrose(concept_id, threshold, max_results)
        
        # Default cosine similarity
        similarities = []
        for other_id, other_concept in self.concepts.items():
            if other_id == concept_id:
                continue
            
            # Get other embedding (from cache)
            other_embedding = self._get_embedding_cached(other_id)
            if other_embedding is None:
                continue
            
            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, other_embedding)
            if similarity >= threshold:
                similarities.append((other_id, similarity))
        
        # Sort and limit
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]
    
    def find_similar_concepts_penrose(
        self,
        concept_id: ConceptID,
        threshold: float = 0.7,
        max_results: int = 10
    ) -> List[Tuple[ConceptID, float]]:
        """Find similar concepts using Penrose O(n^2.32) algorithm"""
        if self.similarity_engine != 'penrose' or not self.penrose_adapter:
            return self.find_similar_concepts(concept_id, threshold, max_results)
        
        # Get query embedding
        concept = self.concepts.get(concept_id)
        if not concept or concept.embedding is None:
            return []
        
        # Build embedding matrix for all concepts
        concept_ids = []
        embeddings = []
        for cid, c in self.concepts.items():
            if c.embedding is not None and cid != concept_id:
                concept_ids.append(cid)
                embeddings.append(c.embedding)
        
        if not embeddings:
            return []
        
        # Use Penrose for batch similarity
        similarities = self.penrose_adapter.batch_similarity(
            query=concept.embedding,
            corpus=np.array(embeddings)
        )
        
        # Filter and sort results
        results = [
            (concept_ids[i], float(sim))
            for i, sim in enumerate(similarities)
            if sim >= threshold
        ]
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def _find_similar_by_metadata(
        self,
        concept: Concept,
        threshold: float,
        max_results: int
    ) -> List[Tuple[ConceptID, float]]:
        """Find similar concepts using metadata"""
        similarities = []
        
        for other_id, other_concept in self.concepts.items():
            if other_id == concept.id:
                continue
            
            # Simple similarity based on category and name overlap
            similarity = 0.0
            
            # Category match
            if concept.category == other_concept.category:
                similarity += 0.5
            
            # Name similarity (word overlap)
            concept_words = set(concept.name.lower().split())
            other_words = set(other_concept.name.lower().split())
            if concept_words and other_words:
                overlap = len(concept_words & other_words)
                similarity += 0.5 * (overlap / max(len(concept_words), len(other_words)))
            
            if similarity >= threshold:
                similarities.append((other_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]
    
    def _cosine_similarity(self, vec1: EmbeddingVector, vec2: EmbeddingVector) -> float:
        """Calculate cosine similarity between vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def extract_concepts_from_text(
        self,
        text: str,
        min_importance: float = 0.5
    ) -> List[ConceptID]:
        """Extract concepts from text and add to mesh"""
        with self._lock:
            # Simple concept extraction (can be enhanced with NLP)
            words = text.lower().split()
            
            # Extract meaningful words
            concepts_found = []
            word_counts = defaultdict(int)
            
            for word in words:
                # Clean word
                word = word.strip('.,!?;:"')
                if len(word) > 2 and word not in STOP_WORDS:
                    word_counts[word] += 1
            
            # Add concepts based on frequency and importance
            for word, count in word_counts.items():
                importance = min(1.0, count / 10.0)  # Simple importance calculation
                if importance >= min_importance:
                    concept_id = self.add_concept(
                        name=word,
                        category="extracted",
                        importance=importance,
                        metadata={'source': 'text_extraction', 'count': count}
                    )
                    concepts_found.append(concept_id)
            
            # Find potential relations (co-occurrence)
            words_list = [w for w in words if w not in STOP_WORDS and len(w) > 2]
            for i in range(len(words_list) - 1):
                word1 = words_list[i].strip('.,!?;:"')
                word2 = words_list[i + 1].strip('.,!?;:"')
                
                if word1 in self.name_index and word2 in self.name_index:
                    id1 = self.name_index[word1]
                    id2 = self.name_index[word2]
                    self.add_relation(id1, id2, "co_occurs", strength=0.5)
            
            return concepts_found
    
    def get_concept_clusters(self, min_size: int = 3) -> List[Set[ConceptID]]:
        """Find clusters of highly connected concepts"""
        with self._lock:
            # Use community detection
            try:
                communities = nx.community.greedy_modularity_communities(
                    self.graph.to_undirected()
                )
                clusters = [set(community) for community in communities if len(community) >= min_size]
                return clusters
            except Exception as e:
                logger.warning(f"Community detection failed: {e}")
                # Fallback to connected components
                try:
                    components = list(nx.weakly_connected_components(self.graph))
                    return [comp for comp in components if len(comp) >= min_size]
                except nx.NetworkXError as e:
                    logger.error(f"Connected components failed: {e}")
                    return []
    
    def calculate_concept_importance(self, concept_id: ConceptID) -> float:
        """Calculate importance based on connections and usage"""
        if concept_id not in self.concepts:
            return 0.0
        
        concept = self.concepts[concept_id]
        base_importance = concept.importance
        
        # Factor in connections
        in_degree = self.graph.in_degree(concept_id) if self.graph.has_node(concept_id) else 0
        out_degree = self.graph.out_degree(concept_id) if self.graph.has_node(concept_id) else 0
        connection_factor = min(1.0, (in_degree + out_degree) / 10.0)
        
        # Factor in access frequency
        access_factor = min(1.0, concept.access_count / 100.0)
        
        # Calculate final importance
        final_importance = (
            0.4 * base_importance +
            0.4 * connection_factor +
            0.2 * access_factor
        )
        
        return final_importance
    
    def prune_mesh(self, importance_threshold: float = 0.1, age_days: int = 30) -> int:
        """Remove low-importance, old concepts"""
        with self._lock:
            current_time = datetime.now()
            concepts_to_remove = []
            
            for concept_id, concept in self.concepts.items():
                # Calculate age
                age = (current_time - concept.created_at).days
                
                # Calculate importance
                importance = self.calculate_concept_importance(concept_id)
                
                # Check if should be pruned
                if importance < importance_threshold and age > age_days:
                    concepts_to_remove.append(concept_id)
            
            # Remove concepts
            removed_count = 0
            for concept_id in concepts_to_remove:
                if self.remove_concept(concept_id):
                    removed_count += 1
            
            logger.info(f"Pruned {removed_count} concepts from mesh")
            return removed_count
    
    def _record_diff(self, diff: ConceptDiff) -> None:
        """Record a diff and notify subscribers"""
        self.diff_history.append(diff)
        
        # Notify subscribers (clean up dead references)
        dead_refs = []
        for callback_id, callback_ref in self.diff_subscribers.items():
            callback = callback_ref()
            if callback is None:
                dead_refs.append(callback_id)
            else:
                try:
                    callback(diff)
                except Exception as e:
                    logger.error(f"Error notifying diff subscriber: {e}")
        
        # Clean up dead references
        for ref_id in dead_refs:
            del self.diff_subscribers[ref_id]
    
    def subscribe_to_diffs(self, callback: Callable[[ConceptDiff], None]) -> CallbackID:
        """Subscribe to concept diff events - returns subscription ID for unsubscribing"""
        callback_id = str(uuid.uuid4())
        self.diff_subscribers[callback_id] = weakref.ref(callback)
        return callback_id
    
    def unsubscribe_from_diffs(self, callback_id: CallbackID) -> bool:
        """Unsubscribe from diff events"""
        if callback_id in self.diff_subscribers:
            del self.diff_subscribers[callback_id]
            return True
        return False
    
    def get_diff_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent diff history"""
        with self._lock:
            history = list(self.diff_history)[-limit:]
            return [asdict(diff) for diff in history]
    
    def _save_mesh(self) -> None:
        """Save mesh to disk with atomic writes"""
        with self._lock:
            try:
                # Save concepts with atomic write
                concepts_file = self.storage_path / "concepts.pkl.gz"
                concepts_temp = self.storage_path / "concepts.pkl.gz.tmp"
                
                with gzip.open(concepts_temp, 'wb') as f:
                    # Convert numpy arrays to lists for serialization
                    concepts_data = {}
                    for cid, concept in self.concepts.items():
                        concept_dict = asdict(concept)
                        if concept.embedding is not None:
                            concept_dict['embedding'] = concept.embedding.tolist()
                        concepts_data[cid] = concept_dict
                    pickle.dump(concepts_data, f)
                
                # Atomic rename
                concepts_temp.replace(concepts_file)
                
                # Save relations with atomic write
                relations_file = self.storage_path / "relations.pkl.gz"
                relations_temp = self.storage_path / "relations.pkl.gz.tmp"
                
                with gzip.open(relations_temp, 'wb') as f:
                    relations_data = [asdict(r) for r in self.relations]
                    pickle.dump(relations_data, f)
                
                relations_temp.replace(relations_file)
                
                # Save indices with atomic write
                indices_file = self.storage_path / "indices.json"
                indices_temp = self.storage_path / "indices.json.tmp"
                
                with open(indices_temp, 'w') as f:
                    json.dump({
                        'name_index': self.name_index,
                        'category_index': {k: list(v) for k, v in self.category_index.items()},
                        'kb_index': self.kb_index
                    }, f)
                
                indices_temp.replace(indices_file)
                
                logger.info(f"Mesh saved to disk: {len(self.concepts)} concepts, {len(self.relations)} relations")
                
            except (IOError, OSError) as e:
                logger.error(f"Failed to save mesh - I/O error: {e}")
                raise
            except pickle.PickleError as e:
                logger.error(f"Failed to save mesh - Pickle error: {e}")
                raise
    
    def _load_mesh(self) -> None:
        """Load mesh from disk"""
        # Load concepts
        concepts_file = self.storage_path / "concepts.pkl.gz"
        if concepts_file.exists():
            try:
                with gzip.open(concepts_file, 'rb') as f:
                    concepts_data = pickle.load(f)
                
                for cid, concept_dict in concepts_data.items():
                    # Convert embedding back to numpy
                    if 'embedding' in concept_dict and concept_dict['embedding'] is not None:
                        concept_dict['embedding'] = np.array(concept_dict['embedding'])
                    
                    # Convert datetime strings back to datetime objects
                    for field in ['created_at', 'last_accessed']:
                        if field in concept_dict and isinstance(concept_dict[field], str):
                            concept_dict[field] = datetime.fromisoformat(concept_dict[field])
                    
                    concept = Concept(**concept_dict)
                    self.concepts[cid] = concept
                    self.graph.add_node(cid, **asdict(concept))
                
                logger.info(f"Loaded {len(self.concepts)} concepts from disk")
            except FileNotFoundError:
                logger.info("No concepts file found, starting with empty mesh")
            except (IOError, OSError) as e:
                logger.error(f"Failed to load concepts - I/O error: {e}")
            except pickle.UnpicklingError as e:
                logger.error(f"Failed to load concepts - Unpickling error: {e}")
            except gzip.BadGzipFile as e:
                logger.error(f"Failed to load concepts - Bad gzip file: {e}")
        
        # Load relations
        relations_file = self.storage_path / "relations.pkl.gz"
        if relations_file.exists():
            try:
                with gzip.open(relations_file, 'rb') as f:
                    relations_data = pickle.load(f)
                
                for relation_dict in relations_data:
                    # Convert datetime strings
                    if 'created_at' in relation_dict and isinstance(relation_dict['created_at'], str):
                        relation_dict['created_at'] = datetime.fromisoformat(relation_dict['created_at'])
                    
                    relation = ConceptRelation(**relation_dict)
                    self.relations.append(relation)
                    self.graph.add_edge(relation.source_id, relation.target_id, **asdict(relation))
                    
                    if relation.bidirectional:
                        self.graph.add_edge(relation.target_id, relation.source_id, **asdict(relation))
                
                logger.info(f"Loaded {len(self.relations)} relations from disk")
            except FileNotFoundError:
                logger.info("No relations file found")
            except (IOError, OSError) as e:
                logger.error(f"Failed to load relations - I/O error: {e}")
            except pickle.UnpicklingError as e:
                logger.error(f"Failed to load relations - Unpickling error: {e}")
            except gzip.BadGzipFile as e:
                logger.error(f"Failed to load relations - Bad gzip file: {e}")
        
        # Load indices
        indices_file = self.storage_path / "indices.json"
        if indices_file.exists():
            try:
                with open(indices_file, 'r') as f:
                    indices_data = json.load(f)
                
                self.name_index = indices_data.get('name_index', {})
                category_index = indices_data.get('category_index', {})
                self.category_index = defaultdict(set)
                for k, v in category_index.items():
                    self.category_index[k] = set(v)
                
                # Load KB index
                self.kb_index = indices_data.get('kb_index', {})
                
                logger.info("Loaded indices from disk")
            except FileNotFoundError:
                logger.info("No indices file found")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to load indices - JSON decode error: {e}")
            except (IOError, OSError) as e:
                logger.error(f"Failed to load indices - I/O error: {e}")
    
    def export_to_json(self, file_path: Path) -> bool:
        """Export mesh to JSON format"""
        with self._lock:
            try:
                export_data = {
                    'concepts': [],
                    'relations': [],
                    'metadata': {
                        'export_time': datetime.now().isoformat(),
                        'concept_count': len(self.concepts),
                        'relation_count': len(self.relations),
                        'version': '2.0'  # Version with all improvements
                    }
                }
                
                # Export concepts
                for concept in self.concepts.values():
                    concept_dict = asdict(concept)
                    # Convert numpy arrays and datetime objects
                    if concept.embedding is not None:
                        concept_dict['embedding'] = concept.embedding.tolist()
                    concept_dict['created_at'] = concept.created_at.isoformat()
                    concept_dict['last_accessed'] = concept.last_accessed.isoformat()
                    export_data['concepts'].append(concept_dict)
                
                # Export relations
                for relation in self.relations:
                    relation_dict = asdict(relation)
                    relation_dict['created_at'] = relation.created_at.isoformat()
                    export_data['relations'].append(relation_dict)
                
                # Write to file atomically
                temp_path = file_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                temp_path.replace(file_path)
                
                logger.info(f"Exported mesh to {file_path}")
                return True
                
            except (IOError, OSError) as e:
                logger.error(f"Failed to export mesh - I/O error: {e}")
                return False
            except json.JSONEncodeError as e:
                logger.error(f"Failed to export mesh - JSON encode error: {e}")
                return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mesh statistics"""
        with self._lock:
            stats = {
                'total_concepts': len(self.concepts),
                'total_relations': len(self.relations),
                'canonical_concepts': len(self.kb_index),
                'categories': dict(self.category_index),
                'graph_density': nx.density(self.graph) if len(self.graph) > 0 else 0,
                'connected_components': nx.number_weakly_connected_components(self.graph) if len(self.graph) > 0 else 0,
                'avg_degree': sum(dict(self.graph.degree()).values()) / len(self.graph) if len(self.graph) > 0 else 0,
                'storage_path': str(self.storage_path),
                'cache_info': self._get_embedding_cached.cache_info()._asdict()
            }
            
            # Category distribution
            category_counts = {}
            for category, concept_ids in self.category_index.items():
                category_counts[category] = len(concept_ids)
            stats['category_distribution'] = category_counts
            
            # Most connected concepts
            if self.concepts:
                degrees = dict(self.graph.degree())
                top_concepts = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
                stats['most_connected'] = [
                    {'id': cid, 'name': self.concepts[cid].name, 'connections': deg}
                    for cid, deg in top_concepts if cid in self.concepts
                ]
            
            return stats
    
    def initialize_user(self, user_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Initialize user-specific state in the ConceptMesh
        
        This method ensures the mesh is properly set up for a specific user,
        including loading seed concepts if needed and setting up user-specific
        configuration.
        
        Args:
            user_id: Optional user identifier for user-specific initialization
            **kwargs: Additional configuration parameters
            
        Returns:
            Dict containing initialization status and mesh statistics
        """
        with self._lock:
            logger.info(f"🚀 Initializing ConceptMesh for user: {user_id or 'default'}")
            
            # Initialize user-specific metadata if provided
            if user_id:
                if not hasattr(self, 'user_metadata'):
                    self.user_metadata = {}
                
                # Set up user-specific configuration
                self.user_metadata[user_id] = {
                    'initialized_at': datetime.now().isoformat(),
                    'user_id': user_id,
                    'config': kwargs
                }
                
                # Create user-specific storage path if needed
                user_storage_path = self.storage_path / f"users/{user_id}"
                user_storage_path.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"📁 Created user storage path: {user_storage_path}")
            
            # Ensure the mesh is populated with seed concepts
            initial_count = self.count()
            if initial_count == 0:
                logger.info("🌱 Mesh empty, loading seed concepts...")
                self.ensure_populated()
                loaded_count = self.count()
                logger.info(f"✅ Loaded {loaded_count} seed concepts")
            else:
                logger.info(f"📊 Mesh already has {initial_count} concepts")
            
            # Get current mesh statistics
            stats = self.get_statistics()
            
            # Create initialization diff for tracking
            diff = ConceptDiff(
                id=str(uuid.uuid4()),
                diff_type="initialize_user",
                concepts=[],
                new_value={
                    'user_id': user_id,
                    'initialization_time': datetime.now().isoformat(),
                    'concept_count': stats['total_concepts'],
                    'relation_count': stats['total_relations']
                },
                metadata={'action': 'user_initialization'}
            )
            self._record_diff(diff)
            
            # Fire initialization event
            try:
                from python.core.event_bus import global_event_bus
                if global_event_bus:
                    global_event_bus.publish("concept_mesh_initialized", {
                        'user_id': user_id,
                        'concept_count': stats['total_concepts'],
                        'initialization_time': datetime.now().isoformat()
                    })
            except ImportError:
                pass
            
            initialization_result = {
                'success': True,
                'user_id': user_id,
                'concept_count': stats['total_concepts'],
                'relation_count': stats['total_relations'],
                'mesh_ready': True,
                'storage_path': str(self.storage_path),
                'initialization_time': datetime.now().isoformat(),
                'message': f"ConceptMesh initialized successfully for {user_id or 'default user'}"
            }
            
            logger.info(f"✅ ConceptMesh initialization complete: {stats['total_concepts']} concepts, {stats['total_relations']} relations")
            return initialization_result
    
    def shutdown(self) -> None:
        """Save mesh and cleanup"""
        self._save_mesh()
        logger.info("ConceptMesh shutdown complete")

    def load_seeds(self, seed_file: Optional[str] = None) -> int:
        """Load seed concepts from file"""
        if seed_file is None:
            # Get seed files from config
            seed_files = self.config.get_seed_files()
            
            for sf in seed_files:
                if os.path.exists(sf):
                    seed_file = sf
                    break
        
        if not seed_file or not os.path.exists(seed_file):
            logger.warning("No seed file found")
            return 0
        
        try:
            with open(seed_file, 'r', encoding='utf-8') as f:
                seeds = json.load(f)
            
            count = 0
            for seed in seeds:
                if isinstance(seed, dict) and 'name' in seed:
                    self.add_concept(
                        name=seed['name'],
                        description=seed.get('description', ''),
                        category=seed.get('category', 'general'),
                        importance=seed.get('priority', seed.get('importance', 0.5)),
                        metadata=seed
                    )
                    count += 1
            
            logger.info(f"Loaded {count} seed concepts from {seed_file}")
            return count
            
        except FileNotFoundError:
            logger.error(f"Seed file not found: {seed_file}")
            return 0
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in seed file: {e}")
            return 0
        except KeyError as e:
            logger.error(f"Missing required field in seed: {e}")
            return 0
    
    def ensure_populated(self) -> None:
        """Ensure mesh has at least some concepts"""
        if self.count() == 0:
            loaded = self.load_seeds()
            if loaded > 0:
                logger.info(f"Concept mesh populated with {loaded} seed concepts")
                # Fire events for lattice
                self.fire_seed_events()
            else:
                logger.warning("Concept mesh remains empty - no seeds loaded")
    
    def fire_seed_events(self) -> None:
        """Publish concept_added events for all concepts"""
        # Try to get event bus from various sources
        event_bus = None
        
        # Try global event bus
        try:
            from python.core.event_bus import global_event_bus
            event_bus = global_event_bus
        except ImportError:
            pass
        
        # Try tori_globals
        if not event_bus:
            try:
                from python.core.tori_globals import event_bus as global_bus
                event_bus = global_bus
            except ImportError:
                pass
        
        if event_bus:
            for concept_id, concept in self.concepts.items():
                event_data = {
                    "id": concept_id,
                    "name": concept.name,
                    "category": concept.category,
                    "importance": concept.importance,
                    "data": asdict(concept)
                }
                try:
                    event_bus.publish("concept_added", event_data)
                    logger.debug(f"Published concept_added event for {concept.name}")
                except Exception as e:
                    logger.error(f"Failed to publish event for {concept.name}: {e}")
            logger.info(f"Published {len(self.concepts)} concept_added events")
        else:
            logger.warning("No event bus available to publish concept events")

    def record_diff(self, diff: ConceptDiff) -> None:
        """Record a diff from external source (e.g., Rust extension)"""
        with self._lock:
            self._record_diff(diff)
            # Apply the diff to update internal state
            self._apply_diff(diff)
    
    def inject_psi_fields(self, 
                         concept_id: str,
                         psi_phase: Union[float, np.ndarray],
                         psi_amplitude: Union[float, np.ndarray],
                         origin: str = "Kretschmann_scalar",
                         coordinates: Optional[Dict[str, np.ndarray]] = None,
                         persistence_mode: str = "persistent",
                         curvature_value: Optional[float] = None,
                         gradient_field: Optional[np.ndarray] = None) -> ConceptID:
        """
        🚀 INJECT ψ-PHASE AND ψ-AMPLITUDE INTO CONCEPT MESH
        
        This weaponizes spacetime geometry into cognitive memory fields!
        
        Args:
            concept_id: Target concept ID or name (auto-creates if missing)
            psi_phase: Phase field from curvature encoding (radians)
            psi_amplitude: Amplitude field (memory density modulation)
            origin: Source of curvature (e.g., "Kretschmann_scalar", "Ricci_tensor")
            coordinates: Mesh coordinates for field interpolation
            persistence_mode: "persistent", "volatile", or "chaotic_collapse"
            curvature_value: Optional raw curvature value
            gradient_field: Optional ∇ψ-phase for soliton coupling
            
        Returns:
            Concept ID with injected phase fields
        """
        with self._lock:
            # Find or create concept
            if concept_id in self.concepts:
                concept = self.concepts[concept_id]
            elif concept_id in self.name_index:
                concept_id = self.name_index[concept_id]
                concept = self.concepts[concept_id]
            else:
                # Auto-create concept for phase injection
                logger.info(f"🌀 Auto-creating concept '{concept_id}' for ψ-field injection")
                new_id = self.add_concept(
                    name=concept_id,
                    description=f"Phase-encoded concept from {origin}",
                    category="phase_encoded",
                    importance=1.0,
                    metadata={
                        'auto_created': True,
                        'phase_origin': origin
                    }
                )
                concept = self.concepts[new_id]
                concept_id = new_id
            
            # Prepare phase metadata
            phase_metadata = {
                'psi_phase': psi_phase.tolist() if isinstance(psi_phase, np.ndarray) else psi_phase,
                'psi_amplitude': psi_amplitude.tolist() if isinstance(psi_amplitude, np.ndarray) else psi_amplitude,
                'origin': origin,
                'timestamp': datetime.now().isoformat(),
                'persistence_mode': persistence_mode,
                'phase_locked': True,
                'phase_version': concept.metadata.get('phase_version', 0) + 1
            }
            
            # Add optional fields
            if coordinates:
                phase_metadata['coordinates'] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                                for k, v in coordinates.items()}
            if curvature_value is not None:
                phase_metadata['curvature_value'] = float(curvature_value)
            if gradient_field is not None:
                phase_metadata['gradient_field'] = gradient_field.tolist() if isinstance(gradient_field, np.ndarray) else gradient_field
            
            # Compute phase statistics
            if isinstance(psi_phase, np.ndarray):
                phase_metadata['phase_stats'] = {
                    'mean': float(np.mean(psi_phase)),
                    'std': float(np.std(psi_phase)),
                    'max': float(np.max(psi_phase)),
                    'min': float(np.min(psi_phase)),
                    'vorticity': float(np.sum(np.gradient(psi_phase))) if psi_phase.ndim == 1 else 0.0
                }
            
            # Update concept metadata
            concept.metadata.update(phase_metadata)
            
            # Update importance based on amplitude
            if isinstance(psi_amplitude, (int, float)):
                # Weight importance by amplitude (memory density)
                concept.importance = max(concept.importance, float(psi_amplitude))
            elif isinstance(psi_amplitude, np.ndarray):
                # Use mean amplitude
                concept.importance = max(concept.importance, float(np.mean(psi_amplitude)))
            
            # Create phase injection diff
            diff = ConceptDiff(
                id=str(uuid.uuid4()),
                diff_type="psi_inject",
                concepts=[concept_id],
                new_value=phase_metadata,
                metadata={
                    'injection_type': 'phase_field',
                    'origin': origin,
                    'persistence': persistence_mode
                }
            )
            self._record_diff(diff)
            
            # Fire phase injection event
            try:
                from python.core.event_bus import global_event_bus
                if global_event_bus:
                    global_event_bus.publish("psi_field_injected", {
                        'concept_id': concept_id,
                        'concept_name': concept.name,
                        'phase_metadata': phase_metadata
                    })
            except ImportError:
                pass
            
            logger.info(f"💉 Injected ψ-fields into concept '{concept.name}' (ID: {concept_id})")
            logger.info(f"   Phase: {phase_metadata.get('phase_stats', {}).get('mean', psi_phase):.3f}")
            logger.info(f"   Amplitude: {np.mean(psi_amplitude) if isinstance(psi_amplitude, np.ndarray) else psi_amplitude:.3f}")
            logger.info(f"   Origin: {origin}, Mode: {persistence_mode}")
            
            return concept_id
    
    def get_psi_fields(self, concept_id: ConceptID) -> Optional[Dict[str, Any]]:
        """
        Retrieve ψ-phase and ψ-amplitude fields for a concept
        
        Returns:
            Dict with psi_phase, psi_amplitude, and metadata, or None if not found
        """
        with self._lock:
            if concept_id not in self.concepts:
                # Try by name
                if concept_id in self.name_index:
                    concept_id = self.name_index[concept_id]
                else:
                    return None
            
            concept = self.concepts[concept_id]
            
            # Extract phase fields from metadata
            if 'psi_phase' in concept.metadata:
                return {
                    'psi_phase': concept.metadata.get('psi_phase'),
                    'psi_amplitude': concept.metadata.get('psi_amplitude'),
                    'origin': concept.metadata.get('origin'),
                    'coordinates': concept.metadata.get('coordinates'),
                    'curvature_value': concept.metadata.get('curvature_value'),
                    'gradient_field': concept.metadata.get('gradient_field'),
                    'phase_stats': concept.metadata.get('phase_stats'),
                    'timestamp': concept.metadata.get('timestamp'),
                    'persistence_mode': concept.metadata.get('persistence_mode'),
                    'phase_version': concept.metadata.get('phase_version', 1)
                }
            
            return None
    
    def find_phase_encoded_concepts(self, origin_filter: Optional[str] = None) -> List[Tuple[ConceptID, Dict[str, Any]]]:
        """
        Find all concepts with injected ψ-fields
        
        Args:
            origin_filter: Optional filter by origin (e.g., "Kretschmann_scalar")
            
        Returns:
            List of (concept_id, phase_metadata) tuples
        """
        with self._lock:
            phase_concepts = []
            
            for concept_id, concept in self.concepts.items():
                if 'psi_phase' in concept.metadata:
                    if origin_filter is None or concept.metadata.get('origin') == origin_filter:
                        phase_data = self.get_psi_fields(concept_id)
                        if phase_data:
                            phase_concepts.append((concept_id, phase_data))
            
            # Sort by phase version (most recent first)
            phase_concepts.sort(key=lambda x: x[1].get('phase_version', 0), reverse=True)
            
            return phase_concepts
    
    def _apply_diff(self, diff: ConceptDiff) -> None:
        """Apply a diff to update the mesh state"""
        try:
            if diff.diff_type == "add" and diff.new_value:
                # Add concept from diff
                concept_data = diff.new_value
                if isinstance(concept_data, dict) and 'name' in concept_data:
                    self.add_concept(
                        name=concept_data['name'],
                        description=concept_data.get('description', ''),
                        category=concept_data.get('category', 'general'),
                        importance=concept_data.get('importance', 1.0),
                        metadata=concept_data.get('metadata', {})
                    )
            
            elif diff.diff_type == "remove" and diff.concepts:
                # Remove concepts
                for concept_id in diff.concepts:
                    self.remove_concept(concept_id)
            
            elif diff.diff_type == "relate" and len(diff.concepts) >= 2 and diff.new_value:
                # Add relation from diff
                relation_data = diff.new_value
                if isinstance(relation_data, dict):
                    self.add_relation(
                        source_id=diff.concepts[0],
                        target_id=diff.concepts[1],
                        relation_type=relation_data.get('relation_type', 'related_to'),
                        strength=relation_data.get('strength', 1.0),
                        bidirectional=relation_data.get('bidirectional', False),
                        metadata=relation_data.get('metadata', {})
                    )
            
        except KeyError as e:
            logger.error(f"Missing required field in diff {diff.id}: {e}")
        except ValueError as e:
            logger.error(f"Invalid value in diff {diff.id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error applying diff {diff.id}: {e}")


    # ========================================================================
    # INTENT ANCHORING METHODS
    # ========================================================================
    
    def find_or_create_anchor(self, text: str, intent_type: str = None) -> Optional[str]:
        """
        Find or create an anchor node for an intent.
        
        Args:
            text: Intent description or user query
            intent_type: Type of intent (optional)
            
        Returns:
            Node ID of anchor node
        """
        # First try to find existing concepts
        extracted = self.extract_concepts_from_text(text)
        
        if extracted:
            # Use the most relevant extracted concept
            return extracted[0]  # First is usually most relevant
        
        # No existing concept found, create a new one
        node_type = f"intent_{intent_type}" if intent_type else "intent"
        anchor_id = self.add_concept(
            name=text[:50],  # Truncate for name
            description=text,
            category=node_type,
            metadata={"auto_created": True, "timestamp": time.time()}
        )
        
        logger.info(f"Created new anchor node {anchor_id} for intent: {text[:50]}")
        return anchor_id
    
    def traverse_context(self, node_id: str, depth: int = 2) -> Dict[str, Any]:
        """
        Traverse semantic context from an anchor node.
        
        Args:
            node_id: Starting node ID
            depth: How deep to traverse
            
        Returns:
            Context dictionary with nodes, relations, and metadata
        """
        if node_id not in self.concepts:
            return {}
        
        context = {
            "anchor": node_id,
            "nodes": [],
            "relations": [],
            "keywords": set(),
            "types": set()
        }
        
        # Get related concepts
        related = self.get_related_concepts(node_id, max_depth=depth)
        
        for concept_id, relation_type, strength in related:
            if concept_id in self.concepts:
                concept = self.concepts[concept_id]
                
                # Add to context
                context["nodes"].append(concept_id)
                context["relations"].append({
                    "from": node_id,
                    "to": concept_id,
                    "type": relation_type,
                    "strength": strength
                })
                
                # Extract keywords from concept
                if concept.name:
                    for word in concept.name.lower().split():
                        if len(word) > 3:  # Skip short words
                            context["keywords"].add(word)
                
                # Track types
                if concept.category:
                    context["types"].add(concept.category)
        
        # Convert sets to lists for serialization
        context["keywords"] = list(context["keywords"])
        context["types"] = list(context["types"])
        
        return context
    
    def calculate_intent_coverage(self, anchor_id: str, covered_concepts: List[str]) -> float:
        """
        Calculate how much of an intent's semantic context is covered.
        
        Args:
            anchor_id: Intent anchor node
            covered_concepts: List of concept IDs that have been addressed
            
        Returns:
            Coverage score between 0 and 1
        """
        context = self.traverse_context(anchor_id, depth=2)
        if not context or not context["nodes"]:
            return 0.0
        
        total_nodes = set(context["nodes"])
        covered_nodes = set(covered_concepts) & total_nodes
        
        # Weight by relation strength
        weighted_total = 0.0
        weighted_covered = 0.0
        
        for relation in context["relations"]:
            weight = relation["strength"]
            weighted_total += weight
            if relation["to"] in covered_nodes:
                weighted_covered += weight
        
        if weighted_total > 0:
            return weighted_covered / weighted_total
        else:
            return len(covered_nodes) / len(total_nodes) if total_nodes else 0.0
    
    def suggest_intent_path(self, from_anchor: str, to_anchor: str) -> List[str]:
        """
        Suggest a path from one intent to another through the mesh.
        
        Args:
            from_anchor: Starting intent anchor
            to_anchor: Target intent anchor
            
        Returns:
            List of node IDs forming a path
        """
        # Simple BFS pathfinding
        if from_anchor not in self.concepts or to_anchor not in self.concepts:
            return []
        
        visited = set()
        queue = [(from_anchor, [from_anchor])]
        
        while queue:
            current, path = queue.pop(0)
            
            if current == to_anchor:
                return path
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Get neighbors
            if current in self.relations:
                for neighbor_id in self.relations[current]:
                    if neighbor_id not in visited:
                        queue.append((neighbor_id, path + [neighbor_id]))
        
        return []  # No path found


class PenroseConceptMesh(ConceptMesh):
    """ConceptMesh with Penrose-accelerated similarity"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        config['similarity_engine'] = 'penrose'
        super().__init__(config)
    
    async def add_relations_from_penrose(
        self, 
        similar: np.ndarray, 
        concept_ids: List[ConceptID], 
        threshold: float = 0.7
    ) -> int:
        """Add relations based on Penrose similarity matrix"""
        n = len(concept_ids)
        relations_added = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if similar[i, j] >= threshold:
                    # Add similarity-based relation
                    success = self.add_relation(
                        concept_ids[i],
                        concept_ids[j],
                        'similar_to',
                        strength=float(similar[i, j]),
                        bidirectional=True,
                        metadata={'similarity_engine': 'penrose'}
                    )
                    if success:
                        relations_added += 1
        
        logger.info(f"Added {relations_added} Penrose-based relations")
        return relations_added


# FastAPI integration helpers
def get_mesh() -> ConceptMesh:
    """Dependency injection for FastAPI"""
    return ConceptMesh.instance()


# Example usage
if __name__ == "__main__":
    # Create mesh with custom config
    config = {
        'storage_path': 'data/test_mesh',
        'embedding_cache_size': 5000,
        'max_diff_history': 500
    }
    mesh = ConceptMesh(config)
    
    # Add some concepts
    ai_id = mesh.add_concept("artificial intelligence", "AI and machine learning", "technology")
    ml_id = mesh.add_concept("machine learning", "Subset of AI", "technology")
    nn_id = mesh.add_concept("neural networks", "Deep learning architecture", "technology")
    
    # Add relations
    mesh.add_relation(ml_id, ai_id, "is_a", strength=0.9)
    mesh.add_relation(nn_id, ml_id, "part_of", strength=0.8)
    mesh.add_relation(ai_id, ml_id, "includes", strength=0.9, bidirectional=True)
    
    # Subscribe to diffs
    def diff_handler(diff: ConceptDiff):
        print(f"Diff recorded: {diff.diff_type} - {diff.concepts}")
    
    subscription_id = mesh.subscribe_to_diffs(diff_handler)
    
    # Extract concepts from text
    text = "Machine learning is revolutionizing artificial intelligence through neural networks"
    extracted = mesh.extract_concepts_from_text(text)
    print(f"Extracted {len(extracted)} concepts from text")
    
    # Find related concepts
    related = mesh.get_related_concepts(ai_id, max_depth=2)
    print(f"\nConcepts related to AI:")
    for concept_id, relation, strength in related:
        if concept_id in mesh.concepts:
            concept = mesh.concepts[concept_id]
            print(f"  - {concept.name} ({relation}, strength: {strength:.2f})")
    
    # Get statistics
    stats = mesh.get_statistics()
    print(f"\nMesh statistics:")
    print(f"  Total concepts: {stats['total_concepts']}")
    print(f"  Total relations: {stats['total_relations']}")
    print(f"  Graph density: {stats['graph_density']:.3f}")
    print(f"  Cache hits: {stats['cache_info']['hits']}")
    
    # Unsubscribe from diffs
    mesh.unsubscribe_from_diffs(subscription_id)
    
    # Export to JSON
    mesh.export_to_json(Path("data/test_mesh/export.json"))
    
    # Shutdown (saves automatically)
    mesh.shutdown()
