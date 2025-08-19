"""
Concept Mesh - Real Implementation (Python)
Handles concept extraction, relationships, and mesh operations
"""

import json
import logging
import os
from typing import Dict, Any, List, Set, Tuple, Optional, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime
import numpy as np
from collections import defaultdict, deque
import networkx as nx
from pathlib import Path
import pickle
import gzip

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Concept:
    """A single concept in the mesh"""
    id: str
    name: str
    description: str = ""
    category: str = "general"
    importance: float = 1.0
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

@dataclass
class ConceptRelation:
    """Relationship between concepts"""
    source_id: str
    target_id: str
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
    concepts: List[str]  # Concept IDs involved
    old_value: Any = None
    new_value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class ConceptMesh:
    """
    Real implementation of concept mesh for knowledge representation
    """
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def instance(cls, config: Optional[Dict[str, Any]] = None) -> 'ConceptMesh':
        """Get or create the singleton instance"""
        if cls._instance is None:
            cls._instance = cls(config or {})
            # Automatically load data on first access
            cls._instance._load_initial_data()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (mainly for testing)"""
        cls._instance = None
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Storage
        self.storage_path = Path(self.config.get('storage_path', 'data/concept_mesh'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Core data structures
        self.concepts: Dict[str, Concept] = {}
        self.relations: List[ConceptRelation] = []
        self.graph = nx.DiGraph()
        
        # Diff tracking
        self.diff_history: deque = deque(maxlen=self.config.get('max_diff_history', 1000))
        self.diff_subscribers: List[Callable] = []
        
        # Indexing
        self.category_index: Dict[str, Set[str]] = defaultdict(set)
        self.name_index: Dict[str, str] = {}  # name -> id mapping
        
        # Performance
        self.cache_embeddings = self.config.get('cache_embeddings', True)
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        # Similarity engine configuration
        self.similarity_engine = config.get('similarity_engine', 'cosine')
        self.penrose_adapter = None
        if self.similarity_engine == 'penrose':
            self._init_penrose()
        
        # Load existing mesh if available
        self._load_mesh()
        
        # Don't log count here - wait until after _load_initial_data()
    
    def _load_initial_data(self):
        """Load initial data from concept_mesh/data.json if available"""
        # Try to load from the standard location
        data_file = Path("concept_mesh/data.json")
        if data_file.exists() and len(self.concepts) == 0:
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict) and 'concepts' in data:
                    for concept in data['concepts']:
                        # Preserve the ID from the file
                        concept_id = concept.get('id', f"concept_{len(self.concepts)}_{hash(concept.get('name', ''))%10000}")
                        
                        # Create concept directly with preserved ID
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
                        
                    logger.info(f"Loaded {len(data['concepts'])} concepts from {data_file} with preserved IDs")
            except Exception as e:
                logger.error(f"Failed to load initial data: {e}")
        
        # Ensure populated with seed concepts if empty
        if self.count() == 0:
            self.ensure_populated()
        
        # Log final count after all loading is done
        logger.info(f"ConceptMesh ready with {len(self.concepts)} concepts")
    
    def _init_penrose(self):
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
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new concept to the mesh"""
        
        # Check if concept already exists
        if name in self.name_index:
            existing_id = self.name_index[name]
            logger.info(f"Concept '{name}' already exists with ID {existing_id}")
            
            # Update access info
            self.concepts[existing_id].last_accessed = datetime.now()
            self.concepts[existing_id].access_count += 1
            
            return existing_id
        
        # Generate ID
        concept_id = f"concept_{len(self.concepts)}_{hash(name) % 10000}"
        
        # Create concept
        concept = Concept(
            id=concept_id,
            name=name,
            description=description,
            category=category,
            importance=importance,
            embedding=embedding,
            metadata=metadata or {}
        )
        
        # Add to storage
        self.concepts[concept_id] = concept
        self.graph.add_node(concept_id, **asdict(concept))
        
        # Update indices
        self.name_index[name] = concept_id
        self.category_index[category].add(concept_id)
        
        # Cache embedding if provided
        if embedding is not None and self.cache_embeddings:
            self.embedding_cache[concept_id] = embedding
        
        # Create diff
        diff = ConceptDiff(
            id=f"diff_{datetime.now().timestamp()}",
            diff_type="add",
            concepts=[concept_id],
            new_value=asdict(concept)
        )
        self._record_diff(diff)
        
        logger.info(f"Added concept: {name} (ID: {concept_id})")
        return concept_id
    
    def remove_concept(self, concept_id: str) -> bool:
        """Remove a concept from the mesh"""
        if concept_id not in self.concepts:
            logger.warning(f"Concept {concept_id} not found")
            return False
        
        concept = self.concepts[concept_id]
        
        # Remove from storage
        del self.concepts[concept_id]
        self.graph.remove_node(concept_id)
        
        # Update indices
        if concept.name in self.name_index:
            del self.name_index[concept.name]
        self.category_index[concept.category].discard(concept_id)
        
        # Remove from cache
        if concept_id in self.embedding_cache:
            del self.embedding_cache[concept_id]
        
        # Remove relations
        self.relations = [
            r for r in self.relations 
            if r.source_id != concept_id and r.target_id != concept_id
        ]
        
        # Create diff
        diff = ConceptDiff(
            id=f"diff_{datetime.now().timestamp()}",
            diff_type="remove",
            concepts=[concept_id],
            old_value=asdict(concept)
        )
        self._record_diff(diff)
        
        logger.info(f"Removed concept: {concept.name} (ID: {concept_id})")
        return True
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        strength: float = 1.0,
        bidirectional: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a relation between concepts"""
        
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
            id=f"diff_{datetime.now().timestamp()}",
            diff_type="relate",
            concepts=[source_id, target_id],
            new_value=asdict(relation)
        )
        self._record_diff(diff)
        
        logger.info(f"Added relation: {source_id} -{relation_type}-> {target_id}")
        return True
    
    def find_concept(self, name: str) -> Optional[Concept]:
        """Find a concept by name"""
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
        concept_id: str,
        relation_type: Optional[str] = None,
        max_depth: int = 1
    ) -> List[Tuple[str, str, float]]:
        """Get concepts related to the given concept"""
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
            except nx.NetworkXError:
                pass
        
        return related
    
    def find_similar_concepts(
        self,
        concept_id: str,
        threshold: float = 0.7,
        max_results: int = 10
    ) -> List[Tuple[str, float]]:
        """Find concepts similar to the given concept"""
        if concept_id not in self.concepts:
            return []
        
        concept = self.concepts[concept_id]
        
        # Get embedding
        if concept_id in self.embedding_cache:
            query_embedding = self.embedding_cache[concept_id]
        elif concept.embedding is not None:
            query_embedding = concept.embedding
        else:
            # Fallback to name/category similarity
            return self._find_similar_by_metadata(concept, threshold, max_results)
        
        # Use Penrose if enabled
        if self.similarity_engine == 'penrose':
            return self.find_similar_concepts_penrose(concept_id, threshold, max_results)
        
        # Default cosine similarity
        # Calculate similarities
        similarities = []
        for other_id, other_concept in self.concepts.items():
            if other_id == concept_id:
                continue
            
            # Get other embedding
            if other_id in self.embedding_cache:
                other_embedding = self.embedding_cache[other_id]
            elif other_concept.embedding is not None:
                other_embedding = other_concept.embedding
            else:
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
        concept_id: str,
        threshold: float = 0.7,
        max_results: int = 10
    ) -> List[Tuple[str, float]]:
        """Find similar concepts using Penrose O(n^2.32) algorithm"""
        if self.similarity_engine != 'penrose':
            return self.find_similar_concepts(concept_id, threshold, max_results)
        
        # Get query embedding
        concept = self.concepts.get(concept_id)
        if not concept or concept.embedding is None:
            return []
        
        # Initialize Penrose if needed
        if not hasattr(self, 'penrose'):
            self._init_penrose_engine()
        
        if not hasattr(self, 'penrose'):
            # Fallback if Penrose not available
            return self.find_similar_concepts(concept_id, threshold, max_results)
        
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
        similarities = self.penrose.batch_similarity(
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
    
    def _init_penrose_engine(self):
        """Initialize Penrose projector for O(n^2.32) similarity"""
        try:
            from python.core.penrose_adapter import PenroseAdapter
            self.penrose = PenroseAdapter.get_instance(
                rank=32,
                embedding_dim=self.config.get('embedding_dim', 768)
            )
            logger.info("✅ Penrose similarity engine initialized")
        except ImportError:
            logger.warning("⚠️ Penrose not available, falling back to cosine")
            self.similarity_engine = 'cosine'
    
    def _find_similar_by_metadata(
        self,
        concept: Concept,
        threshold: float,
        max_results: int
    ) -> List[Tuple[str, float]]:
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
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def extract_concepts_from_text(
        self,
        text: str,
        min_importance: float = 0.5
    ) -> List[str]:
        """Extract concepts from text and add to mesh"""
        with self._lock:
            # Simple concept extraction (can be enhanced with NLP)
            words = text.lower().split()
            
            # Filter stop words (simplified)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                          'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                          'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                          'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                          'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
                          'who', 'when', 'where', 'why', 'how', 'not', 'no', 'yes'}
            
            # Extract meaningful words
            concepts_found = []
            word_counts = defaultdict(int)
            
            for word in words:
                # Clean word
                word = word.strip('.,!?;:"')
                if len(word) > 2 and word not in stop_words:
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
            words_list = [w for w in words if w not in stop_words and len(w) > 2]
            for i in range(len(words_list) - 1):
                word1 = words_list[i].strip('.,!?;:"')
                word2 = words_list[i + 1].strip('.,!?;:"')
                
                if word1 in self.name_index and word2 in self.name_index:
                    id1 = self.name_index[word1]
                    id2 = self.name_index[word2]
                    self.add_relation(id1, id2, "co_occurs", strength=0.5)
            
            return concepts_found
    
    def get_concept_clusters(self, min_size: int = 3) -> List[Set[str]]:
        """Find clusters of highly connected concepts"""
        with self._lock:
            # Use community detection
            try:
                communities = nx.community.greedy_modularity_communities(
                    self.graph.to_undirected()
                )
                clusters = [set(community) for community in communities if len(community) >= min_size]
                return clusters
            except:
                # Fallback to connected components
                components = list(nx.weakly_connected_components(self.graph))
                return [comp for comp in components if len(comp) >= min_size]
    
    def calculate_concept_importance(self, concept_id: str) -> float:
        """Calculate importance based on connections and usage"""
        if concept_id not in self.concepts:
            return 0.0
        
        concept = self.concepts[concept_id]
        base_importance = concept.importance
        
        # Factor in connections
        in_degree = self.graph.in_degree(concept_id)
        out_degree = self.graph.out_degree(concept_id)
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
    
    def prune_mesh(self, importance_threshold: float = 0.1, age_days: int = 30):
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
    
    def _record_diff(self, diff: ConceptDiff):
        """Record a diff and notify subscribers"""
        self.diff_history.append(diff)
        
        # Notify subscribers
        for subscriber in self.diff_subscribers:
            try:
                subscriber(diff)
            except Exception as e:
                logger.error(f"Error notifying diff subscriber: {e}")
    
    def subscribe_to_diffs(self, callback: Callable[[ConceptDiff], None]):
        """Subscribe to concept diff events"""
        self.diff_subscribers.append(callback)
    
    def get_diff_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent diff history"""
        with self._lock:
            history = list(self.diff_history)[-limit:]
            return [asdict(diff) for diff in history]
    
    def _save_mesh(self):
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
                        'category_index': {k: list(v) for k, v in self.category_index.items()}
                    }, f)
                
                indices_temp.replace(indices_file)
                
                logger.info(f"Mesh saved to disk: {len(self.concepts)} concepts, {len(self.relations)} relations")
                
            except Exception as e:
                logger.error(f"Failed to save mesh: {e}")
                raise
    
    def _load_mesh(self):
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
            except Exception as e:
                logger.error(f"Failed to load concepts: {e}")
        
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
            except Exception as e:
                logger.error(f"Failed to load relations: {e}")
        
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
                
                logger.info("Loaded indices from disk")
            except Exception as e:
                logger.error(f"Failed to load indices: {e}")
    
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
                        'relation_count': len(self.relations)
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
                with open(temp_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                temp_path.replace(file_path)
                
                logger.info(f"Exported mesh to {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to export mesh: {e}")
                return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mesh statistics"""
        with self._lock:
            stats = {
                'total_concepts': len(self.concepts),
                'total_relations': len(self.relations),
                'categories': dict(self.category_index),
                'graph_density': nx.density(self.graph) if len(self.graph) > 0 else 0,
                'connected_components': nx.number_weakly_connected_components(self.graph) if len(self.graph) > 0 else 0,
                'avg_degree': sum(dict(self.graph.degree()).values()) / len(self.graph) if len(self.graph) > 0 else 0,
                'storage_path': str(self.storage_path)
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
    
    def shutdown(self):
        """Save mesh and cleanup"""
        self._save_mesh()
        logger.info("ConceptMesh shutdown complete")

    def load_seeds(self, seed_file: str = None):
        """Load seed concepts from file"""
        if seed_file is None:
            # Default seed files
            seed_files = [
                "data/concept_seed_universal.json",
                "ingest_pdf/data/concept_seed_universal.json",
                "data/seed_concepts.json",
                "data/concept_mesh/seed_concepts.json"
            ]
            
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
            
        except Exception as e:
            logger.error(f"Failed to load seeds: {e}")
            return 0
    
    def ensure_populated(self):
        """Ensure mesh has at least some concepts"""
        if self.count() == 0:
            loaded = self.load_seeds()
            if loaded > 0:
                logger.info("Concept mesh populated with %d seed concepts", loaded)
                # Fire events for lattice
                self.fire_seed_events()
            else:
                logger.warning("Concept mesh remains empty - no seeds loaded")
    
    def fire_seed_events(self):
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
                event_bus.publish("concept_added", event_data)
                logger.debug(f"Published concept_added event for {concept.name}")
            logger.info(f"Published {len(self.concepts)} concept_added events")
        else:
            logger.warning("No event bus available to publish concept events")

    def record_diff(self, diff: ConceptDiff):
        """Record a diff from external source (e.g., Rust extension)"""
        with self._lock:
            self._record_diff(diff)
            # Apply the diff to update internal state
            self._apply_diff(diff)
    
    def _apply_diff(self, diff: ConceptDiff):
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
            
        except Exception as e:
            logger.error(f"Failed to apply diff {diff.id}: {e}")


class PenroseConceptMesh(ConceptMesh):
    """ConceptMesh with Penrose-accelerated similarity"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        config['similarity_engine'] = 'penrose'
        super().__init__(config)
        self._init_penrose_engine()
    
    async def add_relations_from_penrose(self, similar: np.ndarray, concept_ids: List[str], threshold: float = 0.7):
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
def get_mesh():
    """Dependency injection for FastAPI"""
    return ConceptMesh.instance()


# Example usage
if __name__ == "__main__":
    # Create mesh
    mesh = ConceptMesh({'storage_path': 'data/test_mesh'})
    
    # Add some concepts
    ai_id = mesh.add_concept("artificial intelligence", "AI and machine learning", "technology")
    ml_id = mesh.add_concept("machine learning", "Subset of AI", "technology")
    nn_id = mesh.add_concept("neural networks", "Deep learning architecture", "technology")
    
    # Add relations
    mesh.add_relation(ml_id, ai_id, "is_a", strength=0.9)
    mesh.add_relation(nn_id, ml_id, "part_of", strength=0.8)
    mesh.add_relation(ai_id, ml_id, "includes", strength=0.9, bidirectional=True)
    
    # Extract concepts from text
    text = "Machine learning is revolutionizing artificial intelligence through neural networks"
    extracted = mesh.extract_concepts_from_text(text)
    print(f"Extracted {len(extracted)} concepts from text")
    
    # Find related concepts
    related = mesh.get_related_concepts(ai_id, max_depth=2)
    print(f"\nConcepts related to AI:")
    for concept_id, relation, strength in related:
        concept = mesh.concepts[concept_id]
        print(f"  - {concept.name} ({relation}, strength: {strength:.2f})")
    
    # Get statistics
    stats = mesh.get_statistics()
    print(f"\nMesh statistics:")
    print(f"  Total concepts: {stats['total_concepts']}")
    print(f"  Total relations: {stats['total_relations']}")
    print(f"  Graph density: {stats['graph_density']:.3f}")
    
    # Export to JSON
    mesh.export_to_json(Path("data/test_mesh/export.json"))
    
    # Shutdown (saves automatically)
    mesh.shutdown()similar_by_metadata(
        self,
        concept: Concept,
        threshold: float,
        max_results: int
    ) -> List[Tuple[str, float]]:
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
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def extract_concepts_from_text(
        self,
        text: str,
        min_importance: float = 0.5
    ) -> List[str]:
        """Extract concepts from text and add to mesh"""
        # Simple concept extraction (can be enhanced with NLP)
        words = text.lower().split()
        
        # Filter stop words (simplified)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                      'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                      'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
                      'who', 'when', 'where', 'why', 'how', 'not', 'no', 'yes'}
        
        # Extract meaningful words
        concepts_found = []
        word_counts = defaultdict(int)
        
        for word in words:
            # Clean word
            word = word.strip('.,!?;:"')
            if len(word) > 2 and word not in stop_words:
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
        words_list = [w for w in words if w not in stop_words and len(w) > 2]
        for i in range(len(words_list) - 1):
            word1 = words_list[i].strip('.,!?;:"')
            word2 = words_list[i + 1].strip('.,!?;:"')
            
            if word1 in self.name_index and word2 in self.name_index:
                id1 = self.name_index[word1]
                id2 = self.name_index[word2]
                self.add_relation(id1, id2, "co_occurs", strength=0.5)
        
        return concepts_found
    
    def get_concept_clusters(self, min_size: int = 3) -> List[Set[str]]:
        """Find clusters of highly connected concepts"""
        # Use community detection
        try:
            communities = nx.community.greedy_modularity_communities(
                self.graph.to_undirected()
            )
            clusters = [set(community) for community in communities if len(community) >= min_size]
            return clusters
        except:
            # Fallback to connected components
            components = list(nx.weakly_connected_components(self.graph))
            return [comp for comp in components if len(comp) >= min_size]
    
    def calculate_concept_importance(self, concept_id: str) -> float:
        """Calculate importance based on connections and usage"""
        if concept_id not in self.concepts:
            return 0.0
        
        concept = self.concepts[concept_id]
        base_importance = concept.importance
        
        # Factor in connections
        in_degree = self.graph.in_degree(concept_id)
        out_degree = self.graph.out_degree(concept_id)
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
    
    def prune_mesh(self, importance_threshold: float = 0.1, age_days: int = 30):
        """Remove low-importance, old concepts"""
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
    
    def _record_diff(self, diff: ConceptDiff):
        """Record a diff and notify subscribers"""
        self.diff_history.append(diff)
        
        # Notify subscribers
        for subscriber in self.diff_subscribers:
            try:
                subscriber(diff)
            except Exception as e:
                logger.error(f"Error notifying diff subscriber: {e}")
    
    def subscribe_to_diffs(self, callback: Callable[[ConceptDiff], None]):
        """Subscribe to concept diff events"""
        self.diff_subscribers.append(callback)
    
    def get_diff_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent diff history"""
        history = list(self.diff_history)[-limit:]
        return [asdict(diff) for diff in history]
    
    def _save_mesh(self):
        """Save mesh to disk"""
        # Save concepts
        concepts_file = self.storage_path / "concepts.pkl.gz"
        with gzip.open(concepts_file, 'wb') as f:
            # Convert numpy arrays to lists for serialization
            concepts_data = {}
            for cid, concept in self.concepts.items():
                concept_dict = asdict(concept)
                if concept.embedding is not None:
                    concept_dict['embedding'] = concept.embedding.tolist()
                concepts_data[cid] = concept_dict
            pickle.dump(concepts_data, f)
        
        # Save relations
        relations_file = self.storage_path / "relations.pkl.gz"
        with gzip.open(relations_file, 'wb') as f:
            relations_data = [asdict(r) for r in self.relations]
            pickle.dump(relations_data, f)
        
        # Save indices
        indices_file = self.storage_path / "indices.json"
        with open(indices_file, 'w') as f:
            json.dump({
                'name_index': self.name_index,
                'category_index': {k: list(v) for k, v in self.category_index.items()}
            }, f)
        
        logger.info("Mesh saved to disk")
    
    def _load_mesh(self):
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
                
                logger.info(f"Loaded {len(self.concepts)} concepts")
            except Exception as e:
                logger.error(f"Failed to load concepts: {e}")
        
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
                
                logger.info(f"Loaded {len(self.relations)} relations")
            except Exception as e:
                logger.error(f"Failed to load relations: {e}")
        
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
                
                logger.info("Loaded indices")
            except Exception as e:
                logger.error(f"Failed to load indices: {e}")
    
    def export_to_json(self, file_path: Path) -> bool:
        """Export mesh to JSON format"""
        try:
            export_data = {
                'concepts': [],
                'relations': [],
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'concept_count': len(self.concepts),
                    'relation_count': len(self.relations)
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
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported mesh to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export mesh: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mesh statistics"""
        stats = {
            'total_concepts': len(self.concepts),
            'total_relations': len(self.relations),
            'categories': dict(self.category_index),
            'graph_density': nx.density(self.graph),
            'connected_components': nx.number_weakly_connected_components(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / len(self.graph) if len(self.graph) > 0 else 0,
            'storage_path': str(self.storage_path)
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
    
    def shutdown(self):
        """Save mesh and cleanup"""
        self._save_mesh()
        logger.info("ConceptMesh shutdown complete")

    def load_seeds(self, seed_file: str = None):
        """Load seed concepts from file"""
        if seed_file is None:
            # Default seed files
            seed_files = [
                "data/concept_seed_universal.json",
                "ingest_pdf/data/concept_seed_universal.json",
                "data/seed_concepts.json"
            ]
            
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
            
        except Exception as e:
            logger.error(f"Failed to load seeds: {e}")
            return 0
    
    def ensure_populated(self):
        """Ensure mesh has at least some concepts"""
        if self.count() == 0:
            loaded = self.load_seeds()
            if loaded > 0:
                logger.info("Concept mesh populated with %d seed concepts", loaded)
                # Fire events for lattice
                self.fire_seed_events()
            else:
                logger.warning("Concept mesh remains empty - no seeds loaded")
    
    def fire_seed_events(self):
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
                event_bus.publish("concept_added", event_data)
                logger.debug(f"Published concept_added event for {concept.name}")
            logger.info(f"Published {len(self.concepts)} concept_added events")
        else:
            logger.warning("No event bus available to publish concept events")


class PenroseConceptMesh(ConceptMesh):
    """ConceptMesh with Penrose-accelerated similarity"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        config['similarity_engine'] = 'penrose'
        super().__init__(config)
        self._init_penrose_engine()
    
    async def add_relations_from_penrose(self, similar: np.ndarray, concept_ids: List[str], threshold: float = 0.7):
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


# Example usage
if __name__ == "__main__":
    # Create mesh
    mesh = ConceptMesh({'storage_path': 'data/test_mesh'})
    
    # Add some concepts
    ai_id = mesh.add_concept("artificial intelligence", "AI and machine learning", "technology")
    ml_id = mesh.add_concept("machine learning", "Subset of AI", "technology")
    nn_id = mesh.add_concept("neural networks", "Deep learning architecture", "technology")
    
    # Add relations
    mesh.add_relation(ml_id, ai_id, "is_a", strength=0.9)
    mesh.add_relation(nn_id, ml_id, "part_of", strength=0.8)
    mesh.add_relation(ai_id, ml_id, "includes", strength=0.9, bidirectional=True)
    
    # Extract concepts from text
    text = "Machine learning is revolutionizing artificial intelligence through neural networks"
    extracted = mesh.extract_concepts_from_text(text)
    print(f"Extracted {len(extracted)} concepts from text")
    
    # Find related concepts
    related = mesh.get_related_concepts(ai_id, max_depth=2)
    print(f"\nConcepts related to AI:")
    for concept_id, relation, strength in related:
        concept = mesh.concepts[concept_id]
        print(f"  - {concept.name} ({relation}, strength: {strength:.2f})")
    
    # Get statistics
    stats = mesh.get_statistics()
    print(f"\nMesh statistics:")
    print(f"  Total concepts: {stats['total_concepts']}")
    print(f"  Total relations: {stats['total_relations']}")
    print(f"  Graph density: {stats['graph_density']:.3f}")
    
    # Export to JSON
    mesh.export_to_json(Path("data/test_mesh/export.json"))
    
    # Shutdown
    mesh.shutdown()
