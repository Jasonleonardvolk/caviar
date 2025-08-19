"""ontology_refactor_engine.py - Implements topological refactoring for ALAN's conceptual graph.

This module provides mechanisms for structural modifications to ALAN's conceptual
graph, enabling higher-order semantic organization. It implements functions for:

1. Merging redundant or highly similar concept nodes
2. Splitting ambiguous concepts into more precise ones
3. Untangling central hub nodes to improve graph topology
4. Detecting redundant phase profiles and ambiguous spectral traces

These capabilities allow ALAN to continuously refactor its knowledge representation
for optimal coherence and expressivity, enabling emergent ontological structures.

References:
- Graph spectral theory
- Topological data analysis
- Knowledge graph refinement techniques
- Phase-coherent ontology engineering
"""

import os
import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import math
from collections import defaultdict, Counter
import uuid
import networkx as nx

try:
    # Try absolute import first
    from models import ConceptTuple
except ImportError:
    # Fallback to relative import
    from .models import ConceptTuple
try:
    # Try absolute import first
    from koopman_phase_graph import get_koopman_phase_graph, ConceptNode
except ImportError:
    # Fallback to relative import
    from .koopman_phase_graph import get_koopman_phase_graph, ConceptNode
try:
    # Try absolute import first
    from memory_sculptor import get_memory_sculptor, ConceptState
except ImportError:
    # Fallback to relative import
    from .memory_sculptor import get_memory_sculptor, ConceptState
try:
    # Try absolute import first
    from spectral_monitor import get_cognitive_spectral_monitor
except ImportError:
    # Fallback to relative import
    from .spectral_monitor import get_cognitive_spectral_monitor

# Configure logger
logger = logging.getLogger("alan_ontology_refactor")

@dataclass
class RefactorOperation:
    """Records information about a refactoring operation."""
    operation_type: str  # "merge", "split", "untangle", etc.
    timestamp: float = field(default_factory=time.time)
    affected_concepts: List[str] = field(default_factory=list)
    created_concepts: List[str] = field(default_factory=list)
    removed_concepts: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_type": self.operation_type,
            "timestamp": self.timestamp,
            "affected_concepts": self.affected_concepts,
            "created_concepts": self.created_concepts,
            "removed_concepts": self.removed_concepts,
            "parameters": self.parameters,
            "metrics": self.metrics
        }


class OntologyRefactorEngine:
    """
    Main class for refactoring ALAN's conceptual ontology.
    
    This class provides mechanisms for structural reorganization of the 
    knowledge graph, focusing on optimizing topology for semantic coherence.
    """
    
    def __init__(
        self,
        merge_similarity_threshold: float = 0.85,  # Threshold for merging similar concepts
        split_ambiguity_threshold: float = 0.5,    # Threshold for detecting ambiguous concepts
        hub_centrality_threshold: float = 0.8,     # Threshold for identifying hub nodes
        spectral_bifurcation_threshold: float = 0.4,  # Threshold for spectral splitting
        max_operations_per_cycle: int = 5,         # Maximum refactor operations per cycle
        log_dir: str = "logs/ontology"
    ):
        """
        Initialize the ontology refactoring engine.
        
        Args:
            merge_similarity_threshold: Threshold for merging similar concepts
            split_ambiguity_threshold: Threshold for detecting ambiguous concepts
            hub_centrality_threshold: Threshold for identifying hub nodes
            spectral_bifurcation_threshold: Threshold for spectral splitting
            max_operations_per_cycle: Maximum refactor operations per cycle
            log_dir: Directory for logging
        """
        self.merge_similarity_threshold = merge_similarity_threshold
        self.split_ambiguity_threshold = split_ambiguity_threshold
        self.hub_centrality_threshold = hub_centrality_threshold
        self.spectral_bifurcation_threshold = spectral_bifurcation_threshold
        self.max_operations_per_cycle = max_operations_per_cycle
        self.log_dir = log_dir
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Get required components
        self.koopman_graph = get_koopman_phase_graph()
        self.memory_sculptor = get_memory_sculptor()
        self.spectral_monitor = get_cognitive_spectral_monitor()
        
        # Track refactoring operations
        self.operations: List[RefactorOperation] = []
        
        # Store temporary operation metrics
        self.last_run_metrics: Dict[str, Any] = {}
        
        logger.info("Ontology refactor engine initialized")
        
    def merge_nodes(
        self,
        concept_id_a: str,
        concept_id_b: str,
        merge_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Merge two similar concepts into a single concept.
        
        Args:
            concept_id_a: First concept ID to merge
            concept_id_b: Second concept ID to merge
            merge_name: Optional name for the merged concept
            
        Returns:
            Dictionary with merge results
        """
        # Get concepts from Koopman graph
        concept_a = self.koopman_graph.get_concept_by_id(concept_id_a)
        concept_b = self.koopman_graph.get_concept_by_id(concept_id_b)
        
        if concept_a is None or concept_b is None:
            return {
                "status": "error",
                "message": f"One or both concepts not found in graph: {concept_id_a}, {concept_id_b}"
            }
            
        # Calculate similarity to verify merge is appropriate
        similarity = np.dot(concept_a.embedding, concept_b.embedding) / (
            np.linalg.norm(concept_a.embedding) * np.linalg.norm(concept_b.embedding)
        )
        
        if similarity < self.merge_similarity_threshold:
            return {
                "status": "rejected",
                "message": f"Concepts not similar enough to merge (similarity: {similarity:.2f})",
                "similarity": similarity
            }
            
        # Create name for merged concept if not provided
        if merge_name is None:
            merge_name = f"{concept_a.name} + {concept_b.name}"
            
        # Create merged embedding (weighted average)
        # Use concept stability as weights if available
        weight_a = 0.5
        weight_b = 0.5
        
        if hasattr(self.memory_sculptor, "concept_states"):
            state_a = self.memory_sculptor.concept_states.get(concept_id_a)
            state_b = self.memory_sculptor.concept_states.get(concept_id_b)
            
            if state_a and state_b:
                total_stability = state_a.stability_score + state_b.stability_score
                if total_stability > 0:
                    weight_a = state_a.stability_score / total_stability
                    weight_b = state_b.stability_score / total_stability
                    
        # Combine embeddings
        merged_embedding = (
            concept_a.embedding * weight_a + 
            concept_b.embedding * weight_b
        )
        
        # Normalize the embedding
        merged_embedding = merged_embedding / np.linalg.norm(merged_embedding)
        
        # Use dominant concept's source information
        source_doc_id = concept_a.source_document_id if weight_a >= weight_b else concept_b.source_document_id
        source_location = concept_a.source_location if weight_a >= weight_b else concept_b.source_location
        
        # Create merged concept
        try:
            merged_concept = self.koopman_graph.create_concept_from_embedding(
                name=merge_name,
                embedding=merged_embedding,
                source_document_id=source_doc_id,
                source_location=source_location
            )
            
            # Combine edges from both concepts
            combined_edges = {}
            
            # Add edges from concept A
            for target_id, weight in concept_a.edges:
                if target_id != concept_id_b:  # Skip edge to concept B
                    combined_edges[target_id] = weight
                    
            # Add edges from concept B
            for target_id, weight in concept_b.edges:
                if target_id != concept_id_a:  # Skip edge to concept A
                    if target_id in combined_edges:
                        # Take max weight if edge exists in both concepts
                        combined_edges[target_id] = max(combined_edges[target_id], weight)
                    else:
                        combined_edges[target_id] = weight
                        
            # Set edges on merged concept
            merged_concept.edges = [(target_id, weight) for target_id, weight in combined_edges.items()]
            
            # Update edges in other concepts to point to merged concept
            for concept_id, concept in self.koopman_graph.concepts.items():
                if concept_id == merged_concept.id:
                    continue
                    
                updated_edges = []
                
                for target_id, weight in concept.edges:
                    if target_id == concept_id_a or target_id == concept_id_b:
                        # Redirect to merged concept
                        updated_edges.append((merged_concept.id, weight))
                    else:
                        # Keep as is
                        updated_edges.append((target_id, weight))
                        
                concept.edges = updated_edges
                
            # Remove old concepts
            if hasattr(self.koopman_graph, "remove_concept"):
                self.koopman_graph.remove_concept(concept_id_a)
                self.koopman_graph.remove_concept(concept_id_b)
            else:
                # Fallback if method doesn't exist
                if hasattr(self.koopman_graph, "concepts"):
                    if concept_id_a in self.koopman_graph.concepts:
                        del self.koopman_graph.concepts[concept_id_a]
                    if concept_id_b in self.koopman_graph.concepts:
                        del self.koopman_graph.concepts[concept_id_b]
                
            # Record operation
            operation = RefactorOperation(
                operation_type="merge",
                affected_concepts=[concept_id_a, concept_id_b],
                created_concepts=[merged_concept.id],
                removed_concepts=[concept_id_a, concept_id_b],
                parameters={
                    "similarity": similarity,
                    "weight_a": weight_a,
                    "weight_b": weight_b
                },
                metrics={
                    "edge_count": len(merged_concept.edges)
                }
            )
            
            self.operations.append(operation)
            
            # Clean up memory sculptor states
            if hasattr(self.memory_sculptor, "concept_states"):
                if concept_id_a in self.memory_sculptor.concept_states:
                    del self.memory_sculptor.concept_states[concept_id_a]
                if concept_id_b in self.memory_sculptor.concept_states:
                    del self.memory_sculptor.concept_states[concept_id_b]
                    
            logger.info(f"Merged concepts {concept_a.name} and {concept_b.name} into {merge_name}")
            
            return {
                "status": "success",
                "message": f"Successfully merged concepts into {merge_name}",
                "merged_id": merged_concept.id,
                "merged_name": merged_name,
                "similarity": similarity,
                "operation": operation.to_dict()
            }
                
        except Exception as e:
            logger.error(f"Error merging concepts: {e}")
            return {
                "status": "error",
                "message": f"Error merging concepts: {str(e)}"
            }
        
    def split_node(
        self,
        concept_id: str,
        split_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Split an ambiguous concept into more precise concepts.
        
        Args:
            concept_id: Concept ID to split
            split_names: Optional names for the split concepts
            
        Returns:
            Dictionary with split results
        """
        # Get concept from Koopman graph
        concept = self.koopman_graph.get_concept_by_id(concept_id)
        
        if concept is None:
            return {
                "status": "error",
                "message": f"Concept {concept_id} not found in graph"
            }
            
        # Check if concept should be split
        # For now, we'll use a simple heuristic: check if it has many diverse edges
        if len(concept.edges) < 5:
            return {
                "status": "rejected",
                "message": f"Concept {concept_id} does not have enough edges to split",
                "edge_count": len(concept.edges)
            }
            
        # Get target concepts
        target_concepts = []
        for target_id, _ in concept.edges:
            target_concept = self.koopman_graph.get_concept_by_id(target_id)
            if target_concept is not None:
                target_concepts.append(target_concept)
                
        if len(target_concepts) < 3:
            return {
                "status": "rejected",
                "message": f"Concept {concept_id} does not have enough valid targets to split",
                "target_count": len(target_concepts)
            }
            
        # Use spectral clustering to identify potential split groups
        # Create a similarity matrix
        n_targets = len(target_concepts)
        similarity_matrix = np.zeros((n_targets, n_targets))
        
        for i in range(n_targets):
            for j in range(n_targets):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Calculate cosine similarity
                    similarity_matrix[i, j] = np.dot(
                        target_concepts[i].embedding, 
                        target_concepts[j].embedding
                    ) / (
                        np.linalg.norm(target_concepts[i].embedding) * 
                        np.linalg.norm(target_concepts[j].embedding)
                    )
                    
        # Create a graph from the similarity matrix
        # Use threshold to create edges
        G = nx.Graph()
        for i in range(n_targets):
            G.add_node(i)
            
        for i in range(n_targets):
            for j in range(i+1, n_targets):
                if similarity_matrix[i, j] > self.spectral_bifurcation_threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i, j])
                    
        # Detect communities in the graph
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
        except Exception as e:
            # Fall back to a simpler algorithm if the main one fails
            communities = []
            connected_components = list(nx.connected_components(G))
            for component in connected_components:
                communities.append(component)
                
        # If we don't find multiple communities, reject the split
        if len(communities) < 2:
            return {
                "status": "rejected",
                "message": f"Could not identify distinct semantic clusters for {concept.name}",
                "communities": len(communities)
            }
            
        # Create names for split concepts if not provided
        if split_names is None or len(split_names) < len(communities):
            split_names = [f"{concept.name}_{i+1}" for i in range(len(communities))]
            
        # Create new concepts, one for each community
        new_concept_ids = []
        try:
            for i, community in enumerate(communities):
                # Get targets in this community
                community_targets = [target_concepts[j] for j in community]
                
                # Create centroid embedding
                community_embedding = np.zeros_like(concept.embedding)
                for target in community_targets:
                    community_embedding += target.embedding
                    
                if len(community_targets) > 0:
                    community_embedding /= len(community_targets)
                    
                # Add some of the original concept embedding
                community_embedding = 0.7 * community_embedding + 0.3 * concept.embedding
                
                # Normalize
                community_embedding = community_embedding / np.linalg.norm(community_embedding)
                
                # Create new concept
                new_concept = self.koopman_graph.create_concept_from_embedding(
                    name=split_names[i],
                    embedding=community_embedding,
                    source_document_id=concept.source_document_id,
                    source_location=concept.source_location
                )
                
                # Add edges to targets in this community
                for target in community_targets:
                    # Find original weight
                    weight = 0.5  # Default
                    for target_id, w in concept.edges:
                        if target_id == target.id:
                            weight = w
                            break
                            
                    # Add edge with original or increased weight
                    new_concept.edges.append((target.id, min(1.0, weight * 1.2)))
                    
                    # Add edge from target to new concept
                    updated_edges = []
                    for edge_target_id, edge_weight in target.edges:
                        if edge_target_id == concept_id:
                            # Replace with new concept
                            updated_edges.append((new_concept.id, edge_weight))
                        else:
                            updated_edges.append((edge_target_id, edge_weight))
                            
                    target.edges = updated_edges
                    
                new_concept_ids.append(new_concept.id)
                
            # Remove original concept
            if hasattr(self.koopman_graph, "remove_concept"):
                self.koopman_graph.remove_concept(concept_id)
            else:
                # Fallback if method doesn't exist
                if hasattr(self.koopman_graph, "concepts") and concept_id in self.koopman_graph.concepts:
                    del self.koopman_graph.concepts[concept_id]
                    
            # Record operation
            operation = RefactorOperation(
                operation_type="split",
                affected_concepts=[concept_id],
                created_concepts=new_concept_ids,
                removed_concepts=[concept_id],
                parameters={
                    "communities": len(communities),
                    "target_count": len(target_concepts),
                    "bifurcation_threshold": self.spectral_bifurcation_threshold
                },
                metrics={
                    "community_sizes": [len(c) for c in communities]
                }
            )
            
            self.operations.append(operation)
            
            # Clean up memory sculptor state
            if hasattr(self.memory_sculptor, "concept_states"):
                if concept_id in self.memory_sculptor.concept_states:
                    del self.memory_sculptor.concept_states[concept_id]
                    
            logger.info(f"Split concept {concept.name} into {len(new_concept_ids)} new concepts")
            
            return {
                "status": "success",
                "message": f"Successfully split concept into {len(new_concept_ids)} new concepts",
                "split_concept_ids": new_concept_ids,
                "split_concept_names": split_names[:len(new_concept_ids)],
                "communities": len(communities),
                "operation": operation.to_dict()
            }
                
        except Exception as e:
            logger.error(f"Error splitting concept: {e}")
            return {
                "status": "error",
                "message": f"Error splitting concept: {str(e)}"
            }
            
    def untangle_hub(
        self,
        concept_id: str,
        max_degree: int = 15  # Maximum number of edges to keep
    ) -> Dict[str, Any]:
        """
        Untangle a hub node by reducing its connections.
        
        Args:
            concept_id: Concept ID of the hub to untangle
            max_degree: Maximum number of edges to keep
            
        Returns:
            Dictionary with untangling results
        """
        # Get concept from Koopman graph
        concept = self.koopman_graph.get_concept_by_id(concept_id)
        
        if concept is None:
            return {
                "status": "error",
                "message": f"Concept {concept_id} not found in graph"
            }
            
        # Check if concept is a hub
        if len(concept.edges) <= max_degree:
            return {
                "status": "rejected",
                "message": f"Concept {concept_id} is not a hub (only {len(concept.edges)} edges)",
                "edge_count": len(concept.edges)
            }
            
        # Calculate centrality based on edge count compared to average
        if not hasattr(self.koopman_graph, "concepts") or not self.koopman_graph.concepts:
            return {
                "status": "error",
                "message": "Cannot calculate centrality without graph concepts"
            }
            
        avg_edge_count = sum(len(c.edges) for c in self.koopman_graph.concepts.values()) / len(self.koopman_graph.concepts)
        centrality = len(concept.edges) / (avg_edge_count * 2)  # Normalize
        
        if centrality < self.hub_centrality_threshold:
            return {
                "status": "rejected",
                "message": f"Concept {concept_id} centrality ({centrality:.2f}) below threshold",
                "centrality": centrality
            }
            
        # Get target concepts and sort by importance
        target_data = []
        for target_id, weight in concept.edges:
            target_concept = self.koopman_graph.get_concept_by_id(target_id)
            if target_concept is not None:
                # Calculate importance score
                # Based on edge weight, target stability, and embedding similarity
                importance = weight
                
                # Add stability factor if available
                if hasattr(self.memory_sculptor, "concept_states"):
                    target_state = self.memory_sculptor.concept_states.get(target_id)
                    if target_state:
                        importance *= (0.5 + 0.5 * target_state.stability_score)
                        
                # Calculate embedding similarity
                similarity = np.dot(concept.embedding, target_concept.embedding) / (
                    np.linalg.norm(concept.embedding) * np.linalg.norm(target_concept.embedding)
                )
                
                # Factor similarity into importance
                importance *= (0.3 + 0.7 * similarity)
                
                target_data.append({
                    "id": target_id,
                    "weight": weight,
                    "importance": importance,
                    "similarity": similarity
                })
                
        # Sort by importance (descending)
        target_data.sort(key=lambda x: x["importance"], reverse=True)
        
        # Keep the most important edges up to max_degree
        important_targets = target_data[:max_degree]
        removed_targets = target_data[max_degree:]
        
        # Update concept edges
        concept.edges = [(t["id"], t["weight"]) for t in important_targets]
        
        # Record operation
        operation = RefactorOperation(
            operation_type="untangle_hub",
            affected_concepts=[concept_id],
            created_concepts=[],
            removed_concepts=[],
            parameters={
                "centrality": centrality,
                "orig_edge_count": len(target_data),
                "max_degree": max_degree
            },
            metrics={
                "removed_edges": len(removed_targets),
                "kept_edges": len(important_targets)
            }
        )
        
        self.operations.append(operation)
        
        logger.info(f"Untangled hub {concept.name}: removed {len(removed_targets)} of {len(target_data)} edges")
        
        return {
            "status": "success",
            "message": f"Successfully untangled hub by removing {len(removed_targets)} edges",
            "original_edge_count": len(target_data),
            "remaining_edge_count": len(important_targets),
            "removed_edge_count": len(removed_targets),
            "centrality": centrality,
            "operation": operation.to_dict()
        }
        
    def find_redundant_clusters(
        self,
        min_redundancy_score: float = 0.75,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        Find clusters of redundant concepts with similar phase profiles.
        
        Args:
            min_redundancy_score: Minimum score to consider clusters redundant
            max_results: Maximum number of redundant clusters to return
            
        Returns:
            Dictionary with redundancy analysis results
        """
        if not hasattr(self.koopman_graph, "concepts") or not self.koopman_graph.concepts:
            return {
                "status": "error",
                "message": "No concepts in graph to analyze"
            }
            
        # Get all concepts
        concepts = list(self.koopman_graph.concepts.values())
        
        # Create similarity matrix
        n_concepts = len(concepts)
        similarity_matrix = np.zeros((n_concepts, n_concepts))
        
        for i in range(n_concepts):
            for j in range(i, n_concepts):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Calculate cosine similarity
                    similarity_matrix[i, j] = np.dot(
                        concepts[i].embedding, 
                        concepts[j].embedding
                    ) / (
                        np.linalg.norm(concepts[i].embedding) * 
                        np.linalg.norm(concepts[j].embedding)
                    )
                    similarity_matrix[j, i] = similarity_matrix[i, j]  # Symmetrical
                    
        # Find clusters of highly similar concepts
        redundant_clusters = []
        
        # Use a greedy approach to find redundant clusters
        remaining_indices = set(range(n_concepts))
        
        while remaining_indices and len(redundant_clusters) < max_results:
            # Start with a random concept
            start_idx = next(iter(remaining_indices))
            
            # Find all concepts similar to this one
            similar_indices = [
                j for j in remaining_indices
                if similarity_matrix[start_idx, j] >= min_redundancy_score
            ]
            
            # Only consider as cluster if at least 2 concepts
            if len(similar_indices) >= 2:
                cluster = [concepts[idx].id for idx in similar_indices]
                avg_similarity = np.mean([
                    similarity_matrix[i, j]
                    for i in similar_indices
                    for j in similar_indices
                    if i != j
                ])
                
                redundant_clusters.append({
                    "concepts": cluster,
                    "concept_names": [concepts[idx].name for idx in similar_indices],
                    "size": len(cluster),
                    "avg_similarity": avg_similarity
                })
                
                # Remove these indices from consideration
                remaining_indices -= set(similar_indices)
            else:
                # No cluster formed, remove the start index
                remaining_indices.remove(start_idx)
                
        return {
            "status": "success",
            "redundant_clusters": redundant_clusters,
            "total_clusters": len(redundant_clusters),
            "analyzed_concepts": n_concepts
        }
        
    def detect_ambiguous_phase_profiles(
        self,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        Detect concepts with ambiguous phase profiles that could benefit from splitting.
        
        Args:
            max_results: Maximum number of ambiguous concepts to return
            
        Returns:
            Dictionary with ambiguity analysis results
        """
        if not hasattr(self.koopman_graph, "concepts") or not self.koopman_graph.concepts:
            return {
                "status": "error",
                "message": "No concepts in graph to analyze"
            }
            
        # Ambiguity indicators:
        # 1. High number of diverse edges
        # 2. Connections to concepts in different clusters
        # 3. Phase instability (if available from memory sculptor)
        
        candidates = []
        
        for concept_id, concept in self.koopman_graph.concepts.items():
            if len(concept.edges) < 5:
                continue  # Skip concepts with few edges
                
            # Get all target concepts
            target_concepts = []
            for target_id, _ in concept.edges:
                target_concept = self.koopman_graph.get_concept_by_id(target_id)
                if target_concept is not None:
                    target_concepts.append(target_concept)
                    
            if len(target_concepts) < 5:
                continue  # Skip if not enough valid targets
                
            # Calculate diversity of targets
            # Use average pairwise similarity as a measure (lower = more diverse)
            similarities = []
            for i in range(len(target_concepts)):
                for j in range(i+1, len(target_concepts)):
                    similarity = np.dot(
                        target_concepts[i].embedding, 
                        target_concepts[j].embedding
                    ) / (
                        np.linalg.norm(target_concepts[i].embedding) * 
                        np.linalg.norm(target_concepts[j].embedding)
                    )
                    similarities.append(similarity)
                    
            if not similarities:
                continue
                
            # Calculate diversity (inverse of average similarity)
            avg_similarity = sum(similarities) / len(similarities)
            diversity = 1.0 - avg_similarity
            
            # Get phase instability if available
            phase_instability = 0.0
            if hasattr(self.memory_sculptor, "concept_states"):
                state = self.memory_sculptor.concept_states.get(concept_id)
                if state:
                    # Use desync count as instability indicator
                    phase_instability = min(1.0, state.phase_desyncs / 20)
                    
            # Calculate overall ambiguity score
            ambiguity_score = 0.4 * diversity + 0.3 * (len(target_concepts) / 20) + 0.3 * phase_instability
            
            # Add if score above threshold
            if ambiguity_score > self.split_ambiguity_threshold:
                candidates.append({
                    "concept_id": concept_id,
                    "name": concept.name,
                    "ambiguity_score": ambiguity_score,
                    "diversity": diversity,
                    "edge_count": len(target_concepts),
                    "phase_instability": phase_instability
                })
                
        # Sort by ambiguity score (descending) and take top results
        candidates.sort(key=lambda x: x["ambiguity_score"], reverse=True)
        top_candidates = candidates[:max_results]
        
        return {
            "status": "success",
            "ambiguous_concepts": top_candidates,
            "total_candidates": len(candidates),
            "analyzed_concepts": len(self.koopman_graph.concepts)
        }
        
    def run_refactor_cycle(self) -> Dict[str, Any]:
        """
        Run a full refactoring cycle on the concept graph.
        
        This performs several refactoring operations:
        1. Merge redundant concepts
        2. Split ambiguous concepts
        3. Untangle hub nodes
        
        Returns:
            Dictionary with refactoring results
        """
        start_time = time.time()
        operations_performed = []
        
        if not hasattr(self.koopman_graph, "concepts") or not self.koopman_graph.concepts:
            return {
                "status": "error",
                "message": "No concepts in graph to refactor"
            }
            
        # Reset last run metrics
        self.last_run_metrics = {}
        
        # 1. Find and merge redundant concepts
        # Find redundant clusters first
        redundant_result = self.find_redundant_clusters(
            min_redundancy_score=self.merge_similarity_threshold
        )
        
        if redundant_result.get("status") == "success":
            # Process some redundant clusters
            redundant_clusters = redundant_result.get("redundant_clusters", [])
            merge_count = 0
            
            for cluster in redundant_clusters:
                if merge_count >= self.max_operations_per_cycle // 3:
                    break  # Limit number of merges
                    
                concepts = cluster.get("concepts", [])
                if len(concepts) >= 2:
                    # Take most similar pair from cluster
                    result = self.merge_nodes(concepts[0], concepts[1])
                    if result.get("status") == "success":
                        operations_performed.append({
                            "type": "merge",
                            "result": result
                        })
                        merge_count += 1
                        
        # 2. Find and split ambiguous concepts
        ambiguous_result = self.detect_ambiguous_phase_profiles()
        
        if ambiguous_result.get("status") == "success":
            ambiguous_concepts = ambiguous_result.get("ambiguous_concepts", [])
            split_count = 0
            
            for concept_data in ambiguous_concepts:
                if split_count >= self.max_operations_per_cycle // 3:
                    break  # Limit number of splits
                    
                concept_id = concept_data.get("concept_id")
                if concept_id:
                    result = self.split_node(concept_id)
                    if result.get("status") == "success":
                        operations_performed.append({
                            "type": "split",
                            "result": result
                        })
                        split_count += 1
                        
        # 3. Find and untangle hub nodes
        # Identify hub concepts with high centrality
        hub_candidates = []
        
        # Calculate average edge count
        avg_edge_count = sum(len(c.edges) for c in self.koopman_graph.concepts.values()) / len(self.koopman_graph.concepts)
        
        for concept_id, concept in self.koopman_graph.concepts.items():
            edge_count = len(concept.edges)
            if edge_count > 3 * avg_edge_count:  # Potential hub
                centrality = edge_count / (avg_edge_count * 2)
                if centrality > self.hub_centrality_threshold:
                    hub_candidates.append({
                        "concept_id": concept_id,
                        "name": concept.name,
                        "edge_count": edge_count,
                        "centrality": centrality
                    })
                    
        # Sort by centrality (descending)
        hub_candidates.sort(key=lambda x: x["centrality"], reverse=True)
        
        # Untangle top hubs
        untangle_count = 0
        
        for hub in hub_candidates:
            if untangle_count >= self.max_operations_per_cycle // 3:
                break  # Limit number of untangles
                
            concept_id = hub.get("concept_id")
            if concept_id:
                result = self.untangle_hub(concept_id)
                if result.get("status") == "success":
                    operations_performed.append({
                        "type": "untangle",
                        "result": result
                    })
                    untangle_count += 1
                    
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Save metrics
        self.last_run_metrics = {
            "merge_count": merge_count,
            "split_count": split_count,
            "untangle_count": untangle_count,
            "elapsed_time": elapsed_time,
            "timestamp": time.time()
        }
        
        return {
            "status": "success",
            "elapsed_time": elapsed_time,
            "operations_performed": operations_performed,
            "merge_count": merge_count,
            "split_count": split_count,
            "untangle_count": untangle_count,
            "total_operations": merge_count + split_count + untangle_count
        }
        
    def get_refactor_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about refactoring operations.
        
        Returns:
            Dictionary with refactoring statistics
        """
        # Count operations by type
        operation_counts = Counter([op.operation_type for op in self.operations])
        
        # Get recent operations
        recent_operations = []
        for op in self.operations[-10:]:
            recent_operations.append(op.to_dict())
            
        return {
            "operation_counts": dict(operation_counts),
            "total_operations": len(self.operations),
            "recent_operations": recent_operations,
            "last_run_metrics": self.last_run_metrics
        }


# Singleton instance
_ontology_refactor_engine_instance = None

def get_ontology_refactor_engine() -> OntologyRefactorEngine:
    """
    Get the singleton instance of the ontology refactor engine.
    
    Returns:
        OntologyRefactorEngine instance
    """
    global _ontology_refactor_engine_instance
    
    if _ontology_refactor_engine_instance is None:
        _ontology_refactor_engine_instance = OntologyRefactorEngine()
        
    return _ontology_refactor_engine_instance
