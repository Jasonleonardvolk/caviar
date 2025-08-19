"""
Graph Backend Abstract Interface
Allows ConceptMesh to use different graph implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Set, Tuple, Optional, Iterator
import numpy as np
from dataclasses import dataclass


@dataclass
class GraphNode:
    """Represents a node in the graph"""
    id: str
    attributes: Dict[str, Any]


@dataclass
class GraphEdge:
    """Represents an edge in the graph"""
    source: str
    target: str
    weight: float = 1.0
    attributes: Dict[str, Any] = None


class GraphBackend(ABC):
    """Abstract base class for graph backends"""
    
    @abstractmethod
    def add_node(self, node_id: str, **attrs) -> None:
        """Add a node to the graph"""
        pass
    
    @abstractmethod
    def add_edge(self, source: str, target: str, weight: float = 1.0, **attrs) -> None:
        """Add an edge to the graph"""
        pass
    
    @abstractmethod
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the graph"""
        pass
    
    @abstractmethod
    def remove_edge(self, source: str, target: str) -> bool:
        """Remove an edge from the graph"""
        pass
    
    @abstractmethod
    def has_node(self, node_id: str) -> bool:
        """Check if a node exists"""
        pass
    
    @abstractmethod
    def has_edge(self, source: str, target: str) -> bool:
        """Check if an edge exists"""
        pass
    
    @abstractmethod
    def get_node_data(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node attributes"""
        pass
    
    @abstractmethod
    def get_edge_data(self, source: str, target: str) -> Optional[Dict[str, Any]]:
        """Get edge attributes"""
        pass
    
    @abstractmethod
    def nodes(self) -> Iterator[str]:
        """Iterate over all node IDs"""
        pass
    
    @abstractmethod
    def edges(self) -> Iterator[Tuple[str, str]]:
        """Iterate over all edges as (source, target) tuples"""
        pass
    
    @abstractmethod
    def neighbors(self, node_id: str) -> List[str]:
        """Get neighbors of a node"""
        pass
    
    @abstractmethod
    def in_degree(self, node_id: str) -> int:
        """Get in-degree of a node"""
        pass
    
    @abstractmethod
    def out_degree(self, node_id: str) -> int:
        """Get out-degree of a node"""
        pass
    
    @abstractmethod
    def number_of_nodes(self) -> int:
        """Get total number of nodes"""
        pass
    
    @abstractmethod
    def number_of_edges(self) -> int:
        """Get total number of edges"""
        pass
    
    # Graph algorithms
    @abstractmethod
    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two nodes"""
        pass
    
    @abstractmethod
    def connected_components(self) -> List[Set[str]]:
        """Find connected components"""
        pass
    
    @abstractmethod
    def density(self) -> float:
        """Calculate graph density"""
        pass
    
    @abstractmethod
    def to_adjacency_matrix(self) -> Tuple[np.ndarray, Dict[str, int]]:
        """Convert to adjacency matrix and node mapping"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Remove all nodes and edges"""
        pass


class GraphBackendFactory:
    """Factory for creating graph backends"""
    
    _backends = {}
    
    @classmethod
    def register(cls, name: str, backend_class: type):
        """Register a new backend"""
        cls._backends[name] = backend_class
    
    @classmethod
    def create(cls, backend_type: str, **kwargs) -> GraphBackend:
        """Create a graph backend instance"""
        if backend_type not in cls._backends:
            raise ValueError(f"Unknown backend type: {backend_type}")
        
        backend_class = cls._backends[backend_type]
        return backend_class(**kwargs)
    
    @classmethod
    def available_backends(cls) -> List[str]:
        """Get list of available backends"""
        return list(cls._backends.keys())
