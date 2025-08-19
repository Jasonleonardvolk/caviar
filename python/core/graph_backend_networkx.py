"""
NetworkX Backend Implementation
Provides NetworkX-based graph operations for ConceptMesh
"""

import networkx as nx
import numpy as np
from typing import Dict, Any, List, Set, Tuple, Optional, Iterator
import logging

from .graph_backend import GraphBackend, GraphBackendFactory

logger = logging.getLogger(__name__)


class NetworkXBackend(GraphBackend):
    """NetworkX implementation of GraphBackend"""
    
    def __init__(self):
        """Initialize NetworkX backend with directed graph"""
        self.graph = nx.DiGraph()
        logger.info("Initialized NetworkX graph backend")
    
    def add_node(self, node_id: str, **attrs) -> None:
        """Add a node to the graph"""
        self.graph.add_node(node_id, **attrs)
    
    def add_edge(self, source: str, target: str, weight: float = 1.0, **attrs) -> None:
        """Add an edge to the graph"""
        self.graph.add_edge(source, target, weight=weight, **attrs)
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the graph"""
        if node_id in self.graph:
            self.graph.remove_node(node_id)
            return True
        return False
    
    def remove_edge(self, source: str, target: str) -> bool:
        """Remove an edge from the graph"""
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)
            return True
        return False
    
    def has_node(self, node_id: str) -> bool:
        """Check if a node exists"""
        return node_id in self.graph
    
    def has_edge(self, source: str, target: str) -> bool:
        """Check if an edge exists"""
        return self.graph.has_edge(source, target)
    
    def get_node_data(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node attributes"""
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None
    
    def get_edge_data(self, source: str, target: str) -> Optional[Dict[str, Any]]:
        """Get edge attributes"""
        if self.graph.has_edge(source, target):
            return dict(self.graph.edges[source, target])
        return None
    
    def nodes(self) -> Iterator[str]:
        """Iterate over all node IDs"""
        return iter(self.graph.nodes())
    
    def edges(self) -> Iterator[Tuple[str, str]]:
        """Iterate over all edges as (source, target) tuples"""
        return iter(self.graph.edges())
    
    def neighbors(self, node_id: str) -> List[str]:
        """Get neighbors of a node (outgoing edges)"""
        if node_id in self.graph:
            return list(self.graph.neighbors(node_id))
        return []
    
    def in_degree(self, node_id: str) -> int:
        """Get in-degree of a node"""
        if node_id in self.graph:
            return self.graph.in_degree(node_id)
        return 0
    
    def out_degree(self, node_id: str) -> int:
        """Get out-degree of a node"""
        if node_id in self.graph:
            return self.graph.out_degree(node_id)
        return 0
    
    def number_of_nodes(self) -> int:
        """Get total number of nodes"""
        return self.graph.number_of_nodes()
    
    def number_of_edges(self) -> int:
        """Get total number of edges"""
        return self.graph.number_of_edges()
    
    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two nodes"""
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None
    
    def connected_components(self) -> List[Set[str]]:
        """Find weakly connected components (for directed graphs)"""
        return [set(component) for component in nx.weakly_connected_components(self.graph)]
    
    def density(self) -> float:
        """Calculate graph density"""
        return nx.density(self.graph)
    
    def to_adjacency_matrix(self) -> Tuple[np.ndarray, Dict[str, int]]:
        """Convert to adjacency matrix and node mapping"""
        nodes = list(self.graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        n = len(nodes)
        adj_matrix = np.zeros((n, n))
        
        for source, target, data in self.graph.edges(data=True):
            i = node_to_idx[source]
            j = node_to_idx[target]
            adj_matrix[i, j] = data.get('weight', 1.0)
        
        return adj_matrix, node_to_idx
    
    def clear(self) -> None:
        """Remove all nodes and edges"""
        self.graph.clear()
    
    # Additional NetworkX-specific methods that might be useful
    
    def degree_centrality(self) -> Dict[str, float]:
        """Calculate degree centrality for all nodes"""
        return nx.degree_centrality(self.graph)
    
    def betweenness_centrality(self) -> Dict[str, float]:
        """Calculate betweenness centrality for all nodes"""
        return nx.betweenness_centrality(self.graph)
    
    def pagerank(self, alpha: float = 0.85) -> Dict[str, float]:
        """Calculate PageRank for all nodes"""
        return nx.pagerank(self.graph, alpha=alpha)
    
    def clustering_coefficient(self) -> Dict[str, float]:
        """Calculate clustering coefficient for all nodes"""
        # Convert to undirected for clustering calculation
        undirected = self.graph.to_undirected()
        return nx.clustering(undirected)
    
    def strongly_connected_components(self) -> List[Set[str]]:
        """Find strongly connected components"""
        return [set(component) for component in nx.strongly_connected_components(self.graph)]
    
    def topological_sort(self) -> Optional[List[str]]:
        """Perform topological sort if graph is a DAG"""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            return None  # Graph has cycles
    
    def minimum_spanning_tree(self) -> 'NetworkXBackend':
        """Get minimum spanning tree (converts to undirected)"""
        undirected = self.graph.to_undirected()
        mst = nx.minimum_spanning_tree(undirected)
        
        # Create new backend with MST
        backend = NetworkXBackend()
        backend.graph = mst.to_directed()
        return backend


# Register the backend
GraphBackendFactory.register('networkx', NetworkXBackend)
