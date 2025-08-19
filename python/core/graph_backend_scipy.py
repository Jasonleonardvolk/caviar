"""
Scipy Sparse Backend Implementation
Provides scipy.sparse based graph operations for better performance
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
from typing import Dict, Any, List, Set, Tuple, Optional, Iterator
import logging
from collections import defaultdict

from .graph_backend import GraphBackend, GraphBackendFactory

logger = logging.getLogger(__name__)


class ScipySparseBackend(GraphBackend):
    """Scipy sparse matrix implementation of GraphBackend"""
    
    def __init__(self, initial_size: int = 1000):
        """
        Initialize scipy sparse backend
        
        Args:
            initial_size: Initial size of adjacency matrix
        """
        self.initial_size = initial_size
        self.current_size = initial_size
        
        # Use DOK (Dictionary of Keys) format for efficient modification
        self.adjacency = sp.dok_matrix((initial_size, initial_size), dtype=np.float32)
        
        # Node mappings
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.next_idx = 0
        
        # Node and edge attributes
        self.node_attrs = {}
        self.edge_attrs = {}  # Key: (source_idx, target_idx)
        
        logger.info(f"Initialized scipy sparse graph backend with size {initial_size}")
    
    def _ensure_capacity(self, required_size: int) -> None:
        """Ensure the adjacency matrix has enough capacity"""
        if required_size > self.current_size:
            new_size = max(required_size, self.current_size * 2)
            self.adjacency.resize((new_size, new_size))
            self.current_size = new_size
            logger.debug(f"Resized adjacency matrix to {new_size}x{new_size}")
    
    def _get_or_create_idx(self, node_id: str) -> int:
        """Get index for node, creating if necessary"""
        if node_id not in self.node_to_idx:
            idx = self.next_idx
            self.node_to_idx[node_id] = idx
            self.idx_to_node[idx] = node_id
            self.next_idx += 1
            
            # Ensure capacity
            self._ensure_capacity(self.next_idx)
            
        return self.node_to_idx[node_id]
    
    def add_node(self, node_id: str, **attrs) -> None:
        """Add a node to the graph"""
        idx = self._get_or_create_idx(node_id)
        self.node_attrs[node_id] = attrs
    
    def add_edge(self, source: str, target: str, weight: float = 1.0, **attrs) -> None:
        """Add an edge to the graph"""
        src_idx = self._get_or_create_idx(source)
        tgt_idx = self._get_or_create_idx(target)
        
        self.adjacency[src_idx, tgt_idx] = weight
        self.edge_attrs[(src_idx, tgt_idx)] = attrs
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the graph"""
        if node_id not in self.node_to_idx:
            return False
        
        idx = self.node_to_idx[node_id]
        
        # Remove all edges to/from this node
        self.adjacency[idx, :] = 0
        self.adjacency[:, idx] = 0
        
        # Remove from mappings
        del self.node_to_idx[node_id]
        del self.idx_to_node[idx]
        if node_id in self.node_attrs:
            del self.node_attrs[node_id]
        
        # Remove edge attributes
        edges_to_remove = []
        for (src, tgt) in self.edge_attrs:
            if src == idx or tgt == idx:
                edges_to_remove.append((src, tgt))
        for edge in edges_to_remove:
            del self.edge_attrs[edge]
        
        return True
    
    def remove_edge(self, source: str, target: str) -> bool:
        """Remove an edge from the graph"""
        if source not in self.node_to_idx or target not in self.node_to_idx:
            return False
        
        src_idx = self.node_to_idx[source]
        tgt_idx = self.node_to_idx[target]
        
        if self.adjacency[src_idx, tgt_idx] == 0:
            return False
        
        self.adjacency[src_idx, tgt_idx] = 0
        if (src_idx, tgt_idx) in self.edge_attrs:
            del self.edge_attrs[(src_idx, tgt_idx)]
        
        return True
    
    def has_node(self, node_id: str) -> bool:
        """Check if a node exists"""
        return node_id in self.node_to_idx
    
    def has_edge(self, source: str, target: str) -> bool:
        """Check if an edge exists"""
        if source not in self.node_to_idx or target not in self.node_to_idx:
            return False
        
        src_idx = self.node_to_idx[source]
        tgt_idx = self.node_to_idx[target]
        
        return self.adjacency[src_idx, tgt_idx] != 0
    
    def get_node_data(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node attributes"""
        return self.node_attrs.get(node_id)
    
    def get_edge_data(self, source: str, target: str) -> Optional[Dict[str, Any]]:
        """Get edge attributes"""
        if source not in self.node_to_idx or target not in self.node_to_idx:
            return None
        
        src_idx = self.node_to_idx[source]
        tgt_idx = self.node_to_idx[target]
        
        if self.adjacency[src_idx, tgt_idx] == 0:
            return None
        
        attrs = self.edge_attrs.get((src_idx, tgt_idx), {})
        attrs['weight'] = float(self.adjacency[src_idx, tgt_idx])
        return attrs
    
    def nodes(self) -> Iterator[str]:
        """Iterate over all node IDs"""
        return iter(self.node_to_idx.keys())
    
    def edges(self) -> Iterator[Tuple[str, str]]:
        """Iterate over all edges as (source, target) tuples"""
        # Convert to COO format for efficient iteration
        coo = self.adjacency.tocoo()
        
        for i in range(coo.nnz):
            src_idx = coo.row[i]
            tgt_idx = coo.col[i]
            
            if src_idx in self.idx_to_node and tgt_idx in self.idx_to_node:
                yield (self.idx_to_node[src_idx], self.idx_to_node[tgt_idx])
    
    def neighbors(self, node_id: str) -> List[str]:
        """Get neighbors of a node (outgoing edges)"""
        if node_id not in self.node_to_idx:
            return []
        
        idx = self.node_to_idx[node_id]
        
        # Get row as CSR for efficient access
        row = self.adjacency.getrow(idx).tocsr()
        neighbor_indices = row.indices
        
        return [self.idx_to_node[i] for i in neighbor_indices if i in self.idx_to_node]
    
    def in_degree(self, node_id: str) -> int:
        """Get in-degree of a node"""
        if node_id not in self.node_to_idx:
            return 0
        
        idx = self.node_to_idx[node_id]
        
        # Get column as CSC for efficient access
        col = self.adjacency.getcol(idx).tocsc()
        return col.nnz
    
    def out_degree(self, node_id: str) -> int:
        """Get out-degree of a node"""
        if node_id not in self.node_to_idx:
            return 0
        
        idx = self.node_to_idx[node_id]
        
        # Get row as CSR for efficient access
        row = self.adjacency.getrow(idx).tocsr()
        return row.nnz
    
    def number_of_nodes(self) -> int:
        """Get total number of nodes"""
        return len(self.node_to_idx)
    
    def number_of_edges(self) -> int:
        """Get total number of edges"""
        return self.adjacency.nnz
    
    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two nodes using Dijkstra's algorithm"""
        if source not in self.node_to_idx or target not in self.node_to_idx:
            return None
        
        src_idx = self.node_to_idx[source]
        tgt_idx = self.node_to_idx[target]
        
        # Convert to CSR format for csgraph
        csr_matrix = self.adjacency.tocsr()
        
        # Compute shortest paths from source
        dist_matrix, predecessors = csgraph.dijkstra(
            csr_matrix,
            indices=src_idx,
            return_predecessors=True,
            directed=True
        )
        
        # Check if target is reachable
        if dist_matrix[tgt_idx] == np.inf:
            return None
        
        # Reconstruct path
        path = []
        current = tgt_idx
        
        while current != src_idx:
            if current == -9999 or current not in self.idx_to_node:
                return None  # No path exists
            
            path.append(self.idx_to_node[current])
            current = predecessors[current]
        
        path.append(self.idx_to_node[src_idx])
        path.reverse()
        
        return path
    
    def connected_components(self) -> List[Set[str]]:
        """Find weakly connected components"""
        # Convert to CSR format
        csr_matrix = self.adjacency.tocsr()
        
        # Find connected components
        n_components, labels = csgraph.connected_components(
            csr_matrix,
            directed=True,
            connection='weak'
        )
        
        # Group nodes by component
        components = defaultdict(set)
        for idx, label in enumerate(labels):
            if idx in self.idx_to_node:
                components[label].add(self.idx_to_node[idx])
        
        return list(components.values())
    
    def density(self) -> float:
        """Calculate graph density"""
        n = self.number_of_nodes()
        if n <= 1:
            return 0.0
        
        m = self.number_of_edges()
        max_edges = n * (n - 1)  # For directed graph
        
        return m / max_edges if max_edges > 0 else 0.0
    
    def to_adjacency_matrix(self) -> Tuple[np.ndarray, Dict[str, int]]:
        """Convert to adjacency matrix and node mapping"""
        # Get active nodes
        node_ids = sorted(self.node_to_idx.keys())
        node_to_new_idx = {node: idx for idx, node in enumerate(node_ids)}
        
        n = len(node_ids)
        adj_matrix = np.zeros((n, n))
        
        # Convert to COO for iteration
        coo = self.adjacency.tocoo()
        
        for i in range(coo.nnz):
            src_idx = coo.row[i]
            tgt_idx = coo.col[i]
            
            if src_idx in self.idx_to_node and tgt_idx in self.idx_to_node:
                src_node = self.idx_to_node[src_idx]
                tgt_node = self.idx_to_node[tgt_idx]
                
                if src_node in node_to_new_idx and tgt_node in node_to_new_idx:
                    new_src = node_to_new_idx[src_node]
                    new_tgt = node_to_new_idx[tgt_node]
                    adj_matrix[new_src, new_tgt] = coo.data[i]
        
        return adj_matrix, node_to_new_idx
    
    def clear(self) -> None:
        """Remove all nodes and edges"""
        self.adjacency = sp.dok_matrix((self.initial_size, self.initial_size), dtype=np.float32)
        self.current_size = self.initial_size
        self.node_to_idx.clear()
        self.idx_to_node.clear()
        self.node_attrs.clear()
        self.edge_attrs.clear()
        self.next_idx = 0
    
    # Additional scipy-specific methods for performance
    
    def batch_add_edges(self, edges: List[Tuple[str, str, float]]) -> None:
        """Add multiple edges efficiently"""
        rows = []
        cols = []
        data = []
        
        for source, target, weight in edges:
            src_idx = self._get_or_create_idx(source)
            tgt_idx = self._get_or_create_idx(target)
            rows.append(src_idx)
            cols.append(tgt_idx)
            data.append(weight)
        
        # Update adjacency matrix
        for i, (r, c, d) in enumerate(zip(rows, cols, data)):
            self.adjacency[r, c] = d
    
    def get_laplacian(self) -> sp.csr_matrix:
        """Get the graph Laplacian matrix"""
        csr = self.adjacency.tocsr()
        return csgraph.laplacian(csr, normed=False)
    
    def pagerank_power_iteration(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> Dict[str, float]:
        """Calculate PageRank using power iteration"""
        n = self.next_idx
        if n == 0:
            return {}
        
        # Initialize PageRank vector
        pr = np.ones(n) / n
        
        # Convert to CSR for efficient computation
        adj = self.adjacency.tocsr()
        
        # Calculate out-degree
        out_degree = np.array(adj.sum(axis=1)).flatten()
        out_degree[out_degree == 0] = 1  # Avoid division by zero
        
        # Create transition matrix
        # P[i,j] = adj[j,i] / out_degree[j]
        transition = adj.T / out_degree
        
        # Power iteration
        for _ in range(max_iter):
            pr_new = (1 - alpha) / n + alpha * transition.dot(pr)
            
            # Check convergence
            if np.abs(pr_new - pr).sum() < tol:
                break
            
            pr = pr_new
        
        # Create result dictionary
        result = {}
        for idx, score in enumerate(pr):
            if idx in self.idx_to_node:
                result[self.idx_to_node[idx]] = float(score)
        
        return result


# Register the backend
GraphBackendFactory.register('scipy', ScipySparseBackend)
