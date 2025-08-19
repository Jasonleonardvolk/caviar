# /goals/analogical_transfer.py
import numpy as np
from typing import Dict, List, Tuple
import networkx as nx
from scipy.sparse.linalg import eigsh

class AnalogicalTransfer:
    """
    Treats knowledge clusters as nodes on a graph manifold.
    Geodesic distance guides cross-domain mapping weights.
    Eigen-decomposition of the graph Laplacian yields transfer kernels.
    """
    
    def __init__(self):
        self.knowledge_graph = nx.Graph()
        self.domain_embeddings = {}
        self.transfer_kernels = {}
    
    def add_knowledge_cluster(self, domain: str, concepts: List[str], 
                            embedding: np.ndarray):
        """Add a knowledge domain with its concepts and embedding."""
        self.knowledge_graph.add_node(domain, concepts=concepts)
        self.domain_embeddings[domain] = embedding / np.linalg.norm(embedding)
        
        # Connect to existing domains based on embedding similarity
        for other_domain, other_embedding in self.domain_embeddings.items():
            if other_domain != domain:
                similarity = np.dot(embedding, other_embedding)
                if similarity > 0.3:  # Threshold for connection
                    weight = 1.0 / (1.0 - similarity + 1e-6)  # Convert to distance
                    self.knowledge_graph.add_edge(domain, other_domain, weight=weight)
    
    def compute_transfer_kernels(self, n_eigenvectors: int = 10):
        """
        Compute transfer kernels via spectral decomposition of graph Laplacian.
        """
        if len(self.knowledge_graph) < 2:
            return
        
        # Compute graph Laplacian
        L = nx.laplacian_matrix(self.knowledge_graph, weight='weight').astype(float)
        
        # Eigen-decomposition (get smallest eigenvalues/vectors)
        eigenvalues, eigenvectors = eigsh(L, k=min(n_eigenvectors, len(self.knowledge_graph)-1), 
                                         which='SM')
        
        # Store eigenvectors as transfer basis
        nodes = list(self.knowledge_graph.nodes())
        for i, node in enumerate(nodes):
            self.transfer_kernels[node] = eigenvectors[i, :]
    
    def get_transfer_weights(self, source_domain: str, 
                           target_domain: str) -> float:
        """
        Calculate transfer weight between domains using geodesic distance.
        """
        if source_domain not in self.knowledge_graph or \
           target_domain not in self.knowledge_graph:
            return 0.0
        
        try:
            # Geodesic (shortest path) distance
            distance = nx.shortest_path_length(self.knowledge_graph, 
                                             source_domain, target_domain, 
                                             weight='weight')
            # Convert distance to transfer weight
            return np.exp(-distance / 2.0)
        except nx.NetworkXNoPath:
            return 0.0
    
    def transfer_strategy(self, source_domain: str, target_domain: str,
                         source_strategy: Dict) -> Dict:
        """
        Transfer a problem-solving strategy from source to target domain.
        """
        weight = self.get_transfer_weights(source_domain, target_domain)
        
        if weight < 0.1:
            return {}  # Too distant for meaningful transfer
        
        # Get domain kernels
        source_kernel = self.transfer_kernels.get(source_domain, np.array([]))
        target_kernel = self.transfer_kernels.get(target_domain, np.array([]))
        
        if len(source_kernel) == 0 or len(target_kernel) == 0:
            return source_strategy  # Direct transfer if no kernels
        
        # Compute transformation in spectral space
        kernel_similarity = np.dot(source_kernel, target_kernel)
        
        # Transform strategy based on spectral alignment
        transferred_strategy = {}
        for key, value in source_strategy.items():
            if isinstance(value, (int, float)):
                # Numerical parameters get scaled by kernel similarity
                transferred_strategy[key] = value * (0.5 + 0.5 * kernel_similarity)
            else:
                # Non-numerical parameters transfer with confidence weight
                transferred_strategy[key] = value
                transferred_strategy[f"{key}_confidence"] = weight * kernel_similarity
        
        return transferred_strategy
    
    def find_analogies(self, domain: str, n_analogies: int = 3) -> List[Tuple[str, float]]:
        """
        Find the most analogous domains for knowledge transfer.
        """
        if domain not in self.knowledge_graph:
            return []
        
        analogies = []
        for other_domain in self.knowledge_graph.nodes():
            if other_domain != domain:
                weight = self.get_transfer_weights(domain, other_domain)
                if weight > 0:
                    analogies.append((other_domain, weight))
        
        # Sort by weight and return top n
        analogies.sort(key=lambda x: x[1], reverse=True)
        return analogies[:n_analogies]
