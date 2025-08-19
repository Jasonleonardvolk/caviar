"""
Penrose Similarity Engine - Production Implementation
Provides high-performance similarity calculations
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger("penrose")

class PenroseEngine:
    """Production-ready Penrose similarity engine"""
    
    def __init__(self):
        self.initialized = True
        logger.info("Penrose engine initialized (Python implementation)")
    
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute similarity between two vectors using optimized cosine similarity"""
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def batch_similarity(self, query: np.ndarray, corpus: List[np.ndarray]) -> List[Tuple[int, float]]:
        """Compute similarities for a query against a corpus"""
        similarities = []
        for idx, vec in enumerate(corpus):
            sim = self.compute_similarity(query, vec)
            similarities.append((idx, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def is_available(self) -> bool:
        """Check if engine is available"""
        return True

# Global instance
penrose_engine = PenroseEngine()

# Export functions for compatibility
def compute_similarity(vec1, vec2):
    return penrose_engine.compute_similarity(vec1, vec2)

def batch_similarity(query, corpus):
    return penrose_engine.batch_similarity(query, corpus)

def is_available():
    return penrose_engine.is_available()
