# core/embedding_client_noauth.py - Embedding client without authentication
import os
import asyncio
import logging
from typing import List
import httpx
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    vectors: np.ndarray
    cache_hits: int = 0
    cache_misses: int = 0
    source: str = "unknown"

class SimpleEmbeddingClient:
    """Simple embedding client without authentication"""
    
    def __init__(self):
        self.url = os.getenv("TORI_EMBED_URL", "http://localhost:8080")
        self.timeout = float(os.getenv("TORI_EMBED_TIMEOUT", "30.0"))
        # Use proper timeout configuration
        timeout_config = httpx.Timeout(self.timeout, connect=5.0)
        self.client = httpx.AsyncClient(timeout=timeout_config)
        logger.info(f"Simple Embedding Client initialized: {self.url}")
        
    async def embed_texts(
        self,
        texts: List[str],
        instruction: str = "Extract key concepts and semantic relationships from the document",
        normalize: bool = True
    ) -> EmbeddingResult:
        """Generate embeddings without auth"""
        
        payload = {
            "texts": texts,
            "instruction": instruction,
            "normalize": normalize
        }
        
        try:
            response = await self.client.post(
                f"{self.url}/embed",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            vectors = np.array(data["embeddings"], dtype=np.float32)
            
            return EmbeddingResult(
                vectors=vectors,
                cache_hits=data.get("cache_hits", 0),
                cache_misses=data.get("cache_misses", 0),
                source="local_qwen3_noauth"
            )
            
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise
    
    async def close(self):
        """Clean up resources"""
        await self.client.aclose()

# Override the original client
_embedding_client = None

async def get_embedding_client() -> SimpleEmbeddingClient:
    """Get or create global embedding client"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = SimpleEmbeddingClient()
    return _embedding_client

async def embed_concepts(texts: List[str], instruction: str = None) -> np.ndarray:
    """Convenience function for concept embedding"""
    client = await get_embedding_client()
    
    if instruction is None:
        instruction = "Extract and represent the semantic meaning of scientific and technical concepts"
    
    result = await client.embed_texts(texts, instruction)
    return result.vectors
