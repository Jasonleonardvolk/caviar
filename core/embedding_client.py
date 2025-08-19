# core/embedding_client.py - TORI embedding client with fallback
import os
import asyncio
import logging
from typing import List, Optional, Union
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

class TORIEmbeddingClient:
    """Production embedding client for TORI with local/cloud fallback"""
    
    def __init__(self):
        self.mode = os.getenv("TORI_EMBED_MODE", "local")
        self.local_url = os.getenv("TORI_EMBED_URL", "http://localhost:8080")
        self.timeout = float(os.getenv("TORI_EMBED_TIMEOUT", "30.0"))
        self.max_retries = int(os.getenv("TORI_EMBED_RETRIES", "3"))
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(timeout=self.timeout)
        
        logger.info(f"TORI Embedding Client initialized in {self.mode} mode")
        
    async def embed_texts(
        self,
        texts: List[str],
        instruction: str = "Extract key concepts and semantic relationships from the document",
        normalize: bool = True
    ) -> EmbeddingResult:
        """Generate embeddings with automatic fallback"""
        
        if self.mode == "local":
            try:
                return await self._embed_local(texts, instruction, normalize)
            except Exception as e:
                logger.warning(f"Local embedding failed: {e}. Falling back to cloud...")
                return await self._embed_cloud_fallback(texts)
        
        elif self.mode == "cloud":
            return await self._embed_cloud_fallback(texts)
        
        else:
            raise ValueError(f"Unknown embed mode: {self.mode}")
    
    async def _embed_local(
        self,
        texts: List[str],
        instruction: str,
        normalize: bool
    ) -> EmbeddingResult:
        """Use local Qwen3 embedding service"""
        
        payload = {
            "texts": texts,
            "instruction": instruction,
            "normalize": normalize
        }
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(
                    f"{self.local_url}/embed",
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                vectors = np.array(data["embeddings"], dtype=np.float32)
                
                return EmbeddingResult(
                    vectors=vectors,
                    cache_hits=data.get("cache_hits", 0),
                    cache_misses=data.get("cache_misses", 0),
                    source="local_qwen3"
                )
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Local embed attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    async def _embed_cloud_fallback(self, texts: List[str]) -> EmbeddingResult:
        """Fallback to OpenAI or other cloud service"""
        try:
            # Import here to avoid dependency when using local mode
            import openai
            
            client = openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            response = await client.embeddings.create(
                input=texts,
                model="text-embedding-3-large"
            )
            
            vectors = np.array([emb.embedding for emb in response.data], dtype=np.float32)
            
            return EmbeddingResult(
                vectors=vectors,
                source="openai_fallback"
            )
            
        except Exception as e:
            logger.error(f"Cloud embedding fallback failed: {e}")
            raise RuntimeError(f"All embedding methods failed. Last error: {e}")
    
    async def health_check(self) -> dict:
        """Check embedding service health"""
        if self.mode == "local":
            try:
                response = await self.client.get(f"{self.local_url}/health")
                return response.json()
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}
        else:
            return {"status": "cloud_mode", "mode": self.mode}
    
    async def close(self):
        """Clean up resources"""
        await self.client.aclose()

# Global client instance
_embedding_client = None

async def get_embedding_client() -> TORIEmbeddingClient:
    """Get or create global embedding client"""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = TORIEmbeddingClient()
    return _embedding_client

async def embed_concepts(texts: List[str], instruction: str = None) -> np.ndarray:
    """Convenience function for concept embedding"""
    client = await get_embedding_client()
    
    if instruction is None:
        instruction = "Extract and represent the semantic meaning of scientific and technical concepts"
    
    result = await client.embed_texts(texts, instruction)
    return result.vectors
