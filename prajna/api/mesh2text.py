"""
Mesh-to-Text Bridge for Saigon LSTM
Converts concept IDs to natural language using Saigon generator
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

class Mesh2TextRequest(BaseModel):
    concept_ids: List[str]
    max_tokens: int = 100
    temperature: float = 0.7
    persona: Optional[dict] = None

class Mesh2TextResponse(BaseModel):
    text: str
    tokens_generated: int
    concept_ids_used: List[str]

# Lazy-loaded Saigon generator
_saigon_generator = None

def get_saigon_generator():
    """Lazy load Saigon LSTM generator"""
    global _saigon_generator
    if _saigon_generator is None:
        try:
            from saigon_generator import SaigonLSTM
            _saigon_generator = SaigonLSTM()
            logger.info("✅ Saigon LSTM loaded successfully")
        except ImportError:
            logger.warning("⚠️ Saigon generator not available - using mock")
            # Mock implementation for development
            class MockSaigon:
                def generate(self, concept_ids, max_tokens=100):
                    return f"Generated text from concepts: {', '.join(concept_ids[:3])}..."
            _saigon_generator = MockSaigon()
    return _saigon_generator

@router.post("/mesh2text", response_model=Mesh2TextResponse)
async def mesh_to_text(request: Mesh2TextRequest):
    """Convert concept mesh IDs to natural language text"""
    try:
        generator = get_saigon_generator()
        
        # Generate text from concepts
        text = await asyncio.to_thread(
            generator.generate,
            request.concept_ids,
            request.max_tokens
        )
        
        return Mesh2TextResponse(
            text=text,
            tokens_generated=len(text.split()),
            concept_ids_used=request.concept_ids[:10]  # Limit to first 10
        )
        
    except Exception as e:
        logger.error(f"Mesh-to-text generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# SSE endpoint for streaming
from sse_starlette.sse import EventSourceResponse

@router.get("/mesh2text/stream")
async def mesh_to_text_stream(concept_ids: str, max_tokens: int = 100):
    """Stream text generation from concept IDs"""
    concepts = concept_ids.split(",")
    
    async def generate():
        generator = get_saigon_generator()
        
        # Mock streaming for now
        words = f"Streaming from {len(concepts)} concepts".split()
        for word in words:
            yield {
                "data": word,
                "event": "token",
                "id": str(asyncio.get_event_loop().time())
            }
            await asyncio.sleep(0.1)
        
        yield {"data": "[DONE]", "event": "done"}
    
    return EventSourceResponse(generate())
