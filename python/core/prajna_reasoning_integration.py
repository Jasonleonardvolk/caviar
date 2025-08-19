"""
Integration script to add reasoning traversal to Prajna API
This shows how to modify your existing Prajna endpoints
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

# Import the reasoning components
from python.core.reasoning_traversal import (
    ConceptMesh, ConceptNode, EdgeType,
    PrajnaResponsePlus
)
from python.core.enhanced_context_builder import (
    EnhancedContextBuilder, PrajnaWithReasoning
)

logger = logging.getLogger(__name__)

# Request/Response models
class ReasoningRequest(BaseModel):
    query: str
    persona: Optional[Dict[str, Any]] = None
    enable_reasoning: bool = True
    enable_inline_attribution: bool = True
    max_reasoning_paths: int = 3
    reasoning_depth: int = 4

class ReasoningResponse(BaseModel):
    text: str
    reasoning_paths: List[Dict[str, Any]]
    sources: List[str]
    confidence: float
    has_reasoning: bool
    graphviz: Optional[str] = None

# Modify your existing Prajna API
def add_reasoning_endpoints(app: FastAPI, prajna_instance=None):
    """Add reasoning-enhanced endpoints to existing FastAPI app"""
    
    # Initialize enhanced Prajna
    enhanced_prajna = PrajnaWithReasoning(prajna_instance)
    
    @app.post("/api/answer_with_reasoning", response_model=ReasoningResponse)
    async def answer_with_reasoning(request: ReasoningRequest):
        """Generate answer with full reasoning traversal and attribution"""
        try:
            # Generate response with reasoning
            response = enhanced_prajna.generate_with_reasoning(
                query=request.query,
                persona=request.persona,
                enable_reasoning=request.enable_reasoning,
                max_paths=request.max_reasoning_paths
            )
            
            # Convert to API response
            return ReasoningResponse(
                text=response.text,
                reasoning_paths=[path.to_dict() for path in response.reasoning_paths],
                sources=response.sources,
                confidence=response.confidence,
                has_reasoning=len(response.reasoning_paths) > 0,
                graphviz=response.to_graphviz() if request.enable_reasoning else None
            )
            
        except Exception as e:
            logger.error(f"Reasoning generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/reasoning/explain/{response_id}")
    async def explain_reasoning(response_id: str):
        """Get detailed explanation of reasoning process"""
        # This would retrieve a stored response and explain its reasoning
        # For now, return a placeholder
        return {
            "response_id": response_id,
            "explanation": "Detailed reasoning explanation would go here"
        }
    
    @app.post("/api/reasoning/validate")
    async def validate_reasoning_path(path: List[Dict[str, Any]]):
        """Validate a reasoning path for logical consistency"""
        # Implement validation logic
        return {
            "valid": True,
            "issues": [],
            "confidence": 0.85
        }
    
    return app

# Standalone script to update existing prajna_api.py
def update_prajna_api():
    """
    Add this to your existing prajna_api.py:
    
    from prajna_reasoning_integration import add_reasoning_endpoints
    
    # After creating your FastAPI app
    app = add_reasoning_endpoints(app, prajna_instance)
    """
    pass

# Example of how to modify existing endpoint
def enhance_existing_answer_endpoint():
    """
    Modify your existing /api/answer endpoint to include reasoning
    """
    code = '''
    @app.post("/api/answer")
    async def answer(request: AnswerRequest):
        """Enhanced answer endpoint with optional reasoning"""
        
        # Check if reasoning is requested
        enable_reasoning = request.metadata.get("enable_reasoning", False)
        
        if enable_reasoning:
            # Use enhanced Prajna with reasoning
            enhanced_response = enhanced_prajna.generate_with_reasoning(
                query=request.user_query,
                persona=request.persona
            )
            
            # Add reasoning to response metadata
            response.metadata["reasoning_paths"] = [
                path.to_dict() for path in enhanced_response.reasoning_paths[:3]
            ]
            response.metadata["reasoning_confidence"] = enhanced_response.confidence
            
            # Add inline attribution to text if enabled
            if request.metadata.get("enable_inline_attribution", True):
                response.text = enhanced_response.text
        else:
            # Use traditional Prajna
            response = await original_answer(request)
        
        return response
    '''
    return code

# Configuration for reasoning integration
REASONING_CONFIG = {
    "enable_by_default": True,
    "max_traversal_depth": 5,
    "min_path_score": 0.1,
    "inline_attribution_format": "[source: {source}]",
    "path_selection_strategy": "highest_confidence",  # or "most_relevant", "shortest_valid"
    "cache_reasoning_paths": True,
    "cache_ttl_seconds": 3600
}

# Initialize concept mesh with common concepts
def initialize_default_concept_mesh() -> ConceptMesh:
    """Initialize a concept mesh with common concepts"""
    mesh = ConceptMesh()
    
    # Add some default concepts that are commonly used
    common_concepts = [
        ("consciousness", "Consciousness", "State of awareness and subjective experience"),
        ("memory", "Memory", "Storage and retrieval of information"),
        ("learning", "Learning", "Process of acquiring new knowledge or skills"),
        ("reasoning", "Reasoning", "Process of drawing conclusions from premises"),
        ("perception", "Perception", "Organization and interpretation of sensory information"),
        ("attention", "Attention", "Selective concentration on aspects of the environment"),
        ("emotion", "Emotion", "Complex psychological state involving subjective experience"),
        ("language", "Language", "System of communication using symbols and rules"),
        ("intelligence", "Intelligence", "Ability to acquire and apply knowledge and skills"),
        ("creativity", "Creativity", "Ability to generate novel and valuable ideas")
    ]
    
    for concept_id, name, description in common_concepts:
        node = ConceptNode(
            id=concept_id,
            name=name,
            description=description,
            sources=["knowledge_base"]
        )
        mesh.add_node(node)
    
    # Add some relationships
    mesh.add_edge("consciousness", "perception", EdgeType.ENABLES, 
                  justification="consciousness enables subjective perception")
    mesh.add_edge("memory", "learning", EdgeType.SUPPORTS,
                  justification="memory supports learning by storing information")
    mesh.add_edge("reasoning", "intelligence", EdgeType.PART_OF,
                  justification="reasoning is a component of intelligence")
    mesh.add_edge("emotion", "reasoning", EdgeType.CONTRADICTS,
                  weight=0.6, justification="strong emotions can impair reasoning")
    mesh.add_edge("language", "reasoning", EdgeType.ENABLES,
                  justification="language enables complex reasoning")
    
    return mesh

# Test the integration
if __name__ == "__main__":
    # Create test app
    from fastapi import FastAPI
    
    app = FastAPI(title="Prajna with Reasoning")
    
    # Add reasoning endpoints
    app = add_reasoning_endpoints(app)
    
    # Test with example
    import asyncio
    
    async def test_reasoning():
        request = ReasoningRequest(
            query="How does consciousness relate to perception?",
            persona={"name": "Philosopher", "style": "analytical"},
            enable_reasoning=True,
            enable_inline_attribution=True
        )
        
        # This would call the actual endpoint
        print("Test request:", request.dict())
    
    asyncio.run(test_reasoning())
