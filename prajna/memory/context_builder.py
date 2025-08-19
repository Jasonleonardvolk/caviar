"""
Context Builder for Prajna
==========================

Builds enhanced context from memory systems for Prajna's responses.
Integrates Soliton Memory and Concept Mesh data.
"""

import logging
from typing import Optional, List, Any
from dataclasses import dataclass

logger = logging.getLogger("prajna.memory.context")

@dataclass
class ContextResult:
    """Result from context building process"""
    text: str = ""
    sources: List[str] = None
    reasoning_result: Optional[Any] = None
    confidence: float = 0.8
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []

async def build_context(
    user_query: str,
    focus_concept: Optional[str] = None,
    conversation_id: Optional[str] = None,
    soliton_memory=None,
    concept_mesh=None,
    enable_reasoning: bool = True,
    **kwargs
) -> ContextResult:
    """
    Build enhanced context for Prajna responses
    
    This function orchestrates context retrieval from multiple memory systems
    and optionally triggers reasoning for complex queries.
    """
    try:
        logger.info(f"🔍 Building context for query: {user_query[:100]}...")
        
        # Initialize context
        context = ContextResult()
        
        # Demo context building
        if user_query:
            context.text = f"Context for query: '{user_query}'"
            
            if focus_concept:
                context.text += f"\nFocus concept: {focus_concept}"
            
            if conversation_id:
                context.text += f"\nConversation context: {conversation_id}"
            
            # Add demo sources
            context.sources = [
                "demo_memory_system",
                "demo_concept_mesh",
                "demo_knowledge_base"
            ]
            
            # Enhanced context for specific topics
            query_lower = user_query.lower()
            
            if "consciousness" in query_lower:
                context.text += "\n\nConcept: Consciousness involves multiple cognitive processes including self-awareness, goal formulation, creative synthesis, causal reasoning, internal debate, and learning integration."
                context.sources.append("consciousness_knowledge_base")
            
            elif "memory" in query_lower:
                context.text += "\n\nMemory Systems: Soliton Memory provides long-term storage, Concept Mesh maps relationships, and context building integrates information for responses."
                context.sources.append("memory_systems_documentation")
            
            elif "reasoning" in query_lower:
                context.text += "\n\nReasoning: Multi-hop cognitive reasoning with support for explanatory, causal, analogical, comparative, and inferential modes."
                context.sources.append("reasoning_engine_docs")
        
        # Simulate memory system integration
        if soliton_memory:
            logger.debug("🧠 Integrating Soliton Memory data...")
            # Future: Actual Soliton Memory integration
            context.text += "\n[Soliton Memory integrated]"
        
        if concept_mesh:
            logger.debug("🕸️ Integrating Concept Mesh data...")
            # Future: Actual Concept Mesh integration
            context.text += "\n[Concept Mesh integrated]"
        
        # Simulate reasoning trigger
        if enable_reasoning and len(user_query) > 50:
            logger.debug("🧠 Reasoning triggered for complex query")
            # Future: Actual reasoning integration
            context.reasoning_result = None  # Would be ReasoningResult
        
        logger.info(f"✅ Context built: {len(context.text)} chars, {len(context.sources)} sources")
        return context
        
    except Exception as e:
        logger.error(f"❌ Context building failed: {e}")
        return ContextResult(
            text=f"Basic context for: {user_query}",
            sources=["fallback"],
            confidence=0.3
        )

if __name__ == "__main__":
    # Test context building
    import asyncio
    
    async def test_context_building():
        queries = [
            "What is consciousness?",
            "How does memory work?", 
            "Explain reasoning systems",
            "Hello world"
        ]
        
        for query in queries:
            print(f"\n🔍 Query: {query}")
            context = await build_context(query)
            print(f"📝 Context: {context.text[:200]}...")
            print(f"📚 Sources: {context.sources}")
            print(f"🎯 Confidence: {context.confidence}")
    
    asyncio.run(test_context_building())
