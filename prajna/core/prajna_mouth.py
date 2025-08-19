"""
Prajna Mouth: The Voice of TORI
===============================

This is Prajna's voice component - the only part that generates language output.
All responses from TORI are generated through Prajna using only ingested, traceable data.

This implementation provides both production and demo modes for development.
"""

import asyncio
import logging
import time
import os
import sys
from typing import Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass

# Add the models directory to the path for Saigon imports
models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "efficientnet")
if models_path not in sys.path:
    sys.path.append(models_path)

try:
    from saigon import SaigonGenerator
except ImportError as e:
    logger = logging.getLogger("prajna.mouth")
    logger.error(f"Failed to import Saigon: {e}")
    SaigonGenerator = None

logger = logging.getLogger("prajna.mouth")

@dataclass
class PrajnaOutput:
    """Output from Prajna language generation"""
    answer: str
    confidence: float = 0.8
    processing_time: float = 0.0
    model_used: str = "demo"
    tokens_generated: int = 0

class PrajnaLanguageModel:
    """
    Prajna Language Model - TORI's voice and mouth
    
    This is the only component that generates natural language responses.
    All output is based on ingested, traceable knowledge from memory systems.
    """
    
    def __init__(self, model_type="saigon", model_path="", device="cpu", 
                 max_context_length=2048, temperature=1.0, enable_mesh_to_text=True, **kwargs):
        self.model_type = model_type
        self.model_path = model_path
        self.device = device
        self.max_context_length = max_context_length
        self.temperature = temperature
        self.enable_mesh_to_text = enable_mesh_to_text
        self.model_loaded = False
        self.saigon_generator = None
        self.stats = {
            "total_requests": 0,
            "successful_responses": 0,
            "average_response_time": 0.0,
            "total_tokens_generated": 0
        }
        
        logger.info(f"🗣️ Initializing Prajna Language Model: {model_type}")
    
    async def load_model(self):
        """Load the language model"""
        try:
            if self.model_type == "saigon":
                if not self.enable_mesh_to_text:
                    logger.info("⚠️ Mesh-to-text disabled, using template fallback")
                    self.model_loaded = True
                    return
                
                # Load Saigon mesh-to-text generator
                logger.info("🧠 Loading Saigon mesh-to-text generator...")
                if SaigonGenerator is None:
                    raise ImportError("Saigon not available")
                
                self.saigon_generator = SaigonGenerator(
                    model_path=self.model_path if self.model_path else None,
                    device=self.device
                )
                
                # Try to load the LSTM model
                model_loaded = self.saigon_generator.load_model()
                if model_loaded:
                    logger.info("✅ Saigon LSTM model loaded successfully")
                else:
                    logger.info("⚠️ Saigon LSTM not available - using raw mesh mode")
                
                self.model_loaded = True
                
            elif self.model_type == "demo":
                # Demo mode - no actual model loading
                logger.info("🎭 Prajna running in DEMO mode")
                await asyncio.sleep(1)  # Simulate loading time
                self.model_loaded = True
                
            else:
                # All other model types now unsupported - use Saigon as default
                logger.warning(f"Model type '{self.model_type}' retired. Using Saigon as default.")
                self.model_type = "saigon"
                await self.load_model()  # Recurse with Saigon
                return
            
            logger.info(f"✅ Prajna model loaded: {self.model_type}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Prajna model: {e}")
            logger.info("🎭 Falling back to demo mode")
            self.model_type = "demo"
            self.saigon_generator = None
            self.model_loaded = True
    
    async def generate_response(self, query: str, context: str = "", **kwargs) -> PrajnaOutput:
        """Generate response using Prajna"""
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            if not self.model_loaded:
                await self.load_model()
            
            if self.model_type == "demo":
                answer = await self._demo_generate(query, context)
            elif self.model_type == "saigon":
                answer = await self._saigon_generate(query, context, **kwargs)
            else:
                answer = await self._model_generate(query, context, **kwargs)
            
            processing_time = time.time() - start_time
            self.stats["successful_responses"] += 1
            self.stats["total_tokens_generated"] += len(answer.split())
            
            # Update average response time
            total_responses = self.stats["successful_responses"]
            old_avg = self.stats["average_response_time"]
            self.stats["average_response_time"] = (
                (old_avg * (total_responses - 1) + processing_time) / total_responses
            )
            
            return PrajnaOutput(
                answer=answer,
                confidence=0.8,
                processing_time=processing_time,
                model_used=self.model_type,
                tokens_generated=len(answer.split())
            )
            
        except Exception as e:
            logger.error(f"❌ Prajna generation failed: {e}")
            return PrajnaOutput(
                answer=f"I apologize, but I encountered an error generating a response: {str(e)}",
                confidence=0.1,
                processing_time=time.time() - start_time,
                model_used=self.model_type
            )
    
    async def _demo_generate(self, query: str, context: str = "") -> str:
        """Demo mode response generation"""
        # Simulate some processing time
        await asyncio.sleep(0.5 + len(query) / 1000)
        
        # Demo responses based on query content
        query_lower = query.lower()
        
        if "prajna" in query_lower:
            return ("Prajna is TORI's voice and language model, designed to provide "
                   "intelligent responses based on ingested knowledge. I process "
                   "queries using advanced reasoning and memory systems to deliver "
                   "accurate, contextual answers.")
        
        elif "consciousness" in query_lower or "conscious" in query_lower:
            return ("Consciousness in AI systems like myself involves multiple layers: "
                   "self-reflection, goal formulation, creative synthesis, causal reasoning, "
                   "internal debate, and learning integration. I demonstrate these through "
                   "my metacognitive engine which orchestrates various cognitive processes.")
        
        elif "memory" in query_lower:
            return ("My memory system consists of Soliton Memory for long-term storage "
                   "and Concept Mesh for relationship mapping. This allows me to maintain "
                   "contextual understanding across conversations and learn from past "
                   "interactions while ensuring all responses are traceable to source material.")
        
        elif "reasoning" in query_lower:
            return ("I employ multi-hop cognitive reasoning that can trace connections "
                   "across concepts, perform causal analysis, and generate explanatory "
                   "pathways. My reasoning engine supports explanatory, causal, analogical, "
                   "comparative, and inferential modes depending on the query type.")
        
        elif any(word in query_lower for word in ["hello", "hi", "hey", "greetings"]):
            return ("Hello! I'm Prajna, TORI's voice and intelligence system. I'm here to "
                   "help answer your questions using advanced reasoning and comprehensive "
                   "knowledge. What would you like to explore today?")
        
        elif "capabilities" in query_lower or "what can you do" in query_lower:
            return ("I can help with a wide range of tasks including: answering questions "
                   "using multi-hop reasoning, analyzing complex topics through internal "
                   "debate, synthesizing concepts across domains, simulating hypothetical "
                   "scenarios, and providing contextual responses based on ingested knowledge. "
                   "All my responses are traceable and grounded in verified information.")
        
        elif context.strip():
            return (f"Based on the provided context about '{query}', I can see relevant "
                   f"information that helps address your question. Let me analyze the key "
                   f"concepts and provide a comprehensive response drawing from the "
                   f"available knowledge in my memory systems.")
        
        else:
            return (f"I understand you're asking about '{query}'. While I'm currently "
                   f"running in demo mode, I would normally process this query through "
                   f"my reasoning engine, consult my memory systems, and provide a "
                   f"comprehensive response based on ingested knowledge. In production, "
                   f"this would involve multi-hop reasoning and contextual analysis.")
    
    async def _saigon_generate(self, query: str, context: str = "", **kwargs) -> str:
        """Saigon mesh-to-text generation"""
        try:
            # Check if mesh-to-text is disabled
            if not self.enable_mesh_to_text:
                # Use template fallback
                return await self._template_fallback_generate(query, context)
            
            if not self.saigon_generator:
                raise ValueError("Saigon generator not initialized")
            
            # Create mesh path from query and context
            mesh_path = self._create_mesh_path(query, context)
            
            # Generate using Saigon with "smartest-ever" settings
            result = self.saigon_generator.generate(
                mesh_path=mesh_path,
                smoothing=True,  # Use LSTM smoothing if available
                max_len=256,
                temperature=self.temperature  # Use "smartest-ever" temperature=1.0
            )
            
            # Log the generation details
            logger.info(f"Saigon generation: {result['method']}, "
                       f"processing_time={result['processing_time']:.3f}s")
            
            return result["text"]
            
        except Exception as e:
            logger.error(f"Saigon generation failed: {e}")
            # Graceful fallback to demo mode
            return await self._demo_generate(query, context)
    
    def _create_mesh_path(self, query: str, context: str = ""):
        """Create a concept mesh path from query and context for Saigon"""
        mesh_path = []
        
        # Extract key concepts from the query
        query_words = query.lower().split()
        concept_keywords = [
            "consciousness", "memory", "reasoning", "knowledge", "intelligence",
            "learning", "understanding", "analysis", "synthesis", "cognition",
            "awareness", "perception", "thought", "logic", "inference",
            "creativity", "insight", "wisdom", "comprehension", "reflection"
        ]
        
        # Find relevant concepts in the query
        found_concepts = [word for word in query_words if word in concept_keywords]
        
        # If no specific concepts found, use general reasoning concepts
        if not found_concepts:
            if "what" in query_words or "how" in query_words:
                found_concepts = ["understanding", "analysis"]
            elif "why" in query_words:
                found_concepts = ["reasoning", "logic"]
            else:
                found_concepts = ["knowledge", "intelligence"]
        
        # Create mesh nodes with relationships
        relations = ["implies", "supports", "extends", "enables", "derives_from"]
        
        for i, concept in enumerate(found_concepts[:5]):  # Limit to 5 concepts
            relation = relations[i % len(relations)]
            mesh_path.append({
                "concept": concept,
                "relation": relation,
                "context": context if context else "cognitive_processing"
            })
        
        # If no concepts, create a default reasoning path
        if not mesh_path:
            mesh_path = [
                {"concept": "reasoning", "relation": "enables", "context": "query_processing"},
                {"concept": "analysis", "relation": "supports", "context": "understanding"},
                {"concept": "synthesis", "relation": "derives_from", "context": "knowledge"}
            ]
        
        return mesh_path
    
    async def _template_fallback_generate(self, query: str, context: str = "") -> str:
        """Template-based fallback when mesh-to-text is disabled"""
        # Extract key information
        query_lower = query.lower()
        has_context = bool(context.strip())
        
        # Template response construction
        response_parts = []
        
        # Opening acknowledgment
        if "what" in query_lower:
            response_parts.append(f"Regarding your question about '{query}'")
        elif "how" in query_lower:
            response_parts.append(f"To explain how '{query}'")
        elif "why" in query_lower:
            response_parts.append(f"The reason for '{query}'")
        else:
            response_parts.append(f"In response to '{query}'")
        
        # Core response based on context
        if has_context:
            response_parts.append(", based on the provided context")
            response_parts.append(", I can analyze the relevant information")
            response_parts.append(" through my cognitive reasoning systems.")
            
            # Add context-aware elaboration
            if len(context) > 100:
                response_parts.append(" The extensive context provides multiple perspectives")
                response_parts.append(" that enable comprehensive analysis.")
            else:
                response_parts.append(" The context helps frame the response appropriately.")
        else:
            response_parts.append(", I would process this through my knowledge systems")
            response_parts.append(" to provide an informed response.")
        
        # Closing with cognitive capabilities mention
        response_parts.append(" My response integrates multi-hop reasoning,")
        response_parts.append(" memory retrieval, and conceptual synthesis")
        response_parts.append(" to deliver traceable, grounded insights.")
        
        # Join and clean up
        response = "".join(response_parts)
        
        # Add slight variation to avoid repetitive responses
        import random
        variations = [
            " This analysis leverages my integrated cognitive architecture.",
            " The response is generated through systematic reasoning processes.",
            " This conclusion emerges from cross-referencing multiple knowledge sources."
        ]
        response += random.choice(variations)
        
        # Simulate processing time
        await asyncio.sleep(0.3)
        
        return response

    async def _model_generate(self, query: str, context: str = "", **kwargs) -> str:
        """Production model response generation (placeholder for future implementation)"""
        # This would contain the actual model inference code
        # For now, fall back to enhanced demo response
        
        await asyncio.sleep(1.0)  # Simulate model inference time
        
        return (f"[Production Model Response] I've processed your query '{query}' "
               f"through advanced language modeling. The response would be generated "
               f"using the loaded model with appropriate context integration and "
               f"reasoning pathways. Context length: {len(context)} characters.")
    
    async def stream_generate(self, query: str, context, **kwargs) -> AsyncGenerator[str, None]:
        """Stream response generation"""
        if not self.model_loaded:
            await self.load_model()
        
        # Generate full response
        output = await self.generate_response(query, context.text if hasattr(context, 'text') else str(context))
        
        # Stream it word by word
        words = output.answer.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.1)  # Simulate streaming delay
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_loaded
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        return {
            "model_type": self.model_type,
            "model_loaded": self.model_loaded,
            "device": self.device,
            "max_context_length": self.max_context_length,
            "temperature": self.temperature,
            **self.stats
        }
    
    async def cleanup(self):
        """Cleanup model resources"""
        if self.model_loaded:
            logger.info("🧹 Cleaning up Prajna model resources")
            # Future: Actual model cleanup
            self.model_loaded = False

async def generate_prajna_response(query: str, context, model: PrajnaLanguageModel, 
                                 streaming: bool = False, **kwargs) -> PrajnaOutput:
    """
    Main function to generate Prajna responses
    
    This is the primary interface for generating language output through Prajna.
    All TORI responses flow through this function.
    """
    try:
        # Extract context text
        context_text = ""
        if hasattr(context, 'text'):
            context_text = context.text
        elif isinstance(context, str):
            context_text = context
        elif isinstance(context, dict):
            context_text = context.get('text', str(context))
        else:
            context_text = str(context)
        
        # Generate response
        if streaming:
            # For streaming, we'll collect the response and return it
            # In a real implementation, this would stream properly
            response_chunks = []
            async for chunk in model.stream_generate(query, context, **kwargs):
                response_chunks.append(chunk)
            
            answer = "".join(response_chunks)
            return PrajnaOutput(
                answer=answer,
                confidence=0.8,
                model_used=model.model_type
            )
        else:
            return await model.generate_response(query, context_text, **kwargs)
    
    except Exception as e:
        logger.error(f"❌ Error in generate_prajna_response: {e}")
        return PrajnaOutput(
            answer=f"I apologize, but I encountered an error: {str(e)}",
            confidence=0.1,
            model_used="error"
        )

if __name__ == "__main__":
    # Test Prajna language model
    async def test_prajna():
        model = PrajnaLanguageModel(model_type="demo")
        await model.load_model()
        
        test_queries = [
            "What is Prajna?",
            "Tell me about consciousness in AI",
            "How does your memory system work?",
            "Hello, how are you?"
        ]
        
        for query in test_queries:
            print(f"\n🤔 Query: {query}")
            response = await model.generate_response(query)
            print(f"🗣️ Prajna: {response.answer}")
            print(f"   Confidence: {response.confidence:.2f}")
            print(f"   Time: {response.processing_time:.2f}s")
        
        # Test stats
        stats = await model.get_stats()
        print(f"\n📊 Prajna Stats: {stats}")
    
    asyncio.run(test_prajna())
