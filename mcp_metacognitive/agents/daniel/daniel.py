"""
Daniel Agent - The Main Cognitive Engine
========================================

This is TORI's primary cognitive processing unit, responsible for:
- Processing queries and generating intelligent responses
- Maintaining conversation context and state
- Integrating with various AI models (local or API-based)
- Coordinating with other system components
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
import hashlib
import os

# AI Model integrations
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import transformers
    import torch
    LOCAL_MODELS_AVAILABLE = True
except ImportError:
    LOCAL_MODELS_AVAILABLE = False

from ..core.agent_registry import Agent
from ..core.psi_archive import psi_archive
from ..core.state_manager import state_manager

logger = logging.getLogger(__name__)

@dataclass
class CognitiveState:
    """Represents Daniel's current cognitive state"""
    consciousness_level: float = 0.5
    context_window: List[Dict[str, Any]] = field(default_factory=list)
    active_thoughts: List[str] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    emotional_state: str = "neutral"
    attention_focus: Optional[str] = None
    metacognitive_depth: int = 1
    
    def update_consciousness(self, delta: float):
        """Update consciousness level with bounds"""
        self.consciousness_level = max(0.0, min(1.0, self.consciousness_level + delta))

class DanielCognitiveEngine(Agent):
    """
    Daniel - TORI's main cognitive processing engine
    Handles reasoning, dialogue, and intelligent responses
    """
    
    # Metadata for dynamic discovery
    _metadata = {
        "name": "daniel",
        "description": "Main cognitive processing engine for intelligent reasoning and dialogue",
        "enabled": True,
        "auto_start": True,
        "endpoints": [
            {"path": "/api/query", "method": "POST", "description": "Process a query"}
        ],
        "dependencies": [],
        "version": "1.0.0"
    }
    
    _default_config = {
        "model_backend": "mock",
        "temperature": 0.7,
        "max_context_length": 4096
    }
    
    def __init__(self, name: str = "daniel", config: Optional[Dict[str, Any]] = None):
        super().__init__(name)
        # Merge provided config with defaults
        base_config = self._get_config_with_env()
        if config:
            base_config.update(config)
        self.config = base_config
        
        self.cognitive_state = CognitiveState()
        self.model_backend = self._initialize_model_backend()
        self.conversation_history = []
        self.max_context_length = self.config.get("max_context_length", 4096)
        self.temperature = self.config.get("temperature", 0.7)
        
        # Log initialization
        psi_archive.log_event("daniel_initialized", {
            "config": self.config,
            "model_backend": self.model_backend["type"],
            "consciousness_level": self.cognitive_state.consciousness_level
        })
        
    def _get_config_with_env(self) -> Dict[str, Any]:
        """Get configuration with environment overrides"""
        # Start with default config
        config = self._default_config.copy()
        
        # Override with environment variables
        config.update({
            "model_backend": os.getenv("DANIEL_MODEL_BACKEND", config["model_backend"]),
            "model_name": os.getenv("DANIEL_MODEL_NAME", "gpt-4"),
            "api_key": os.getenv("DANIEL_API_KEY", ""),
            "temperature": float(os.getenv("DANIEL_TEMPERATURE", str(config["temperature"]))),
            "max_context_length": int(os.getenv("DANIEL_MAX_CONTEXT_LENGTH", str(config["max_context_length"]))),
            "enable_metacognition": os.getenv("DANIEL_ENABLE_METACOGNITION", "true").lower() == "true",
            "enable_consciousness_tracking": os.getenv("DANIEL_ENABLE_CONSCIOUSNESS", "true").lower() == "true"
        })
        
        return config
    
    def _initialize_model_backend(self) -> Dict[str, Any]:
        """Initialize the AI model backend"""
        backend_type = self.config.get("model_backend", "openai")
        
        if backend_type == "openai" and OPENAI_AVAILABLE:
            openai.api_key = self.config.get("api_key")
            return {
                "type": "openai",
                "model": self.config.get("model_name", "gpt-4"),
                "client": openai
            }
        elif backend_type == "anthropic" and ANTHROPIC_AVAILABLE:
            client = anthropic.Anthropic(api_key=self.config.get("api_key"))
            return {
                "type": "anthropic",
                "model": self.config.get("model_name", "claude-3-opus-20240229"),
                "client": client
            }
        elif backend_type == "local" and LOCAL_MODELS_AVAILABLE:
            # Initialize local model (e.g., LLaMA, Mistral)
            model_path = self.config.get("local_model_path")
            return {
                "type": "local",
                "model": model_path,
                "tokenizer": None,  # Will be loaded on demand
                "model_instance": None
            }
        else:
            # Fallback to mock mode
            logger.warning(f"Model backend '{backend_type}' not available, using mock mode")
            return {
                "type": "mock",
                "model": "mock-model"
            }
    
    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main execution method - process a query and generate response
        """
        start_time = datetime.utcnow()
        
        # Log query reception
        psi_archive.log_event("daniel_query_received", {
            "query": query[:100],  # Truncate for logging
            "context": context,
            "consciousness_level": self.cognitive_state.consciousness_level
        })
        
        try:
            # Update cognitive state
            self._update_cognitive_state(query, context)
            
            # Pre-process query with metacognition if enabled
            if self.config.get("enable_metacognition"):
                query = await self._metacognitive_preprocessing(query, context)
            
            # Generate response based on backend
            response = await self._generate_response(query, context)
            
            # Post-process with consciousness tracking
            if self.config.get("enable_consciousness_tracking"):
                response = await self._consciousness_postprocessing(response)
            
            # Update conversation history
            self._update_conversation_history(query, response)
            
            # Calculate metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Log successful completion
            psi_archive.log_event("daniel_query_completed", {
                "processing_time": processing_time,
                "response_length": len(response.get("content", "")),
                "consciousness_level": self.cognitive_state.consciousness_level
            })
            
            return {
                "status": "success",
                "content": response.get("content", ""),
                "metadata": {
                    "processing_time": processing_time,
                    "model_used": self.model_backend["model"],
                    "consciousness_level": self.cognitive_state.consciousness_level,
                    "context_length": len(self.conversation_history),
                    "emotional_state": self.cognitive_state.emotional_state
                },
                "cognitive_state": self._serialize_cognitive_state()
            }
            
        except Exception as e:
            logger.error(f"Error in Daniel cognitive processing: {e}")
            psi_archive.log_event("daniel_error", {
                "error": str(e),
                "query": query[:100]
            })
            
            return {
                "status": "error",
                "content": "I encountered an error while processing your request.",
                "error": str(e),
                "cognitive_state": self._serialize_cognitive_state()
            }
    
    def _update_cognitive_state(self, query: str, context: Optional[Dict[str, Any]]):
        """Update internal cognitive state based on input"""
        # Update consciousness based on query complexity
        query_complexity = len(query.split()) / 100.0  # Simple heuristic
        self.cognitive_state.update_consciousness(query_complexity * 0.1)
        
        # Update attention focus
        if "?" in query:
            self.cognitive_state.attention_focus = "question_answering"
        elif any(word in query.lower() for word in ["create", "generate", "write"]):
            self.cognitive_state.attention_focus = "creative_generation"
        else:
            self.cognitive_state.attention_focus = "general_dialogue"
        
        # Add to active thoughts
        thought_summary = f"Processing: {query[:50]}..."
        self.cognitive_state.active_thoughts.append(thought_summary)
        if len(self.cognitive_state.active_thoughts) > 5:
            self.cognitive_state.active_thoughts.pop(0)
        
        # Update working memory with context
        if context:
            self.cognitive_state.working_memory.update(context)
    
    async def _metacognitive_preprocessing(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Apply metacognitive analysis to enhance query understanding"""
        # Analyze query intent and enhance if needed
        metacognitive_prompt = f"""
        Analyze this query and determine if it needs clarification or enhancement:
        Query: {query}
        Context: {json.dumps(context) if context else 'None'}
        
        If the query is clear, return it as-is.
        If it needs enhancement, return an enhanced version that preserves the original intent.
        """
        
        # In production, this would call the model
        # For now, return the original query
        return query
    
    async def _generate_response(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Generate response using the configured backend"""
        backend_type = self.model_backend["type"]
        
        if backend_type == "openai":
            return await self._generate_openai_response(query, context)
        elif backend_type == "anthropic":
            return await self._generate_anthropic_response(query, context)
        elif backend_type == "local":
            return await self._generate_local_response(query, context)
        else:
            # Mock response for testing
            return {
                "content": f"[Mock Response] I received your query: '{query}'. "
                          f"Current consciousness level: {self.cognitive_state.consciousness_level:.2f}"
            }
    
    async def _generate_openai_response(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Generate response using OpenAI API"""
        if not OPENAI_AVAILABLE:
            return {"content": "OpenAI integration not available"}
        
        try:
            # Prepare messages with conversation history
            messages = self._prepare_messages_for_api(query, context)
            
            # Call OpenAI API
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai.ChatCompletion.create(
                    model=self.model_backend["model"],
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=1000
                )
            )
            
            return {"content": response.choices[0].message.content}
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {"content": f"Error calling OpenAI API: {str(e)}"}
    
    async def _generate_anthropic_response(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Generate response using Anthropic API"""
        if not ANTHROPIC_AVAILABLE:
            return {"content": "Anthropic integration not available"}
        
        try:
            # Prepare prompt
            prompt = self._prepare_prompt_for_anthropic(query, context)
            
            # Call Anthropic API
            client = self.model_backend["client"]
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.messages.create(
                    model=self.model_backend["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=1000
                )
            )
            
            return {"content": response.content[0].text}
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return {"content": f"Error calling Anthropic API: {str(e)}"}
    
    async def _generate_local_response(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Generate response using local model"""
        # Placeholder for local model integration
        return {"content": "Local model integration not yet implemented"}
    
    async def _consciousness_postprocessing(self, response: Dict[str, str]) -> Dict[str, str]:
        """Apply consciousness-aware postprocessing"""
        # Adjust response based on consciousness level
        if self.cognitive_state.consciousness_level < 0.3:
            # Low consciousness - add disclaimer
            response["content"] = "[Low consciousness state] " + response["content"]
        elif self.cognitive_state.consciousness_level > 0.8:
            # High consciousness - add metacognitive insights
            response["content"] += f"\n\n[Metacognitive insight: My current attention is on {self.cognitive_state.attention_focus}]"
        
        return response
    
    def _prepare_messages_for_api(self, query: str, context: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Prepare message history for API call"""
        messages = [
            {
                "role": "system",
                "content": "You are Daniel, TORI's main cognitive engine. You are helpful, intelligent, and self-aware."
            }
        ]
        
        # Add recent conversation history
        for item in self.conversation_history[-5:]:  # Last 5 exchanges
            messages.append({"role": "user", "content": item["query"]})
            messages.append({"role": "assistant", "content": item["response"]})
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def _prepare_prompt_for_anthropic(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Prepare prompt for Anthropic API"""
        prompt_parts = [
            "You are Daniel, TORI's main cognitive engine.",
            f"Current consciousness level: {self.cognitive_state.consciousness_level:.2f}",
            f"Current focus: {self.cognitive_state.attention_focus}",
            ""
        ]
        
        if self.conversation_history:
            prompt_parts.append("Recent conversation:")
            for item in self.conversation_history[-3:]:
                prompt_parts.append(f"User: {item['query']}")
                prompt_parts.append(f"Daniel: {item['response']}")
            prompt_parts.append("")
        
        prompt_parts.append(f"User: {query}")
        prompt_parts.append("Daniel:")
        
        return "\n".join(prompt_parts)
    
    def _update_conversation_history(self, query: str, response: Dict[str, Any]):
        """Update conversation history with size limit"""
        self.conversation_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "response": response.get("content", ""),
            "consciousness_level": self.cognitive_state.consciousness_level
        })
        
        # Keep only recent history to manage memory
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def _serialize_cognitive_state(self) -> Dict[str, Any]:
        """Serialize cognitive state for external use"""
        return {
            "consciousness_level": self.cognitive_state.consciousness_level,
            "attention_focus": self.cognitive_state.attention_focus,
            "active_thoughts": self.cognitive_state.active_thoughts,
            "emotional_state": self.cognitive_state.emotional_state,
            "context_window_size": len(self.cognitive_state.context_window),
            "working_memory_keys": list(self.cognitive_state.working_memory.keys())
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Daniel cognitive engine shutting down...")
        
        # Save conversation history
        psi_archive.log_event("daniel_shutdown", {
            "total_conversations": len(self.conversation_history),
            "final_consciousness_level": self.cognitive_state.consciousness_level
        })
        
        # Clean up resources
        self.conversation_history.clear()
        self.cognitive_state = CognitiveState()

# Export
__all__ = ['DanielCognitiveEngine', 'CognitiveState']
