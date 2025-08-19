"""
TORI/KHA Memory Vault V2 Integration
Shows how to integrate the improved memory vault with the enhanced launcher
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import sys

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))

from memory_vault_v2 import UnifiedMemoryVaultV2, MemoryType

logger = logging.getLogger(__name__)


class MemoryVaultService:
    """
    Service wrapper for Memory Vault V2 that integrates with enhanced_launcher.py
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.vault: Optional[UnifiedMemoryVaultV2] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the memory vault service"""
        if self._initialized:
            return
        
        # Default configuration
        default_config = {
            'storage_path': 'data/memory_vault_v2',
            'max_working_memory': 100,
            'ghost_memory_ttl': 3600,
            'batch_size': 50,
            'packfile_threshold': 1000
        }
        
        # Merge with provided config
        vault_config = {**default_config, **self.config}
        
        # Create and initialize vault
        self.vault = UnifiedMemoryVaultV2(vault_config)
        await self.vault.initialize()
        
        self._initialized = True
        logger.info("Memory Vault V2 service initialized")
    
    async def store_agent_context(self, agent_id: str, context: Dict[str, Any]):
        """Store agent context in working memory"""
        if not self._initialized:
            await self.initialize()
        
        return await self.vault.store(
            content=context,
            memory_type=MemoryType.WORKING,
            metadata={
                'agent_id': agent_id,
                'type': 'agent_context',
                'tags': ['agent', 'context', agent_id]
            },
            importance=0.8
        )
    
    async def store_conversation(self, conversation_id: str, messages: list):
        """Store conversation history"""
        if not self._initialized:
            await self.initialize()
        
        return await self.vault.store(
            content={'messages': messages},
            memory_type=MemoryType.EPISODIC,
            metadata={
                'conversation_id': conversation_id,
                'message_count': len(messages),
                'tags': ['conversation', 'history']
            }
        )
    
    async def store_learned_fact(self, fact: str, source: str, confidence: float = 0.8):
        """Store a learned fact"""
        if not self._initialized:
            await self.initialize()
        
        return await self.vault.store(
            content=fact,
            memory_type=MemoryType.SEMANTIC,
            metadata={
                'source': source,
                'confidence': confidence,
                'tags': ['fact', 'learned']
            },
            importance=confidence
        )
    
    async def get_agent_context(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve agent context"""
        if not self._initialized:
            await self.initialize()
        
        results = await self.vault.search(
            tags=['agent', agent_id],
            memory_type=MemoryType.WORKING,
            max_results=1
        )
        
        if results:
            memory = results[0]
            return memory.content
        return None
    
    async def search_conversations(self, query: str = None, limit: int = 10):
        """Search conversation history"""
        if not self._initialized:
            await self.initialize()
        
        return await self.vault.search(
            query=query,
            tags=['conversation'],
            memory_type=MemoryType.EPISODIC,
            max_results=limit
        )
    
    async def find_related_facts(self, embedding, threshold: float = 0.7):
        """Find facts related to given embedding"""
        if not self._initialized:
            await self.initialize()
        
        similar = await self.vault.find_similar(
            embedding=embedding,
            memory_type=MemoryType.SEMANTIC,
            threshold=threshold,
            max_results=10
        )
        
        return [(m.content, score) for m, score in similar]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get memory vault statistics"""
        if not self._initialized:
            await self.initialize()
        
        return await self.vault.get_statistics()
    
    async def save_all(self):
        """Save all memories - called by enhanced_launcher on shutdown"""
        if self._initialized and self.vault:
            await self.vault.save_all()
            logger.info("Memory Vault V2 saved all data")
    
    async def shutdown(self):
        """Shutdown the memory vault service"""
        if self._initialized and self.vault:
            await self.vault.shutdown()
            self._initialized = False
            logger.info("Memory Vault V2 service shut down")


# Integration with enhanced_launcher.py
class MemoryEnabledAgent:
    """Example agent that uses the memory vault"""
    
    def __init__(self, agent_id: str, memory_service: MemoryVaultService):
        self.agent_id = agent_id
        self.memory = memory_service
        self.context = {}
    
    async def initialize(self):
        """Initialize agent with stored context"""
        # Load previous context if available
        stored_context = await self.memory.get_agent_context(self.agent_id)
        if stored_context:
            self.context = stored_context
            logger.info(f"Loaded context for agent {self.agent_id}")
        else:
            self.context = {
                'agent_id': self.agent_id,
                'created_at': asyncio.get_event_loop().time(),
                'interactions': 0
            }
    
    async def process_message(self, message: str) -> str:
        """Process a message with memory"""
        # Update interaction count
        self.context['interactions'] += 1
        self.context['last_message'] = message
        
        # Store context update
        await self.memory.store_agent_context(self.agent_id, self.context)
        
        # Search for related facts
        # (In real implementation, you'd generate embedding from message)
        # facts = await self.memory.find_related_facts(message_embedding)
        
        response = f"Processed message #{self.context['interactions']}: {message}"
        return response
    
    async def learn_fact(self, fact: str):
        """Learn a new fact"""
        await self.memory.store_learned_fact(
            fact=fact,
            source=self.agent_id,
            confidence=0.9
        )
        logger.info(f"Agent {self.agent_id} learned: {fact}")


# Example integration function for enhanced_launcher.py
async def create_memory_service() -> MemoryVaultService:
    """
    Factory function to create memory service for enhanced_launcher.py
    
    This would be called in the launcher's initialization:
    
    ```python
    # In enhanced_launcher.py
    async def initialize_services():
        memory_service = await create_memory_service()
        return {'memory': memory_service}
    ```
    """
    service = MemoryVaultService({
        'storage_path': 'data/tori_kha_memories',
        'max_working_memory': 200,
        'ghost_memory_ttl': 7200,  # 2 hours
        'batch_size': 100
    })
    
    await service.initialize()
    return service


# Standalone test
async def test_integration():
    """Test the integration"""
    logging.basicConfig(level=logging.INFO)
    
    # Create memory service
    memory_service = await create_memory_service()
    
    # Create an agent
    agent = MemoryEnabledAgent("test_agent_001", memory_service)
    await agent.initialize()
    
    # Process some messages
    messages = [
        "Hello, I'm testing the memory system",
        "Store this important fact: Python asyncio is powerful",
        "What do you remember about our conversation?"
    ]
    
    for msg in messages:
        response = await agent.process_message(msg)
        print(f"User: {msg}")
        print(f"Agent: {response}\n")
    
    # Learn some facts
    await agent.learn_fact("The memory vault uses async/await throughout")
    await agent.learn_fact("Deduplication prevents storing duplicate memories")
    
    # Store a conversation
    conversation_id = "conv_001"
    await memory_service.store_conversation(conversation_id, messages)
    
    # Get statistics
    stats = await memory_service.get_statistics()
    print(f"\nMemory Statistics:")
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Working memory: {stats['working_memory_count']}")
    print(f"  Storage size: {stats['total_size_mb']:.2f} MB")
    
    # Search conversations
    conversations = await memory_service.search_conversations(limit=5)
    print(f"\nFound {len(conversations)} conversations")
    
    # Graceful shutdown
    await memory_service.shutdown()


if __name__ == "__main__":
    asyncio.run(test_integration())
