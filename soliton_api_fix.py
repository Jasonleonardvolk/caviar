"""
Fix for soliton API - adds missing async functions
"""

# Add this to the end of mcp_metacognitive/core/soliton_memory.py

# Global client instance
_soliton_client = None

def get_client():
    """Get or create the global soliton client"""
    global _soliton_client
    if _soliton_client is None:
        _soliton_client = SolitonMemoryClient()
    return _soliton_client

# Async wrapper functions for API compatibility
async def initialize_user(user_id: str) -> bool:
    """Initialize soliton memory for a user"""
    try:
        client = get_client()
        # The client already initializes on creation
        return True
    except Exception as e:
        logging.error(f"Failed to initialize soliton for user {user_id}: {e}")
        return False

async def get_user_stats(user_id: str) -> Dict[str, Any]:
    """Get memory statistics for a user"""
    try:
        client = get_client()
        stats = client.get_memory_stats()
        return stats
    except Exception as e:
        logging.error(f"Failed to get stats for user {user_id}: {e}")
        return {
            "total_memories": 0,
            "active_waves": 0,
            "average_strength": 0.0,
            "cluster_count": 0,
            "status": "error"
        }

async def store_memory(user_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
    """Store a memory for a user"""
    try:
        client = get_client()
        success = client.store_memory(content, metadata)
        return success
    except Exception as e:
        logging.error(f"Failed to store memory for user {user_id}: {e}")
        return False

async def recall_memories(user_id: str, query: str, limit: int = 5) -> List[Dict]:
    """Recall memories for a user based on query"""
    try:
        client = get_client()
        memories = client.recall_memories(query, limit)
        return memories
    except Exception as e:
        logging.error(f"Failed to recall memories for user {user_id}: {e}")
        return []
