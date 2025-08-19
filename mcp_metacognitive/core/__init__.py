"""
Core MCP Metacognitive module exports
"""

# Export the correct class name for compatibility
try:
    from .soliton_memory import (
        SolitonMemoryClient as SolitonMemoryLattice,
        SolitonMemoryClient as SolitonMemory,
        SolitonMemoryClient, 
        UnifiedSolitonMemory,
        VaultStatus,
        ContentType
    )
    
    # Re-export all the important functions
    from .soliton_memory import (
        initialize_user,
        store_memory,
        find_related_memories,
        get_user_stats,
        record_phase_change,
        check_health,
        verify_connectivity,
        soliton_client
    )
    
    __all__ = [
        'SolitonMemoryLattice',  # Alias for backward compatibility
        'SolitonMemory',         # Another alias for backward compatibility
        'SolitonMemoryClient',
        'UnifiedSolitonMemory',
        'VaultStatus',
        'ContentType',
        'initialize_user',
        'store_memory',
        'find_related_memories',
        'get_user_stats',
        'record_phase_change',
        'check_health',
        'verify_connectivity',
        'soliton_client'
    ]
    
except ImportError as e:
    print(f"⚠️ Failed to import soliton_memory components: {e}")
    # Provide dummy classes to prevent total failure
    class SolitonMemoryLattice:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("SolitonMemoryLattice not available - check imports")
    
    SolitonMemoryClient = SolitonMemoryLattice
    UnifiedSolitonMemory = SolitonMemoryLattice
