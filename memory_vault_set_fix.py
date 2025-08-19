"""
Patch for the UnifiedMemoryVault to fix set serialization issue
This modifies the MemoryEntry.to_dict() method to handle sets properly
"""

# Add this to python/core/memory_vault.py in the MemoryEntry class:

def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary with full BPS metadata! 📝"""
    # Manual conversion to handle sets properly
    result = {
        'id': self.id,
        'content': self.content,
        'memory_type': self.memory_type.value,
        'embedding': self.embedding.tolist() if self.embedding is not None else None,
        'metadata': self.metadata,
        'timestamp': self.timestamp,
        'access_count': self.access_count,
        'last_accessed': self.last_accessed,
        'decay_rate': self.decay_rate,
        'importance': self.importance,
        'topological_charge': self.topological_charge,
        'energy_content': self.energy_content,
        'bps_compliant': self.bps_compliant,
        'phase_coherence': self.phase_coherence,
        'stability_measure': self.stability_measure,
        # Convert sets to lists for JSON serialization
        'tags': list(self.tags),
        'bps_tags': list(self.bps_tags),
        'creation_time': self.creation_time,
        'processing_time': self.processing_time,
        'bps_metrics': self.compute_bps_metrics(),
        'config_available': BPS_CONFIG_AVAILABLE
    }
    return result