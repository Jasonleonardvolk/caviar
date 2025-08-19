"""
Soliton Memory SDK
Public API for interacting with soliton memory systems
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

# Import core modules
from python.core.soliton_memory_integration import EnhancedSolitonMemory
from python.core.hot_swap_laplacian import HotSwapLaplacian
from python.core.oscillator_lattice import get_global_lattice
from python.core.physics_instrumentation import PhysicsMonitor


class SolitonType(Enum):
    """Types of solitons"""
    BRIGHT = "bright"
    DARK = "dark"


class TopologyType(Enum):
    """Available lattice topologies"""
    KAGOME = "kagome"
    HEXAGONAL = "hexagonal"
    SQUARE = "square"
    SMALL_WORLD = "small_world"
    ALL_TO_ALL = "all_to_all"


@dataclass
class MemoryEntry:
    """Public memory entry representation"""
    id: str
    content: str
    soliton_type: SolitonType
    phase: float
    amplitude: float
    frequency: float
    concepts: List[str]
    metadata: Dict[str, Any]


@dataclass
class SystemStatus:
    """System status information"""
    total_memories: int
    bright_solitons: int
    dark_solitons: int
    current_topology: str
    is_morphing: bool
    morph_progress: float
    total_energy: float
    order_parameter: float
    active_oscillators: int


class SolitonMemorySDK:
    """
    Main SDK interface for soliton memory systems
    
    This provides a simplified, stable API for applications to interact
    with the soliton memory engine without needing to understand the
    internal implementation details.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SDK
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize core components
        self._memory = EnhancedSolitonMemory()
        self._lattice = get_global_lattice()
        self._hot_swap = HotSwapLaplacian()
        self._monitor = PhysicsMonitor("SDK")
        
        # Apply configuration
        self._apply_config()
    
    def _apply_config(self):
        """Apply configuration settings"""
        if 'default_topology' in self.config:
            self._hot_swap.switch_topology(self.config['default_topology'])
        
        if 'harvest_efficiency' in self.config:
            self._hot_swap.energy_harvest_efficiency = self.config['harvest_efficiency']
    
    # === Memory Operations ===
    
    def store_memory(
        self,
        content: str,
        concepts: Optional[List[str]] = None,
        soliton_type: SolitonType = SolitonType.BRIGHT,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a memory in the soliton system
        
        Args:
            content: The content to store
            concepts: Associated concepts/tags
            soliton_type: Type of soliton to create
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        # Store using internal API
        memory_id = self._memory.store(
            content=content,
            memory_type=soliton_type.value,
            metadata={
                'concepts': concepts or [],
                'sdk_metadata': metadata or {},
                'source': 'sdk'
            }
        )
        
        return memory_id
    
    def retrieve_memories(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.5
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Retrieve memories matching a query
        
        Args:
            query: Search query
            limit: Maximum number of results
            threshold: Minimum similarity score
            
        Returns:
            List of (memory_entry, score) tuples
        """
        # Retrieve using internal API
        results = self._memory.retrieve(query, top_k=limit)
        
        # Convert to public format
        public_results = []
        for mem_id, score, entry in results:
            if score < threshold:
                continue
            
            public_entry = MemoryEntry(
                id=mem_id,
                content=entry.content,
                soliton_type=SolitonType(entry.memory_type),
                phase=entry.phase,
                amplitude=entry.amplitude,
                frequency=entry.frequency,
                concepts=entry.metadata.get('concepts', []),
                metadata=entry.metadata
            )
            
            public_results.append((public_entry, score))
        
        return public_results
    
    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID"""
        if memory_id not in self._memory.memory_entries:
            return None
        
        entry = self._memory.memory_entries[memory_id]
        
        return MemoryEntry(
            id=memory_id,
            content=entry.content,
            soliton_type=SolitonType(entry.memory_type),
            phase=entry.phase,
            amplitude=entry.amplitude,
            frequency=entry.frequency,
            concepts=entry.metadata.get('concepts', []),
            metadata=entry.metadata
        )
    
    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing memory
        
        Returns:
            True if updated, False if memory not found
        """
        if memory_id not in self._memory.memory_entries:
            return False
        
        entry = self._memory.memory_entries[memory_id]
        
        if content is not None:
            entry.content = content
        
        if concepts is not None:
            entry.metadata['concepts'] = concepts
        
        if metadata is not None:
            entry.metadata['sdk_metadata'] = metadata
        
        # Update timestamp
        import time
        entry.metadata['last_updated'] = time.time()
        
        return True
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory"""
        if memory_id in self._memory.memory_entries:
            del self._memory.memory_entries[memory_id]
            return True
        return False
    
    # === Topology Operations ===
    
    def get_current_topology(self) -> TopologyType:
        """Get current lattice topology"""
        return TopologyType(self._hot_swap.current_topology)
    
    def switch_topology(
        self,
        topology: TopologyType,
        gradual: bool = True,
        rate: float = 0.02
    ) -> None:
        """
        Switch to a new lattice topology
        
        Args:
            topology: Target topology
            gradual: If True, morph gradually; if False, switch immediately
            rate: Morphing rate (if gradual)
        """
        if gradual:
            self._hot_swap.initiate_morph(topology.value, blend_rate=rate)
        else:
            self._hot_swap.switch_topology(topology.value)
    
    def is_morphing(self) -> bool:
        """Check if topology is currently morphing"""
        return self._hot_swap.is_morphing
    
    def get_morph_progress(self) -> float:
        """Get morphing progress (0.0 to 1.0)"""
        return self._hot_swap.morph_progress if self._hot_swap.is_morphing else 1.0
    
    # === System Operations ===
    
    def trigger_consolidation(self) -> Dict[str, int]:
        """
        Trigger memory consolidation
        
        Returns:
            Dictionary with consolidation results
        """
        fused = self._memory._perform_memory_fusion()
        split = self._memory._perform_memory_fission()
        
        return {
            'memories_fused': fused,
            'memories_split': split
        }
    
    def get_status(self) -> SystemStatus:
        """Get comprehensive system status"""
        # Count memory types
        total = len(self._memory.memory_entries)
        dark = sum(1 for e in self._memory.memory_entries.values() 
                  if e.memory_type == 'dark')
        bright = total - dark
        
        # Get lattice metrics
        order_param = 0.0
        active_osc = 0
        total_energy = 0.0
        
        if hasattr(self._lattice, 'order_parameter'):
            order_param = self._lattice.order_parameter()
        
        if hasattr(self._lattice, 'oscillators'):
            active_osc = sum(1 for o in self._lattice.oscillators 
                           if o.get('active', True))
        
        if hasattr(self._lattice, 'compute_total_energy'):
            total_energy = self._lattice.compute_total_energy()
        
        return SystemStatus(
            total_memories=total,
            bright_solitons=bright,
            dark_solitons=dark,
            current_topology=self._hot_swap.current_topology,
            is_morphing=self._hot_swap.is_morphing,
            morph_progress=self._hot_swap.morph_progress,
            total_energy=total_energy,
            order_parameter=order_param,
            active_oscillators=active_osc
        )
    
    def get_energy_metrics(self) -> Dict[str, float]:
        """Get energy-related metrics"""
        return {
            'total_energy': self._lattice.compute_total_energy() 
                          if hasattr(self._lattice, 'compute_total_energy') else 0.0,
            'harvested_energy': self._hot_swap.total_harvested_energy,
            'harvest_efficiency': self._hot_swap.energy_harvest_efficiency,
            'energy_threshold': getattr(self._hot_swap, 'energy_threshold', 1000.0)
        }
    
    # === Batch Operations ===
    
    def batch_store(
        self,
        items: List[Dict[str, Any]],
        default_type: SolitonType = SolitonType.BRIGHT
    ) -> List[str]:
        """
        Store multiple memories at once
        
        Args:
            items: List of dictionaries with 'content', 'concepts', etc.
            default_type: Default soliton type if not specified
            
        Returns:
            List of memory IDs
        """
        memory_ids = []
        
        for item in items:
            memory_id = self.store_memory(
                content=item['content'],
                concepts=item.get('concepts'),
                soliton_type=item.get('type', default_type),
                metadata=item.get('metadata')
            )
            memory_ids.append(memory_id)
        
        return memory_ids
    
    def search_by_concepts(
        self,
        concepts: List[str],
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search memories by concept tags"""
        results = []
        
        for mem_id, entry in self._memory.memory_entries.items():
            entry_concepts = entry.metadata.get('concepts', [])
            
            # Check for concept overlap
            if any(c in entry_concepts for c in concepts):
                public_entry = self.get_memory(mem_id)
                if public_entry:
                    results.append(public_entry)
        
        return results[:limit]
    
    # === Export/Import ===
    
    def export_state(self) -> Dict[str, Any]:
        """Export complete system state"""
        return {
            'memories': {
                mem_id: {
                    'content': entry.content,
                    'type': entry.memory_type,
                    'phase': entry.phase,
                    'amplitude': entry.amplitude,
                    'frequency': entry.frequency,
                    'metadata': entry.metadata
                }
                for mem_id, entry in self._memory.memory_entries.items()
            },
            'topology': self._hot_swap.current_topology,
            'config': self.config
        }
    
    def import_state(self, state: Dict[str, Any], merge: bool = False):
        """Import system state"""
        if not merge:
            self._memory.memory_entries.clear()
        
        # Import memories
        for mem_id, mem_data in state.get('memories', {}).items():
            if mem_id not in self._memory.memory_entries:
                # Create memory entry
                self._memory.memory_entries[mem_id] = type('MemoryEntry', (), mem_data)()
        
        # Set topology
        if 'topology' in state:
            self._hot_swap.switch_topology(state['topology'])


# Convenience functions for quick access
def create_sdk(config: Optional[Dict[str, Any]] = None) -> SolitonMemorySDK:
    """Create a new SDK instance"""
    return SolitonMemorySDK(config)


def quick_store(content: str, **kwargs) -> str:
    """Quick store using global SDK instance"""
    global _global_sdk
    if '_global_sdk' not in globals():
        _global_sdk = create_sdk()
    return _global_sdk.store_memory(content, **kwargs)


def quick_retrieve(query: str, **kwargs) -> List[Tuple[MemoryEntry, float]]:
    """Quick retrieve using global SDK instance"""
    global _global_sdk
    if '_global_sdk' not in globals():
        _global_sdk = create_sdk()
    return _global_sdk.retrieve_memories(query, **kwargs)
