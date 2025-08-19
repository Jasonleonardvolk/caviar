"""
Cognitive State Manager
======================

Manages the global cognitive state and components for the MCP server.
Enhanced with REAL TORI filtering and Soliton Memory system.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import json
import asyncio
from pathlib import Path
import aiofiles
from collections import deque
import logging

from .config import config
from .real_tori_bridge import RealTORIFilter, FilteredContent
from .soliton_memory import SolitonMemoryLattice, SolitonMemory, VaultStatus, ContentType

# Import TORI cognitive framework
from cog import (
    MetaCognitiveManifold,
    ReflectiveOperator,
    SelfModificationOperator,
    CuriosityFunctional,
    TransferMorphism,
    CognitiveDynamics,
    ConsciousnessMonitor,
    LyapunovStabilizer,
    MetacognitiveTower,
    KnowledgeSheaf,
    compute_iit_phi,
    compute_free_energy,
    find_fixed_point,
    set_random_seed
)

logger = logging.getLogger(__name__)


class CognitiveStateManager:
    """
    Thread-safe singleton manager for cognitive state and components.
    🏆 Enhanced with REAL TORI filtering and 🌊 Soliton Memory system
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the cognitive state manager."""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.config = config
        
        # Initialize state
        self.dimension = config.cognitive_dimension
        self.current_state = np.random.randn(self.dimension) * 0.1
        self.state_lock = asyncio.Lock()
        
        # Initialize REAL TORI filtering
        self.tori_filter = RealTORIFilter()
        
        # Initialize soliton memory system
        self.soliton_lattice = SolitonMemoryLattice(f"mcp_user_{self.dimension}")
        
        # Initialize components
        self._initialize_components()
        
        # History and tracking
        self.state_history = deque(maxlen=1000)
        self.event_history = deque(maxlen=500)
        self.session_start = datetime.now()
        
        # Persistence
        self.persistence_enabled = True
        self.last_save_time = None
        
        logger.info(f"Cognitive State Manager initialized with dimension {self.dimension}")
        logger.info(f"🏆 REAL TORI filtering enabled: {self.tori_filter.real_tori_available}")
        logger.info(f"🌊 Soliton memory lattice initialized for infinite context")
    
    def _initialize_components(self):
        """Initialize all cognitive components."""
        # Manifold
        self.manifold = MetaCognitiveManifold(
            self.dimension,
            metric=self.config.manifold_metric
        )
        
        # Define log posterior for reflection
        def log_posterior(s):
            return -0.5 * np.sum(s**2) - compute_free_energy(s)
        
        # Reflective operator
        self.reflective_op = ReflectiveOperator(
            self.manifold,
            log_posterior,
            step_size=0.01,
            momentum=0.9
        )
        
        # Self-modification operator
        self.self_mod_op = SelfModificationOperator(
            self.manifold,
            compute_free_energy,
            compute_iit_phi,
            iit_weight=1.0,
            step_size=0.01,
            max_iter=100
        )
        
        # Curiosity functional
        self.curiosity = CuriosityFunctional(
            self.manifold,
            decay_const=1.0,
            exploration_bonus=0.5,
            memory_capacity=100
        )
        
        # Cognitive dynamics
        self.dynamics = CognitiveDynamics(
            self.manifold,
            self.reflective_op,
            self.curiosity,
            noise_sigma=0.1
        )
        
        # Transfer morphism (if GUDHI available)
        try:
            self.transfer = TransferMorphism(
                homology_max_edge=1.0,
                homology_dim=2
            )
        except ImportError:
            logger.warning("GUDHI not available, transfer morphism disabled")
            self.transfer = None
        
        # Consciousness monitor
        self.consciousness_monitor = ConsciousnessMonitor(
            phi_threshold=config.consciousness_threshold,
            history_size=100
        )
        
        # Lyapunov stabilizer
        self.stabilizer = LyapunovStabilizer(
            compute_free_energy,
            stability_margin=0.01,
            intervention_threshold=0.1
        )
        
        # Metacognitive tower
        self.tower = MetacognitiveTower(
            base_dim=self.dimension,
            levels=config.max_metacognitive_levels,
            metric=config.manifold_metric
        )
        
        # Knowledge sheaf
        self.sheaf = KnowledgeSheaf(self.manifold)
        self._setup_knowledge_regions()
    
    def _setup_knowledge_regions(self):
        """Set up default knowledge sheaf regions."""
        regions = {
            'perception': lambda s: {
                'sensory': s[:self.dimension//4].tolist(),
                'attention': float(np.mean(np.abs(s[:self.dimension//4])))
            },
            'reasoning': lambda s: {
                'logic': s[self.dimension//4:self.dimension//2].tolist(),
                'coherence': float(np.std(s[self.dimension//4:self.dimension//2]))
            },
            'memory': lambda s: {
                'trace': s[self.dimension//2:3*self.dimension//4].tolist(),
                'strength': float(np.linalg.norm(s[self.dimension//2:3*self.dimension//4]))
            },
            'action': lambda s: {
                'potential': s[3*self.dimension//4:].tolist(),
                'readiness': float(np.max(np.abs(s[3*self.dimension//4:])))
            }
        }
        
        for region, func in regions.items():
            self.sheaf.add_section(region, func)
    
    async def get_current_state(self) -> Dict[str, Any]:
        """Get current cognitive state with metadata."""
        async with self.state_lock:
            return {
                'state': self.current_state.tolist(),
                'dimension': self.dimension,
                'phi': float(compute_iit_phi(self.current_state)),
                'free_energy': float(compute_free_energy(self.current_state)),
                'timestamp': datetime.now().isoformat(),
                'session_duration': (datetime.now() - self.session_start).total_seconds()
            }
    
    async def update_state(self, new_state: np.ndarray, 
                          source: str = "unknown",
                          metadata: Optional[Dict] = None,
                          filter_content: bool = True):
        """Update cognitive state with tracking and optional TORI filtering."""
        async with self.state_lock:
            old_state = self.current_state.copy()
            
            # Apply REAL TORI filtering if requested
            if filter_content:
                try:
                    # Create content for filtering
                    state_content = {
                        'state': new_state.tolist(),
                        'source': source,
                        'metadata': metadata or {}
                    }
                    
                    # Filter through REAL TORI
                    filtered_content = await self.tori_filter.filter_input(state_content)
                    
                    # Log filtering result
                    if isinstance(filtered_content, str) and "BLOCKED" in filtered_content:
                        logger.warning(f"🚨 TORI blocked state update from source: {source}")
                        # Don't update state if blocked
                        return
                    
                    # Update metadata with filtering info
                    if metadata is None:
                        metadata = {}
                    metadata['tori_filtered'] = True
                    metadata['tori_available'] = self.tori_filter.real_tori_available
                    
                except Exception as e:
                    logger.error(f"TORI filtering error: {e}")
                    # Continue with unfiltered update but log the issue
                    if metadata is None:
                        metadata = {}
                    metadata['tori_error'] = str(e)
            
            self.current_state = new_state.copy()
            
            # Compute metrics
            phi = compute_iit_phi(new_state)
            free_energy = compute_free_energy(new_state)
            
            # Create history entry
            entry = {
                'timestamp': datetime.now().isoformat(),
                'state': new_state.tolist(),
                'phi': float(phi),
                'free_energy': float(free_energy),
                'source': source,
                'metadata': metadata or {}
            }
            
            self.state_history.append(entry)
            
            # Check consciousness preservation
            preserved = self.consciousness_monitor.check_preservation(
                old_state, new_state
            )
            
            # Store state change in soliton memory for infinite context
            try:
                state_change_content = {
                    'old_state_summary': {
                        'phi': float(compute_iit_phi(old_state)),
                        'free_energy': float(compute_free_energy(old_state)),
                        'norm': float(np.linalg.norm(old_state))
                    },
                    'new_state_summary': {
                        'phi': float(phi),
                        'free_energy': float(free_energy),
                        'norm': float(np.linalg.norm(new_state))
                    },
                    'source': source,
                    'consciousness_preserved': preserved,
                    'metadata': metadata or {}
                }
                
                # Store in soliton memory
                memory_id = await self.soliton_lattice.store_memory(
                    concept_id=f"state_change_{source}",
                    content=json.dumps(state_change_content),
                    importance=min(1.0, abs(float(phi - compute_iit_phi(old_state))) + 0.1),
                    content_type=ContentType.COGNITIVE_STATE
                )
                
                logger.debug(f"Stored state change in soliton memory: {memory_id}")
                
            except Exception as e:
                logger.error(f"Failed to store state change in soliton memory: {e}")
            
            # Log event
            self._log_event('state_update', {
                'source': source,
                'consciousness_preserved': preserved,
                'phi_change': float(phi - compute_iit_phi(old_state)),
                'free_energy_change': float(free_energy - compute_free_energy(old_state)),
                'tori_filtered': filter_content,
                'soliton_stored': True
            })
            
            # Auto-save if enabled
            if self.persistence_enabled:
                await self._auto_save()
    
    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event to history."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'data': data
        }
        self.event_history.append(event)
        logger.debug(f"Event logged: {event_type}")
    
    async def _auto_save(self):
        """Auto-save state if enough time has passed."""
        now = datetime.now()
        if (self.last_save_time is None or 
            (now - self.last_save_time).total_seconds() > 60):
            await self.save_state()
            self.last_save_time = now
    
    async def save_state(self, filename: Optional[str] = None):
        """Save current state to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cognitive_state_{timestamp}.json"
        
        filepath = config.state_persistence_path / filename
        
        # Get soliton memory stats for saving
        memory_stats = await self.soliton_lattice.get_memory_statistics()
        
        state_data = {
            'timestamp': datetime.now().isoformat(),
            'dimension': self.dimension,
            'current_state': self.current_state.tolist(),
            'config': {
                'manifold_metric': config.manifold_metric,
                'consciousness_threshold': config.consciousness_threshold,
                'max_metacognitive_levels': config.max_metacognitive_levels
            },
            'metrics': {
                'phi': float(compute_iit_phi(self.current_state)),
                'free_energy': float(compute_free_energy(self.current_state))
            },
            'consciousness_stats': self.consciousness_monitor.get_statistics(),
            'history_length': len(self.state_history),
            'event_count': len(self.event_history),
            'tori_filter_active': self.tori_filter.real_tori_available,
            'soliton_memory_stats': memory_stats
        }
        
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(state_data, indent=2))
        
        # Also save soliton memory lattice
        soliton_filepath = filepath.with_suffix('.soliton.json')
        self.soliton_lattice.save_to_file(str(soliton_filepath))
        
        logger.info(f"State saved to {filepath}")
        logger.info(f"Soliton memory saved to {soliton_filepath}")
        self._log_event('state_saved', {
            'filename': str(filepath),
            'soliton_filename': str(soliton_filepath)
        })
    
    async def load_state(self, filename: str):
        """Load state from file."""
        filepath = config.state_persistence_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"State file not found: {filepath}")
        
        async with aiofiles.open(filepath, 'r') as f:
            content = await f.read()
            state_data = json.loads(content)
        
        # Validate dimension
        if state_data['dimension'] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: file has {state_data['dimension']}, "
                f"but manager has {self.dimension}"
            )
        
        # Load state
        new_state = np.array(state_data['current_state'])
        await self.update_state(new_state, source='load_state', metadata={
            'filename': filename,
            'original_timestamp': state_data['timestamp']
        })
        
        # Try to load soliton memory if available
        soliton_filepath = filepath.with_suffix('.soliton.json')
        if soliton_filepath.exists():
            try:
                self.soliton_lattice = SolitonMemoryLattice.load_from_file(str(soliton_filepath))
                logger.info(f"Soliton memory loaded from {soliton_filepath}")
            except Exception as e:
                logger.error(f"Failed to load soliton memory: {e}")
        
        logger.info(f"State loaded from {filepath}")
    
    async def get_recent_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent state history."""
        async with self.state_lock:
            return list(self.state_history)[-n:]
    
    async def get_trajectory(self, n: int = 50) -> np.ndarray:
        """Get recent trajectory as numpy array."""
        async with self.state_lock:
            if len(self.state_history) < 2:
                return np.array([self.current_state])
            
            states = [h['state'] for h in list(self.state_history)[-n:]]
            return np.array(states)
    
    async def reset(self, seed: Optional[int] = None):
        """Reset cognitive state to initial conditions."""
        async with self.state_lock:
            if seed is not None:
                set_random_seed(seed)
            
            self.current_state = np.random.randn(self.dimension) * 0.1
            self.state_history.clear()
            self.event_history.clear()
            self.session_start = datetime.now()
            
            # Reset components
            self.reflective_op.reset_momentum()
            self.curiosity.reset_memory()
            
            # Reset soliton memory lattice
            self.soliton_lattice = SolitonMemoryLattice(f"mcp_user_{self.dimension}")
            
            self._log_event('reset', {'seed': seed})
            logger.info("Cognitive state reset")
            logger.info("🌊 Soliton memory lattice reset")
    
    def get_component(self, component_name: str) -> Any:
        """Get a cognitive component by name."""
        components = {
            'manifold': self.manifold,
            'reflective': self.reflective_op,
            'self_modification': self.self_mod_op,
            'curiosity': self.curiosity,
            'dynamics': self.dynamics,
            'transfer': self.transfer,
            'consciousness_monitor': self.consciousness_monitor,
            'stabilizer': self.stabilizer,
            'tower': self.tower,
            'sheaf': self.sheaf,
            'tori_filter': self.tori_filter,
            'soliton_lattice': self.soliton_lattice
        }
        
        if component_name not in components:
            raise ValueError(f"Unknown component: {component_name}")
        
        return components[component_name]
    
    async def store_memory(self, concept_id: str, content: str, 
                          importance: float = 1.0,
                          content_type: ContentType = ContentType.TEXT,
                          filter_content: bool = True) -> str:
        """Store memory in soliton lattice with optional TORI filtering"""
        try:
            # Apply REAL TORI filtering if requested
            filtered_content = content
            if filter_content:
                filtered_content = await self.tori_filter.filter_input(content)
                
                # Check if content was blocked
                if isinstance(filtered_content, str) and "BLOCKED" in filtered_content:
                    logger.warning(f"🚨 TORI blocked memory storage for concept: {concept_id}")
                    # Store a safe placeholder instead
                    filtered_content = "[Content filtered by TORI for safety]"
            
            # Store in soliton memory
            memory_id = await self.soliton_lattice.store_memory(
                concept_id=concept_id,
                content=filtered_content,
                importance=importance,
                content_type=content_type
            )
            
            self._log_event('memory_stored', {
                'memory_id': memory_id,
                'concept_id': concept_id,
                'content_length': len(content),
                'filtered_length': len(filtered_content),
                'importance': importance,
                'content_type': content_type.value,
                'tori_filtered': filter_content
            })
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise
    
    async def recall_memories(self, concept_id: str, max_results: int = 5,
                             include_vaulted: bool = False) -> List[SolitonMemory]:
        """Recall related memories using soliton phase correlation"""
        try:
            memories = await self.soliton_lattice.find_related_memories(
                concept_id=concept_id,
                max_results=max_results,
                include_vaulted=include_vaulted
            )
            
            self._log_event('memory_recalled', {
                'concept_id': concept_id,
                'memories_found': len(memories),
                'include_vaulted': include_vaulted
            })
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to recall memories: {e}")
            return []
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory and system statistics"""
        try:
            # Get soliton memory stats
            memory_stats = await self.soliton_lattice.get_memory_statistics()
            
            # Get TORI filter stats
            tori_stats = {
                'real_tori_available': self.tori_filter.real_tori_available,
                'filter_active': True
            }
            
            # Get cognitive stats
            current = await self.get_current_state()
            cognitive_stats = {
                'current_phi': current['phi'],
                'current_free_energy': current['free_energy'],
                'session_duration': current['session_duration'],
                'state_updates': len(self.state_history),
                'events_logged': len(self.event_history)
            }
            
            return {
                'memory_system': memory_stats,
                'tori_filtering': tori_stats,
                'cognitive_system': cognitive_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}")
            return {}
    
    async def filter_content(self, content: Any, filter_type: str = "input") -> Any:
        """Apply REAL TORI filtering to arbitrary content"""
        try:
            if filter_type == "input":
                return await self.tori_filter.filter_input(content)
            elif filter_type == "output":
                return await self.tori_filter.filter_output(content)
            elif filter_type == "error":
                return await self.tori_filter.filter_error(str(content))
            else:
                logger.warning(f"Unknown filter type: {filter_type}")
                return content
                
        except Exception as e:
            logger.error(f"Content filtering failed: {e}")
            return "[CONTENT UNAVAILABLE - FILTER ERROR]"


# Global instance
state_manager = CognitiveStateManager()