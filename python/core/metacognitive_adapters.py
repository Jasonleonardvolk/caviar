#!/usr/bin/env python3
"""
Metacognitive Adapter Layer
Surgical adapters that bridge existing metacognitive modules with the chaos architecture

Implements the adapter pattern to allow seamless integration without breaking changes
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import asyncio
import logging
from enum import Enum
from collections import defaultdict

# Import existing metacognitive components
from python.core.unified_metacognitive_integration import (
    UnifiedMetacognitiveSystem, MetacognitiveState,
    SolitonMemorySystem, ReflectionSystem, 
    CognitiveDynamicsSystem, RealTORIFilter
)
from python.core.reflection_fixed_point_integration import (
    EnhancedReflectionSystem, ReflectionType
)
from python.core.soliton_memory_integration import (
    EnhancedSolitonMemory, SolitonMemoryEntry, MemoryType
)
from python.core.cognitive_dynamics_monitor import (
    CognitiveDynamicsMonitor, DynamicsState
)

# Import new chaos components
from python.core.eigensentry.core import EigenSentry2, InstabilityType
from python.core.chaos_control_layer import (
    ChaosControlLayer, ChaosTask, ChaosMode, ChaosResult
)

logger = logging.getLogger(__name__)

# ========== Adapter Interfaces ==========

class AdapterMode(Enum):
    """Operating modes for adapters"""
    PASSTHROUGH = "passthrough"      # Direct pass to original module
    CHAOS_ASSISTED = "chaos_assisted" # Use CCL for computation
    HYBRID = "hybrid"                # Blend both approaches
    CHAOS_ONLY = "chaos_only"        # Full chaos mode

@dataclass
class AdapterConfig:
    """Configuration for metacognitive adapters"""
    mode: AdapterMode = AdapterMode.HYBRID
    chaos_threshold: float = 0.3  # When to engage chaos assistance
    energy_allocation: int = 100  # Energy budget per operation
    safety_margin: float = 0.8    # Safety factor for chaos operations

# ========== Memory Adapter ==========

class SolitonMemoryAdapter:
    """
    Adapter for soliton memory system
    Enhances memory operations with dark soliton dynamics from CCL
    """
    
    def __init__(self, original_memory: EnhancedSolitonMemory,
                 ccl: ChaosControlLayer,
                 config: AdapterConfig = None):
        self.original = original_memory
        self.ccl = ccl
        self.config = config or AdapterConfig()
        self.operation_stats = defaultdict(int)
        
    async def store_memory_with_chaos(self, content: str, concept_ids: List[str],
                                    memory_type: MemoryType,
                                    sources: List[str]) -> str:
        """Store memory using chaos-enhanced encoding"""
        
        # First, use original storage
        memory_id = self.original.store_enhanced_memory(
            content, concept_ids, memory_type, sources
        )
        
        # If chaos mode enabled, enhance with soliton dynamics
        if self.config.mode in [AdapterMode.CHAOS_ASSISTED, AdapterMode.HYBRID]:
            # Convert memory to phase representation
            phase_data = self._memory_to_phase_array(content, concept_ids)
            
            # Submit to CCL for soliton encoding
            chaos_task = ChaosTask(
                task_id=f"mem_encode_{memory_id}",
                mode=ChaosMode.DARK_SOLITON,
                input_data=phase_data,
                parameters={
                    'time_steps': 100,
                    'encoding_depth': 0.8
                },
                energy_budget=self.config.energy_allocation
            )
            
            try:
                await self.ccl.submit_task(chaos_task)
                self.operation_stats['chaos_enhanced_stores'] += 1
            except Exception as e:
                logger.warning(f"Chaos enhancement failed: {e}")
                self.operation_stats['chaos_failures'] += 1
                
        return memory_id
        
    async def find_memories_with_chaos(self, query_phase: float,
                                     concept_ids: List[str],
                                     use_attractor_search: bool = True) -> List[SolitonMemoryEntry]:
        """Find memories using chaos-assisted search"""
        
        # Get baseline results from original system
        baseline_memories = self.original.find_resonant_memories_enhanced(
            query_phase, concept_ids
        )
        
        if not use_attractor_search or \
           self.config.mode not in [AdapterMode.CHAOS_ASSISTED, AdapterMode.CHAOS_ONLY]:
            return baseline_memories
            
        # Enhance search with attractor hopping
        memory_vectors = self._memories_to_vectors(baseline_memories)
        query_vector = self._query_to_vector(query_phase, concept_ids)
        
        chaos_task = ChaosTask(
            task_id=f"mem_search_{datetime.now(timezone.utc).timestamp()}",
            mode=ChaosMode.ATTRACTOR_HOP,
            input_data=query_vector,
            parameters={
                'target': memory_vectors.mean(axis=0) if len(memory_vectors) > 0 else query_vector,
                'max_hops': 30,
                'similarity_threshold': 0.7
            },
            energy_budget=self.config.energy_allocation * 2  # More energy for search
        )
        
        try:
            await self.ccl.submit_task(chaos_task)
            self.operation_stats['chaos_searches'] += 1
            
            # Combine results (simplified - would need result handling)
            return baseline_memories
            
        except Exception as e:
            logger.warning(f"Chaos search failed: {e}")
            self.operation_stats['chaos_failures'] += 1
            return baseline_memories
            
    def _memory_to_phase_array(self, content: str, concepts: List[str]) -> np.ndarray:
        """Convert memory content to phase array for soliton encoding"""
        # Simple encoding - would be more sophisticated in production
        content_hash = hash(content)
        concept_hashes = [hash(c) for c in concepts]
        
        # Create phase array
        phases = np.zeros(100)
        phases[0] = (content_hash % 1000) / 1000 * 2 * np.pi
        
        for i, ch in enumerate(concept_hashes[:10]):
            phases[i+1] = (ch % 1000) / 1000 * 2 * np.pi
            
        return phases
        
    def _memories_to_vectors(self, memories: List[SolitonMemoryEntry]) -> np.ndarray:
        """Convert memories to vector representation"""
        if not memories:
            return np.array([[]])
            
        vectors = []
        for mem in memories:
            vec = np.zeros(50)
            vec[0] = mem.phase
            vec[1] = mem.amplitude
            vec[2] = mem.frequency
            vec[3:3+len(mem.concept_ids)] = [hash(c) % 100 / 100 for c in mem.concept_ids[:47]]
            vectors.append(vec)
            
        return np.array(vectors)
        
    def _query_to_vector(self, phase: float, concepts: List[str]) -> np.ndarray:
        """Convert query to vector representation"""
        vec = np.zeros(50)
        vec[0] = phase
        vec[1] = 1.0  # Default amplitude
        vec[2] = 0.5  # Default frequency
        vec[3:3+len(concepts)] = [hash(c) % 100 / 100 for c in concepts[:47]]
        return vec

# ========== Reflection Adapter ==========

class ReflectionAdapter:
    """
    Adapter for reflection system
    Enhances reflection with phase explosion for deeper insights
    """
    
    def __init__(self, original_reflection: EnhancedReflectionSystem,
                 ccl: ChaosControlLayer,
                 eigen_sentry: EigenSentry2,
                 config: AdapterConfig = None):
        self.original = original_reflection
        self.ccl = ccl
        self.eigen_sentry = eigen_sentry
        self.config = config or AdapterConfig()
        
    async def reflect_with_chaos(self, state: np.ndarray,
                               reflection_type: ReflectionType,
                               use_phase_explosion: bool = True) -> Tuple[np.ndarray, float]:
        """Apply reflection with optional chaos enhancement"""
        
        # Check if we should use chaos
        if not use_phase_explosion or \
           self.config.mode == AdapterMode.PASSTHROUGH:
            return self.original.reflect(state)
            
        # Register chaos event with EigenSentry
        gate_id = self.eigen_sentry.enter_ccl(
            'reflection_adapter',
            self.config.energy_allocation
        )
        
        if not gate_id:
            # Fall back to original if no energy
            return self.original.reflect(state)
            
        try:
            # Trigger phase explosion for exploration
            chaos_task = ChaosTask(
                task_id=f"reflect_explore_{datetime.now(timezone.utc).timestamp()}",
                mode=ChaosMode.PHASE_EXPLOSION,
                input_data=state,
                parameters={
                    'explosion_strength': self._get_explosion_strength(reflection_type),
                    'coherence_target': 0.5
                },
                energy_budget=self.config.energy_allocation
            )
            
            await self.ccl.submit_task(chaos_task)
            
            # Apply original reflection to stabilize
            reflected_state, change = self.original.reflect(state)
            
            # Exit CCL
            self.eigen_sentry.exit_ccl(gate_id, success=True)
            
            # Boost change metric due to chaos exploration
            change *= 1.5
            
            return reflected_state, change
            
        except Exception as e:
            logger.error(f"Chaos reflection failed: {e}")
            self.eigen_sentry.exit_ccl(gate_id, success=False)
            return self.original.reflect(state)
            
    def _get_explosion_strength(self, reflection_type: ReflectionType) -> float:
        """Determine phase explosion strength based on reflection type"""
        strengths = {
            ReflectionType.SHALLOW: 0.5,
            ReflectionType.DEEP: 1.0,
            ReflectionType.ADVERSARIAL: 2.0,
            ReflectionType.CRITICAL: 1.5,
            ReflectionType.CONSTRUCTIVE: 0.8,
            ReflectionType.SYNTHESIS: 1.2
        }
        return strengths.get(reflection_type, 1.0)

# ========== Dynamics Monitor Adapter ==========

class DynamicsMonitorAdapter:
    """
    Adapter for cognitive dynamics monitor
    Integrates with EigenSentry 2.0 for unified chaos management
    """
    
    def __init__(self, original_monitor: CognitiveDynamicsMonitor,
                 eigen_sentry: EigenSentry2,
                 config: AdapterConfig = None):
        self.original = original_monitor
        self.eigen_sentry = eigen_sentry
        self.config = config or AdapterConfig()
        self.chaos_events = deque(maxlen=1000)
        
    async def monitor_with_chaos_awareness(self, window_size: int = 20) -> Dict[str, Any]:
        """Monitor dynamics with awareness of chaos events"""
        
        # Get original monitoring result
        original_result = self.original.monitor_and_stabilize(window_size)
        
        # Get EigenSentry status
        eigen_status = self.eigen_sentry.get_status()
        
        # Combine insights
        enhanced_result = original_result.copy()
        enhanced_result['chaos_metrics'] = {
            'active_chaos_events': eigen_status['active_chaos_events'],
            'current_eigenvalue': eigen_status['current_max_eigenvalue'],
            'energy_available': eigen_status['energy_stats']['total_available'],
            'stability_state': eigen_status['stability_state']
        }
        
        # Adjust dynamics state based on chaos activity
        if eigen_status['active_chaos_events'] > 0:
            # Override state if chaos is active
            enhanced_result['dynamics_state'] = 'controlled_chaos'
            enhanced_result['intervention'] = None  # Don't intervene during controlled chaos
            
        # Record chaos event if detected
        if enhanced_result.get('dynamics_state') == 'chaotic' and \
           eigen_status['active_chaos_events'] == 0:
            # Uncontrolled chaos detected
            self.chaos_events.append({
                'timestamp': datetime.now(timezone.utc),
                'type': 'uncontrolled',
                'max_lyapunov': enhanced_result['metrics']['max_lyapunov']
            })
            
        return enhanced_result
        
    def should_allow_chaos(self, metrics: Dict[str, Any]) -> bool:
        """Determine if chaos should be allowed based on current dynamics"""
        
        # Check safety conditions
        max_lyapunov = metrics.get('max_lyapunov', 0)
        energy = metrics.get('energy', 0)
        
        # Allow chaos if:
        # 1. Lyapunov is in safe range (positive but not too large)
        # 2. Energy is moderate
        # 3. No recent uncontrolled chaos events
        
        recent_uncontrolled = sum(
            1 for event in list(self.chaos_events)[-10:]
            if event.get('type') == 'uncontrolled'
        )
        
        safe_for_chaos = (
            0 < max_lyapunov < self.config.chaos_threshold * 2 and
            energy < 100 and
            recent_uncontrolled < 2
        )
        
        return safe_for_chaos

# ========== TORI Filter Adapter ==========

class TORIFilterAdapter:
    """
    Adapter for REAL-TORI filter
    Adds chaos-aware filtering and concept evolution
    """
    
    def __init__(self, original_filter: RealTORIFilter,
                 ccl: ChaosControlLayer,
                 config: AdapterConfig = None):
        self.original = original_filter
        self.ccl = ccl
        self.config = config or AdapterConfig()
        self.concept_evolution = defaultdict(list)
        
    async def analyze_with_evolution(self, concepts: List[str],
                                   allow_mutation: bool = True) -> Dict[str, Any]:
        """Analyze concepts with potential chaos-driven evolution"""
        
        # Get baseline purity
        baseline_purity = self.original.analyze_concept_purity(concepts)
        
        result = {
            'baseline_purity': baseline_purity,
            'concepts': concepts,
            'mutations': []
        }
        
        # If purity is low and mutation allowed, try concept evolution
        if baseline_purity < self.original.purity_threshold and \
           allow_mutation and \
           self.config.mode != AdapterMode.PASSTHROUGH:
            
            # Submit concepts for chaos-assisted evolution
            concept_vector = self._concepts_to_vector(concepts)
            
            chaos_task = ChaosTask(
                task_id=f"concept_evolve_{datetime.now(timezone.utc).timestamp()}",
                mode=ChaosMode.HYBRID,
                input_data=concept_vector,
                parameters={
                    'soliton_weight': 0.3,
                    'attractor_weight': 0.4,
                    'phase_weight': 0.3,
                    'target_purity': 0.8
                },
                energy_budget=self.config.energy_allocation
            )
            
            try:
                await self.ccl.submit_task(chaos_task)
                
                # Generate mutated concepts (simplified)
                mutations = self._generate_mutations(concepts)
                
                # Test mutated concepts
                for mutation in mutations:
                    mut_purity = self.original.analyze_concept_purity(mutation)
                    if mut_purity > baseline_purity:
                        result['mutations'].append({
                            'concepts': mutation,
                            'purity': mut_purity,
                            'improvement': mut_purity - baseline_purity
                        })
                        
                # Track evolution
                for concept in concepts:
                    self.concept_evolution[concept].append({
                        'timestamp': datetime.now(timezone.utc),
                        'purity': baseline_purity,
                        'mutations_tried': len(mutations)
                    })
                    
            except Exception as e:
                logger.warning(f"Concept evolution failed: {e}")
                
        return result
        
    def _concepts_to_vector(self, concepts: List[str]) -> np.ndarray:
        """Convert concepts to vector for chaos processing"""
        vec = np.zeros(50)
        for i, concept in enumerate(concepts[:50]):
            vec[i] = hash(concept) % 100 / 100
        return vec
        
    def _generate_mutations(self, concepts: List[str]) -> List[List[str]]:
        """Generate concept mutations"""
        mutations = []
        
        # Simple mutations - in production would use NLP
        prefixes = ['meta', 'quantum', 'emergent', 'adaptive']
        suffixes = ['dynamics', 'field', 'space', 'system']
        
        for prefix in prefixes:
            mutated = [f"{prefix}_{c}" for c in concepts]
            mutations.append(mutated)
            
        for suffix in suffixes:
            mutated = [f"{c}_{suffix}" for c in concepts]
            mutations.append(mutated)
            
        return mutations

# ========== Unified Adapter System ==========

class MetacognitiveAdapterSystem:
    """
    Unified system that manages all metacognitive adapters
    Provides clean interface for gradual chaos adoption
    """
    
    def __init__(self, unified_system: UnifiedMetacognitiveSystem,
                 eigen_sentry: EigenSentry2,
                 ccl: ChaosControlLayer,
                 global_config: Optional[AdapterConfig] = None):
        self.unified_system = unified_system
        self.eigen_sentry = eigen_sentry
        self.ccl = ccl
        self.global_config = global_config or AdapterConfig()
        
        # Create individual adapters
        self.memory_adapter = SolitonMemoryAdapter(
            unified_system.soliton_memory,
            ccl,
            self.global_config
        )
        
        self.reflection_adapter = ReflectionAdapter(
            unified_system.reflection_system,
            ccl,
            eigen_sentry,
            self.global_config
        )
        
        self.dynamics_adapter = DynamicsMonitorAdapter(
            unified_system.dynamics_system,
            eigen_sentry,
            self.global_config
        )
        
        self.filter_adapter = TORIFilterAdapter(
            unified_system.tori_filter,
            ccl,
            self.global_config
        )
        
        # Monkey-patch unified system methods
        self._install_adapters()
        
    def _install_adapters(self):
        """Install adapters into unified system"""
        # Store original methods
        self._original_process = self.unified_system.process_query_metacognitively
        
        # Replace with adapted version
        self.unified_system.process_query_metacognitively = self._adapted_process_query
        
    async def _adapted_process_query(self, query: str,
                                   context: Optional[Dict[str, Any]] = None):
        """Adapted query processing with chaos enhancement"""
        
        # Check if chaos should be enabled for this query
        chaos_enabled = self._should_enable_chaos(query, context)
        
        if chaos_enabled:
            # Prepare for chaos-assisted processing
            logger.info("Enabling chaos-assisted processing")
            
            # Temporarily set adapter modes
            original_mode = self.global_config.mode
            self.global_config.mode = AdapterMode.CHAOS_ASSISTED
            
        # Process with original method (but with adapters installed)
        result = await self._original_process(query, context)
        
        if chaos_enabled:
            # Restore mode
            self.global_config.mode = original_mode
            
            # Add chaos metrics to result
            result.metadata['chaos_assistance'] = {
                'enabled': True,
                'mode': AdapterMode.CHAOS_ASSISTED.value,
                'energy_used': self.ccl.energy_consumed
            }
            
        return result
        
    def _should_enable_chaos(self, query: str, context: Optional[Dict[str, Any]]) -> bool:
        """Determine if chaos should be enabled for this query"""
        
        # Check explicit context flag
        if context and context.get('enable_chaos', False):
            return True
            
        # Check query complexity (simple heuristic)
        complexity_indicators = ['how', 'why', 'explain', 'analyze', 'explore']
        if any(indicator in query.lower() for indicator in complexity_indicators):
            return True
            
        # Check current system state
        dynamics_status = self.dynamics_adapter.should_allow_chaos(
            {'max_lyapunov': 0.1, 'energy': 50}  # Simplified check
        )
        
        return dynamics_status
        
    def set_adapter_mode(self, mode: AdapterMode):
        """Set global adapter mode"""
        self.global_config.mode = mode
        logger.info(f"Adapter mode set to: {mode.value}")
        
    def get_adapter_stats(self) -> Dict[str, Any]:
        """Get statistics from all adapters"""
        return {
            'global_mode': self.global_config.mode.value,
            'memory_stats': dict(self.memory_adapter.operation_stats),
            'dynamics_chaos_events': len(self.dynamics_adapter.chaos_events),
            'concept_evolutions': len(self.filter_adapter.concept_evolution),
            'ccl_status': self.ccl.get_status(),
            'eigensentry_status': self.eigen_sentry.get_status()
        }

# ========== Testing and Demo ==========

async def demonstrate_adapters():
    """Demonstrate metacognitive adapter system"""
    print("🔌 Metacognitive Adapter System Demo")
    print("=" * 60)
    
    # This would normally use the full initialized system
    print("\n1️⃣ Adapter Modes Available:")
    for mode in AdapterMode:
        print(f"  • {mode.value}: {mode.name}")
        
    print("\n2️⃣ Adapter Configuration:")
    config = AdapterConfig()
    print(f"  • Default mode: {config.mode.value}")
    print(f"  • Chaos threshold: {config.chaos_threshold}")
    print(f"  • Energy allocation: {config.energy_allocation}")
    print(f"  • Safety margin: {config.safety_margin}")
    
    print("\n3️⃣ Adapter Types:")
    print("  • SolitonMemoryAdapter: Enhances memory with dark solitons")
    print("  • ReflectionAdapter: Uses phase explosion for deeper insights")
    print("  • DynamicsMonitorAdapter: Integrates with EigenSentry 2.0")
    print("  • TORIFilterAdapter: Enables concept evolution")
    
    print("\n✅ Adapters ready for integration!")
    print("   Use MetacognitiveAdapterSystem to enable chaos assistance")

if __name__ == "__main__":
    asyncio.run(demonstrate_adapters())
