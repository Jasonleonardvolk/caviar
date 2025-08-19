#!/usr/bin/env python3
"""
Unified Metacognitive Integration Module
Bridges MCP metacognitive components with TORI's reasoning system
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import json
import asyncio
from enum import Enum
from pathlib import Path

# Import our existing components
from python.core.reasoning_traversal import (
    ConceptMesh, ConceptNode, ReasoningPath,
    PrajnaResponsePlus, PrajnaReasoningIntegration
)
from python.core.temporal_reasoning_integration import (
    TemporalConceptMesh, TemporalReasoningAnalyzer
)
try:
    from python.core.intent_driven_reasoning import (
        ReasoningIntent, PathStrategy, IntentAwarePrajna,
        CognitiveResolutionEngine, SelfReflectiveReasoner
    )
except ImportError:
    # Provide minimal implementations if intent module is not available
    import logging
    logging.warning("Intent-driven reasoning module not available, using stubs")
    
    class ReasoningIntent:
        EXPLAIN = "explain"
        PREDICT = "predict"
        GENERATE = "generate"
        VALIDATE = "validate"
    
    class PathStrategy:
        SHORTEST = "shortest"
        MOST_CONFIDENT = "most_confident"
        MOST_DIVERSE = "most_diverse"
    
    class IntentAwarePrajna:
        def __init__(self, *args, **kwargs):
            pass
    
    class CognitiveResolutionEngine:
        def __init__(self, *args, **kwargs):
            pass
    
    class SelfReflectiveReasoner:
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)

# ========== Core Integration Classes ==========

class MetacognitiveState(Enum):
    """States of the metacognitive system"""
    STABLE = "stable"
    REFLECTING = "reflecting"
    CONFLICTED = "conflicted"
    EVOLVING = "evolving"
    CHAOTIC = "chaotic"
    CONVERGED = "converged"

@dataclass
class MetacognitiveContext:
    """Context for metacognitive processing"""
    current_state: np.ndarray
    reasoning_paths: List[ReasoningPath]
    memory_resonance: float = 0.0
    stability_score: float = 1.0
    reflection_depth: int = 0
    phase_alignment: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

# ========== REAL-TORI Filter Integration ==========

class RealTORIFilter:
    """Content filtering and concept purity analysis"""
    
    def __init__(self, purity_threshold: float = 0.7):
        self.purity_threshold = purity_threshold
        self.rogue_concepts = set()
        self.audit_log = []
    
    def analyze_concept_purity(self, concepts: List[str], 
                             context: Optional[Dict[str, Any]] = None) -> float:
        """Analyze purity of concepts for reasoning"""
        if not concepts:
            return 1.0
        
        # Calculate purity score
        total_score = 0.0
        for concept in concepts:
            # Check against known rogue concepts
            if concept.lower() in self.rogue_concepts:
                score = 0.0
            else:
                # Simple heuristic - could be enhanced
                score = 1.0
                
                # Penalize overly abstract or vague concepts
                if len(concept) < 3:
                    score *= 0.5
                
                # Bonus for specific technical concepts
                if any(term in concept.lower() for term in ['quantum', 'entropy', 'algorithm']):
                    score *= 1.2
            
            total_score += score
        
        purity = min(1.0, total_score / len(concepts))
        
        # Log audit
        self._log_audit("concept_purity", {
            "concepts": concepts,
            "purity": purity,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return purity
    
    def is_rogue_concept_contextual(self, concept: str, 
                                   context: Dict[str, Any]) -> bool:
        """Check if concept is rogue in given context"""
        # Context-aware rogue detection
        if concept.lower() in self.rogue_concepts:
            # Check if context allows it
            if context.get("allow_experimental", False):
                return False
            return True
        
        # Check for contextual inappropriateness
        intent = context.get("intent", "")
        if intent == "technical" and "emotion" in concept.lower():
            return True  # Emotional concepts in technical context
        
        return False
    
    def analyze_content_quality(self, content: str) -> Dict[str, float]:
        """Analyze quality metrics of content"""
        metrics = {
            "clarity": self._assess_clarity(content),
            "coherence": self._assess_coherence(content),
            "technical_depth": self._assess_technical_depth(content),
            "safety": self._assess_safety(content)
        }
        
        self._log_audit("content_quality", {
            "content_preview": content[:100],
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return metrics
    
    def _assess_clarity(self, content: str) -> float:
        """Simple clarity assessment"""
        # Penalize very short or very long content
        length = len(content.split())
        if length < 10:
            return 0.3
        elif length > 1000:
            return 0.7
        else:
            return 0.9
    
    def _assess_coherence(self, content: str) -> float:
        """Simple coherence assessment"""
        # Check for logical connectors
        connectors = ['therefore', 'because', 'thus', 'hence', 'implies']
        connector_count = sum(1 for conn in connectors if conn in content.lower())
        return min(1.0, 0.5 + connector_count * 0.1)
    
    def _assess_technical_depth(self, content: str) -> float:
        """Simple technical depth assessment"""
        technical_terms = ['algorithm', 'entropy', 'quantum', 'neural', 'optimization']
        term_count = sum(1 for term in technical_terms if term in content.lower())
        return min(1.0, 0.3 + term_count * 0.15)
    
    def _assess_safety(self, content: str) -> float:
        """Simple safety assessment"""
        unsafe_patterns = ['hack', 'exploit', 'illegal', 'harmful']
        if any(pattern in content.lower() for pattern in unsafe_patterns):
            return 0.2
        return 1.0
    
    def _log_audit(self, event_type: str, data: Dict[str, Any]):
        """Log audit event"""
        self.audit_log.append({
            "type": event_type,
            "data": data
        })

# ========== Cognitive State Manager Integration ==========

class CognitiveStateManager:
    """Manages cognitive state evolution and history"""
    
    def __init__(self, state_dim: int = 100):
        self.state_dim = state_dim
        self.current_state = np.zeros(state_dim)
        self.state_history = []
        self.event_log = []
        self.max_history = 1000
    
    def update_state(self, new_state: Optional[np.ndarray] = None,
                    delta: Optional[np.ndarray] = None,
                    metadata: Optional[Dict[str, Any]] = None):
        """Update cognitive state"""
        if new_state is not None:
            self.current_state = new_state
        elif delta is not None:
            self.current_state += delta
        
        # Normalize to prevent explosion
        norm = np.linalg.norm(self.current_state)
        if norm > 10.0:
            self.current_state = self.current_state / norm * 10.0
        
        # Record in history
        self.state_history.append({
            "state": self.current_state.copy(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        })
        
        # Trim history if needed
        if len(self.state_history) > self.max_history:
            self.state_history = self.state_history[-self.max_history:]
        
        self._log_event("state_update", metadata)
    
    def get_state(self) -> np.ndarray:
        """Get current state"""
        return self.current_state.copy()
    
    def get_trajectory(self, n: int = 10) -> List[np.ndarray]:
        """Get recent state trajectory"""
        recent = self.state_history[-n:] if len(self.state_history) >= n else self.state_history
        return [entry["state"] for entry in recent]
    
    def get_state_history(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get state history"""
        if n is None:
            return self.state_history
        return self.state_history[-n:]
    
    def compute_state_from_reasoning(self, reasoning_paths: List[ReasoningPath]) -> np.ndarray:
        """Convert reasoning paths to state vector"""
        state = np.zeros(self.state_dim)
        
        for i, path in enumerate(reasoning_paths[:10]):  # Use top 10 paths
            # Encode path properties into state
            base_idx = i * 10
            if base_idx + 10 <= self.state_dim:
                state[base_idx] = path.score
                state[base_idx + 1] = path.confidence
                state[base_idx + 2] = len(path.chain)
                state[base_idx + 3] = 1.0 if path.path_type == "causal" else 0.5
                
                # Encode some node information
                for j, node in enumerate(path.chain[:3]):
                    if base_idx + 4 + j < self.state_dim:
                        # Simple hash to float
                        state[base_idx + 4 + j] = hash(node.id) % 1000 / 1000.0
        
        return state
    
    def _log_event(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        """Log cognitive event"""
        self.event_log.append({
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {}
        })

# ========== Soliton Memory Integration ==========

class SolitonMemorySystem:
    """Infinite context memory with phase-based retrieval"""
    
    def __init__(self, lattice_size: int = 1000):
        self.lattice_size = lattice_size
        self.memory_lattice = {}
        self.phase_index = {}
        self.vault_status = {}
    
    def store_memory(self, content: str, concept_id: str, 
                    phase: float = 0.0, metadata: Optional[Dict[str, Any]] = None):
        """Store memory in soliton lattice"""
        memory_id = f"mem_{len(self.memory_lattice)}"
        
        # Create soliton wave packet
        memory_entry = {
            "id": memory_id,
            "content": content,
            "concept_id": concept_id,
            "phase": phase,
            "amplitude": 1.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "resonance_count": 0
        }
        
        self.memory_lattice[memory_id] = memory_entry
        
        # Index by phase
        phase_key = int(phase * 100) / 100  # Quantize to 0.01 rad
        if phase_key not in self.phase_index:
            self.phase_index[phase_key] = []
        self.phase_index[phase_key].append(memory_id)
        
        return memory_id
    
    def find_resonant_memories(self, target_phase: float, 
                              threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Find memories that resonate with target phase"""
        resonant = []
        
        for phase_key, memory_ids in self.phase_index.items():
            # Check phase alignment
            phase_diff = abs(phase_key - target_phase)
            if phase_diff < threshold or phase_diff > (2 * np.pi - threshold):
                for mem_id in memory_ids:
                    if mem_id in self.memory_lattice:
                        memory = self.memory_lattice[mem_id]
                        memory["resonance_strength"] = 1.0 - phase_diff / threshold
                        resonant.append(memory)
        
        # Sort by resonance strength
        resonant.sort(key=lambda m: m["resonance_strength"], reverse=True)
        
        # Update resonance counts
        for memory in resonant:
            self.memory_lattice[memory["id"]]["resonance_count"] += 1
        
        return resonant
    
    def detect_memory_dissonance(self, new_content: str, 
                               related_memories: List[Dict[str, Any]]) -> float:
        """Detect dissonance between new content and existing memories"""
        if not related_memories:
            return 0.0
        
        # Simple dissonance metric based on content similarity
        dissonance_scores = []
        
        for memory in related_memories:
            # Check for contradictions (simple heuristic)
            old_content = memory["content"].lower()
            new_content_lower = new_content.lower()
            
            # Look for opposing terms
            if ("not" in new_content_lower and "not" not in old_content) or \
               ("not" in old_content and "not" not in new_content_lower):
                dissonance_scores.append(0.8)
            elif any(neg in new_content_lower for neg in ["false", "incorrect", "wrong"]):
                if not any(neg in old_content for neg in ["false", "incorrect", "wrong"]):
                    dissonance_scores.append(0.6)
            else:
                dissonance_scores.append(0.1)
        
        return np.mean(dissonance_scores) if dissonance_scores else 0.0
    
    def vault_traumatic_memory(self, memory_id: str, vault_phase: float = np.pi/2):
        """Vault traumatic or problematic memory at phase shift"""
        if memory_id in self.memory_lattice:
            memory = self.memory_lattice[memory_id]
            
            # Shift phase to vault
            old_phase = memory["phase"]
            memory["phase"] = (old_phase + vault_phase) % (2 * np.pi)
            memory["vaulted"] = True
            memory["vault_timestamp"] = datetime.now(timezone.utc).isoformat()
            
            # Update phase index
            old_key = int(old_phase * 100) / 100
            new_key = int(memory["phase"] * 100) / 100
            
            if old_key in self.phase_index and memory_id in self.phase_index[old_key]:
                self.phase_index[old_key].remove(memory_id)
            
            if new_key not in self.phase_index:
                self.phase_index[new_key] = []
            self.phase_index[new_key].append(memory_id)
            
            self.vault_status[memory_id] = {
                "reason": "traumatic_content",
                "original_phase": old_phase,
                "vault_phase": memory["phase"]
            }

# ========== Reflection Tools Integration ==========

class ReflectionSystem:
    """Self-reflection and fixed-point finding"""
    
    def __init__(self, state_manager: CognitiveStateManager):
        self.state_manager = state_manager
        self.reflection_history = []
    
    def reflect(self, state: np.ndarray, steps: int = 1, 
               momentum: float = 0.9) -> Tuple[np.ndarray, float]:
        """Apply reflection operator to state"""
        reflected_state = state.copy()
        total_change = 0.0
        
        for step in range(steps):
            # Compute reflection gradient (simplified)
            gradient = self._compute_reflection_gradient(reflected_state)
            
            # Apply with momentum
            delta = momentum * gradient + (1 - momentum) * np.random.randn(*state.shape) * 0.01
            reflected_state += delta
            
            # Track change
            total_change += np.linalg.norm(delta)
        
        # Record reflection
        self.reflection_history.append({
            "original_state": state,
            "reflected_state": reflected_state,
            "total_change": total_change,
            "steps": steps,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return reflected_state, total_change
    
    def find_fixed_point(self, initial_state: np.ndarray, 
                        max_iterations: int = 10,
                        tolerance: float = 0.01) -> Tuple[np.ndarray, int, bool]:
        """Find fixed point through iterative reflection"""
        state = initial_state.copy()
        
        for iteration in range(max_iterations):
            new_state, change = self.reflect(state, steps=1)
            
            if change < tolerance:
                # Converged to fixed point
                return new_state, iteration + 1, True
            
            state = new_state
        
        # Did not converge
        return state, max_iterations, False
    
    def _compute_reflection_gradient(self, state: np.ndarray) -> np.ndarray:
        """Compute gradient for reflection (simplified)"""
        # Center-pulling force
        center_force = -state * 0.1
        
        # Smoothing force (reduce high-frequency components)
        if len(state) > 2:
            smooth_force = np.zeros_like(state)
            smooth_force[1:-1] = (state[:-2] + state[2:]) / 2 - state[1:-1]
            smooth_force *= 0.2
        else:
            smooth_force = np.zeros_like(state)
        
        # Random exploration
        explore_force = np.random.randn(*state.shape) * 0.05
        
        return center_force + smooth_force + explore_force

# ========== Dynamics and Stabilization Integration ==========

class CognitiveDynamicsSystem:
    """Monitor and stabilize cognitive dynamics"""
    
    def __init__(self, state_manager: CognitiveStateManager):
        self.state_manager = state_manager
        self.lyapunov_threshold = 0.5
    
    def compute_lyapunov_exponents(self, trajectory: List[np.ndarray], 
                                  k: int = 3) -> np.ndarray:
        """Compute Lyapunov exponents for trajectory"""
        if len(trajectory) < 2:
            return np.zeros(k)
        
        # Simplified Lyapunov calculation
        exponents = []
        
        for i in range(min(k, len(trajectory) - 1)):
            # Measure divergence rate
            diffs = []
            for j in range(len(trajectory) - 1):
                if j + i + 1 < len(trajectory):
                    diff = np.linalg.norm(trajectory[j + i + 1] - trajectory[j])
                    if diff > 0:
                        diffs.append(np.log(diff))
            
            if diffs:
                exponents.append(np.mean(diffs))
            else:
                exponents.append(0.0)
        
        return np.array(exponents[:k])
    
    def detect_chaos(self, window_size: int = 20) -> Tuple[bool, float]:
        """Detect chaotic behavior in recent trajectory"""
        trajectory = self.state_manager.get_trajectory(window_size)
        
        if len(trajectory) < 5:
            return False, 0.0
        
        # Compute Lyapunov exponents
        exponents = self.compute_lyapunov_exponents(trajectory)
        max_exponent = np.max(exponents) if len(exponents) > 0 else 0.0
        
        # Chaos detected if largest Lyapunov exponent is positive and large
        is_chaotic = max_exponent > self.lyapunov_threshold
        
        return is_chaotic, max_exponent
    
    def stabilize(self, state: np.ndarray, target_state: Optional[np.ndarray] = None,
                 strength: float = 0.3) -> np.ndarray:
        """Apply stabilization to state"""
        if target_state is None:
            # Use center of recent trajectory as target
            trajectory = self.state_manager.get_trajectory(10)
            if trajectory:
                target_state = np.mean(trajectory, axis=0)
            else:
                target_state = np.zeros_like(state)
        
        # Apply stabilizing force toward target
        stabilized = state + strength * (target_state - state)
        
        # Add small damping
        stabilized *= 0.95
        
        return stabilized
    
    def compute_energy(self, state: np.ndarray) -> float:
        """Compute energy/magnitude of state"""
        return np.linalg.norm(state)
    
    def analyze_stability(self, window_size: int = 50) -> Dict[str, Any]:
        """Comprehensive stability analysis"""
        trajectory = self.state_manager.get_trajectory(window_size)
        
        if len(trajectory) < 2:
            return {
                "stable": True,
                "chaotic": False,
                "energy_variance": 0.0,
                "max_lyapunov": 0.0,
                "recommendation": "Insufficient data"
            }
        
        # Check for chaos
        is_chaotic, max_lyapunov = self.detect_chaos(window_size)
        
        # Compute energy variance
        energies = [self.compute_energy(state) for state in trajectory]
        energy_variance = np.var(energies)
        
        # Determine stability
        stable = not is_chaotic and energy_variance < 1.0
        
        # Generate recommendation
        if is_chaotic:
            recommendation = "Apply strong stabilization"
        elif energy_variance > 0.5:
            recommendation = "Apply mild stabilization"
        else:
            recommendation = "System is stable"
        
        return {
            "stable": stable,
            "chaotic": is_chaotic,
            "energy_variance": float(energy_variance),
            "max_lyapunov": float(max_lyapunov),
            "trajectory_length": len(trajectory),
            "recommendation": recommendation
        }

# ========== Unified Metacognitive Orchestrator ==========

class UnifiedMetacognitiveSystem:
    """Orchestrates all metacognitive components"""
    
    def __init__(self, mesh: TemporalConceptMesh, 
                 enable_all_systems: bool = True):
        self.mesh = mesh
        
        # Initialize all subsystems
        self.tori_filter = RealTORIFilter()
        self.state_manager = CognitiveStateManager()
        self.soliton_memory = SolitonMemorySystem()
        self.reflection_system = ReflectionSystem(self.state_manager)
        self.dynamics_system = CognitiveDynamicsSystem(self.state_manager)
        
        # Existing reasoning components
        self.intent_prajna = IntentAwarePrajna(mesh)
        
        # Configuration
        self.enable_filtering = enable_all_systems
        self.enable_memory = enable_all_systems
        self.enable_reflection = enable_all_systems
        self.enable_dynamics = enable_all_systems
        
        # Metacognitive state
        self.meta_state = MetacognitiveState.STABLE
        self.processing_history = []
    
    async def process_query_metacognitively(self, query: str,
                                          context: Optional[Dict[str, Any]] = None) -> PrajnaResponsePlus:
        """Process query through full metacognitive pipeline"""
        
        start_time = datetime.now(timezone.utc)
        processing_context = MetacognitiveContext(
            current_state=self.state_manager.get_state(),
            reasoning_paths=[],
            metadata={"query": query, "start_time": start_time}
        )
        
        try:
            # Phase 1: Filtering and Concept Analysis
            if self.enable_filtering:
                concepts = self.intent_prajna._extract_anchor_concepts(query)
                purity = self.tori_filter.analyze_concept_purity(concepts, context)
                
                if purity < self.tori_filter.purity_threshold:
                    # Concepts need cleaning
                    logger.warning(f"Low concept purity: {purity}")
                    # Could implement concept cleaning here
                
                processing_context.metadata["concept_purity"] = purity
            
            # Phase 2: Initial Reasoning
            self.meta_state = MetacognitiveState.EVOLVING
            initial_response = self.intent_prajna.generate_intent_aware_response(
                query, context, concepts if self.enable_filtering else None
            )
            
            processing_context.reasoning_paths = initial_response.reasoning_paths
            
            # Convert reasoning to state
            reasoning_state = self.state_manager.compute_state_from_reasoning(
                initial_response.reasoning_paths
            )
            self.state_manager.update_state(new_state=reasoning_state, 
                                          metadata={"phase": "initial_reasoning"})
            
            # Phase 3: Memory Resonance Check
            if self.enable_memory and initial_response.reasoning_paths:
                # Use primary concept phase for memory lookup
                primary_phase = hash(query) % (2 * np.pi)
                resonant_memories = self.soliton_memory.find_resonant_memories(
                    primary_phase, threshold=0.3
                )
                
                if resonant_memories:
                    # Check for dissonance
                    dissonance = self.soliton_memory.detect_memory_dissonance(
                        initial_response.text, resonant_memories[:5]
                    )
                    processing_context.memory_resonance = 1.0 - dissonance
                    
                    if dissonance > 0.5:
                        logger.warning(f"High memory dissonance detected: {dissonance}")
                        self.meta_state = MetacognitiveState.CONFLICTED
                
                # Store new insight
                self.soliton_memory.store_memory(
                    initial_response.text,
                    concepts[0] if concepts else "general",
                    primary_phase,
                    {"query": query, "confidence": initial_response.confidence}
                )
            
            # Phase 4: Stability Check
            if self.enable_dynamics:
                stability_analysis = self.dynamics_system.analyze_stability()
                
                if stability_analysis["chaotic"]:
                    self.meta_state = MetacognitiveState.CHAOTIC
                    logger.warning("Chaotic dynamics detected")
                    
                    # Apply stabilization
                    stabilized_state = self.dynamics_system.stabilize(
                        self.state_manager.get_state()
                    )
                    self.state_manager.update_state(
                        new_state=stabilized_state,
                        metadata={"phase": "stabilization"}
                    )
                
                processing_context.stability_score = \
                    1.0 - min(1.0, stability_analysis["max_lyapunov"])
            
            # Phase 5: Reflective Processing
            if self.enable_reflection and \
               (self.meta_state == MetacognitiveState.CONFLICTED or \
                context and context.get("deep_reflection", False)):
                
                self.meta_state = MetacognitiveState.REFLECTING
                
                # Find fixed point through reflection
                current_state = self.state_manager.get_state()
                fixed_state, iterations, converged = \
                    self.reflection_system.find_fixed_point(
                        current_state, max_iterations=5
                    )
                
                if converged:
                    self.state_manager.update_state(
                        new_state=fixed_state,
                        metadata={"phase": "reflection_converged", 
                                "iterations": iterations}
                    )
                    self.meta_state = MetacognitiveState.CONVERGED
                    processing_context.reflection_depth = iterations
                else:
                    logger.warning("Reflection did not converge")
            
            # Phase 6: Final Response Generation
            final_response = self._generate_final_response(
                initial_response, processing_context
            )
            
            # Record processing
            self.processing_history.append({
                "query": query,
                "timestamp": start_time.isoformat(),
                "duration": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "meta_state": self.meta_state.value,
                "memory_resonance": processing_context.memory_resonance,
                "stability_score": processing_context.stability_score,
                "reflection_depth": processing_context.reflection_depth
            })
            
            return final_response
            
        except Exception as e:
            logger.error(f"Metacognitive processing failed: {e}")
            self.meta_state = MetacognitiveState.STABLE
            raise
    
    def _generate_final_response(self, initial_response: PrajnaResponsePlus,
                               context: MetacognitiveContext) -> PrajnaResponsePlus:
        """Generate final response with metacognitive enhancements"""
        
        # Enhanced response text
        enhanced_text = initial_response.text
        
        # Add metacognitive insights if significant
        insights = []
        
        if context.memory_resonance < 0.5:
            insights.append("Note: This conclusion differs from previous understanding.")
        
        if context.stability_score < 0.7:
            insights.append("Caution: This reasoning involved complex dynamics.")
        
        if context.reflection_depth > 3:
            insights.append(f"This required {context.reflection_depth} rounds of reflection to stabilize.")
        
        if insights:
            enhanced_text += "\n\n" + " ".join(insights)
        
        # Update metadata
        initial_response.metadata.update({
            "metacognitive_state": self.meta_state.value,
            "memory_resonance": context.memory_resonance,
            "stability_score": context.stability_score,
            "reflection_depth": context.reflection_depth,
            "concept_purity": context.metadata.get("concept_purity", 1.0)
        })
        
        return initial_response
    
    def get_metacognitive_report(self) -> Dict[str, Any]:
        """Generate comprehensive metacognitive report"""
        
        # Recent processing stats
        recent_processing = self.processing_history[-10:] if self.processing_history else []
        
        if recent_processing:
            avg_duration = np.mean([p["duration"] for p in recent_processing])
            avg_resonance = np.mean([p["memory_resonance"] for p in recent_processing])
            avg_stability = np.mean([p["stability_score"] for p in recent_processing])
        else:
            avg_duration = avg_resonance = avg_stability = 0.0
        
        # Current stability
        stability = self.dynamics_system.analyze_stability()
        
        # Memory stats
        memory_count = len(self.soliton_memory.memory_lattice)
        vaulted_count = len(self.soliton_memory.vault_status)
        
        return {
            "current_state": self.meta_state.value,
            "processing_stats": {
                "total_queries": len(self.processing_history),
                "average_duration": avg_duration,
                "average_resonance": avg_resonance,
                "average_stability": avg_stability
            },
            "stability_analysis": stability,
            "memory_stats": {
                "total_memories": memory_count,
                "vaulted_memories": vaulted_count,
                "phase_distribution": len(self.soliton_memory.phase_index)
            },
            "filter_stats": {
                "rogue_concepts": len(self.tori_filter.rogue_concepts),
                "audit_events": len(self.tori_filter.audit_log)
            },
            "recommendations": self._generate_recommendations(stability, avg_resonance)
        }
    
    def _generate_recommendations(self, stability: Dict[str, Any], 
                                avg_resonance: float) -> List[str]:
        """Generate system recommendations"""
        recommendations = []
        
        if stability["chaotic"]:
            recommendations.append("System showing chaotic behavior - increase stabilization")
        
        if avg_resonance < 0.3:
            recommendations.append("Low memory resonance - consider memory consolidation")
        
        if len(self.soliton_memory.vault_status) > 10:
            recommendations.append("Many vaulted memories - review and process trauma")
        
        if not recommendations:
            recommendations.append("System operating within normal parameters")
        
        return recommendations

# ========== Testing and Demo ==========

async def demonstrate_unified_metacognition():
    """Demonstrate the unified metacognitive system"""
    print("🧠 Unified Metacognitive System Demo")
    print("=" * 60)
    
    # Create test mesh
    mesh = TemporalConceptMesh()
    
    # Add test concepts
    consciousness = ConceptNode("consciousness", "Consciousness",
                              "Subjective experience and awareness",
                              ["neuroscience_2024", "philosophy_2024"])
    qualia = ConceptNode("qualia", "Qualia", 
                        "Subjective conscious experiences",
                        ["philosophy_2023"])
    emergence = ConceptNode("emergence", "Emergence",
                          "Complex properties arising from simple rules",
                          ["complexity_2024"])
    
    for node in [consciousness, qualia, emergence]:
        mesh.add_node(node)
    
    mesh.add_temporal_edge("consciousness", "qualia", EdgeType.IMPLIES,
                          justification="consciousness gives rise to qualia")
    mesh.add_temporal_edge("emergence", "consciousness", EdgeType.ENABLES,
                          justification="emergence enables consciousness")
    
    # Initialize metacognitive system
    meta_system = UnifiedMetacognitiveSystem(mesh)
    
    # Test queries
    test_queries = [
        {
            "query": "How does consciousness emerge from physical processes?",
            "context": {"deep_reflection": True}
        },
        {
            "query": "What is the relationship between qualia and emergence?",
            "context": {"intent": "causal"}
        },
        {
            "query": "Can consciousness exist without qualia?",
            "context": {"allow_experimental": True}
        }
    ]
    
    for i, test in enumerate(test_queries):
        print(f"\n{'='*60}")
        print(f"Query {i+1}: {test['query']}")
        print("-" * 60)
        
        # Process metacognitively
        response = await meta_system.process_query_metacognitively(
            test["query"], test["context"]
        )
        
        print(f"\nResponse: {response.text}")
        print(f"\nMetacognitive State: {meta_system.meta_state.value}")
        print(f"Memory Resonance: {response.metadata.get('memory_resonance', 0):.2f}")
        print(f"Stability Score: {response.metadata.get('stability_score', 0):.2f}")
        print(f"Reflection Depth: {response.metadata.get('reflection_depth', 0)}")
        
        # Small delay between queries
        await asyncio.sleep(0.1)
    
    # Show final report
    print(f"\n{'='*60}")
    print("📊 Metacognitive System Report")
    print("-" * 60)
    
    report = meta_system.get_metacognitive_report()
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    # Run the demo
    asyncio.run(demonstrate_unified_metacognition())
