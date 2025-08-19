"""phase_reasoning.py - Implements phase-coherent reasoning for ALAN.

This module provides the core implementation of ALAN's phase-coherent reasoning
engine, which uses phase synchronization dynamics as the foundation for logical
inference. Rather than operating on static truth values, this reasoning system
evaluates logical relationships based on how concepts synchronize their phases
in a dynamical system.

Key features:
1. Premise synchronization: Creates a phase-locked cluster of premise concepts
2. Inference testing: Determines if conclusions can join premise clusters without disruption
3. Coherence metrics: Evaluates validity based on phase stability and synchronization
4. Modal reasoning: Supports necessary vs. possible truths through attractor analysis

This approach enables a "post-probabilistic" reasoning framework where validity
emerges naturally from dynamical properties rather than being assigned through
rules or statistical models.
"""

import os
import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import math
import random
from collections import defaultdict
import uuid

try:
    # Try absolute import first
    from models import ConceptTuple
except ImportError:
    # Fallback to relative import
    from .models import ConceptTuple
try:
    # Try absolute import first
    from koopman_phase_graph import get_koopman_phase_graph, ConceptNode
except ImportError:
    # Fallback to relative import
    from .koopman_phase_graph import get_koopman_phase_graph, ConceptNode
try:
    # Try absolute import first
    from memory_sculptor import get_memory_sculptor, ConceptState
except ImportError:
    # Fallback to relative import
    from .memory_sculptor import get_memory_sculptor, ConceptState
try:
    # Try absolute import first
    from ontology_refactor_engine import get_ontology_refactor_engine
except ImportError:
    # Fallback to relative import
    from .ontology_refactor_engine import get_ontology_refactor_engine
try:
    # Try absolute import first
    from ghost_label_synthesizer import get_ghost_label_synthesizer
except ImportError:
    # Fallback to relative import
    from .ghost_label_synthesizer import get_ghost_label_synthesizer

# Configure logger
logger = logging.getLogger("alan_phase_reasoning")

# Data structures for phase-coherent reasoning
@dataclass
class PremiseNetwork:
    """Represents a network of phase-coupled premise concepts."""
    premise_ids: List[str]                   # IDs of concepts serving as premises
    phase_states: Dict[str, float]           # Current phase of each oscillator
    natural_frequencies: Dict[str, float]    # Intrinsic frequency of each oscillator
    coupling_matrix: np.ndarray              # Coupling strengths between oscillators
    coherence_history: List[float] = field(default_factory=list)  # History of coherence during stabilization
    is_stable: bool = False                  # Whether network has reached stability
    stabilization_time: float = 0.0          # Time taken to reach stability
    koopman_signature: Optional[np.ndarray] = None  # Koopman modes of stable network


@dataclass
class InferenceResult:
    """Represents the result of testing a conclusion against premises."""
    premise_ids: List[str]              # Premise concepts
    conclusion_id: str                  # Tested conclusion
    is_coherent: bool                   # Whether conclusion coherently joins premises
    coherence_score: float              # Quantitative coherence measure
    phase_disruption: float             # How much conclusion disrupts premise coherence
    stabilization_time: float           # Time to reach stability with conclusion
    probability_analog: float           # Value between 0-1 analogous to probability
    convergence_trajectory: List[float] = field(default_factory=list)  # Coherence values during simulation
    modal_status: str = "unknown"       # "necessary", "possible", "impossible"


@dataclass
class PossibleWorld:
    """Represents a possible world as a stable phase configuration."""
    core_concepts: List[str]            # Concepts defining this world
    phase_configuration: Dict[str, float]  # Stable phase values
    coherence_value: float              # Internal coherence
    basin_size: float                   # Relative size of basin of attraction
    compatible_concepts: List[str] = field(default_factory=list)  # Concepts that can join without disruption
    incompatible_concepts: List[str] = field(default_factory=list)  # Concepts that disrupt coherence


@dataclass
class PhaseValidity:
    """Represents the validity of an inference based on phase coherence."""
    logical_validity: float             # Traditional logical validity analog
    coherence_validity: float           # Based on phase coherence
    resilience_validity: float          # Based on stability against perturbations
    contextual_validity: float          # Validity within specific context
    global_validity: float              # Validity across all contexts
    conflicts: List[Tuple[str, str]] = field(default_factory=list)  # Pairs of concepts with phase conflicts
    supporting_concepts: List[str] = field(default_factory=list)  # Concepts enhancing conclusion validity
    aggregate_score: float = 0.0        # Overall validity assessment


@dataclass
class ReasoningResult:
    """Represents the result of a reasoning process."""
    premises: List[str]                # Premise concepts
    candidates: List[str]              # Candidate conclusions
    inferences: Dict[str, InferenceResult] = field(default_factory=dict)  # Results for each conclusion
    coherent_concepts: List[str] = field(default_factory=list)  # Concepts that joined coherently
    incoherent_concepts: List[str] = field(default_factory=list)  # Concepts that disrupted coherence
    reasoning_time: float = 0.0        # Time taken for reasoning process
    context_depth: int = 0             # Depth of context considered


@dataclass
class ModalResult:
    """Represents the modal status of a concept."""
    concept_id: str                    # Concept ID
    status: str                        # "necessary", "possible", "impossible", "contingent"
    necessity_degree: float            # Degree of necessity (0-1)
    possibility_degree: float          # Degree of possibility (0-1)
    compatible_contexts: List[str] = field(default_factory=list)  # Contexts where concept is compatible
    incompatible_contexts: List[str] = field(default_factory=list)  # Contexts where concept is incompatible


class PremiseSynchronizer:
    """
    Establishes a stable phase-locked cluster representing premises of an inference.
    """
    
    def __init__(
        self,
        simulation_steps: int = 1000,
        dt: float = 0.1,
        stability_threshold: float = 0.95,
        frequency_range: Tuple[float, float] = (0.8, 1.2)
    ):
        """
        Initialize the premise synchronizer.
        
        Args:
            simulation_steps: Maximum steps for synchronization simulation
            dt: Time step for simulation
            stability_threshold: Coherence threshold for stability
            frequency_range: Range of natural frequencies for oscillators
        """
        self.simulation_steps = simulation_steps
        self.dt = dt
        self.stability_threshold = stability_threshold
        self.frequency_range = frequency_range
        
        # Get Koopman graph
        self.koopman_graph = get_koopman_phase_graph()
        
    def create_premise_network(self, premise_ids: List[str]) -> PremiseNetwork:
        """
        Create a phase-coupled network from premise concepts.
        
        Args:
            premise_ids: List of concept IDs to use as premises
            
        Returns:
            PremiseNetwork object
        """
        # Check that all premises exist
        valid_premise_ids = []
        for premise_id in premise_ids:
            concept = self.koopman_graph.get_concept_by_id(premise_id)
            if concept is not None:
                valid_premise_ids.append(premise_id)
            else:
                logger.warning(f"Premise concept {premise_id} not found in Koopman graph")
                
        if not valid_premise_ids:
            raise ValueError("No valid premise concepts found")
            
        # Initialize phase states randomly
        phase_states = {
            premise_id: 2 * math.pi * random.random()
            for premise_id in valid_premise_ids
        }
        
        # Assign natural frequencies
        natural_frequencies = {
            premise_id: self.frequency_range[0] + random.random() * (self.frequency_range[1] - self.frequency_range[0])
            for premise_id in valid_premise_ids
        }
        
        # Create coupling matrix based on concept similarities
        n = len(valid_premise_ids)
        coupling_matrix = np.zeros((n, n))
        
        for i, id1 in enumerate(valid_premise_ids):
            concept1 = self.koopman_graph.get_concept_by_id(id1)
            
            for j, id2 in enumerate(valid_premise_ids):
                if i != j:
                    concept2 = self.koopman_graph.get_concept_by_id(id2)
                    
                    # Calculate similarity-based coupling
                    similarity = np.dot(concept1.embedding, concept2.embedding) / (
                        np.linalg.norm(concept1.embedding) * np.linalg.norm(concept2.embedding)
                    )
                    
                    # Use similarity as coupling strength
                    coupling_matrix[i, j] = max(0.1, similarity)
                    
                    # Check if there's an explicit edge
                    for target_id, weight in concept1.edges:
                        if target_id == id2:
                            # Use max of similarity and explicit edge weight
                            coupling_matrix[i, j] = max(coupling_matrix[i, j], weight)
                            break
        
        # Create premise network
        premise_network = PremiseNetwork(
            premise_ids=valid_premise_ids,
            phase_states=phase_states,
            natural_frequencies=natural_frequencies,
            coupling_matrix=coupling_matrix
        )
        
        return premise_network
        
    def stabilize_premises(self, network: PremiseNetwork) -> PremiseNetwork:
        """
        Run phase coupling simulation until the premise network stabilizes.
        
        Args:
            network: Premise network to stabilize
            
        Returns:
            Stabilized PremiseNetwork
        """
        start_time = time.time()
        
        premise_ids = network.premise_ids
        phase_states = network.phase_states
        natural_frequencies = network.natural_frequencies
        coupling_matrix = network.coupling_matrix
        coherence_history = []
        
        # Run phase coupling simulation
        for step in range(self.simulation_steps):
            # Calculate current coherence
            coherence = self._calculate_coherence(phase_states)
            coherence_history.append(coherence)
            
            # Check if stable
            if step > 20 and coherence > self.stability_threshold:
                # Calculate stability over last 20 steps
                recent_coherence = coherence_history[-20:]
                coherence_std = np.std(recent_coherence)
                
                if coherence_std < 0.01:
                    # Stable state reached
                    network.is_stable = True
                    network.stabilization_time = time.time() - start_time
                    network.coherence_history = coherence_history
                    
                    # Calculate Koopman signature (simple approximation)
                    network.koopman_signature = self._approximate_koopman_signature(network)
                    
                    logger.info(f"Premise network stabilized after {step} steps with coherence {coherence:.3f}")
                    return network
            
            # Update phases
            new_phase_states = {}
            
            for i, id1 in enumerate(premise_ids):
                # Get current phase and frequency
                phase = phase_states[id1]
                freq = natural_frequencies[id1]
                
                # Calculate coupling effects
                coupling_sum = 0.0
                for j, id2 in enumerate(premise_ids):
                    if i != j:
                        coupling_sum += coupling_matrix[i, j] * math.sin(phase_states[id2] - phase)
                
                # Update phase
                new_phase = phase + self.dt * (freq + coupling_sum)
                # Normalize to [0, 2π]
                new_phase = new_phase % (2 * math.pi)
                new_phase_states[id1] = new_phase
                
            # Update network phases
            phase_states = new_phase_states
            network.phase_states = phase_states
            
        # Max steps reached without stability
        network.is_stable = False
        network.stabilization_time = time.time() - start_time
        network.coherence_history = coherence_history
        
        logger.warning(f"Premise network did not stabilize after {self.simulation_steps} steps")
        return network
    
    def _calculate_coherence(self, phase_states: Dict[str, float]) -> float:
        """
        Calculate Kuramoto order parameter as coherence measure.
        
        Args:
            phase_states: Dictionary mapping concept IDs to phase values
            
        Returns:
            Coherence value between 0 and 1
        """
        if not phase_states:
            return 0.0
            
        # Calculate complex order parameter
        sum_real = 0.0
        sum_imag = 0.0
        
        for phase in phase_states.values():
            sum_real += math.cos(phase)
            sum_imag += math.sin(phase)
            
        # Normalize by number of oscillators
        n = len(phase_states)
        sum_real /= n
        sum_imag /= n
        
        # Calculate magnitude (r)
        r = math.sqrt(sum_real ** 2 + sum_imag ** 2)
        
        return r
    
    def _approximate_koopman_signature(self, network: PremiseNetwork) -> np.ndarray:
        """
        Create a simple approximation of Koopman modes from phase configuration.
        
        This is a simplified representation - a full implementation would involve
        proper Koopman decomposition using more sophisticated techniques.
        
        Args:
            network: Stabilized premise network
            
        Returns:
            Approximate Koopman signature
        """
        # For this simplified implementation, we'll use the phases directly
        phase_values = np.array(list(network.phase_states.values()))
        
        # Create a simple Koopman-inspired signature
        # In a full implementation, this would involve proper DMD or EDMD
        signature = np.zeros(len(phase_values) * 2)
        
        # Add both sine and cosine components
        signature[:len(phase_values)] = np.cos(phase_values)
        signature[len(phase_values):] = np.sin(phase_values)
        
        return signature


class InferenceTester:
    """
    Tests whether a candidate conclusion can join a premise network without
    disrupting phase coherence.
    """
    
    def __init__(
        self,
        simulation_steps: int = 500,
        dt: float = 0.1,
        coherence_threshold: float = 0.8,
        perturbation_strength: float = 0.1
    ):
        """
        Initialize the inference tester.
        
        Args:
            simulation_steps: Maximum steps for inference simulation
            dt: Time step for simulation
            coherence_threshold: Threshold for coherent inference
            perturbation_strength: Strength of perturbations for stability testing
        """
        self.simulation_steps = simulation_steps
        self.dt = dt
        self.coherence_threshold = coherence_threshold
        self.perturbation_strength = perturbation_strength
        
        # Get Koopman graph
        self.koopman_graph = get_koopman_phase_graph()
        
    def test_conclusion(
        self, 
        premise_network: PremiseNetwork, 
        conclusion_id: str
    ) -> InferenceResult:
        """
        Test whether a conclusion concept can join a premise network coherently.
        
        Args:
            premise_network: Stabilized premise network
            conclusion_id: ID of concept to test as conclusion
            
        Returns:
            InferenceResult object
        """
        # Check if premise network is stable
        if not premise_network.is_stable:
            raise ValueError("Premise network must be stabilized before testing conclusions")
            
        # Get conclusion concept
        conclusion_concept = self.koopman_graph.get_concept_by_id(conclusion_id)
        if conclusion_concept is None:
            raise ValueError(f"Conclusion concept {conclusion_id} not found in Koopman graph")
            
        # Create extended network including conclusion
        extended_network = self._extend_network(premise_network, conclusion_id)
        
        # Simulate coupling dynamics with conclusion
        start_time = time.time()
        convergence_trajectory = []
        final_coherence = 0.0
        disruption_measure = 0.0
        
        # Get original coherence
        original_coherence = self._calculate_coherence(premise_network.phase_states)
        
        # Add conclusion and simulate
        for step in range(self.simulation_steps):
            # Calculate current coherence
            coherence = self._calculate_coherence(extended_network["phase_states"])
            convergence_trajectory.append(coherence)
            
            # Update based on dynamics
            extended_network = self._update_network(extended_network, self.dt)
            
            # Check if stable with good coherence
            if step > 20 and coherence > self.coherence_threshold:
                # Calculate stability over last 20 steps
                recent_coherence = convergence_trajectory[-20:]
                coherence_std = np.std(recent_coherence)
                
                if coherence_std < 0.01:
                    # Stable coherent state reached
                    final_coherence = coherence
                    
                    # Calculate how much conclusion disrupted original coherence
                    # (in a negative sense - positive means improved coherence)
                    disruption_measure = original_coherence - coherence
                    
                    break
        
        # If we reached the end without stability, use final state
        if final_coherence == 0.0:
            final_coherence = coherence
            disruption_measure = original_coherence - coherence
        
        # Calculate probability analog
        probability_analog = max(0.0, min(1.0, 
            (final_coherence - self.coherence_threshold) / 
            (1.0 - self.coherence_threshold)
        ))
        
        # Create result
        result = InferenceResult(
            premise_ids=premise_network.premise_ids,
            conclusion_id=conclusion_id,
            is_coherent=final_coherence >= self.coherence_threshold,
            coherence_score=final_coherence,
            phase_disruption=disruption_measure,
            stabilization_time=time.time() - start_time,
            probability_analog=probability_analog,
            convergence_trajectory=convergence_trajectory
        )
        
        logger.info(f"Conclusion {conclusion_id} tested with coherence {final_coherence:.3f}")
        return result
    
    def rank_potential_conclusions(
        self, 
        premise_network: PremiseNetwork, 
        candidate_ids: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Rank multiple potential conclusions by coherence.
        
        Args:
            premise_network: Stabilized premise network
            candidate_ids: List of concept IDs to test as conclusions
            
        Returns:
            List of (concept_id, coherence_score) tuples, sorted by score
        """
        results = []
        
        for candidate_id in candidate_ids:
            try:
                result = self.test_conclusion(premise_network, candidate_id)
                results.append((candidate_id, result.coherence_score))
            except ValueError as e:
                logger.warning(f"Skipping candidate {candidate_id}: {e}")
                
        # Sort by coherence (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _extend_network(
        self, 
        premise_network: PremiseNetwork, 
        conclusion_id: str
    ) -> Dict[str, Any]:
        """
        Extend a premise network to include a conclusion concept.
        
        Args:
            premise_network: Original premise network
            conclusion_id: Concept ID to add
            
        Returns:
            Extended network as dictionary
        """
        premise_ids = premise_network.premise_ids
        conclusion_concept = self.koopman_graph.get_concept_by_id(conclusion_id)
        
        # Create extended lists
        extended_ids = premise_ids + [conclusion_id]
        
        # Copy phase states and add conclusion
        extended_phases = premise_network.phase_states.copy()
        extended_phases[conclusion_id] = 2 * math.pi * random.random()
        
        # Copy natural frequencies and add conclusion
        extended_frequencies = premise_network.natural_frequencies.copy()
        extended_frequencies[conclusion_id] = 0.9 + 0.2 * random.random()
        
        # Create extended coupling matrix
        old_n = len(premise_ids)
        new_n = old_n + 1
        extended_coupling = np.zeros((new_n, new_n))
        
        # Copy original couplings
        extended_coupling[:old_n, :old_n] = premise_network.coupling_matrix
        
        # Add couplings to/from conclusion
        for i, premise_id in enumerate(premise_ids):
            premise_concept = self.koopman_graph.get_concept_by_id(premise_id)
            
            # Calculate similarity-based coupling
            similarity = np.dot(conclusion_concept.embedding, premise_concept.embedding) / (
                np.linalg.norm(conclusion_concept.embedding) * np.linalg.norm(premise_concept.embedding)
            )
            
            # Set coupling in both directions
            extended_coupling[i, old_n] = max(0.1, similarity)
            extended_coupling[old_n, i] = max(0.1, similarity)
            
            # Check for explicit edges
            for target_id, weight in premise_concept.edges:
                if target_id == conclusion_id:
                    extended_coupling[i, old_n] = max(extended_coupling[i, old_n], weight)
                    
            for target_id, weight in conclusion_concept.edges:
                if target_id == premise_id:
                    extended_coupling[old_n, i] = max(extended_coupling[old_n, i], weight)
                    
        return {
            "ids": extended_ids,
            "phase_states": extended_phases,
            "natural_frequencies": extended_frequencies,
            "coupling_matrix": extended_coupling
        }
    
    def _update_network(
        self, 
        network: Dict[str, Any], 
        dt: float
    ) -> Dict[str, Any]:
        """
        Update phase states based on coupled oscillator dynamics.
        
        Args:
            network: Network dictionary
            dt: Time step
            
        Returns:
            Updated network
        """
        ids = network["ids"]
        phase_states = network["phase_states"]
        natural_frequencies = network["natural_frequencies"]
        coupling_matrix = network["coupling_matrix"]
        
        # Update phases
        new_phase_states = {}
        
        for i, id1 in enumerate(ids):
            # Get current phase and frequency
            phase = phase_states[id1]
            freq = natural_frequencies[id1]
            
            # Calculate coupling effects
            coupling_sum = 0.0
            for j, id2 in enumerate(ids):
                if i != j:
                    coupling_sum += coupling_matrix[i, j] * math.sin(phase_states[id2] - phase)
            
            # Update phase
            new_phase = phase + dt * (freq + coupling_sum)
            # Normalize to [0, 2π]
            new_phase = new_phase % (2 * math.pi)
            new_phase_states[id1] = new_phase
            
        # Update network phases
        network["phase_states"] = new_phase_states
        return network
    
    def _calculate_coherence(self, phase_states: Dict[str, float]) -> float:
        """
        Calculate Kuramoto order parameter as coherence measure.
        
        Args:
            phase_states: Dictionary mapping concept IDs to phase values
            
        Returns:
            Coherence value between 0 and 1
        """
        if not phase_states:
            return 0.0
            
        # Calculate complex order parameter
        sum_real = 0.0
        sum_imag = 0.0
        
        for phase in phase_states.values():
            sum_real += math.cos(phase)
            sum_imag += math.sin(phase)
            
        # Normalize by number of oscillators
        n = len(phase_states)
        sum_real /= n
        sum_imag /= n
        
        # Calculate magnitude (r)
        r = math.sqrt(sum_real ** 2 + sum_imag ** 2)
        
        return r


class ModalReasoner:
    """
    Extends reasoning to modal concepts (necessity, possibility, contingency).
    """
    
    def __init__(
        self,
        world_count: int = 5,
        simulation_steps: int = 500,
        dt: float = 0.1,
        coherence_threshold: float = 0.8
    ):
        """
        Initialize the modal reasoner.
        
        Args:
            world_count: Number of possible worlds to generate
            simulation_steps: Maximum steps for simulation
            dt: Time step for simulation
            coherence_threshold: Threshold for coherent states
        """
        self.world_count = world_count
        self.simulation_steps = simulation_steps
        self.dt = dt
        self.coherence_threshold = coherence_threshold
        
        # Get Koopman graph
        self.koopman_graph = get_koopman_phase_graph()
        
        # Other needed components
        self.premise_sync = PremiseSynchronizer()
        self.inference_test = InferenceTester()
        
    def identify_modal_status(
        self, 
        concept_id: str, 
        context_ids: List[str]
    ) -> ModalResult:
        """
        Determine the modal status of a concept within a context.
        
        Args:
            concept_id: Concept ID to check
            context_ids: Context concept IDs
            
        Returns:
            ModalResult object
        """
        # Generate possible worlds
        possible_worlds = self.generate_possible_worlds(context_ids, self.world_count)
        
        # Check compatibility with each world
        compatible_worlds = []
        incompatible_worlds = []
        
        for world in possible_worlds:
            # Create a premise network from this world
            premise_network = self._world_to_network(world)
            
            # Test conclusion in this world
            try:
                result = self.inference_test.test_conclusion(premise_network, concept_id)
                
                if result.is_coherent:
                    compatible_worlds.append(world)
                else:
                    incompatible_worlds.append(world)
            except ValueError:
                # If testing fails, count as incompatible
                incompatible_worlds.append(world)
                
        # Calculate modal metrics
        total_worlds = len(possible_worlds)
        compatible_count = len(compatible_worlds)
        
        necessity_degree = compatible_count / total_worlds if total_worlds > 0 else 0.0
        possibility_degree = 1.0 if compatible_count > 0 else 0.0
        
        # Determine modal status
        if necessity_degree == 1.0:
            status = "necessary"
        elif possibility_degree == 0.0:
            status = "impossible"
        else:
            status = "contingent"
            
        # Get compatible and incompatible contexts
        compatible_contexts = []
        for world in compatible_worlds:
            compatible_contexts.extend(world.core_concepts)
            
        incompatible_contexts = []
        for world in incompatible_worlds:
            incompatible_contexts.extend(world.core_concepts)
            
        # Remove duplicates and context concepts themselves
        compatible_contexts = list(set([c for c in compatible_contexts if c != concept_id and c not in context_ids]))
        incompatible_contexts = list(set([c for c in incompatible_contexts if c != concept_id and c not in context_ids]))
        
        return ModalResult(
            concept_id=concept_id,
            status=status,
            necessity_degree=necessity_degree,
            possibility_degree=possibility_degree,
            compatible_contexts=compatible_contexts[:10],  # Limit to top 10
            incompatible_contexts=incompatible_contexts[:10]  # Limit to top 10
        )
    
    def generate_possible_worlds(
        self, 
        seed_concepts: List[str], 
        count: int
    ) -> List[PossibleWorld]:
        """
        Generate different synchronization patterns representing possible worlds.
        
        Args:
            seed_concepts: Core concepts defining the domain
            count: Number of worlds to generate
            
        Returns:
            List of PossibleWorld objects
        """
        worlds = []
        
        # Get all concepts for testing compatibility
        all_concepts = list(self.koopman_graph.concepts.keys())
        test_concepts = [c for c in all_concepts if c not in seed_concepts]
        
        # Generate worlds with different initial conditions
        for i in range(count):
            # Create a premise network from seed concepts
            premise_network = self.premise_sync.create_premise_network(seed_concepts)
            
            # Randomize initial phases for variety
            for concept_id in premise_network.phase_states:
                premise_network.phase_states[concept_id] = 2 * math.pi * random.random()
                
            # Stabilize the network
            premise_network = self.premise_sync.stabilize_premises(premise_network)
            
            if not premise_network.is_stable:
                # Skip unstable worlds
                continue
                
            # Create possible world from stable network
            phase_config = premise_network.phase_states.copy()
            coherence = self._calculate_coherence(phase_config)
            
            # Estimate basin size based on stability (simplified)
            basin_size = coherence ** 2
            
            # Create the world
            world = PossibleWorld(
                core_concepts=seed_concepts.copy(),
                phase_configuration=phase_config,
                coherence_value=coherence,
                basin_size=basin_size
            )
            
            # Test compatibility with a sample of other concepts
            self._test_world_compatibility(world, random.sample(test_concepts, min(20, len(test_concepts))))
            
            worlds.append(world)
            
        # If we couldn't create enough worlds, duplicate with variations
        while len(worlds) < count and worlds:
            # Duplicate a world with modifications
            base_world = random.choice(worlds)
            
            # Create a modified phase configuration
            modified_phases = {}
            for concept_id, phase in base_world.phase_configuration.items():
                # Add small random perturbation
                modified_phases[concept_id] = (phase + 0.2 * random.random()) % (2 * math.pi)
                
            # Create modified world
            modified_world = PossibleWorld(
                core_concepts=base_world.core_concepts.copy(),
                phase_configuration=modified_phases,
                coherence_value=base_world.coherence_value * 0.9,  # Slightly less coherent
                basin_size=base_world.basin_size * 0.8  # Smaller basin
            )
            
            # Test compatibility with a sample of other concepts
            self._test_world_compatibility(modified_world, random.sample(test_concepts, min(10, len(test_concepts))))
            
            worlds.append(modified_world)
            
        return worlds
    
    def _world_to_network(self, world: PossibleWorld) -> PremiseNetwork:
        """
        Convert a possible world to a premise network.
        
        Args:
            world: PossibleWorld object
            
        Returns:
            Corresponding PremiseNetwork
        """
        # Create a premise network
        premise_network = self.premise_sync.create_premise_network(world.core_concepts)
        
        # Override phases with the world's configuration
        for concept_id, phase in world.phase_configuration.items():
            if concept_id in premise_network.phase_states:
                premise_network.phase_states[concept_id] = phase
                
        # Mark as stable
        premise_network.is_stable = True
        
        return premise_network
    
    def _test_world_compatibility(
        self,
        world: PossibleWorld,
        test_concepts: List[str]
    ) -> None:
        """
        Test which concepts are compatible with a possible world.
        
        Args:
            world: PossibleWorld to test
            test_concepts: Concepts to test for compatibility
        """
        # Convert world to premise network
        premise_network = self._world_to_network(world)
        
        # Test each concept
        for concept_id in test_concepts:
            try:
                result = self.inference_test.test_conclusion(premise_network, concept_id)
                
                if result.is_coherent:
                    world.compatible_concepts.append(concept_id)
                else:
                    world.incompatible_concepts.append(concept_id)
            except ValueError:
                # Skip if testing fails
                pass
                
    def _calculate_coherence(self, phase_states: Dict[str, float]) -> float:
        """
        Calculate Kuramoto order parameter as coherence measure.
        
        Args:
            phase_states: Dictionary mapping concept IDs to phase values
            
        Returns:
            Coherence value between 0 and 1
        """
        if not phase_states:
            return 0.0
            
        # Calculate complex order parameter
        sum_real = 0.0
        sum_imag = 0.0
        
        for phase in phase_states.values():
            sum_real += math.cos(phase)
            sum_imag += math.sin(phase)
            
        # Normalize by number of oscillators
        n = len(phase_states)
        sum_real /= n
        sum_imag /= n
        
        # Calculate magnitude (r)
        r = math.sqrt(sum_real ** 2 + sum_imag ** 2)
        
        return r


class CoherenceMetricsCalculator:
    """
    Provides sophisticated metrics for evaluating reasoning validity beyond binary true/false.
    """
    
    def __init__(
        self,
        perturbation_strength: float = 0.1,
        perturbation_trials: int = 10,
        simulation_steps: int = 200
    ):
        """
        Initialize the coherence metrics calculator.
        
        Args:
            perturbation_strength: Strength of perturbations for resilience testing
            perturbation_trials: Number of perturbation trials
            simulation_steps: Maximum steps for simulations
        """
        self.perturbation_strength = perturbation_strength
        self.perturbation_trials = perturbation_trials
        self.simulation_steps = simulation_steps
        
        # Other components
        self.premise_sync = PremiseSynchronizer()
        
    def calculate_phase_validity(
        self,
        premise_network: PremiseNetwork,
        conclusion_id: str
    ) -> PhaseValidity:
        """
        Calculate comprehensive validity assessment for an inference.
        
        Args:
            premise_network: Stabilized premise network
            conclusion_id: Conclusion concept ID
            
        Returns:
            PhaseValidity object with comprehensive metrics
        """
        # Get inference tester for basic coherence
        inference_tester = InferenceTester()
        
        # Test basic coherence
        inference_result = inference_tester.test_conclusion(premise_network, conclusion_id)
        coherence_validity = inference_result.coherence_score
        
        # Calculate logical validity analog
        logical_validity = 1.0 if inference_result.is_coherent else 0.0
        
        # Test resilience to perturbations
        resilience_validity = self._test_resilience(premise_network, conclusion_id)
        
        # Calculate contextual validity
        contextual_validity = self._calculate_contextual_validity(premise_network, conclusion_id)
        
        # Calculate global validity
        global_validity = self._calculate_global_validity(premise_network, conclusion_id)
        
        # Detect phase conflicts
        conflicts = self._detect_phase_conflicts(premise_network, conclusion_id)
        
        # Identify supporting concepts
        supporting_concepts = self._identify_supporting_concepts(premise_network, conclusion_id)
        
        # Calculate aggregate score
        aggregate_score = 0.25 * logical_validity + 0.25 * coherence_validity + \
                          0.2 * resilience_validity + 0.15 * contextual_validity + \
                          0.15 * global_validity
        
        # Create result
        return PhaseValidity(
            logical_validity=logical_validity,
            coherence_validity=coherence_validity,
            resilience_validity=resilience_validity,
            contextual_validity=contextual_validity,
            global_validity=global_validity,
            conflicts=conflicts,
            supporting_concepts=supporting_concepts,
            aggregate_score=aggregate_score
        )
        
    def _test_resilience(
        self,
        premise_network: PremiseNetwork,
        conclusion_id: str
    ) -> float:
        """
        Test how resilient an inference is to perturbations.
        
        Args:
            premise_network: Stabilized premise network
            conclusion_id: Conclusion concept ID
            
        Returns:
            Resilience score (0-1)
        """
        # Create a combined network with conclusion
        inference_tester = InferenceTester()
        extended_network = inference_tester._extend_network(premise_network, conclusion_id)
        
        # Run multiple perturbation trials
        recoveries = 0
        
        for _ in range(self.perturbation_trials):
            # Create perturbed copy
            perturbed_network = {
                "ids": extended_network["ids"].copy(),
                "phase_states": extended_network["phase_states"].copy(),
                "natural_frequencies": extended_network["natural_frequencies"].copy(),
                "coupling_matrix": extended_network["coupling_matrix"].copy()
            }
            
            # Apply random perturbations to phases
            for concept_id in perturbed_network["phase_states"]:
                perturbation = self.perturbation_strength * (2 * random.random() - 1) * math.pi
                perturbed_network["phase_states"][concept_id] = \
                    (perturbed_network["phase_states"][concept_id] + perturbation) % (2 * math.pi)
                    
            # Simulate recovery
            final_coherence = 0.0
            for step in range(self.simulation_steps):
                # Update network
                perturbed_network = inference_tester._update_network(perturbed_network, 0.1)
                
                # Calculate coherence
                coherence = inference_tester._calculate_coherence(perturbed_network["phase_states"])
                
                if step > self.simulation_steps - 10:
                    final_coherence += coherence / 10  # Average over last 10 steps
                    
            # Check if recovered to coherent state
            if final_coherence >= 0.8:
                recoveries += 1
                
        # Calculate resilience score
        resilience = recoveries / self.perturbation_trials
        
        return resilience
        
    def _calculate_contextual_validity(
        self,
        premise_network: PremiseNetwork,
        conclusion_id: str
    ) -> float:
        """
        Calculate how valid a conclusion is within the specific context.
        
        Args:
            premise_network: Stabilized premise network
            conclusion_id: Conclusion concept ID
            
        Returns:
            Contextual validity score (0-1)
        """
        # For a simplified implementation, use embedding similarity
        koopman_graph = get_koopman_phase_graph()
        conclusion_concept = koopman_graph.get_concept_by_id(conclusion_id)
        
        if conclusion_concept is None:
            return 0.0
            
        # Calculate average similarity to premises
        similarities = []
        for premise_id in premise_network.premise_ids:
            premise_concept = koopman_graph.get_concept_by_id(premise_id)
            
            if premise_concept is not None:
                similarity = np.dot(conclusion_concept.embedding, premise_concept.embedding) / (
                    np.linalg.norm(conclusion_concept.embedding) * np.linalg.norm(premise_concept.embedding)
                )
                similarities.append(similarity)
                
        if not similarities:
            return 0.0
            
        avg_similarity = sum(similarities) / len(similarities)
        
        # Transform to validity score (similarity of 0.5+ starts to be valid)
        contextual_validity = max(0.0, min(1.0, (avg_similarity - 0.5) * 2))
        
        return contextual_validity
        
    def _calculate_global_validity(
        self,
        premise_network: PremiseNetwork,
        conclusion_id: str
    ) -> float:
        """
        Calculate how valid a conclusion is across broader contexts.
        
        Args:
            premise_network: Stabilized premise network
            conclusion_id: Conclusion concept ID
            
        Returns:
            Global validity score (0-1)
        """
        # For a simplified implementation, estimate based on connectivity
        koopman_graph = get_koopman_phase_graph()
        conclusion_concept = koopman_graph.get_concept_by_id(conclusion_id)
        
        if conclusion_concept is None or not hasattr(koopman_graph, "concepts"):
            return 0.0
            
        # Get stability from memory sculptor if available
        memory_sculptor = get_memory_sculptor()
        stability = 0.5  # Default
        
        if hasattr(memory_sculptor, "concept_states"):
            state = memory_sculptor.concept_states.get(conclusion_id)
            if state:
                stability = state.stability_score
                
        # Calculate connectivity ratio
        n_concepts = len(koopman_graph.concepts)
        if n_concepts <= 1:
            return 0.5  # Default if only one concept
            
        connectivity = len(conclusion_concept.edges) / (n_concepts - 1)
        
        # Combine stability and connectivity for global validity
        global_validity = 0.7 * stability + 0.3 * min(1.0, connectivity * 5)
        
        return global_validity
        
    def _detect_phase_conflicts(
        self,
        premise_network: PremiseNetwork,
        conclusion_id: str
    ) -> List[Tuple[str, str]]:
        """
        Identify concepts with phase interference.
        
        Args:
            premise_network: Stabilized premise network
            conclusion_id: Conclusion concept ID
            
        Returns:
            List of (concept_id1, concept_id2) pairs with conflicts
        """
        # For a simplified implementation, detect based on edge existence
        koopman_graph = get_koopman_phase_graph()
        conclusion_concept = koopman_graph.get_concept_by_id(conclusion_id)
        
        if conclusion_concept is None:
            return []
            
        # Get directly connected concepts to conclusion
        connected_ids = [target_id for target_id, _ in conclusion_concept.edges]
        
        # Find pairs of premises that are connected to conclusion but not to each other
        conflicts = []
        
        for i, premise_id1 in enumerate(premise_network.premise_ids):
            if premise_id1 not in connected_ids:
                continue
                
            premise_concept1 = koopman_graph.get_concept_by_id(premise_id1)
            if premise_concept1 is None:
                continue
                
            for premise_id2 in premise_network.premise_ids[i+1:]:
                if premise_id2 not in connected_ids:
                    continue
                    
                # Check if these premises are connected to each other
                if not any(target_id == premise_id2 for target_id, _ in premise_concept1.edges):
                    # These premises could create a conflict
                    conflicts.append((premise_id1, premise_id2))
                    
        return conflicts
        
    def _identify_supporting_concepts(
        self,
        premise_network: PremiseNetwork,
        conclusion_id: str
    ) -> List[str]:
        """
        Identify concepts that enhance conclusion validity.
        
        Args:
            premise_network: Stabilized premise network
            conclusion_id: Conclusion concept ID
            
        Returns:
            List of supporting concept IDs
        """
        # For a simplified implementation, use edge weights
        koopman_graph = get_koopman_phase_graph()
        conclusion_concept = koopman_graph.get_concept_by_id(conclusion_id)
        
        if conclusion_concept is None:
            return []
            
        # Get edge weights to premises
        supporting_concepts = []
        
        for premise_id in premise_network.premise_ids:
            # Check edges from conclusion to premise
            for target_id, weight in conclusion_concept.edges:
                if target_id == premise_id and weight >= 0.7:
                    supporting_concepts.append(premise_id)
                    break
                    
        return supporting_concepts


class PhaseReasoner:
    """
    Main class orchestrating the overall reasoning process.
    """
    
    def __init__(self):
        """Initialize the phase reasoner."""
        # Component instances
        self.premise_sync = PremiseSynchronizer()
        self.inference_test = InferenceTester()
        self.modal_reasoner = ModalReasoner()
        self.coherence_calc = CoherenceMetricsCalculator()
        
        # Integration components
        self.koopman_graph = get_koopman_phase_graph()
        self.memory_sculptor = get_memory_sculptor()
        
    def derive_conclusions(
        self,
        premises: List[str],
        candidates: List[str],
        context_depth: int = 0,
        threshold: float = 0.7
    ) -> ReasoningResult:
        """
        Perform phase-coherent reasoning to derive conclusions from premises.
        
        Args:
            premises: List of premise concept IDs
            candidates: List of candidate conclusion concept IDs
            context_depth: Depth of context to consider (0 = premises only)
            threshold: Coherence threshold for valid conclusions
            
        Returns:
            ReasoningResult object
        """
        start_time = time.time()
        
        # Create expanded premise set if context requested
        if context_depth > 0:
            expanded_premises = self._expand_context(premises, context_depth)
        else:
            expanded_premises = premises
            
        # Create and stabilize premise network
        premise_network = self.premise_sync.create_premise_network(expanded_premises)
        premise_network = self.premise_sync.stabilize_premises(premise_network)
        
        # Test each candidate conclusion
        inferences = {}
        coherent_concepts = []
        incoherent_concepts = []
        
        for candidate_id in candidates:
            try:
                # Test conclusion
                result = self.inference_test.test_conclusion(premise_network, candidate_id)
                
                # Get modal status
                modal_result = self.modal_reasoner.identify_modal_status(candidate_id, expanded_premises[:3])
                result.modal_status = modal_result.status
                
                # Add to results
                inferences[candidate_id] = result
                
                if result.is_coherent:
                    coherent_concepts.append(candidate_id)
                    # Update memory sculptor state
                    self._update_concept_states(result, True)
                else:
                    incoherent_concepts.append(candidate_id)
                    # Update memory sculptor state
                    self._update_concept_states(result, False)
            except ValueError as e:
                logger.warning(f"Error testing conclusion {candidate_id}: {e}")
                
        # Create reasoning result
        result = ReasoningResult(
            premises=premises,
            candidates=candidates,
            inferences=inferences,
            coherent_concepts=coherent_concepts,
            incoherent_concepts=incoherent_concepts,
            reasoning_time=time.time() - start_time,
            context_depth=context_depth
        )
        
        # Inform ontology refactoring
        self._inform_ontology_refactoring(result)
        
        return result
    
    def explain_inference(
        self,
        inference_result: InferenceResult
    ) -> Dict[str, Any]:
        """
        Generate an explanation of the reasoning behind an inference.
        
        Args:
            inference_result: Result of an inference
            
        Returns:
            Dictionary with explanation
        """
        # Calculate comprehensive phase validity
        premise_network = self.premise_sync.create_premise_network(inference_result.premise_ids)
        premise_network = self.premise_sync.stabilize_premises(premise_network)
        
        validity = self.coherence_calc.calculate_phase_validity(
            premise_network, inference_result.conclusion_id
        )
        
        # Get concept information
        koopman_graph = get_koopman_phase_graph()
        premise_concepts = []
        
        for premise_id in inference_result.premise_ids:
            concept = koopman_graph.get_concept_by_id(premise_id)
            if concept is not None:
                premise_concepts.append({
                    "id": premise_id,
                    "name": concept.name
                })
                
        conclusion_concept = koopman_graph.get_concept_by_id(inference_result.conclusion_id)
        conclusion_name = conclusion_concept.name if conclusion_concept else inference_result.conclusion_id
        
        # Create explanation
        explanation = {
            "premises": premise_concepts,
            "conclusion": {
                "id": inference_result.conclusion_id,
                "name": conclusion_name
            },
            "is_valid": inference_result.is_coherent,
            "coherence_score": inference_result.coherence_score,
            "phase_validity": {
                "logical_validity": validity.logical_validity,
                "coherence_validity": validity.coherence_validity,
                "resilience_validity": validity.resilience_validity,
                "contextual_validity": validity.contextual_validity,
                "global_validity": validity.global_validity,
                "aggregate_score": validity.aggregate_score
            },
            "conflicts": validity.conflicts,
            "supporting_concepts": validity.supporting_concepts,
            "modal_status": inference_result.modal_status,
            "probability_analog": inference_result.probability_analog,
            "stabilization_time": inference_result.stabilization_time
        }
        
        return explanation
    
    def _expand_context(
        self,
        premise_ids: List[str],
        depth: int
    ) -> List[str]:
        """
        Expand premises with related concepts for context.
        
        Args:
            premise_ids: Original premise concept IDs
            depth: Depth of expansion
            
        Returns:
            Expanded list of concept IDs
        """
        if depth <= 0:
            return premise_ids
            
        # Get all directly connected concepts
        expanded_ids = set(premise_ids)
        
        for premise_id in premise_ids:
            concept = self.koopman_graph.get_concept_by_id(premise_id)
            if concept is not None:
                # Add directly connected concepts
                for target_id, weight in concept.edges:
                    if weight >= 0.6:  # Only add strong connections
                        expanded_ids.add(target_id)
                        
        # If depth > 1, recursively expand
        if depth > 1:
            return self._expand_context(list(expanded_ids), depth - 1)
            
        return list(expanded_ids)
    
    def _update_concept_states(
        self,
        inference_result: InferenceResult,
        is_coherent: bool
    ) -> None:
        """
        Update Memory Sculptor concept states based on reasoning.
        
        Args:
            inference_result: Result of an inference
            is_coherent: Whether the inference was coherent
        """
        if not hasattr(self.memory_sculptor, "update_concept_state"):
            return
            
        # Update conclusion state
        if is_coherent:
            # Coherent conclusion resonates with premises
            for premise_id in inference_result.premise_ids:
                self.memory_sculptor.update_concept_state(
                    concept_id=inference_result.conclusion_id,
                    detected_desync=False,
                    resonated_with=premise_id,
                    resonance_strength=inference_result.coherence_score
                )
                
                # Also update premise states
                self.memory_sculptor.update_concept_state(
                    concept_id=premise_id,
                    detected_desync=False,
                    resonated_with=inference_result.conclusion_id,
                    resonance_strength=inference_result.coherence_score
                )
        else:
            # Incoherent conclusion desyncs
            self.memory_sculptor.update_concept_state(
                concept_id=inference_result.conclusion_id,
                detected_desync=True
            )
    
    def _inform_ontology_refactoring(
        self,
        reasoning_result: ReasoningResult
    ) -> None:
        """
        Inform Ontology Refactor Engine based on reasoning outcomes.
        
        Args:
            reasoning_result: Result of reasoning process
        """
        refactor_engine = get_ontology_refactor_engine()
        
        # Find potential merge candidates (concepts that always synchronize together)
        coherent_pairs = []
        
        for conclusion_id in reasoning_result.coherent_concepts:
            for premise_id in reasoning_result.premises:
                result = reasoning_result.inferences.get(conclusion_id)
                if result and result.coherence_score > 0.9:
                    coherent_pairs.append((premise_id, conclusion_id))
                    
        # If multiple conclusions were coherent with the same premise,
        # suggest them as merge candidates
        merge_candidates = defaultdict(list)
        
        for premise_id, conclusion_id in coherent_pairs:
            merge_candidates[premise_id].append(conclusion_id)
            
        for premise_id, conclusions in merge_candidates.items():
            if len(conclusions) >= 2:
                # Suggest first two as merge candidates
                if hasattr(refactor_engine, "merge_nodes"):
                    # Just inform for now, don't actually merge
                    logger.info(f"Suggesting merge candidates: {conclusions[0]} and {conclusions[1]}")


# Singleton instance
_phase_reasoner_instance = None

def get_phase_reasoner() -> PhaseReasoner:
    """
    Get the singleton instance of the phase reasoner.
    
    Returns:
        PhaseReasoner instance
    """
    global _phase_reasoner_instance
    
    if _phase_reasoner_instance is None:
        _phase_reasoner_instance = PhaseReasoner()
        
    return _phase_reasoner_instance
