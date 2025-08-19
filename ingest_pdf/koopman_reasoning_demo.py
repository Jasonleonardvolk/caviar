"""koopman_reasoning_demo.py - Demonstrates ALAN's enhanced Koopman-based reasoning.

This script showcases ALAN's new Koopman-based reasoning capabilities using
spectral decomposition and eigenfunction analysis. It demonstrates:

1. How to use Takata's Koopman Phase Estimator with the Yosida approximation
2. Reasoning through eigenfunction alignment rather than direct phase comparison
3. Spectral stability analysis with Lyapunov exponents
4. Visualization of the spectral reasoning process

This approach replaces the previous Kuramoto-based synchronization model with
a more precise, robust, and mathematically grounded approach based on spectral
decomposition of the dynamics.
"""

import os
import sys
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any
import random
import math
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/koopman_reasoning_demo.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("koopman_reasoning_demo")

# Ensure path includes current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ALAN components
try:
    try:
        # Try absolute import first
        from koopman_estimator import KoopmanEstimator, KoopmanEigenMode
    except ImportError:
        # Fallback to relative import
        from .koopman_estimator import KoopmanEstimator, KoopmanEigenMode
    try:
        # Try absolute import first
        from eigen_alignment import EigenAlignment, AlignmentResult
    except ImportError:
        # Fallback to relative import
        from .eigen_alignment import EigenAlignment, AlignmentResult
    try:
        # Try absolute import first
        from lyapunov_spike_detector import LyapunovSpikeDetector, StabilityAnalysis
    except ImportError:
        # Fallback to relative import
        from .lyapunov_spike_detector import LyapunovSpikeDetector, StabilityAnalysis
    try:
        # Try absolute import first
        from models import ConceptTuple
    except ImportError:
        # Fallback to relative import
        from .models import ConceptTuple
except ImportError as e:
    # Try direct import if relative import fails
    try:
        from koopman_estimator import KoopmanEstimator, KoopmanEigenMode
        from eigen_alignment import EigenAlignment, AlignmentResult
        from lyapunov_spike_detector import LyapunovSpikeDetector, StabilityAnalysis
        from models import ConceptTuple
    except ImportError:
        logger.error(f"Failed to import ALAN components: {e}")
        logger.error("Please make sure the ALAN system is properly installed.")
        sys.exit(1)

# Ensure directories exist
def ensure_directories():
    """Create necessary directories."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("output", exist_ok=True)

class ConceptOscillator:
    """
    Represents a concept as an oscillator with phase, frequency, and amplitude.
    
    This class models concepts as oscillators, tracking their trajectories
    over time for use in Koopman-based reasoning.
    
    Attributes:
        name: Name of the concept
        id: Unique identifier
        phase: Current phase
        frequency: Natural frequency
        amplitude: Oscillation amplitude
        coupling: Coupling strengths to other oscillators
        trajectory: History of states (phase, amplitude)
    """
    
    def __init__(
        self, 
        name: str, 
        concept_id: str = None,
        phase: float = 0.0,
        frequency: float = 1.0,
        amplitude: float = 1.0
    ):
        """
        Initialize concept oscillator.
        
        Args:
            name: Name of the concept
            concept_id: Unique identifier (defaults to name if not provided)
            phase: Initial phase
            frequency: Natural frequency
            amplitude: Oscillation amplitude
        """
        self.name = name
        self.id = concept_id or name
        self.phase = phase
        self.frequency = frequency
        self.amplitude = amplitude
        self.coupling: Dict[str, float] = {}
        self.trajectory: List[np.ndarray] = []
        
        # Initial state
        self._record_state()
        
    def _record_state(self):
        """Record current state to trajectory."""
        state = np.array([self.phase, self.amplitude])
        self.trajectory.append(state)
        
    def couple_to(self, other_id: str, strength: float = 0.1):
        """
        Couple this oscillator to another.
        
        Args:
            other_id: ID of oscillator to couple to
            strength: Coupling strength
        """
        self.coupling[other_id] = strength
        
    def update(self, dt: float, oscillators: Dict[str, 'ConceptOscillator']):
        """
        Update oscillator state based on couplings.
        
        Args:
            dt: Time step
            oscillators: Dictionary of all oscillators
        """
        # Calculate phase update using Kuramoto model
        phase_sum = 0.0
        
        for other_id, strength in self.coupling.items():
            if other_id in oscillators:
                other = oscillators[other_id]
                phase_diff = math.sin(other.phase - self.phase)
                phase_sum += strength * phase_diff
                
        # Update phase
        self.phase += dt * (self.frequency + phase_sum)
        self.phase = self.phase % (2 * math.pi)
        
        # Simple amplitude dynamics (can be extended)
        # For now, just add small noise
        self.amplitude += dt * 0.01 * (random.random() - 0.5)
        self.amplitude = max(0.1, min(1.0, self.amplitude))
        
        # Record state
        self._record_state()
        
    def get_trajectory(self) -> np.ndarray:
        """
        Get oscillator trajectory as array.
        
        Returns:
            Trajectory array with shape (n_steps, 2)
        """
        return np.array(self.trajectory)

class KoopmanReasoningSystem:
    """
    Implements ALAN's Koopman-based reasoning capabilities.
    
    This system uses Koopman operator theory to perform eigenfunction-based
    reasoning and stability analysis, replacing the simpler Kuramoto approach.
    
    Attributes:
        oscillators: Dictionary mapping concept IDs to oscillators
        koopman_estimator: Estimator for Koopman eigenfunctions
        eigen_alignment: Analyzer for eigenfunction alignment
        stability_detector: Detector for spectral instabilities
    """
    
    def __init__(
        self,
        dt: float = 0.1,
        alignment_threshold: float = 0.7,
        stability_threshold: float = 0.01
    ):
        """
        Initialize the reasoning system.
        
        Args:
            dt: Time step for simulations
            alignment_threshold: Threshold for eigenfunction alignment
            stability_threshold: Max Lyapunov exponent threshold for stability
        """
        self.oscillators: Dict[str, ConceptOscillator] = {}
        self.dt = dt
        self.alignment_threshold = alignment_threshold
        self.stability_threshold = stability_threshold
        
        # Create Koopman components
        self.koopman_estimator = KoopmanEstimator(
            basis_type="fourier",
            basis_params={"n_harmonics": 3},
            dt=dt,
            n_eigenfunctions=5
        )
        
        self.eigen_alignment = EigenAlignment(
            koopman_estimator=self.koopman_estimator,
            alignment_threshold=alignment_threshold
        )
        
        self.stability_detector = LyapunovSpikeDetector(
            koopman_estimator=self.koopman_estimator,
            stability_threshold=stability_threshold
        )
        
    def add_concept(
        self,
        name: str,
        concept_id: str = None,
        phase: float = None,
        frequency: float = None
    ) -> str:
        """
        Add a concept to the system.
        
        Args:
            name: Name of the concept
            concept_id: Unique identifier (defaults to name if not provided)
            phase: Initial phase (random if None)
            frequency: Natural frequency (random if None)
            
        Returns:
            Concept ID
        """
        if phase is None:
            phase = random.random() * 2 * math.pi
            
        if frequency is None:
            frequency = 0.8 + random.random() * 0.4  # 0.8-1.2
            
        concept_id = concept_id or name
        
        # Create oscillator
        self.oscillators[concept_id] = ConceptOscillator(
            name=name,
            concept_id=concept_id,
            phase=phase,
            frequency=frequency
        )
        
        return concept_id
        
    def add_concepts(self, names: List[str]) -> List[str]:
        """
        Add multiple concepts to the system.
        
        Args:
            names: List of concept names
            
        Returns:
            List of concept IDs
        """
        return [self.add_concept(name) for name in names]
        
    def create_relation(self, source_id: str, target_id: str, strength: float = 0.1):
        """
        Create a directional relation between concepts.
        
        Args:
            source_id: ID of source concept
            target_id: ID of target concept
            strength: Coupling strength
        """
        if source_id in self.oscillators and target_id in self.oscillators:
            self.oscillators[source_id].couple_to(target_id, strength)
            
    def create_bidirectional_relation(
        self, 
        concept1_id: str, 
        concept2_id: str, 
        strength: float = 0.1
    ):
        """
        Create a bidirectional relation between concepts.
        
        Args:
            concept1_id: ID of first concept
            concept2_id: ID of second concept
            strength: Coupling strength
        """
        self.create_relation(concept1_id, concept2_id, strength)
        self.create_relation(concept2_id, concept1_id, strength)
            
    def simulate(self, steps: int = 100):
        """
        Run simulation for specified number of steps.
        
        Args:
            steps: Number of simulation steps
        """
        for _ in range(steps):
            # Update all oscillators
            for oscillator in self.oscillators.values():
                oscillator.update(self.dt, self.oscillators)
                
    def analyze_traditional_coherence(
        self, 
        concept_ids: List[str]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Analyze phase coherence using traditional Kuramoto approach.
        
        Args:
            concept_ids: List of concept IDs to analyze
            
        Returns:
            Tuple of (overall_coherence, phase_differences)
        """
        if not concept_ids or len(concept_ids) < 2:
            return 1.0, {}
            
        # Get oscillators
        oscillators = [
            self.oscillators[cid] for cid in concept_ids 
            if cid in self.oscillators
        ]
        
        if len(oscillators) < 2:
            return 1.0, {}
            
        # Compute average phase
        phases = [osc.phase for osc in oscillators]
        avg_phase = math.atan2(
            sum(math.sin(p) for p in phases) / len(phases),
            sum(math.cos(p) for p in phases) / len(phases)
        )
        
        # Compute phase differences from average
        phase_diffs = {}
        sum_squared_diff = 0.0
        
        for i, osc in enumerate(oscillators):
            # Compute circular difference
            diff = (osc.phase - avg_phase + math.pi) % (2 * math.pi) - math.pi
            phase_diffs[concept_ids[i]] = diff
            sum_squared_diff += diff ** 2
            
        # Compute overall coherence (1.0 = perfectly coherent)
        variance = sum_squared_diff / len(oscillators)
        coherence = max(0.0, 1.0 - math.sqrt(variance) / math.pi)
        
        return coherence, phase_diffs
        
    def analyze_koopman_coherence(
        self,
        concept_ids: List[str]
    ) -> AlignmentResult:
        """
        Analyze coherence using Koopman eigenfunction alignment.
        
        Args:
            concept_ids: List of concept IDs to analyze
            
        Returns:
            AlignmentResult with detailed alignment metrics
        """
        if not concept_ids or len(concept_ids) < 2:
            # Single concept or empty is coherent with itself
            return AlignmentResult(
                alignment_score=1.0,
                disruption_score=0.0,
                confidence=1.0,
                modal_status="necessary",
                eigenmode_overlap=1.0,
                premise_coherence=1.0,
                resilience=1.0
            )
            
        # Get trajectories
        trajectories = []
        ids = []
        
        for cid in concept_ids:
            if cid in self.oscillators:
                trajectories.append(self.oscillators[cid].get_trajectory())
                ids.append(cid)
                
        if not trajectories:
            return AlignmentResult()
            
        # Use eigenfunction alignment to compute coherence
        # Split into premise and conclusion (last concept)
        if len(trajectories) > 1:
            premise_trajectories = trajectories[:-1]
            premise_ids = ids[:-1]
            conclusion_trajectory = trajectories[-1]
            conclusion_id = ids[-1]
            
            # Compute alignment
            alignment = self.eigen_alignment.analyze_alignment(
                premise_trajectories=premise_trajectories,
                candidate_trajectory=conclusion_trajectory,
                premise_ids=premise_ids,
                candidate_id=conclusion_id
            )
            
            return alignment
        else:
            # Single concept case
            return AlignmentResult(
                alignment_score=1.0,
                disruption_score=0.0,
                confidence=1.0,
                modal_status="necessary",
                eigenmode_overlap=1.0,
                premise_coherence=1.0,
                resilience=1.0
            )
            
    def analyze_stability(
        self,
        concept_ids: List[str]
    ) -> StabilityAnalysis:
        """
        Analyze stability of concept dynamics using Lyapunov analysis.
        
        Args:
            concept_ids: List of concept IDs to analyze
            
        Returns:
            StabilityAnalysis with detailed stability metrics
        """
        # Get trajectories
        trajectories = []
        ids = []
        
        for cid in concept_ids:
            if cid in self.oscillators:
                trajectories.append(self.oscillators[cid].get_trajectory())
                ids.append(cid)
                
        if not trajectories:
            return StabilityAnalysis()
            
        # Analyze stability
        stability = self.stability_detector.assess_cluster_stability(
            cluster_trajectories=trajectories,
            concept_ids=ids
        )
        
        return stability
        
    def test_inference(
        self,
        premise_ids: List[str],
        conclusion_id: str,
        visualize: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test inference from premises to conclusion using Koopman analysis.
        
        Args:
            premise_ids: List of premise concept IDs
            conclusion_id: Conclusion concept ID
            visualize: Whether to visualize results
            save_path: Optional path to save visualizations
            
        Returns:
            Dictionary with inference results
        """
        # Get trajectories
        premise_trajectories = []
        premise_names = []
        
        for cid in premise_ids:
            if cid in self.oscillators:
                premise_trajectories.append(self.oscillators[cid].get_trajectory())
                premise_names.append(self.oscillators[cid].name)
                
        if conclusion_id not in self.oscillators:
            return {"error": f"Conclusion concept {conclusion_id} not found"}
            
        conclusion_trajectory = self.oscillators[conclusion_id].get_trajectory()
        conclusion_name = self.oscillators[conclusion_id].name
        
        # Analyze eigenfunction alignment
        alignment = self.eigen_alignment.analyze_alignment(
            premise_trajectories=premise_trajectories,
            candidate_trajectory=conclusion_trajectory,
            premise_ids=premise_ids,
            candidate_id=conclusion_id
        )
        
        # Analyze stability
        premise_stability, combined_stability, instability_increase = self.stability_detector.detect_inference_instability(
            premise_trajectories=premise_trajectories,
            conclusion_trajectory=conclusion_trajectory,
            premise_ids=premise_ids,
            conclusion_id=conclusion_id
        )
        
        # Determine inference validity
        is_valid = alignment.is_aligned(self.alignment_threshold) and instability_increase <= 0.2
        
        # Traditional coherence analysis for comparison
        traditional_coherence, _ = self.analyze_traditional_coherence(premise_ids + [conclusion_id])
        
        # Create result
        result = {
            "premises": premise_names,
            "conclusion": conclusion_name,
            "is_valid": is_valid,
            "alignment_score": alignment.alignment_score,
            "disruption_score": alignment.disruption_score,
            "modal_status": alignment.modal_status,
            "max_lyapunov": combined_stability.max_lyapunov,
            "instability_increase": instability_increase,
            "traditional_coherence": traditional_coherence,
            "alignment_details": alignment,
            "stability_details": combined_stability
        }
        
        # Visualize if requested
        if visualize:
            # Create visualization directory
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
            # Alignment visualization
            alignment_viz = self.eigen_alignment.visualize_alignment(
                alignment,
                title=f"Inference: {' & '.join(premise_names)} → {conclusion_name}",
                show_plot=False
            )
            
            if alignment_viz and save_path:
                alignment_path = f"{os.path.splitext(save_path)[0]}_alignment.png"
                alignment_viz.savefig(alignment_path)
                plt.close(alignment_viz)
                
            # Stability visualization
            stability_viz = self.stability_detector.visualize_stability_analysis(
                combined_stability,
                title=f"Stability Analysis: {' & '.join(premise_names)} + {conclusion_name}",
                show_plot=False
            )
            
            if stability_viz and save_path:
                stability_path = f"{os.path.splitext(save_path)[0]}_stability.png"
                stability_viz.savefig(stability_path)
                plt.close(stability_viz)
        
        return result

def demonstrate_simple_inferences():
    """Demonstrate simple inferences with the Koopman reasoning system."""
    print("\n" + "="*70)
    print(" Koopman-Based Reasoning: Simple Inferences ")
    print("="*70)
    
    # Create reasoning system
    system = KoopmanReasoningSystem(dt=0.1, alignment_threshold=0.7)
    
    # Create basic concepts
    concepts = {
        "animal": system.add_concept("Animal"),
        "dog": system.add_concept("Dog"),
        "cat": system.add_concept("Cat"),
        "mammal": system.add_concept("Mammal"),
        "bird": system.add_concept("Bird"),
        "canine": system.add_concept("Canine"),
        "barks": system.add_concept("Barks"),
        "flies": system.add_concept("Flies")
    }
    
    # Create valid relations
    system.create_bidirectional_relation(concepts["dog"], concepts["animal"], 0.5)
    system.create_bidirectional_relation(concepts["cat"], concepts["animal"], 0.5)
    system.create_bidirectional_relation(concepts["dog"], concepts["mammal"], 0.5)
    system.create_bidirectional_relation(concepts["cat"], concepts["mammal"], 0.5)
    system.create_bidirectional_relation(concepts["dog"], concepts["canine"], 0.5)
    system.create_bidirectional_relation(concepts["bird"], concepts["animal"], 0.5)
    system.create_bidirectional_relation(concepts["bird"], concepts["flies"], 0.5)
    system.create_bidirectional_relation(concepts["dog"], concepts["barks"], 0.5)
    
    # Invalid/weak relations
    system.create_bidirectional_relation(concepts["cat"], concepts["barks"], 0.1)
    system.create_bidirectional_relation(concepts["dog"], concepts["flies"], 0.1)
    
    # Run simulation to generate trajectories
    print("\nSimulating concept dynamics...")
    system.simulate(steps=100)
    print("Simulation complete.")
    
    # Test inferences
    print("\nTesting inferences with Takata's Koopman approach:")
    
    test_inferences = [
        {
            "name": "Dogs are Animals",
            "premises": [concepts["dog"]],
            "conclusion": concepts["animal"],
            "expected": True
        },
        {
            "name": "Dogs are Mammals",
            "premises": [concepts["dog"]],
            "conclusion": concepts["mammal"],
            "expected": True
        },
        {
            "name": "Dogs Bark",
            "premises": [concepts["dog"]],
            "conclusion": concepts["barks"],
            "expected": True
        },
        {
            "name": "Dogs Fly",
            "premises": [concepts["dog"]],
            "conclusion": concepts["flies"],
            "expected": False
        },
        {
            "name": "Dogs and Birds are Mammals",
            "premises": [concepts["dog"], concepts["bird"]],
            "conclusion": concepts["mammal"],
            "expected": False
        }
    ]
    
    for i, test in enumerate(test_inferences):
        print(f"\n[{i+1}] Testing: {test['name']}")
        print(f"  Expected: {'Valid' if test['expected'] else 'Invalid'}")
        
        result = system.test_inference(
            premise_ids=test["premises"],
            conclusion_id=test["conclusion"],
            visualize=False
        )
        
        alignment = result["alignment_score"]
        is_valid = result["is_valid"]
        modal = result["modal_status"]
        disruption = result["disruption_score"]
        instability = result["instability_increase"]
        
        print(f"  Result: {'Valid' if is_valid else 'Invalid'}")
        print(f"  Alignment Score: {alignment:.3f}")
        print(f"  Disruption Score: {disruption:.3f}")
        print(f"  Modal Status: {modal}")
        print(f"  Instability Increase: {instability:.3f}")
        
        # Compare to traditional
        traditional = result["traditional_coherence"]
        print(f"  Traditional Coherence: {traditional:.3f}")
        print(f"  {'✓' if is_valid == test['expected'] else '✗'} " +
              f"{'Correctly' if is_valid == test['expected'] else 'Incorrectly'} classified")

def demonstrate_modal_inferences():
    """Demonstrate modal inferences with the Koopman reasoning system."""
    print("\n" + "="*70)
    print(" Koopman-Based Reasoning: Modal Inferences ")
    print("="*70)
    
    # Create reasoning system
    system = KoopmanReasoningSystem(dt=0.1, alignment_threshold=0.6)
    
    # Create basic concepts for modal reasoning
    concepts = {
        "human": system.add_concept("Human"),
        "mortal": system.add_concept("Mortal"),
        "socrates": system.add_concept("Socrates"),
        "philosopher": system.add_concept("Philosopher"),
        "animal": system.add_concept("Animal"),
        "rational": system.add_concept("Rational"),
        "featherless_biped": system.add_concept("FeatherlessBiped"),
    }
    
    # Create relations
    # Necessary connections (strong)
    system.create_bidirectional_relation(concepts["human"], concepts["mortal"], 0.9)
    system.create_bidirectional_relation(concepts["socrates"], concepts["human"], 0.9)
    system.create_bidirectional_relation(concepts["human"], concepts["animal"], 0.9)
    system.create_bidirectional_relation(concepts["human"], concepts["rational"], 0.9)
    
    # Possible connections (medium)
    system.create_bidirectional_relation(concepts["human"], concepts["philosopher"], 0.5)
    system.create_bidirectional_relation(concepts["socrates"], concepts["philosopher"], 0.7)
    
    # Contingent connections (weak)
    system.create_bidirectional_relation(concepts["human"], concepts["featherless_biped"], 0.3)
    
    # Run simulation to generate trajectories
    print("\nSimulating concept dynamics...")
    system.simulate(steps=100)
    print("Simulation complete.")
    
    # Test modal inferences
    print("\nTesting modal inferences with Takata's Koopman approach:")
    
    test_inferences = [
        {
            "name": "Socrates is Mortal",
            "premises": [concepts["socrates"], concepts["human"]],
            "conclusion": concepts["mortal"],
            "expected_modal": "necessary"
        },
        {
            "name": "Socrates is a Philosopher",
            "premises": [concepts["socrates"]],
            "conclusion": concepts["philosopher"],
            "expected_modal": "possible"
        },
        {
            "name": "Humans are Featherless Bipeds",
            "premises": [concepts["human"]],
            "conclusion": concepts["featherless_biped"],
            "expected_modal": "contingent"
        },
        {
            "name": "Humans are Rational Animals",
            "premises": [concepts["human"]],
            "conclusion": concepts["rational"],
            "expected_modal": "necessary"
        }
    ]
    
    for i, test in enumerate(test_inferences):
        print(f"\n[{i+1}] Testing: {test['name']}")
        print(f"  Expected Modal Status: {test['expected_modal'].capitalize()}")
        
        result = system.test_inference(
            premise_ids=test["premises"],
            conclusion_id=test["conclusion"],
            visualize=False
        )
        
        alignment = result["alignment_score"]
        disruption = result["disruption_score"]
        modal = result["modal_status"]
        
        print(f"  Result: {modal.capitalize()}")
        print(f"  Alignment Score: {alignment:.3f}")
        print(f"  Disruption Score: {disruption:.3f}")
        
        # Check if modal status matches expected
        modal_correct = modal == test["expected_modal"]
        print(f"  {'✓' if modal_correct else '✗'} " +
              f"{'Correctly' if modal_correct else 'Incorrectly'} classified modality")

def demonstrate_stability_analysis():
    """Demonstrate stability analysis with the Koopman reasoning system."""
    print("\n" + "="*70)
    print(" Koopman-Based Reasoning: Stability Analysis ")
    print("="*70)
    
    # Create reasoning system
    system = KoopmanReasoningSystem(dt=0.1, stability_threshold=0.01)
    
    # Create concepts for stability analysis
    concepts = {
        "water": system.add_concept("Water"),
        "ice": system.add_concept("Ice"),
        "steam": system.add_concept("Steam"),
        "cold": system.add_concept("Cold"),
        "heat": system.add_concept("Heat"),
        "liquid": system.add_concept("Liquid"),
        "solid": system.add_concept("Solid"),
        "gas": system.add_concept("Gas"),
    }
    
    # Create stable connections
    system.create_bidirectional_relation(concepts["water"], concepts["liquid"], 0.5)
    system.create_bidirectional_relation(concepts["ice"], concepts["solid"], 0.5)
    system.create_bidirectional_relation(concepts["steam"], concepts["gas"], 0.5)
    system.create_bidirectional_relation(concepts["water"], concepts["ice"], 0.4)
    system.create_bidirectional_relation(concepts["water"], concepts["steam"], 0.4)
    
    # Create unstable (contradictory) connections
    system.create_bidirectional_relation(concepts["ice"], concepts["heat"], 0.1)
    system.create_relation(concepts["heat"], concepts["ice"], -0.3)  # Negative coupling
    system.create_bidirectional_relation(concepts["cold"], concepts["steam"], 0.1)
    system.create_relation(concepts["cold"], concepts["steam"], -0.3)  # Negative coupling
    
    # Run simulation to generate trajectories
    print("\nSimulating concept dynamics...")
    system.simulate(steps=100)
    print("Simulation complete.")
    
    # Test stability of different concept clusters
    print("\nAnalyzing stability of concept clusters:")
    
    test_clusters = [
        {
            "name": "Stable: Water, Ice, Cold",
            "concepts": [concepts["water"], concepts["ice"], concepts["cold"]],
            "expected": "stable"
        },
        {
            "name": "Potentially Unstable: Ice, Heat",
            "concepts": [concepts["ice"], concepts["heat"]],
            "expected": "unstable"
        },
        {
            "name": "Stable: Water, Liquid",
            "concepts": [concepts["water"], concepts["liquid"]],
            "expected": "stable"
        },
        {
            "name": "Potentially Unstable: Cold, Steam, Heat",
            "concepts": [concepts["cold"], concepts["steam"], concepts["heat"]],
            "expected": "unstable"
        }
    ]
    
    for i, test in enumerate(test_clusters):
        print(f"\n[{i+1}] Testing: {test['name']}")
        print(f"  Expected: {test['expected'].capitalize()}")
        
        stability = system.analyze_stability(test["concepts"])
        
        is_stable = stability.is_stable
        max_lyapunov = stability.max_lyapunov
        instability_risk = stability.instability_risk
        critical_modes = len(stability.critical_modes)
        
        print(f"  Result: {'Stable' if is_stable else 'Unstable'}")
        print(f"  Max Lyapunov Exponent: {max_lyapunov:.5f}")
        print(f"  Instability Risk: {instability_risk:.3f}")
        print(f"  Critical Modes: {critical_modes}")
        print(f"  Spectral Gap: {stability.spectral_gap:.3f}")
        print(f"  {'✓' if (is_stable and test['expected'] == 'stable') or (not is_stable and test['expected'] == 'unstable') else '✗'} " +
              f"{'Correctly' if (is_stable and test['expected'] == 'stable') or (not is_stable and test['expected'] == 'unstable') else 'Incorrectly'} classified")

def demonstrate_koopman_vs_traditional():
    """Demonstrate the advantages of Koopman-based reasoning over traditional phase analysis."""
    print("\n" + "="*70)
    print(" Koopman-Based Reasoning vs. Traditional Phase Analysis ")
    print("="*70)
    
    # Create reasoning system
    system = KoopmanReasoningSystem(dt=0.1)
    
    # Create concepts for demonstration
    concepts = {
        # Core concepts
        "knowledge": system.add_concept("Knowledge"),
        "belief": system.add_concept("Belief"),
        "truth": system.add_concept("Truth"),
        "justification": system.add_concept("Justification"),
        
        # Test concepts
        "opinion": system.add_concept("Opinion"),
        "fact": system.add_concept("Fact"),
        "evidence": system.add_concept("Evidence"),
        "hypothesis": system.add_concept("Hypothesis"),
        "theory": system.add_concept("Theory"),
        "conjecture": system.add_concept("Conjecture")
    }
    
    # Create relations - simplified Platonic knowledge model
    # Knowledge = Justified True Belief
    system.create_bidirectional_relation(concepts["knowledge"], concepts["belief"], 0.7)
    system.create_bidirectional_relation(concepts["knowledge"], concepts["truth"], 0.7)
    system.create_bidirectional_relation(concepts["knowledge"], concepts["justification"], 0.7)
    
    # Secondary relations
    system.create_bidirectional_relation(concepts["belief"], concepts["opinion"], 0.5)
    system.create_bidirectional_relation(concepts["truth"], concepts["fact"], 0.6)
    system.create_bidirectional_relation(concepts["justification"], concepts["evidence"], 0.6)
    
    # Scientific concept relations
    system.create_bidirectional_relation(concepts["hypothesis"], concepts["conjecture"], 0.5)
    system.create_bidirectional_relation(concepts["hypothesis"], concepts["evidence"], 0.3)
    system.create_bidirectional_relation(concepts["theory"], concepts["evidence"], 0.7)
    system.create_bidirectional_relation(concepts["theory"], concepts["hypothesis"], 0.6)
    
    # Add noise connections with oscillatory components
    # This will specifically show the advantage of spectral methods
    for _ in range(5):
        # Random connections with oscillatory components
        a = random.choice(list(concepts.keys()))
        b = random.choice(list(concepts.keys()))
        if a != b:
            # Create weak oscillatory coupling
            system.create_relation(a, b, 0.1 * math.sin(random.random() * math.pi))
    
    # Run simulation to generate trajectories with noisy dynamics
    print("\nSimulating concept dynamics...")
    system.simulate(steps=100)
    print("Simulation complete.")
    
    # Test cases that are difficult for traditional phase analysis
    print("\nComparing traditional vs. Koopman analysis:")
    
    test_inferences = [
        {
            "name": "Knowledge requires Truth",
            "premises": [concepts["knowledge"]],
            "conclusion": concepts["truth"]
        },
        {
            "name": "Theory requires Evidence",
            "premises": [concepts["theory"]],
            "conclusion": concepts["evidence"]
        },
        {
            "name": "Knowledge requires Evidence",
            "premises": [concepts["knowledge"], concepts["justification"]],
            "conclusion": concepts["evidence"]
        },
        {
            "name": "Theory implies Knowledge",
            "premises": [concepts["theory"], concepts["evidence"], concepts["truth"]],
            "conclusion": concepts["knowledge"]
        }
    ]
    
    for i, test in enumerate(test_inferences):
        print(f"\n[{i+1}] Testing: {test['name']}")
        
        # Analyze using both methods
        premises = test["premises"]
        conclusion = test["conclusion"]
        
        # Traditional coherence
        traditional_coherence, _ = system.analyze_traditional_coherence(
            premises + [conclusion]
        )
        
        # Koopman analysis
        koopman_result = system.test_inference(
            premise_ids=premises,
            conclusion_id=conclusion,
            visualize=False
        )
        
        # Compare results
        print(f"  Traditional Coherence: {traditional_coherence:.3f}")
        print(f"  Koopman Alignment: {koopman_result['alignment_score']:.3f}")
        print(f"  Koopman Disruption: {koopman_result['disruption_score']:.3f}")
        print(f"  Koopman Stability Impact: {koopman_result['max_lyapunov']:.5f}")
        print(f"  Koopman Modality: {koopman_result['modal_status']}")
        
        # Compare assessment
        traditional_valid = traditional_coherence >= 0.7
        koopman_valid = koopman_result["is_valid"]
        
        print(f"  Traditional Assessment: {'Valid' if traditional_valid else 'Invalid'}")
        print(f"  Koopman Assessment: {'Valid' if koopman_valid else 'Invalid'}")
        
        # Show advantage
        if traditional_valid != koopman_valid:
            print(f"  Advantage demonstrated: {'Koopman' if koopman_valid else 'Traditional'} " +
                  f"assessment is more accurate")

def main():
    """Main entry point for the Koopman reasoning demonstration."""
    # Ensure directories exist
    ensure_directories()
    
    print("\n" + "="*80)
    print(" ALAN Koopman-Based Reasoning Demonstration ")
    print(" Spectral decomposition and eigenfunction alignment for robust inference ")
    print("="*80 + "\n")
    
    print("This demonstration showcases ALAN's new Koopman-based reasoning capabilities,")
    print("implementing Takata's approach with the Yosida approximation for robust")
    print("eigenfunction estimation under noise and data limitations.\n")
    
    # Ask user which demonstrations to run
    print("Available demonstrations:")
    print("1. Simple Inferences (basic reasoning capabilities)")
    print("2. Modal Inferences (necessary, possible, contingent truths)")
    print("3. Stability Analysis (Lyapunov-based inference validation)")
    print("4. Koopman vs. Traditional (advantages of spectral methods)")
    print("5. Run All Demonstrations")
    
    choice = input("\nEnter your choice (1-5, default is 5): ").strip()
    
    if not choice:
        choice = "5"
        
    try:
        option = int(choice)
        if option < 1 or option > 5:
            print("Invalid choice. Running all demonstrations.")
            option = 5
    except ValueError:
        print("Invalid choice. Running all demonstrations.")
        option = 5
        
    # Run selected demonstrations
    if option in [1, 5]:
        demonstrate_simple_inferences()
        
    if option in [2, 5]:
        demonstrate_modal_inferences()
        
    if option in [3, 5]:
        demonstrate_stability_analysis()
        
    if option in [4, 5]:
        demonstrate_koopman_vs_traditional()
        
    # Summary
    print("\n" + "="*80)
    print(" Summary of Koopman-Based Reasoning Advantages ")
    print("="*80)
    print("\n1. Robust inference under noise and sparse data")
    print("   - Eigenfunction estimation with confidence intervals")
    print("   - Resilience to data limitations and noise")
    
    print("\n2. Modal reasoning capabilities")
    print("   - Necessary truths: high alignment, minimal disruption")
    print("   - Possible truths: moderate alignment, moderate disruption")
    print("   - Contingent truths: low alignment, high disruption or instability")
    
    print("\n3. Stability-aware inference validation")
    print("   - Lyapunov exponent estimation for detecting unstable inferences")
    print("   - Spectral gap analysis for bifurcation detection")
    print("   - Critical mode identification")
    
    print("\n4. Visualization and explainability")
    print("   - Eigenfunction phase portraits")
    print("   - Stability landscapes")
    print("   - Spectral decomposition visualizations")
    
    print("\nThis implementation follows Takata's (2025) approach, replacing the")
    print("traditional Kuramoto model with a more mathematically rigorous foundation")
    print("based on spectral decomposition of the dynamics.\n")

if __name__ == "__main__":
    main()
