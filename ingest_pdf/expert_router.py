"""expert_router.py - Implements selective activation of cognitive modules based on context.

This module provides mechanisms for intelligently routing cognitive processes through
the appropriate expert subsystems, ensuring that only relevant modules are co-activated
to maintain coherence while preventing interference. It enables ALAN to:
- Score module compatibility with current context using SVD-based methods
- Selectively activate only relevant modules
- Prevent cross-talk and interference between unrelated processes
- Dynamically adjust routing based on coherence feedback

References:
- Fleshman & Van Durme (2025) SpectR approach for dynamic expert routing
- Modular activation for coherent reasoning
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import logging
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import heapq
from collections import defaultdict
import warnings

try:
    # Try absolute import first
    from models import ConceptTuple
except ImportError:
    # Fallback to relative import
    from .models import ConceptTuple
try:
    # Try absolute import first
    from spectral_monitor import get_cognitive_spectral_monitor
except ImportError:
    # Fallback to relative import
    from .spectral_monitor import get_cognitive_spectral_monitor
try:
    # Try absolute import first
    from fractality import get_cognitive_fractal_analyzer
except ImportError:
    # Fallback to relative import
    from .fractality import get_cognitive_fractal_analyzer
try:
    # Try absolute import first
    from introspection import get_introspection_system
except ImportError:
    # Fallback to relative import
    from .introspection import get_introspection_system

# Configure logger
logger = logging.getLogger("alan_expert_router")

@dataclass
class ExpertModule:
    """Represents a cognitive expert module that can be selectively activated."""
    id: str  # Unique identifier for the module
    name: str  # Human-readable name
    domain: str  # Domain of expertise (reasoning, memory, perception, etc.)
    description: str  # Description of the module's function
    embedding: np.ndarray  # Vector representation of the module's domain
    activation_threshold: float = 0.5  # Minimum compatibility score to activate
    energy_cost: float = 1.0  # Computational/energy cost of activation
    activation_count: int = 0  # Number of times this module has been activated
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain,
            "description": self.description,
            "embedding": self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else None,
            "activation_threshold": float(self.activation_threshold),
            "energy_cost": float(self.energy_cost),
            "activation_count": self.activation_count,
            "metadata": self.metadata
        }


@dataclass
class RoutingContext:
    """Represents the current cognitive context for routing decisions."""
    query_embedding: np.ndarray  # Embedding of the current query or context
    active_concepts: List[str] = field(default_factory=list)  # Currently active concept IDs
    task_domain: str = "general"  # Current task domain
    task_phase: str = "processing"  # Current phase of task (e.g., input, processing, output)
    desired_module_count: int = 3  # Number of modules to activate
    coherence_requirement: float = 0.7  # Required coherence between modules
    constraints: Dict[str, Any] = field(default_factory=dict)  # Additional constraints
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_embedding": self.query_embedding.tolist() if isinstance(self.query_embedding, np.ndarray) else None,
            "active_concepts": self.active_concepts,
            "task_domain": self.task_domain,
            "task_phase": self.task_phase,
            "desired_module_count": self.desired_module_count,
            "coherence_requirement": float(self.coherence_requirement),
            "constraints": self.constraints
        }


@dataclass
class RoutingDecision:
    """Represents a decision about which modules to activate."""
    timestamp: datetime = field(default_factory=datetime.now)
    selected_modules: List[str] = field(default_factory=list)  # IDs of selected modules
    compatibility_scores: Dict[str, float] = field(default_factory=dict)  # Module ID -> score
    coherence_score: float = 0.0  # Overall coherence of selected modules
    energy_cost: float = 0.0  # Total energy cost of activation
    reasoning: str = ""  # Explanation for the routing decision
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "selected_modules": self.selected_modules,
            "compatibility_scores": {k: float(v) for k, v in self.compatibility_scores.items()},
            "coherence_score": float(self.coherence_score),
            "energy_cost": float(self.energy_cost),
            "reasoning": self.reasoning,
            "metadata": self.metadata
        }


class ContextCompatibilityScorer:
    """
    Scores the compatibility of expert modules with the current context.
    
    Based on SVD-based methods from Fleshman & Van Durme (2025), this class
    evaluates how well each expert module matches the current cognitive context.
    """
    
    def __init__(
        self, 
        similarity_metric: str = "cosine",
        top_k_svd: int = 8
    ):
        """
        Initialize the context compatibility scorer.
        
        Args:
            similarity_metric: Metric for computing similarity ("cosine", "dot", "euclidean")
            top_k_svd: Number of singular values/vectors to use for SVD-based scoring
        """
        self.similarity_metric = similarity_metric
        self.top_k_svd = top_k_svd
        
    def compute_embeddings_similarity(
        self, 
        query_embedding: np.ndarray,
        module_embedding: np.ndarray
    ) -> float:
        """
        Compute similarity between query and module embeddings.
        
        Args:
            query_embedding: Embedding of the query/context
            module_embedding: Embedding of the module
            
        Returns:
            Similarity score (0-1)
        """
        # Ensure embeddings are normalized for cosine similarity
        if self.similarity_metric == "cosine":
            query_norm = np.linalg.norm(query_embedding)
            module_norm = np.linalg.norm(module_embedding)
            
            if query_norm < 1e-10 or module_norm < 1e-10:
                return 0.0
                
            query_embedding = query_embedding / query_norm
            module_embedding = module_embedding / module_norm
            
            similarity = np.dot(query_embedding, module_embedding)
            # Ensure the result is in [-1, 1] due to floating point errors
            similarity = max(-1.0, min(1.0, similarity))
            # Map from [-1, 1] to [0, 1]
            return (similarity + 1) / 2
            
        elif self.similarity_metric == "dot":
            # Simple dot product (not normalized)
            similarity = np.dot(query_embedding, module_embedding)
            # Normalize to [0, 1] based on typical ranges (assumes embeddings are normalized)
            return max(0.0, min(1.0, similarity))
            
        elif self.similarity_metric == "euclidean":
            # Euclidean distance (smaller is better)
            distance = np.linalg.norm(query_embedding - module_embedding)
            # Convert to similarity score (1 when distance is 0, 0 when distance is large)
            similarity = 1.0 / (1.0 + distance)
            return similarity
            
        else:
            logger.warning(f"Unknown similarity metric: {self.similarity_metric}, using cosine")
            # Default to cosine
            cos_sim = np.dot(query_embedding, module_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(module_embedding))
            return (cos_sim + 1) / 2
    
    def compute_svd_compatibility(
        self, 
        query_embedding: np.ndarray,
        module_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute compatibility scores using SVD-based method from SpectR.
        
        Args:
            query_embedding: Embedding of the query/context
            module_embeddings: Dictionary mapping module IDs to embeddings
            
        Returns:
            Dictionary mapping module IDs to compatibility scores
        """
        if not module_embeddings:
            return {}
            
        # Create matrix of module embeddings
        module_ids = list(module_embeddings.keys())
        embedding_dim = next(iter(module_embeddings.values())).shape[0]
        
        # Create matrix where each row is a module embedding
        X = np.zeros((len(module_ids), embedding_dim))
        for i, module_id in enumerate(module_ids):
            X[i] = module_embeddings[module_id]
            
        # Compute SVD
        try:
            # Use sparse SVD for efficiency
            k = min(self.top_k_svd, min(X.shape) - 1)
            U, S, Vt = svds(X, k=k)
            
            # Sort by singular values in decreasing order
            idx = np.argsort(-S)
            U = U[:, idx]
            S = S[idx]
            Vt = Vt[idx]
            
            # Compute query projection onto singular vectors
            query_proj = query_embedding @ Vt.T
            
            # Compute module projections
            module_projs = U * S
            
            # Compute compatibility scores as similarity between projections
            scores = {}
            for i, module_id in enumerate(module_ids):
                # Use cosine similarity in the SVD space
                score = np.dot(query_proj, module_projs[i]) / (
                    np.linalg.norm(query_proj) * np.linalg.norm(module_projs[i]))
                
                # Map from [-1, 1] to [0, 1]
                scores[module_id] = (score + 1) / 2
                
        except Exception as e:
            logger.warning(f"SVD computation failed: {str(e)}")
            # Fall back to direct similarity computation
            scores = {}
            for module_id, embedding in module_embeddings.items():
                scores[module_id] = self.compute_embeddings_similarity(
                    query_embedding, embedding)
                
        return scores
    
    def compute_module_compatibility(
        self, 
        context: RoutingContext,
        modules: List[ExpertModule]
    ) -> Dict[str, float]:
        """
        Compute compatibility scores for all modules with the given context.
        
        Args:
            context: Current routing context
            modules: List of available expert modules
            
        Returns:
            Dictionary mapping module IDs to compatibility scores
        """
        # Extract module embeddings
        module_embeddings = {
            module.id: module.embedding for module in modules
        }
        
        # Compute SVD-based compatibility
        svd_scores = self.compute_svd_compatibility(
            context.query_embedding, module_embeddings)
        
        # Compute direct similarity scores
        direct_scores = {}
        for module in modules:
            direct_scores[module.id] = self.compute_embeddings_similarity(
                context.query_embedding, module.embedding)
            
        # Combine scores (weighted average: 70% SVD, 30% direct)
        combined_scores = {}
        for module_id in module_embeddings:
            svd_score = svd_scores.get(module_id, 0.0)
            direct_score = direct_scores.get(module_id, 0.0)
            combined_scores[module_id] = 0.7 * svd_score + 0.3 * direct_score
            
        # Apply domain-specific adjustments
        for module in modules:
            score = combined_scores.get(module.id, 0.0)
            
            # Boost score if module domain matches task domain
            if module.domain == context.task_domain:
                score *= 1.2
                
            # Apply any custom constraints from the context
            domain_constraints = context.constraints.get("domains", {})
            if module.domain in domain_constraints:
                modifier = domain_constraints[module.domain]
                score *= modifier
                
            # Ensure score is in [0, 1]
            combined_scores[module.id] = max(0.0, min(1.0, score))
            
        return combined_scores


class SparseActivationController:
    """
    Controls which modules are activated based on compatibility and coherence.
    
    This class implements the sparse activation strategy, ensuring that only
    mutually compatible modules are co-activated to maintain coherence.
    """
    
    def __init__(
        self, 
        max_active_modules: int = 5,
        coherence_threshold: float = 0.6,
        interference_penalty: float = 0.3
    ):
        """
        Initialize the sparse activation controller.
        
        Args:
            max_active_modules: Maximum number of modules to activate simultaneously
            coherence_threshold: Minimum required coherence between modules
            interference_penalty: Penalty for potential interference between modules
        """
        self.max_active_modules = max_active_modules
        self.coherence_threshold = coherence_threshold
        self.interference_penalty = interference_penalty
        
        # Keep track of activation history
        self.activation_history = []
        
    def compute_module_coherence(
        self, 
        module1: ExpertModule,
        module2: ExpertModule
    ) -> float:
        """
        Compute coherence score between two modules.
        
        Args:
            module1, module2: Expert modules to compare
            
        Returns:
            Coherence score (0-1)
        """
        # Simple coherence based on embedding similarity
        embedding1 = module1.embedding
        embedding2 = module2.embedding
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        # Map to [0, 1]
        coherence = (similarity + 1) / 2
        
        # Apply domain-based adjustment
        if module1.domain == module2.domain:
            # Modules in same domain are assumed to be more coherent
            coherence = min(1.0, coherence * 1.2)
        else:
            # Cross-domain modules might have more interference
            coherence = max(0.0, coherence - self.interference_penalty)
            
        return coherence
    
    def compute_overall_coherence(
        self, 
        modules: List[ExpertModule]
    ) -> float:
        """
        Compute overall coherence score for a set of modules.
        
        Args:
            modules: List of expert modules
            
        Returns:
            Overall coherence score (0-1)
        """
        if len(modules) <= 1:
            return 1.0  # Perfect coherence with 0 or 1 module
            
        # Compute pairwise coherence for all module pairs
        coherence_sum = 0.0
        pair_count = 0
        
        for i in range(len(modules)):
            for j in range(i+1, len(modules)):
                coherence = self.compute_module_coherence(modules[i], modules[j])
                coherence_sum += coherence
                pair_count += 1
                
        # Average coherence across all pairs
        return coherence_sum / pair_count if pair_count > 0 else 1.0
    
    def select_modules(
        self, 
        context: RoutingContext,
        modules: List[ExpertModule],
        compatibility_scores: Dict[str, float]
    ) -> List[ExpertModule]:
        """
        Select which modules to activate based on compatibility and coherence.
        
        Args:
            context: Current routing context
            modules: List of available expert modules
            compatibility_scores: Module compatibility scores with context
            
        Returns:
            List of selected modules to activate
        """
        # Create module lookup dictionary
        module_dict = {module.id: module for module in modules}
        
        # Filter modules by activation threshold
        eligible_modules = []
        for module_id, score in compatibility_scores.items():
            module = module_dict.get(module_id)
            if module and score >= module.activation_threshold:
                eligible_modules.append(module)
                
        if not eligible_modules:
            logger.warning("No modules meet activation threshold")
            # Return a small number of highest-scoring modules anyway
            sorted_modules = sorted(
                modules, 
                key=lambda m: compatibility_scores.get(m.id, 0.0),
                reverse=True
            )
            return sorted_modules[:min(3, len(sorted_modules))]
            
        # Start with the highest compatibility module
        sorted_eligible = sorted(
            eligible_modules,
            key=lambda m: compatibility_scores.get(m.id, 0.0),
            reverse=True
        )
        
        selected = [sorted_eligible[0]]
        remaining = sorted_eligible[1:]
        
        # Add modules one by one, ensuring coherence is maintained
        target_count = min(
            context.desired_module_count,
            self.max_active_modules,
            len(eligible_modules)
        )
        
        while len(selected) < target_count and remaining:
            # Try adding each remaining module and measure coherence
            best_next = None
            best_coherence = 0.0
            
            for candidate in remaining:
                # Test coherence with this candidate added
                test_modules = selected + [candidate]
                coherence = self.compute_overall_coherence(test_modules)
                
                # Weight by compatibility score
                weighted_coherence = coherence * compatibility_scores.get(candidate.id, 0.0)
                
                if weighted_coherence > best_coherence:
                    best_coherence = weighted_coherence
                    best_next = candidate
                    
            # If best candidate meets coherence threshold, add it
            if best_next and best_coherence >= self.coherence_threshold:
                selected.append(best_next)
                remaining.remove(best_next)
            else:
                # No more modules can be added while maintaining coherence
                break
                
        # Record activation in history
        self.activation_history.append({
            "timestamp": datetime.now().isoformat(),
            "context": context.to_dict(),
            "selected_modules": [module.id for module in selected],
            "coherence": self.compute_overall_coherence(selected)
        })
        
        return selected
        
    def get_recent_activations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent activation history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of recent activation records
        """
        return self.activation_history[-limit:]


class ConceptRoutingOptimizer:
    """
    Dynamically adjusts routing based on coherence feedback.
    
    This class uses feedback from spectral monitoring and introspection
    to continuously optimize the routing of concepts to expert modules.
    """
    
    def __init__(
        self, 
        learning_rate: float = 0.1,
        adjustment_threshold: float = 0.2,
        coherence_history_size: int = 50
    ):
        """
        Initialize the concept routing optimizer.
        
        Args:
            learning_rate: Rate at which to adjust routing parameters
            adjustment_threshold: Minimum coherence change to trigger adjustment
            coherence_history_size: Number of coherence measurements to track
        """
        self.learning_rate = learning_rate
        self.adjustment_threshold = adjustment_threshold
        self.coherence_history = []
        self.coherence_history_size = coherence_history_size
        
        # Access monitoring systems
        self.spectral_monitor = get_cognitive_spectral_monitor()
        self.introspection = get_introspection_system()
        self.fractal_analyzer = get_cognitive_fractal_analyzer()
        
        # Track optimization actions
        self.optimization_history = []
        
    def record_coherence_measurement(
        self, 
        coherence: float,
        routing_decision: RoutingDecision
    ) -> None:
        """
        Record a coherence measurement for optimization.
        
        Args:
            coherence: Measured coherence (0-1)
            routing_decision: The routing decision that led to this coherence
        """
        # Add to history
        self.coherence_history.append({
            "timestamp": datetime.now().isoformat(),
            "coherence": coherence,
            "routing_decision": routing_decision.to_dict()
        })
        
        # Trim history if needed
        if len(self.coherence_history) > self.coherence_history_size:
            self.coherence_history = self.coherence_history[-self.coherence_history_size:]
            
    def detect_coherence_drift(self) -> Dict[str, Any]:
        """
        Detect if there's been a significant drift in coherence.
        
        Returns:
            Dictionary with drift analysis
        """
        if len(self.coherence_history) < 10:
            return {"status": "insufficient_data"}
            
        # Calculate recent vs older coherence
        mid_point = len(self.coherence_history) // 2
        
        recent = self.coherence_history[-mid_point:]
        older = self.coherence_history[:mid_point]
        
        recent_avg = sum(r["coherence"] for r in recent) / len(recent)
        older_avg = sum(o["coherence"] for o in older) / len(older)
        
        # Calculate drift
        drift = recent_avg - older_avg
        percent_change = (drift / older_avg) * 100 if older_avg > 0 else float('inf')
        
        # Determine if adjustment is needed
        needs_adjustment = abs(drift) >= self.adjustment_threshold
        
        return {
            "status": "analyzed",
            "drift": drift,
            "percent_change": percent_change,
            "recent_coherence": recent_avg,
            "older_coherence": older_avg,
            "needs_adjustment": needs_adjustment,
            "adjustment_direction": "increase" if drift < 0 else "decrease"
        }
        
    def suggest_routing_adjustments(
        self, 
        active_modules: List[ExpertModule]
    ) -> Dict[str, Any]:
        """
        Suggest adjustments to routing parameters based on coherence feedback.
        
        Args:
            active_modules: Currently active expert modules
            
        Returns:
            Dictionary with suggested adjustments
        """
        # Check for coherence drift
        drift_analysis = self.detect_coherence_drift()
        
        if drift_analysis.get("status") != "analyzed" or not drift_analysis.get("needs_adjustment"):
            return {"status": "no_adjustment_needed"}
            
        # Get current spectral state
        spectral_state = self.spectral_monitor.get_current_spectral_state()
        
        # Get introspection health
        introspection_health = self.introspection.get_system_health()
        
        # Determine adjustment direction and magnitude
        drift = drift_analysis["drift"]
        direction = drift_analysis["adjustment_direction"]
        magnitude = min(1.0, abs(drift) / self.adjustment_threshold)
        
        # Define adjustment actions
        adjustments = []
        
        if direction == "increase":  # Need to increase coherence
            # Suggest reducing active module count
            adjustments.append({
                "type": "reduce_active_modules",
                "description": "Reduce number of simultaneously active modules",
                "current": len(active_modules),
                "suggested": max(1, len(active_modules) - 1),
                "rationale": "Too many active modules may cause interference"
            })
            
            # Suggest increasing coherence threshold
            adjustments.append({
                "type": "increase_coherence_threshold",
                "description": "Increase required coherence between modules",
                "magnitude": magnitude * self.learning_rate,
                "rationale": "Higher threshold will ensure more compatible modules"
            })
            
            # Check if any domain is overrepresented
            domain_counts = defaultdict(int)
            for module in active_modules:
                domain_counts[module.domain] += 1
                
            # Identify most common domain
            if domain_counts:
                most_common = max(domain_counts.items(), key=lambda x: x[1])
                if most_common[1] > 1:  # If there's more than one module of same domain
                    adjustments.append({
                        "type": "reduce_domain_weight",
                        "description": f"Reduce prominence of '{most_common[0]}' domain",
                        "domain": most_common[0],
                        "current_count": most_common[1],
                        "rationale": "Domain saturation may reduce cognitive diversity"
                    })
        else:  # Need to decrease coherence (add more diversity)
            # Suggest increasing active module count
            adjustments.append({
                "type": "increase_active_modules",
                "description": "Increase number of simultaneously active modules",
                "current": len(active_modules),
                "suggested": min(8, len(active_modules) + 1),
                "rationale": "More cognitive diversity needed"
            })
            
            # Suggest decreasing coherence threshold
            adjustments.append({
                "type": "decrease_coherence_threshold",
                "description": "Decrease required coherence between modules",
                "magnitude": magnitude * self.learning_rate,
                "rationale": "Lower threshold will allow more diverse modules"
            })
            
            # Check if any additional domains could be included
            current_domains = {module.domain for module in active_modules}
            if len(current_domains) < 3:  # If we have fewer than 3 domains
                adjustments.append({
                    "type": "increase_domain_diversity",
                    "description": "Include more diverse domain experts",
                    "current_domains": list(current_domains),
                    "rationale": "Broaden cognitive coverage across domains"
                })
                
        # Record optimization suggestion
        self.optimization_history.append({
            "timestamp": datetime.now().isoformat(),
            "drift_analysis": drift_analysis,
            "spectral_state": spectral_state,
            "introspection_health": introspection_health,
            "suggested_adjustments": adjustments
        })
        
        return {
            "status": "adjustment_suggested",
            "adjustments": adjustments,
            "analysis": {
                "drift": drift_analysis,
                "spectral_state": spectral_state.get("status") == "success" and spectral_state or None,
                "health": introspection_health
            }
        }
        
    def apply_routing_adjustment(
        self, 
        adjustment_type: str,
        controller: SparseActivationController,
        magnitude: float = 0.1
    ) -> None:
        """
        Apply a suggested routing adjustment.
        
        Args:
            adjustment_type: Type of adjustment to apply
            controller: The SparseActivationController to adjust
            magnitude: Magnitude of the adjustment (0-1)
        """
        # Ensure magnitude is in [0, 1]
        magnitude = max(0.0, min(1.0, magnitude))
        
        # Apply the adjustment
        if adjustment_type == "increase_coherence_threshold":
            # Increase coherence threshold
            current = controller.coherence_threshold
            new_value = min(1.0, current + magnitude * 0.2)  # Max 20% increase
            controller.coherence_threshold = new_value
            logger.info(f"Increased coherence threshold: {current:.2f} -> {new_value:.2f}")
            
        elif adjustment_type == "decrease_coherence_threshold":
            # Decrease coherence threshold
            current = controller.coherence_threshold
            new_value = max(0.3, current - magnitude * 0.2)  # Min 0.3
            controller.coherence_threshold = new_value
            logger.info(f"Decreased coherence threshold: {current:.2f} -> {new_value:.2f}")
            
        elif adjustment_type == "increase_active_modules":
            # Increase max active modules
            current = controller.max_active_modules
            new_value = min(10, current + 1)
            controller.max_active_modules = new_value
            logger.info(f"Increased max active modules: {current} -> {new_value}")
            
        elif adjustment_type == "reduce_active_modules":
            # Reduce max active modules
            current = controller.max_active_modules
            new_value = max(1, current - 1)
            controller.max_active_modules = new_value
            logger.info(f"Reduced max active modules: {current} -> {new_value}")
            
        elif adjustment_type == "adjust_interference_penalty":
            # Adjust interference penalty
            current = controller.interference_penalty
            new_value = max(0.1, min(0.5, current + (magnitude * 0.2)))
            controller.interference_penalty = new_value
            logger.info(f"Adjusted interference penalty: {current:.2f} -> {new_value:.2f}")
            
        else:
            logger.warning(f"Unknown adjustment type: {adjustment_type}")


class ExpertRoutingManager:
    """
    Main class integrating all expert routing components.
    
    This provides a unified interface for routing cognitive processes
    through the appropriate expert subsystems and maintaining coherence.
    """
    
    def __init__(self, embedding_dim: int = 768):
        """
        Initialize the expert routing manager.
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        # Initialize components
        self.compatibility_scorer = ContextCompatibilityScorer()
        self.activation_controller = SparseActivationController()
        self.routing_optimizer = ConceptRoutingOptimizer()
        
        # Store expert modules
        self.modules = []
        
        # Track routing decisions
        self.routing_history = []
        
        # Set embedding dimension
        self.embedding_dim = embedding_dim
        
        logger.info("Expert routing manager initialized")
        
    def register_module(
        self, 
        module: ExpertModule
    ) -> None:
        """
        Register an expert module.
        
        Args:
            module: Module to register
        """
        # Check if module already exists
        if any(m.id == module.id for m in self.modules):
            logger.warning(f"Module with ID '{module.id}' already exists, replacing")
            # Remove existing module
            self.modules = [m for m in self.modules if m.id != module.id]
            
        # Add the module
        self.modules.append(module)
        logger.info(f"Registered expert module: {module.name} ({module.domain})")
        
    def register_modules(
        self, 
        modules: List[ExpertModule]
    ) -> None:
        """
        Register multiple expert modules.
        
        Args:
            modules: List of modules to register
        """
        for module in modules:
            self.register_module(module)
            
    def create_standard_modules(self) -> None:
        """Create a set of standard expert modules."""
        # Create some initial modules for different domains
        
        # Reasoning domain modules
        reasoning_embedding = np.random.randn(self.embedding_dim)
        reasoning_embedding = reasoning_embedding / np.linalg.norm(reasoning_embedding)
        self.register_module(ExpertModule(
            id="reasoning_formal",
            name="Formal Reasoning",
            domain="reasoning",
            description="Handles formal logical reasoning with strict rule application",
            embedding=reasoning_embedding,
            activation_threshold=0.5,
            energy_cost=1.2
        ))
        
        deductive_embedding = np.random.randn(self.embedding_dim)
        deductive_embedding = deductive_embedding / np.linalg.norm(deductive_embedding)
        self.register_module(ExpertModule(
            id="reasoning_deductive",
            name="Deductive Reasoning",
            domain="reasoning",
            description="Applies deductive logic to draw conclusions from premises",
            embedding=deductive_embedding,
            activation_threshold=0.45,
            energy_cost=1.0
        ))
        
        # Memory domain modules
        memory_embedding = np.random.randn(self.embedding_dim)
        memory_embedding = memory_embedding / np.linalg.norm(memory_embedding)
        self.register_module(ExpertModule(
            id="memory_episodic",
            name="Episodic Memory",
            domain="memory",
            description="Manages storage and retrieval of episode-like memory structures",
            embedding=memory_embedding,
            activation_threshold=0.4,
            energy_cost=0.8
        ))
        
        semantic_embedding = np.random.randn(self.embedding_dim)
        semantic_embedding = semantic_embedding / np.linalg.norm(semantic_embedding)
        self.register_module(ExpertModule(
            id="memory_semantic",
            name="Semantic Memory",
            domain="memory",
            description="Handles concept relationships and semantic networks",
            embedding=semantic_embedding,
            activation_threshold=0.5,
            energy_cost=0.9
        ))
        
        # Pattern recognition modules
        pattern_embedding = np.random.randn(self.embedding_dim)
        pattern_embedding = pattern_embedding / np.linalg.norm(pattern_embedding)
        self.register_module(ExpertModule(
            id="pattern_recognition",
            name="Pattern Recognition",
            domain="perception",
            description="Identifies recurring patterns and structures in data",
            embedding=pattern_embedding,
            activation_threshold=0.6,
            energy_cost=1.1
        ))
        
        # Koopman dynamics module
        koopman_embedding = np.random.randn(self.embedding_dim)
        koopman_embedding = koopman_embedding / np.linalg.norm(koopman_embedding)
        self.register_module(ExpertModule(
            id="koopman_dynamics",
            name="Koopman Mode Analysis",
            domain="dynamics",
            description="Analyzes system dynamics using Koopman operator theory",
            embedding=koopman_embedding,
            activation_threshold=0.7,
            energy_cost=1.5
        ))
        
        # Spectral analysis module
        spectral_embedding = np.random.randn(self.embedding_dim)
        spectral_embedding = spectral_embedding / np.linalg.norm(spectral_embedding)
        self.register_module(ExpertModule(
            id="spectral_analysis",
            name="Spectral Analysis",
            domain="dynamics",
            description="Performs frequency-domain analysis of cognitive states",
            embedding=spectral_embedding,
            activation_threshold=0.6,
            energy_cost=1.3
        ))
        
        # Phase synchronization module
        phase_embedding = np.random.randn(self.embedding_dim)
        phase_embedding = phase_embedding / np.linalg.norm(phase_embedding)
        self.register_module(ExpertModule(
            id="phase_sync",
            name="Phase Synchronization",
            domain="coherence",
            description="Maintains phase coherence across cognitive subsystems",
            embedding=phase_embedding,
            activation_threshold=0.6,
            energy_cost=1.0
        ))
        
        logger.info(f"Created {len(self.modules)} standard modules")
    
    def route_context(
        self, 
        context: RoutingContext
    ) -> RoutingDecision:
        """
        Route a cognitive context to the appropriate expert modules.
        
        Args:
            context: Current routing context
            
        Returns:
            Routing decision
        """
        # Compute compatibility scores
        compatibility_scores = self.compatibility_scorer.compute_module_compatibility(
            context, self.modules)
        
        # Select modules to activate
        selected_modules = self.activation_controller.select_modules(
            context, self.modules, compatibility_scores)
        
        # Calculate overall coherence
        coherence_score = self.activation_controller.compute_overall_coherence(selected_modules)
        
        # Calculate total energy cost
        energy_cost = sum(module.energy_cost for module in selected_modules)
        
        # Create routing decision
        decision = RoutingDecision(
            selected_modules=[module.id for module in selected_modules],
            compatibility_scores=compatibility_scores,
            coherence_score=coherence_score,
            energy_cost=energy_cost,
            reasoning=f"Selected {len(selected_modules)} modules with overall coherence {coherence_score:.2f}",
            metadata={
                "context": context.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Record decision in history
        self.routing_history.append(decision)
        
        # Record coherence for optimization
        self.routing_optimizer.record_coherence_measurement(coherence_score, decision)
        
        # Update activation counts
        for module in selected_modules:
            module.activation_count += 1
            
        return decision
    
    def create_routing_context(
        self, 
        query_embedding: np.ndarray,
        active_concepts: Optional[List[str]] = None,
        task_domain: str = "general",
        task_phase: str = "processing",
        desired_module_count: int = 3
    ) -> RoutingContext:
        """
        Create a routing context from a query embedding.
        
        Args:
            query_embedding: Embedding of the query/context
            active_concepts: Optional list of currently active concept IDs
            task_domain: Domain of the current task
            task_phase: Current phase of the task
            desired_module_count: Number of modules to activate
            
        Returns:
            New RoutingContext
        """
        return RoutingContext(
            query_embedding=query_embedding,
            active_concepts=active_concepts or [],
            task_domain=task_domain,
            task_phase=task_phase,
            desired_module_count=desired_module_count
        )
    
    def optimize_routing(self) -> Dict[str, Any]:
        """
        Optimize routing parameters based on coherence feedback.
        
        Returns:
            Dictionary with optimization results
        """
        # Get all currently active modules from the most recent routing decision
        if not self.routing_history:
            return {"status": "no_routing_history"}
            
        active_module_ids = self.routing_history[-1].selected_modules
        active_modules = [m for m in self.modules if m.id in active_module_ids]
        
        # Get adjustment suggestions
        adjustments = self.routing_optimizer.suggest_routing_adjustments(active_modules)
        
        if adjustments.get("status") != "adjustment_suggested":
            return adjustments
            
        # Apply top adjustment
        if adjustments.get("adjustments"):
            top_adjustment = adjustments["adjustments"][0]
            self.routing_optimizer.apply_routing_adjustment(
                top_adjustment["type"],
                self.activation_controller
            )
            
            # Record application
            adjustments["applied"] = top_adjustment["type"]
            
        return adjustments
    
    def get_module_by_id(self, module_id: str) -> Optional[ExpertModule]:
        """
        Get a module by its ID.
        
        Args:
            module_id: ID of the module to retrieve
            
        Returns:
            ExpertModule if found, None otherwise
        """
        for module in self.modules:
            if module.id == module_id:
                return module
        return None
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about routing decisions.
        
        Returns:
            Dictionary with routing statistics
        """
        if not self.routing_history:
            return {"status": "no_routing_history"}
            
        # Modules by activation count
        module_activations = {}
        for module in self.modules:
            module_activations[module.id] = module.activation_count
            
        # Recent coherence trend
        coherence_values = [d.coherence_score for d in self.routing_history[-10:]]
        avg_coherence = sum(coherence_values) / len(coherence_values) if coherence_values else 0
        
        # Domain distribution
        domain_counts = defaultdict(int)
        for decision in self.routing_history[-20:]:
            for module_id in decision.selected_modules:
                module = self.get_module_by_id(module_id)
                if module:
                    domain_counts[module.domain] += 1
                    
        # Average energy cost
        energy_costs = [d.energy_cost for d in self.routing_history[-10:]]
        avg_energy = sum(energy_costs) / len(energy_costs) if energy_costs else 0
        
        return {
            "status": "success",
            "total_routing_decisions": len(self.routing_history),
            "module_activations": module_activations,
            "average_coherence": avg_coherence,
            "domain_distribution": dict(domain_counts),
            "average_energy_cost": avg_energy
        }


# Singleton instance for easy access
_expert_routing_manager = None

def get_expert_routing_manager(embedding_dim: int = 768) -> ExpertRoutingManager:
    """
    Get or create the singleton expert routing manager.
    
    Args:
        embedding_dim: Dimension of embeddings
        
    Returns:
        ExpertRoutingManager instance
    """
    global _expert_routing_manager
    if _expert_routing_manager is None:
        _expert_routing_manager = ExpertRoutingManager(embedding_dim)
        # Create standard modules
        _expert_routing_manager.create_standard_modules()
    return _expert_routing_manager
