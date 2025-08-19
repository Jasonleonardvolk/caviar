"""
ALAN-ELFIN Stability Integration

This module provides integration between ALAN's concept network and
the ELFIN stability framework, enabling real-time monitoring and
verification of stability properties in ALAN's cognitive architecture.
"""

import os
import sys
import time
import logging
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path

# Ensure parent directory is in path
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    # ELFIN stability imports
    from alan_backend.elfin.stability.lyapunov import (
        LyapunovFunction, PolynomialLyapunov, NeuralLyapunov, 
        CLFFunction, CompositeLyapunov
    )
    from alan_backend.elfin.stability.jit_guard import StabilityGuard
    from alan_backend.elfin.stability.incremental_verifier import (
        ProofCache, ParallelVerifier, LyapunovVerifier
    )
    from alan_backend.elfin.visualization.dashboard import DashboardServer
    
    # ALAN imports
    from alan_backend.core.concept_network import ConceptNetwork
    from alan_backend.core.concept import Concept
except ImportError:
    print("Error: Required modules not found.")
    print("Please ensure both ALAN and ELFIN frameworks are installed.")
    
    # Minimal implementations for standalone testing
    class LyapunovFunction:
        def __init__(self, name):
            self.name = name
            self.domain_ids = []
    
    class PolynomialLyapunov(LyapunovFunction):
        def __init__(self, name, Q, domain_ids=None):
            super().__init__(name)
            self.Q = Q
            self.domain_ids = domain_ids or []
    
    class StabilityGuard:
        def __init__(self, lyap, threshold=0, callback=None):
            self.lyap = lyap
            self.threshold = threshold
            self.callback = callback
            self.violations = 0
    
    class Concept:
        def __init__(self, id, name):
            self.id = id
            self.name = name
            self.state = {}
    
    class ConceptNetwork:
        def __init__(self):
            self.concepts = {}


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StabilityStatus:
    """Status of concept stability."""
    
    UNKNOWN = "unknown"
    STABLE = "stable"
    UNSTABLE = "unstable"
    VERIFIED = "verified"
    VIOLATED = "violated"


class ConceptStabilityMonitor:
    """
    Monitor stability of ALAN concepts using ELFIN stability framework.
    
    This class provides integration between ALAN's concept network and
    the ELFIN stability framework, enabling real-time monitoring and
    verification of stability properties in ALAN's cognitive architecture.
    """
    
    def __init__(
        self,
        concept_network: Optional[ConceptNetwork] = None,
        dashboard_host: str = "localhost",
        dashboard_port: int = 5000,
        enable_dashboard: bool = True
    ):
        """
        Initialize concept stability monitor.
        
        Args:
            concept_network: ALAN concept network
            dashboard_host: Host for dashboard server
            dashboard_port: Port for dashboard server
            enable_dashboard: Whether to enable the dashboard
        """
        self.concept_network = concept_network
        
        # Stability components
        self.lyapunov_functions = {}
        self.stability_guards = {}
        self.concept_states = {}
        self.concept_stability = {}
        
        # Verification components
        self.verifier = LyapunovVerifier()
        self.proof_cache = ProofCache()
        self.parallel_verifier = ParallelVerifier(
            verifier=self.verifier,
            cache=self.proof_cache
        )
        
        # Dashboard
        self.dashboard = None
        self.dashboard_host = dashboard_host
        self.dashboard_port = dashboard_port
        
        if enable_dashboard:
            self.init_dashboard()
        
        # Monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        self.monitoring_interval = 0.1  # seconds
        
        # Callbacks
        self.stability_callbacks = []
    
    def init_dashboard(self):
        """Initialize dashboard server."""
        try:
            from alan_backend.elfin.visualization.dashboard import (
                DashboardServer, create_dashboard_files
            )
            
            # Create dashboard files
            create_dashboard_files()
            
            # Create dashboard server
            self.dashboard = DashboardServer(
                host=self.dashboard_host,
                port=self.dashboard_port
            )
            
            # Start dashboard server
            self.dashboard.start(debug=False)
            
            logger.info(
                f"Dashboard server started at "
                f"http://{self.dashboard_host}:{self.dashboard_port}"
            )
            
        except ImportError:
            logger.warning(
                "Dashboard not available. "
                "Please ensure visualization components are installed."
            )
    
    def register_concept(
        self,
        concept: Union[Concept, str],
        state_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Register concept for stability monitoring.
        
        Args:
            concept: Concept or concept ID
            state_mapping: Mapping of concept state keys to state vector indices
        """
        # Get concept
        if isinstance(concept, str):
            if self.concept_network is None:
                raise ValueError("Concept network not provided")
            
            concept_id = concept
            concept = self.concept_network.get_concept(concept_id)
            
            if concept is None:
                raise ValueError(f"Concept not found: {concept_id}")
        else:
            concept_id = concept.id
        
        # Default state mapping
        if state_mapping is None:
            # Use all numeric state values as state vector
            state_mapping = {}
            
            # Initialize with empty state
            if hasattr(concept, "state") and isinstance(concept.state, dict):
                for key, value in concept.state.items():
                    if isinstance(value, (int, float)):
                        state_mapping[key] = len(state_mapping)
        
        # Register concept
        self.concept_states[concept_id] = {
            "concept": concept,
            "state_mapping": state_mapping,
            "current_state": np.zeros(len(state_mapping)),
            "prev_state": np.zeros(len(state_mapping)),
            "history": []
        }
        
        self.concept_stability[concept_id] = {
            "status": StabilityStatus.UNKNOWN,
            "lyapunov_value": 0.0,
            "violations": 0
        }
        
        logger.info(f"Registered concept for stability monitoring: {concept_id}")
        
        # Create default Lyapunov function for concept
        self.create_default_lyapunov(concept_id)
    
    def create_default_lyapunov(self, concept_id: str):
        """
        Create default Lyapunov function for concept.
        
        Args:
            concept_id: Concept ID
        """
        # Get concept state
        concept_state = self.concept_states.get(concept_id)
        if concept_state is None:
            raise ValueError(f"Concept not registered: {concept_id}")
        
        # Get state dimension
        state_dim = len(concept_state["state_mapping"])
        
        if state_dim == 0:
            logger.warning(
                f"Concept has no numeric state values, "
                f"cannot create Lyapunov function: {concept_id}"
            )
            return
        
        # Create default Lyapunov function (quadratic)
        lyap_name = f"V_{concept_id}"
        
        # Use identity matrix for Q (simple quadratic Lyapunov function)
        Q = np.eye(state_dim)
        
        lyap = PolynomialLyapunov(
            name=lyap_name,
            Q=Q,
            domain_ids=[concept_id]
        )
        
        # Register Lyapunov function
        self.register_lyapunov_function(lyap)
        
        # Create stability guard
        guard = StabilityGuard(
            lyap=lyap,
            threshold=0.0,
            callback=self._stability_violation_callback
        )
        
        # Register stability guard
        self.register_stability_guard(guard, concept_id)
    
    def register_lyapunov_function(self, lyap: LyapunovFunction):
        """
        Register Lyapunov function for stability monitoring.
        
        Args:
            lyap: Lyapunov function
        """
        lyap_name = getattr(lyap, "name", str(lyap))
        self.lyapunov_functions[lyap_name] = lyap
        
        # Register with dashboard if available
        if self.dashboard:
            self.dashboard.register_lyapunov_function(lyap)
        
        logger.info(f"Registered Lyapunov function: {lyap_name}")
    
    def register_stability_guard(
        self,
        guard: StabilityGuard,
        concept_id: Optional[str] = None
    ):
        """
        Register stability guard for concept.
        
        Args:
            guard: Stability guard
            concept_id: Concept ID (if different from Lyapunov domain)
        """
        # Get Lyapunov function domain
        if concept_id is None and hasattr(guard.lyap, "domain_ids"):
            if guard.lyap.domain_ids:
                concept_id = guard.lyap.domain_ids[0]
        
        if concept_id is None:
            concept_id = str(guard.lyap)
        
        # Register guard
        self.stability_guards[concept_id] = guard
        
        # Register with dashboard if available
        if self.dashboard:
            self.dashboard.register_stability_guard(guard, concept_id)
        
        logger.info(f"Registered stability guard for concept: {concept_id}")
    
    def register_stability_callback(
        self,
        callback: Callable[[str, Dict[str, Any]], None]
    ):
        """
        Register callback for stability status changes.
        
        Args:
            callback: Callback function
        """
        self.stability_callbacks.append(callback)
    
    def _stability_violation_callback(self, x_prev, x, guard):
        """
        Callback for stability violations.
        
        Args:
            x_prev: Previous state
            x: Current state
            guard: Stability guard
        """
        # Find concept for this guard
        concept_id = None
        for cid, g in self.stability_guards.items():
            if g == guard:
                concept_id = cid
                break
        
        if concept_id is None:
            logger.warning("Stability violation detected for unknown guard")
            return
        
        # Update concept stability status
        if concept_id in self.concept_stability:
            self.concept_stability[concept_id]["status"] = StabilityStatus.UNSTABLE
            self.concept_stability[concept_id]["violations"] += 1
        
        # Record violation in dashboard
        if self.dashboard:
            self.dashboard.record_stability_violation(concept_id, x_prev, x)
        
        # Log violation
        logger.warning(f"Stability violation detected for concept: {concept_id}")
        logger.warning(f"  Previous state: {x_prev}")
        logger.warning(f"  Current state: {x}")
        logger.warning(f"  Total violations: {guard.violations}")
        
        # Call stability callbacks
        for callback in self.stability_callbacks:
            try:
                callback(concept_id, {
                    "status": StabilityStatus.UNSTABLE,
                    "prev_state": x_prev,
                    "current_state": x,
                    "violations": guard.violations
                })
            except Exception as e:
                logger.error(f"Error in stability callback: {e}")
    
    def update_concept_state(self, concept_id: str):
        """
        Update concept state from concept network.
        
        Args:
            concept_id: Concept ID
        """
        # Get concept state
        concept_state = self.concept_states.get(concept_id)
        if concept_state is None:
            return
        
        # Get concept
        concept = concept_state["concept"]
        if concept is None:
            return
        
        # Check if concept network is available (for fresh concept state)
        if self.concept_network:
            concept = self.concept_network.get_concept(concept_id)
            if concept is None:
                return
        
        # Get state mapping
        state_mapping = concept_state["state_mapping"]
        
        # Update previous state
        concept_state["prev_state"] = concept_state["current_state"].copy()
        
        # Extract state vector from concept state
        state_vector = np.zeros(len(state_mapping))
        
        if hasattr(concept, "state") and isinstance(concept.state, dict):
            for key, idx in state_mapping.items():
                if key in concept.state:
                    value = concept.state[key]
                    if isinstance(value, (int, float)):
                        state_vector[idx] = value
        
        # Update current state
        concept_state["current_state"] = state_vector
        
        # Add to history (limit to 1000 points)
        concept_state["history"].append({
            "time": time.time(),
            "state": state_vector.copy()
        })
        
        if len(concept_state["history"]) > 1000:
            concept_state["history"] = concept_state["history"][-1000:]
        
        # Update dashboard
        if self.dashboard:
            self.dashboard.update_system_state(state_vector, concept_id)
        
        # Check stability
        guard = self.stability_guards.get(concept_id)
        if guard:
            prev_state = concept_state["prev_state"]
            current_state = concept_state["current_state"]
            
            is_stable = guard.step(prev_state, current_state)
            
            # Update stability status
            if is_stable:
                self.concept_stability[concept_id]["status"] = StabilityStatus.STABLE
            else:
                self.concept_stability[concept_id]["status"] = StabilityStatus.UNSTABLE
            
            # Update Lyapunov value
            if hasattr(guard.lyap, "evaluate"):
                lyap_value = guard.lyap.evaluate(current_state)
                self.concept_stability[concept_id]["lyapunov_value"] = lyap_value
    
    def update_all_concepts(self):
        """Update all registered concept states."""
        for concept_id in self.concept_states:
            self.update_concept_state(concept_id)
    
    def verify_concept_stability(self, concept_id: str):
        """
        Verify stability of concept using formal methods.
        
        Args:
            concept_id: Concept ID
            
        Returns:
            Verification result
        """
        # Get concept state
        concept_state = self.concept_states.get(concept_id)
        if concept_state is None:
            raise ValueError(f"Concept not registered: {concept_id}")
        
        # Get Lyapunov function
        lyap = None
        lyap_name = f"V_{concept_id}"
        
        if lyap_name in self.lyapunov_functions:
            lyap = self.lyapunov_functions[lyap_name]
        else:
            # Try to find a Lyapunov function for this concept
            for name, function in self.lyapunov_functions.items():
                if hasattr(function, "domain_ids") and concept_id in function.domain_ids:
                    lyap = function
                    break
        
        if lyap is None:
            raise ValueError(f"No Lyapunov function for concept: {concept_id}")
        
        # Create dynamics function
        # This would typically come from the concept's dynamics model
        # Here we use a simple linear model for demonstration
        
        # Extract state dimension
        state_dim = len(concept_state["state_mapping"])
        
        # Create simple linear dynamics model (stable)
        A = np.eye(state_dim) * 0.9  # Stable linear system (eigenvalues < 1)
        
        def linear_dynamics(x):
            return A @ x
        
        # Verify stability
        logger.info(f"Verifying stability for concept: {concept_id}")
        
        result = self.parallel_verifier.verify(lyap, linear_dynamics)
        
        # Update concept stability status
        if result.status.name == "VERIFIED":
            self.concept_stability[concept_id]["status"] = StabilityStatus.VERIFIED
        elif result.status.name == "REFUTED":
            self.concept_stability[concept_id]["status"] = StabilityStatus.VIOLATED
        
        return result
    
    def verify_all_concepts(self):
        """Verify stability of all registered concepts."""
        results = {}
        
        for concept_id in self.concept_states:
            try:
                result = self.verify_concept_stability(concept_id)
                results[concept_id] = result
            except Exception as e:
                logger.error(f"Error verifying concept stability: {e}")
                results[concept_id] = None
        
        return results
    
    def start_monitoring(self):
        """Start concept stability monitoring thread."""
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread already running")
            return
        
        self.monitoring_active = True
        
        # Create monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        
        # Start thread
        self.monitoring_thread.start()
        
        logger.info("Started concept stability monitoring")
    
    def stop_monitoring(self):
        """Stop concept stability monitoring thread."""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        self.monitoring_thread = None
        
        logger.info("Stopped concept stability monitoring")
    
    def _monitoring_loop(self):
        """Monitoring thread loop."""
        logger.info("Concept stability monitoring loop started")
        
        while self.monitoring_active:
            try:
                # Update all concept states
                self.update_all_concepts()
                
                # Sleep for interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in concept stability monitoring: {e}")
                time.sleep(1.0)
    
    def get_concept_stability(self, concept_id: str) -> Dict[str, Any]:
        """
        Get stability status of concept.
        
        Args:
            concept_id: Concept ID
            
        Returns:
            Stability status
        """
        if concept_id not in self.concept_stability:
            raise ValueError(f"Concept not registered: {concept_id}")
        
        return self.concept_stability[concept_id]
    
    def get_all_concept_stability(self) -> Dict[str, Dict[str, Any]]:
        """
        Get stability status of all concepts.
        
        Returns:
            Dictionary of concept ID to stability status
        """
        return self.concept_stability
    
    def close(self):
        """Close monitor and release resources."""
        # Stop monitoring
        self.stop_monitoring()
        
        # Stop dashboard
        if self.dashboard:
            self.dashboard.stop()
        
        logger.info("Concept stability monitor closed")


def demo():
    """Run a demonstration of concept stability monitoring."""
    # Create a mock concept network
    concept_network = ConceptNetwork()
    
    # Create some mock concepts
    concept1 = Concept("concept1", "Pendulum")
    concept1.state = {"angle": 0.1, "velocity": 0.0}
    
    concept2 = Concept("concept2", "Cart-Pole")
    concept2.state = {"position": 0.0, "angle": 0.1, "velocity": 0.0, "angular_velocity": 0.0}
    
    # Add concepts to network
    concept_network.concepts = {
        "concept1": concept1,
        "concept2": concept2
    }
    
    # Create concept stability monitor
    monitor = ConceptStabilityMonitor(
        concept_network=concept_network,
        enable_dashboard=True
    )
    
    # Register concepts
    monitor.register_concept("concept1")
    monitor.register_concept("concept2")
    
    # Register callback
    def stability_callback(concept_id, status):
        print(f"Stability status changed for {concept_id}: {status}")
    
    monitor.register_stability_callback(stability_callback)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulation loop
    try:
        print("Running concept stability monitoring demo...")
        print("Press Ctrl+C to exit.")
        
        for i in range(100):
            # Update concept states with some dynamics
            time.sleep(0.1)
            
            # Pendulum dynamics
            angle = concept1.state["angle"]
            velocity = concept1.state["velocity"]
            
            # Simple pendulum dynamics
            angle_new = angle + 0.1 * velocity
            velocity_new = velocity - 0.1 * np.sin(angle)
            
            concept1.state["angle"] = angle_new
            concept1.state["velocity"] = velocity_new
            
            # Cart-pole dynamics
            position = concept2.state["position"]
            angle = concept2.state["angle"]
            velocity = concept2.state["velocity"]
            angular_velocity = concept2.state["angular_velocity"]
            
            # Simple cart-pole dynamics
            position_new = position + 0.1 * velocity
            angle_new = angle + 0.1 * angular_velocity
            velocity_new = velocity
            angular_velocity_new = angular_velocity - 0.2 * np.sin(angle)
            
            concept2.state["position"] = position_new
            concept2.state["angle"] = angle_new
            concept2.state["velocity"] = velocity_new
            concept2.state["angular_velocity"] = angular_velocity_new
            
            # Print status every 10 steps
            if i % 10 == 0:
                stability = monitor.get_all_concept_stability()
                
                print(f"Step {i}:")
                for concept_id, status in stability.items():
                    print(f"  {concept_id}: {status}")
        
        # Verify stability of concepts
        print("\nVerifying concept stability...")
        results = monitor.verify_all_concepts()
        
        for concept_id, result in results.items():
            print(f"  {concept_id}: {result.status.name if result else 'ERROR'}")
        
        # Keep monitoring active for dashboard
        print("\nContinuing stability monitoring for dashboard...")
        print("Press Ctrl+C to exit.")
        
        while True:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up
        monitor.close()


if __name__ == "__main__":
    demo()
