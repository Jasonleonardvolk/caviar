#!/usr/bin/env python3
"""
Chaos Control Layer (CCL) Integration Harness
Orchestrates dark-soliton furnaces, topological waveguides, and Lyapunov pumps
"""

import asyncio
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import json
import logging
from pathlib import Path

# Import CCL components
from .lyap_pump import LyapunovGatedPump, LyapunovEstimator
# Note: furnace_kernel would be imported from compiled bindings
# from .bindings import DarkSolitonFurnace

logger = logging.getLogger(__name__)

@dataclass
class CCLConfig:
    """CCL configuration parameters"""
    lattice_config_path: str = "ccl/alpha_lattice_geom.json"
    max_lyapunov: float = 0.05
    target_lyapunov: float = 0.02
    kerr_coefficient: float = -1.0  # Negative for dark solitons
    dispersion: float = -0.5  # Anomalous dispersion
    energy_threshold: int = 100
    safety_margin: float = 0.1

class ChaosControlLayer:
    """
    Main CCL orchestrator - manages controlled chaos generation
    """
    
    def __init__(self, 
                 eigen_sentry,
                 energy_broker,
                 topo_switch,
                 config: Optional[CCLConfig] = None):
        
        self.eigen_sentry = eigen_sentry
        self.energy_broker = energy_broker
        self.topo_switch = topo_switch
        self.config = config or CCLConfig()
        
        # Load lattice geometry
        self.lattice = self._load_lattice_geometry()
        
        # Initialize components
        self.lyap_estimator = LyapunovEstimator(
            dim=self.lattice['total_sites']
        )
        self.lyap_pump = LyapunovGatedPump(
            target_lyapunov=self.config.target_lyapunov,
            lambda_threshold=self.config.max_lyapunov
        )
        
        # Dark soliton furnace (placeholder for C++ binding)
        self.furnace = None  # Would be: DarkSolitonFurnace(...)
        
        # State tracking
        self.active_sessions: Dict[str, Any] = {}
        self.total_chaos_generated = 0.0
        self.running = False
        
    def _load_lattice_geometry(self) -> dict:
        """Load alpha lattice geometry from JSON"""
        path = Path(self.config.lattice_config_path)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        else:
            # Default geometry
            return {
                'lattice_type': 'kagome_2d',
                'total_sites': 1200,
                'system_size': [20, 20]
            }
            
    async def start(self):
        """Start CCL background processes"""
        self.running = True
        logger.info("Chaos Control Layer started")
        
        # Start monitoring loop
        asyncio.create_task(self._monitor_loop())
        
    async def stop(self):
        """Stop CCL"""
        self.running = False
        
        # Close all active sessions
        for session_id in list(self.active_sessions.keys()):
            await self.exit_chaos_session(session_id)
            
        logger.info("Chaos Control Layer stopped")
        
    async def enter_chaos_session(self, 
                                 module_id: str, 
                                 purpose: str,
                                 required_energy: int) -> Optional[str]:
        """
        Enter a controlled chaos session
        
        Returns:
            Session ID if successful, None otherwise
        """
        # Request CCL entry through topological switch
        gate_id = await self.topo_switch.enter_ccl(module_id, required_energy)
        if not gate_id:
            return None
            
        # Create session
        session_id = f"chaos_{module_id}_{len(self.active_sessions)}"
        
        # Initialize dark soliton state
        initial_state = self._create_dark_soliton_seed()
        
        self.active_sessions[session_id] = {
            'module_id': module_id,
            'gate_id': gate_id,
            'purpose': purpose,
            'energy_allocated': required_energy,
            'energy_used': 0,
            'state': initial_state,
            'lyapunov_history': [],
            'start_time': asyncio.get_event_loop().time()
        }
        
        logger.info(f"Chaos session started: {session_id} for {purpose}")
        return session_id
        
    async def exit_chaos_session(self, session_id: str) -> Dict[str, Any]:
        """Exit chaos session and return results"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")
            
        session = self.active_sessions[session_id]
        
        # Calculate unused energy
        unused_energy = session['energy_allocated'] - session['energy_used']
        
        # Exit through topological gate
        await self.topo_switch.exit_ccl(session['gate_id'], unused_energy)
        
        # Extract results
        results = {
            'chaos_generated': session['energy_used'],
            'final_state': session['state'],
            'lyapunov_trajectory': session['lyapunov_history'],
            'duration': asyncio.get_event_loop().time() - session['start_time']
        }
        
        # Clean up
        del self.active_sessions[session_id]
        
        logger.info(f"Chaos session ended: {session_id}")
        return results
        
    async def evolve_chaos(self, 
                          session_id: str, 
                          steps: int = 10) -> np.ndarray:
        """
        Evolve chaos dynamics for specified steps
        
        Returns:
            Evolved state vector
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session ID: {session_id}")
            
        session = self.active_sessions[session_id]
        state = session['state']
        
        for step in range(steps):
            # Check energy budget
            if session['energy_used'] >= session['energy_allocated']:
                logger.warning(f"Energy budget exhausted for {session_id}")
                break
                
            # Evolve dynamics (placeholder for actual furnace evolution)
            state, jacobian = self._evolve_dark_soliton(state)
            
            # Update Lyapunov estimate
            spectrum = self.lyap_estimator.update(jacobian)
            max_lyap = np.max(spectrum)
            session['lyapunov_history'].append(max_lyap)
            
            # Apply Lyapunov-gated feedback
            gain = self.lyap_pump.compute_gain(max_lyap)
            state = self.lyap_pump.apply_feedback(state, gain)
            
            # Track energy usage
            session['energy_used'] += np.sum(np.abs(state)**2) * 0.01
            
        session['state'] = state
        self.total_chaos_generated += session['energy_used']
        
        return state
        
    def _create_dark_soliton_seed(self) -> np.ndarray:
        """Create initial dark soliton state"""
        x = np.linspace(-10, 10, self.lattice['total_sites'])
        
        # Dark soliton: tanh profile with phase jump
        amplitude = np.tanh(x)
        phase = np.pi * (1 + np.sign(x)) / 2
        
        state = amplitude * np.exp(1j * phase)
        return state
        
    def _evolve_dark_soliton(self, 
                            state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve dark soliton dynamics (placeholder for C++ kernel)
        
        Returns:
            (evolved_state, jacobian)
        """
        # Simplified evolution for demonstration
        # In production: would call self.furnace.evolve()
        
        # Nonlinear Schrödinger-like evolution
        dt = 0.01
        N = len(state)
        
        # Laplacian (discrete)
        laplacian = np.zeros_like(state)
        laplacian[1:-1] = state[2:] - 2*state[1:-1] + state[:-2]
        
        # Nonlinear term
        nonlinear = self.config.kerr_coefficient * np.abs(state)**2 * state
        
        # Evolution
        state_new = state + 1j * dt * (
            self.config.dispersion * laplacian + nonlinear
        )
        
        # Approximate Jacobian
        jacobian = np.eye(N) * (1 + dt * self.config.kerr_coefficient)
        
        return state_new, jacobian
        
    async def _monitor_loop(self):
        """Background monitoring of active chaos sessions"""
        while self.running:
            for session_id, session in self.active_sessions.items():
                # Check Lyapunov history
                if session['lyapunov_history']:
                    recent_lyap = np.mean(session['lyapunov_history'][-10:])
                    
                    # Emergency brake if chaos too high
                    if recent_lyap > self.config.max_lyapunov * 1.5:
                        logger.error(f"Emergency brake for {session_id}: λ={recent_lyap}")
                        await self.exit_chaos_session(session_id)
                        
            await asyncio.sleep(1.0)
            
    def get_status(self) -> Dict[str, Any]:
        """Get CCL status"""
        return {
            'running': self.running,
            'active_sessions': len(self.active_sessions),
            'total_chaos_generated': self.total_chaos_generated,
            'lattice_info': {
                'type': self.lattice['lattice_type'],
                'sites': self.lattice['total_sites']
            },
            'pump_status': self.lyap_pump.get_status()
        }
        
# Test function
async def test_ccl():
    """Test the Chaos Control Layer"""
    print("Testing Chaos Control Layer")
    print("=" * 50)
    
    # Mock dependencies
    class MockEigenSentry:
        pass
        
    class MockEnergyBroker:
        def request(self, module_id, energy, purpose):
            return True
        def refund(self, module_id, energy):
            pass
            
    class MockTopoSwitch:
        async def enter_ccl(self, module_id, energy):
            return f"gate_{module_id}"
        async def exit_ccl(self, gate_id, unused):
            pass
            
    # Create CCL
    ccl = ChaosControlLayer(
        eigen_sentry=MockEigenSentry(),
        energy_broker=MockEnergyBroker(),
        topo_switch=MockTopoSwitch()
    )
    
    await ccl.start()
    
    # Start chaos session
    session_id = await ccl.enter_chaos_session(
        module_id="test_module",
        purpose="exploration",
        required_energy=200
    )
    
    print(f"Started session: {session_id}")
    
    # Evolve chaos
    for i in range(5):
        state = await ccl.evolve_chaos(session_id, steps=10)
        print(f"Evolution {i}: energy={np.sum(np.abs(state)**2):.2f}")
        
    # Get results
    results = await ccl.exit_chaos_session(session_id)
    print(f"\nResults: chaos_generated={results['chaos_generated']:.2f}")
    print(f"Duration: {results['duration']:.2f}s")
    
    # Status
    print(f"\nFinal status: {ccl.get_status()}")
    
    await ccl.stop()
    
if __name__ == "__main__":
    asyncio.run(test_ccl())
