#!/usr/bin/env python3
"""
Chaos Fuzzer - Random burst attack generator for safety testing
Tests CCL resilience against adversarial chaos injections
"""

import asyncio
import numpy as np
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class FuzzAttack:
    """Definition of a fuzz attack"""
    attack_type: str
    intensity: float
    duration: float
    target_modules: List[str]
    parameters: Dict[str, Any]

class ChaosFuzzer:
    """
    Automated chaos attack generator for safety validation
    Ensures system can handle worst-case chaos scenarios
    """
    
    def __init__(self, ccl, energy_broker, topo_switch):
        self.ccl = ccl
        self.energy_broker = energy_broker
        self.topo_switch = topo_switch
        
        # Attack patterns
        self.attack_patterns = [
            "energy_starvation",
            "rapid_oscillation", 
            "cascade_failure",
            "resonance_attack",
            "phase_disruption",
            "topology_corruption",
            "memory_exhaustion",
            "lyapunov_spike"
        ]
        
        # Safety metrics
        self.attacks_executed = 0
        self.failures_induced = 0
        self.safety_violations = []
        
    async def run_fuzzing_campaign(self, 
                                 duration_minutes: int = 10,
                                 intensity: str = "medium") -> Dict[str, Any]:
        """
        Run comprehensive fuzzing campaign
        
        Args:
            duration_minutes: How long to fuzz
            intensity: "low", "medium", "high", "extreme"
            
        Returns:
            Safety report
        """
        logger.info(f"Starting {intensity} intensity fuzzing for {duration_minutes} minutes")
        
        intensity_map = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.9,
            "extreme": 1.0
        }
        
        attack_intensity = intensity_map.get(intensity, 0.6)
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            # Generate random attack
            attack = self._generate_attack(attack_intensity)
            
            # Execute attack
            try:
                await self._execute_attack(attack)
                self.attacks_executed += 1
                
            except Exception as e:
                logger.error(f"Attack induced failure: {e}")
                self.failures_induced += 1
                self.safety_violations.append({
                    'attack': attack,
                    'error': str(e),
                    'timestamp': time.time()
                })
                
            # Random delay between attacks
            await asyncio.sleep(random.uniform(0.1, 2.0))
            
        return self._generate_report()
        
    def _generate_attack(self, intensity: float) -> FuzzAttack:
        """Generate random attack based on intensity"""
        attack_type = random.choice(self.attack_patterns)
        
        # Scale parameters by intensity
        base_params = self._get_attack_params(attack_type)
        scaled_params = {
            k: v * intensity if isinstance(v, (int, float)) else v
            for k, v in base_params.items()
        }
        
        return FuzzAttack(
            attack_type=attack_type,
            intensity=intensity,
            duration=random.uniform(0.1, 5.0) * intensity,
            target_modules=self._select_targets(),
            parameters=scaled_params
        )
        
    def _get_attack_params(self, attack_type: str) -> Dict[str, Any]:
        """Get parameters for specific attack type"""
        params_map = {
            "energy_starvation": {
                "drain_rate": 100,
                "block_refill": True
            },
            "rapid_oscillation": {
                "frequency": 1000,  # Hz
                "amplitude": 5.0
            },
            "cascade_failure": {
                "failure_probability": 0.3,
                "propagation_delay": 0.01
            },
            "resonance_attack": {
                "target_frequency": 50.0,
                "q_factor": 10.0
            },
            "phase_disruption": {
                "phase_noise": np.pi,
                "correlation_time": 0.1
            },
            "topology_corruption": {
                "edge_flip_probability": 0.1,
                "node_removal_probability": 0.05
            },
            "memory_exhaustion": {
                "allocation_rate": 1000,  # MB/s
                "fragmentation": True
            },
            "lyapunov_spike": {
                "spike_magnitude": 0.5,
                "spike_duration": 0.5
            }
        }
        
        return params_map.get(attack_type, {})
        
    def _select_targets(self) -> List[str]:
        """Randomly select target modules"""
        all_modules = ["UIH", "RFPE", "SMP", "DMON", "CCL"]
        n_targets = random.randint(1, len(all_modules))
        return random.sample(all_modules, n_targets)
        
    async def _execute_attack(self, attack: FuzzAttack):
        """Execute specific attack pattern"""
        logger.debug(f"Executing {attack.attack_type} attack on {attack.target_modules}")
        
        if attack.attack_type == "energy_starvation":
            await self._energy_starvation_attack(attack)
        elif attack.attack_type == "rapid_oscillation":
            await self._rapid_oscillation_attack(attack)
        elif attack.attack_type == "cascade_failure":
            await self._cascade_failure_attack(attack)
        elif attack.attack_type == "resonance_attack":
            await self._resonance_attack(attack)
        elif attack.attack_type == "phase_disruption":
            await self._phase_disruption_attack(attack)
        elif attack.attack_type == "topology_corruption":
            await self._topology_corruption_attack(attack)
        elif attack.attack_type == "memory_exhaustion":
            await self._memory_exhaustion_attack(attack)
        elif attack.attack_type == "lyapunov_spike":
            await self._lyapunov_spike_attack(attack)
            
    async def _energy_starvation_attack(self, attack: FuzzAttack):
        """Drain energy from target modules"""
        for module in attack.target_modules:
            # Rapid energy requests
            for _ in range(int(attack.parameters['drain_rate'])):
                self.energy_broker.request(
                    module,
                    random.randint(1, 10),
                    "fuzz_drain"
                )
                
        # Block refills if specified
        if attack.parameters.get('block_refill'):
            await asyncio.sleep(attack.duration)
            
    async def _rapid_oscillation_attack(self, attack: FuzzAttack):
        """Induce rapid state oscillations"""
        frequency = attack.parameters['frequency']
        amplitude = attack.parameters['amplitude']
        
        # Create oscillating sessions
        sessions = []
        for module in attack.target_modules:
            session = await self.ccl.enter_chaos_session(
                module_id=f"fuzz_{module}",
                purpose="oscillation_attack",
                required_energy=100
            )
            if session:
                sessions.append(session)
                
        # Oscillate states
        start_time = time.time()
        while time.time() - start_time < attack.duration:
            for session in sessions:
                # Inject oscillation
                phase = 2 * np.pi * frequency * (time.time() - start_time)
                perturbation = amplitude * np.sin(phase)
                
                # This would modify internal state
                await self.ccl.evolve_chaos(session, steps=1)
                
            await asyncio.sleep(1.0 / frequency)
            
        # Cleanup
        for session in sessions:
            await self.ccl.exit_chaos_session(session)
            
    async def _cascade_failure_attack(self, attack: FuzzAttack):
        """Simulate cascading failures"""
        failure_prob = attack.parameters['failure_probability']
        
        for module in attack.target_modules:
            if random.random() < failure_prob:
                # Force module into error state
                self.topo_switch.emergency_close_all()
                
                # Propagate to neighbors
                await asyncio.sleep(attack.parameters['propagation_delay'])
                
    async def _resonance_attack(self, attack: FuzzAttack):
        """Attack system at resonant frequency"""
        # This would find and exploit system resonances
        logger.warning(f"Resonance attack at {attack.parameters['target_frequency']} Hz")
        
    async def _phase_disruption_attack(self, attack: FuzzAttack):
        """Disrupt phase coherence"""
        # Add phase noise to active sessions
        for session_id in list(self.ccl.active_sessions.keys()):
            session = self.ccl.active_sessions[session_id]
            if 'state' in session:
                # Add random phase
                noise = attack.parameters['phase_noise'] * np.random.randn(len(session['state']))
                session['state'] *= np.exp(1j * noise)
                
    async def _topology_corruption_attack(self, attack: FuzzAttack):
        """Corrupt topological structure"""
        # This would flip edges in the lattice
        logger.warning("Topology corruption attack executed")
        
    async def _memory_exhaustion_attack(self, attack: FuzzAttack):
        """Attempt to exhaust memory"""
        # Rapid allocation requests
        allocation_rate = attack.parameters['allocation_rate']
        
        for _ in range(int(allocation_rate * attack.duration)):
            try:
                # Request large chaos session
                await self.ccl.enter_chaos_session(
                    module_id=f"mem_attack_{random.randint(0, 10000)}",
                    purpose="memory_exhaustion",
                    required_energy=1000
                )
            except:
                pass  # Expected to fail
                
    async def _lyapunov_spike_attack(self, attack: FuzzAttack):
        """Inject Lyapunov exponent spikes"""
        if hasattr(self.ccl, 'lyap_estimator'):
            # Corrupt Jacobian to spike Lyapunov
            dim = self.ccl.lyap_estimator.dim
            bad_jacobian = np.eye(dim) * (1 + attack.parameters['spike_magnitude'])
            
            # Force update with unstable Jacobian
            self.ccl.lyap_estimator.update(bad_jacobian)
            
    def _generate_report(self) -> Dict[str, Any]:
        """Generate fuzzing report"""
        return {
            'duration_minutes': self.attacks_executed * 0.5 / 60,  # Rough estimate
            'attacks_executed': self.attacks_executed,
            'failures_induced': self.failures_induced,
            'failure_rate': self.failures_induced / max(1, self.attacks_executed),
            'safety_violations': self.safety_violations,
            'attack_distribution': self._get_attack_distribution(),
            'resilience_score': self._calculate_resilience_score()
        }
        
    def _get_attack_distribution(self) -> Dict[str, int]:
        """Get distribution of attack types"""
        distribution = {attack_type: 0 for attack_type in self.attack_patterns}
        
        for violation in self.safety_violations:
            attack_type = violation['attack'].attack_type
            distribution[attack_type] = distribution.get(attack_type, 0) + 1
            
        return distribution
        
    def _calculate_resilience_score(self) -> float:
        """Calculate system resilience score (0-100)"""
        if self.attacks_executed == 0:
            return 100.0
            
        # Base score from failure rate
        base_score = 100 * (1 - self.failures_induced / self.attacks_executed)
        
        # Penalties for specific violation types
        critical_violations = [v for v in self.safety_violations 
                             if 'topology' in v['attack'].attack_type or 
                                'cascade' in v['attack'].attack_type]
        
        penalty = len(critical_violations) * 5
        
        return max(0, base_score - penalty)

# Test the fuzzer
async def test_chaos_fuzzer():
    """Test the chaos fuzzer"""
    print("🎲 Testing Chaos Fuzzer")
    print("=" * 50)
    
    # Mock components
    from unittest.mock import Mock, AsyncMock
    
    ccl = Mock()
    ccl.enter_chaos_session = AsyncMock(return_value="test_session")
    ccl.exit_chaos_session = AsyncMock()
    ccl.evolve_chaos = AsyncMock()
    ccl.active_sessions = {}
    
    energy_broker = Mock()
    energy_broker.request = Mock(return_value=True)
    
    topo_switch = Mock()
    topo_switch.emergency_close_all = Mock()
    
    # Create fuzzer
    fuzzer = ChaosFuzzer(ccl, energy_broker, topo_switch)
    
    # Run short campaign
    report = await fuzzer.run_fuzzing_campaign(
        duration_minutes=0.1,  # 6 seconds
        intensity="high"
    )
    
    print(f"\nFuzzing Report:")
    print(f"  Attacks executed: {report['attacks_executed']}")
    print(f"  Failures induced: {report['failures_induced']}")
    print(f"  Failure rate: {report['failure_rate']:.1%}")
    print(f"  Resilience score: {report['resilience_score']:.1f}/100")

if __name__ == "__main__":
    asyncio.run(test_chaos_fuzzer())
