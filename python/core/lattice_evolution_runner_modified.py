#!/usr/bin/env python3
"""
Lattice Evolution Runner
High-performance async runner for oscillator lattice evolution
"""

import asyncio
import numpy as np
import logging
from typing import Dict, Any, Optional

from oscillator_lattice import OscillatorLattice
from coupling_matrix import CouplingMatrix
from chaos_control_layer import ChaosControlLayer
from nightly_growth_engine import NightlyGrowthEngine
from hot_swap_laplacian import integrate_hot_swap_with_ccl

logger = logging.getLogger(__name__)

class LatticeEvolutionRunner:
    """
    Manages continuous evolution of the oscillator lattice.
    Handles efficient batch updates and adaptive timesteps.
    """
    
    def __init__(self, lattice: OscillatorLattice, ccl: Optional[ChaosControlLayer] = None):
        self.lattice = lattice
        self.ccl = ccl
        
        # Evolution parameters
        self.dt = 0.01
        self.adaptive_timestep = True
        self.min_dt = 0.001
        self.max_dt = 0.1
        
        # Performance parameters
        self.update_interval = 0.1  # 100ms between updates
        self.batch_size = 100  # Update oscillators in batches
        self.is_running = False
        self.growth_engine = None
        
    async def start(self):
        """Start the evolution runner"""
        if self.is_running:
            logger.warning("Evolution runner already running")
            return
            
        # Initialize hot-swap and growth engine
        if hasattr(self, 'ccl') and self.ccl:
            hot_swap = await integrate_hot_swap_with_ccl(self.ccl)
            
            # Create growth engine
            from soliton_memory_integration import EnhancedSolitonMemory
            memory_system = self.ccl.memory_system if hasattr(self.ccl, 'memory_system') else EnhancedSolitonMemory()
            
            self.growth_engine = NightlyGrowthEngine(memory_system, hot_swap)
            await self.growth_engine.start()
            
        self.is_running = True
        logger.info("Starting lattice evolution runner")
        
        # Start evolution loop
        asyncio.create_task(self._evolution_loop())
        
    async def stop(self):
        """Stop the evolution runner"""
        logger.info("Stopping lattice evolution runner")
        
        self.is_running = False
        
        # Stop growth engine
        if self.growth_engine:
            self.growth_engine.enabled = False
            logger.info("Stopped growth engine")
            
        logger.info("Stopped lattice evolution runner")
        
    async def _evolution_loop(self):
        """Main evolution loop"""
        last_update = asyncio.get_event_loop().time()
        
        while self.is_running:
            try:
                current_time = asyncio.get_event_loop().time()
                dt = current_time - last_update
                
                if dt >= self.update_interval:
                    # Perform evolution step
                    await self._evolution_step(dt)
                    last_update = current_time
                
                # Update in batches for efficiency
                await self._batch_update(dt)
                
                # Check for topology morphing
                if hasattr(self.lattice, 'target_topology') and self.lattice.target_topology:
                    self.lattice.step_topology_blend()
                    if self.lattice.blend_progress >= 1.0:
                        self.lattice.target_topology = None
                        
                # Adaptive timestep based on system energy
                if self.adaptive_timestep:
                    self._adjust_timestep()
                
                # Apply CCL if available
                if self.ccl:
                    self.ccl.step()
                
                # Small sleep to prevent CPU hogging
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Evolution loop error: {e}")
                await asyncio.sleep(0.1)  # Backoff on error
                
    async def _evolution_step(self, dt: float):
        """Perform one evolution step"""
        # Update global lattice parameters
        self.lattice.step()
        
        # Record metrics
        if hasattr(self.lattice, 'metrics'):
            self.lattice.metrics['evolution_dt'] = dt
            self.lattice.metrics['timestep'] = self.dt
            
    async def _batch_update(self, dt: float):
        """Update oscillators in batches"""
        n_oscillators = len(self.lattice.oscillators)
        
        for i in range(0, n_oscillators, self.batch_size):
            batch_end = min(i + self.batch_size, n_oscillators)
            
            # Update batch of oscillators
            for j in range(i, batch_end):
                osc = self.lattice.oscillators[j]
                
                # Simple phase evolution
                if 'phase' in osc:
                    osc['phase'] += osc.get('natural_freq', 0.1) * self.dt
                    osc['phase'] %= 2 * np.pi
                    
                # Amplitude dynamics
                if 'amplitude' in osc:
                    # Simple damping
                    osc['amplitude'] *= (1 - 0.01 * self.dt)
                    
            # Yield to prevent blocking
            if (j - i) % 10 == 0:
                await asyncio.sleep(0)
                
    def _adjust_timestep(self):
        """Adjust timestep based on system dynamics"""
        # Get system energy
        energy = getattr(self.lattice, 'total_energy', 1.0)
        
        # Adjust timestep inversely with energy
        # Higher energy = smaller timestep for stability
        if energy > 10.0:
            self.dt = max(self.min_dt, self.dt * 0.9)
        elif energy < 1.0:
            self.dt = min(self.max_dt, self.dt * 1.1)
            
    def get_status(self) -> Dict[str, Any]:
        """Get current status of evolution runner"""
        status = {
            'is_running': self.is_running,
            'timestep': self.dt,
            'adaptive': self.adaptive_timestep,
            'update_interval': self.update_interval,
            'batch_size': self.batch_size,
            'oscillator_count': len(self.lattice.oscillators)
        }
        
        # Add growth engine status if available
        if self.growth_engine:
            status['growth_engine'] = self.growth_engine.get_status()
            
        return status

async def demo_evolution():
    """Demo the evolution runner"""
    print("🌊 Lattice Evolution Runner Demo")
    print("=" * 50)
    
    # Create lattice
    lattice = OscillatorLattice(size=100)
    
    # Create CCL
    ccl = ChaosControlLayer(
        lattice=lattice,
        target_entropy=2.0,
        energy_budget=1000.0
    )
    
    # Create runner
    runner = LatticeEvolutionRunner(lattice, ccl)
    
    # Start evolution
    await runner.start()
    print("✅ Evolution started")
    
    # Run for a bit
    for i in range(5):
        await asyncio.sleep(1)
        status = runner.get_status()
        print(f"\n📊 Status at t={i+1}s:")
        print(f"   Timestep: {status['timestep']:.4f}")
        print(f"   Oscillators: {status['oscillator_count']}")
        print(f"   Order parameter: {lattice.order_parameter():.3f}")
        print(f"   Entropy: {lattice.phase_entropy():.3f}")
        
        if 'growth_engine' in status:
            print(f"   Growth Engine: {status['growth_engine']['enabled']}")
            print(f"   Next nightly run: {status['growth_engine']['next_run']}")
        
    # Stop evolution
    await runner.stop()
    print("\n✅ Evolution stopped")

if __name__ == "__main__":
    asyncio.run(demo_evolution())