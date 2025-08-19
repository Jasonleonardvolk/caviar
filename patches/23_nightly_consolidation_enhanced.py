#!/usr/bin/env python3
"""
Enhanced Nightly Consolidation with Topology Switching
Implements the complete nightly self-growth cycle with soliton voting
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Dict, Any, Optional

from python.core.soliton_memory_integration import EnhancedSolitonMemory
from python.core.oscillator_lattice import get_global_lattice
from python.core.hot_swap_laplacian import HotSwapLaplacian
from python.core.topology_policy import TopologyPolicy
from patches.22_soliton_interactions import SolitonInteractionEngine, SolitonVotingSystem

logger = logging.getLogger(__name__)


class NightlyConsolidationEngine:
    """
    Orchestrates nightly memory consolidation with topology morphing,
    soliton interactions, and self-optimization
    """
    
    def __init__(
        self,
        memory_system: EnhancedSolitonMemory,
        hot_swap: Optional[HotSwapLaplacian] = None,
        consolidation_hour: int = 3  # 3 AM by default
    ):
        self.memory_system = memory_system
        self.hot_swap = hot_swap or HotSwapLaplacian()
        self.consolidation_hour = consolidation_hour
        
        # Sub-engines
        self.interaction_engine = SolitonInteractionEngine(memory_system)
        self.voting_system = SolitonVotingSystem(memory_system)
        
        # State
        self.is_running = False
        self.last_consolidation = None
        self.consolidation_task = None
        
    async def start(self):
        """Start the nightly consolidation scheduler"""
        if self.is_running:
            logger.warning("Nightly consolidation already running")
            return
            
        self.is_running = True
        self.consolidation_task = asyncio.create_task(self._scheduler_loop())
        logger.info(f"Nightly consolidation engine started (scheduled for {self.consolidation_hour}:00)")
        
    async def stop(self):
        """Stop the consolidation scheduler"""
        self.is_running = False
        if self.consolidation_task:
            self.consolidation_task.cancel()
            try:
                await self.consolidation_task
            except asyncio.CancelledError:
                pass
        logger.info("Nightly consolidation engine stopped")
        
    async def _scheduler_loop(self):
        """Main scheduling loop"""
        while self.is_running:
            try:
                # Calculate time until next consolidation
                now = datetime.now()
                next_run = now.replace(
                    hour=self.consolidation_hour,
                    minute=0,
                    second=0,
                    microsecond=0
                )
                
                # If time has passed today, schedule for tomorrow
                if next_run <= now:
                    next_run += timedelta(days=1)
                
                wait_seconds = (next_run - now).total_seconds()
                logger.info(f"Next consolidation in {wait_seconds/3600:.1f} hours")
                
                # Wait until consolidation time
                await asyncio.sleep(wait_seconds)
                
                # Run consolidation
                await self.run_consolidation()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                # Wait an hour before retrying
                await asyncio.sleep(3600)
    
    async def run_consolidation(self):
        """Execute the complete nightly consolidation cycle"""
        start_time = datetime.now()
        logger.info("=== Starting Nightly Memory Consolidation ===")
        
        try:
            # Phase 1: Switch to consolidation topology (all-to-all for maximum interaction)
            logger.info("Phase 1: Switching to consolidation topology")
            original_topology = self.hot_swap.current_topology
            
            if self.hot_swap:
                # Morph to all-to-all topology for consolidation
                self.hot_swap.initiate_morph("all_to_all", blend_rate=0.05)
                
                # Wait for morph to complete
                while self.hot_swap.is_morphing:
                    await asyncio.sleep(0.1)
                
                # Apply temporary damping for stability
                lattice = get_global_lattice()
                lattice.global_damping = 0.1
            
            # Phase 2: Let system equilibrate
            logger.info("Phase 2: System equilibration")
            await asyncio.sleep(2.0)
            
            # Phase 3: Soliton voting
            logger.info("Phase 3: Soliton voting")
            votes = self.voting_system.compute_concept_votes()
            voting_results = self.voting_system.apply_voting_decisions(votes)
            logger.info(f"Voting results: {voting_results}")
            
            # Phase 4: Soliton interactions (fusion, fission, collision)
            logger.info("Phase 4: Soliton interactions")
            interaction_results = await self.interaction_engine.run_consolidation_cycle()
            
            # Phase 5: Memory crystallization (heat-based organization)
            logger.info("Phase 5: Memory crystallization")
            crystallization_results = await self._crystallize_memories()
            
            # Phase 6: Comfort-based lattice optimization
            logger.info("Phase 6: Lattice self-optimization")
            optimization_results = await self._optimize_lattice_comfort()
            
            # Phase 7: Return to stable topology
            logger.info("Phase 7: Returning to stable topology")
            if self.hot_swap:
                # Remove damping
                lattice.global_damping = 0.0
                
                # Morph back to stable Kagome topology
                self.hot_swap.initiate_morph("kagome", blend_rate=0.05)
                
                # Wait for completion
                while self.hot_swap.is_morphing:
                    await asyncio.sleep(0.1)
            
            # Summary
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"=== Consolidation Complete in {duration:.1f}s ===")
            logger.info(f"Results: {interaction_results}, "
                       f"Crystallized: {crystallization_results['migrated']}, "
                       f"Optimized: {optimization_results['adjustments']}")
            
            self.last_consolidation = start_time
            
        except Exception as e:
            logger.error(f"Consolidation error: {e}")
            # Ensure we return to stable state
            if self.hot_swap:
                self.hot_swap.current_topology = original_topology
                lattice = get_global_lattice()
                lattice.global_damping = 0.0
    
    async def _crystallize_memories(self) -> Dict[str, int]:
        """
        Crystallize memories based on heat (access patterns)
        Hot memories migrate to stable positions, cold memories decay
        """
        results = {'migrated': 0, 'decayed': 0}
        
        # Sort memories by heat
        memories_by_heat = []
        for mem_id, entry in self.memory_system.memory_entries.items():
            heat = getattr(entry, 'heat', 0.0)
            memories_by_heat.append((heat, mem_id, entry))
        
        memories_by_heat.sort(reverse=True)
        
        # Hot memories (top 20%) get stability boost
        hot_threshold = int(len(memories_by_heat) * 0.2)
        for i, (heat, mem_id, entry) in enumerate(memories_by_heat):
            if i < hot_threshold and heat > 0.7:
                # Boost stability for hot memories
                entry.stability = min(1.0, entry.stability * 1.2)
                results['migrated'] += 1
                logger.debug(f"Boosted stability for hot memory {mem_id}")
            
            elif heat < 0.1:
                # Decay cold memories
                entry.amplitude *= 0.9
                entry.heat *= 0.95
                
                # Remove if amplitude too low
                if entry.amplitude < 0.1:
                    # Mark for removal
                    entry.vault_status = VaultStatus.QUARANTINE
                    results['decayed'] += 1
                    logger.debug(f"Decayed cold memory {mem_id}")
        
        return results
    
    async def _optimize_lattice_comfort(self) -> Dict[str, int]:
        """
        Optimize lattice based on soliton comfort metrics
        Adjust local couplings to reduce stress and improve stability
        """
        results = {'adjustments': 0}
        lattice = get_global_lattice()
        
        if not hasattr(lattice, 'coupling_matrix'):
            return results
        
        # Analyze comfort for each memory
        comfort_feedback = []
        
        for mem_id, entry in self.memory_system.memory_entries.items():
            if 'oscillator_idx' not in entry.metadata:
                continue
            
            idx = entry.metadata['oscillator_idx']
            if idx >= len(lattice.oscillators):
                continue
            
            # Simple comfort metric based on coupling stress
            stress = 0.0
            flux = 0.0
            
            # Check all couplings for this oscillator
            for j in range(len(lattice.oscillators)):
                if j != idx and hasattr(lattice, 'K'):
                    coupling = lattice.K[idx, j] if idx < len(lattice.K) else 0
                    if coupling > 0:
                        # Phase difference creates stress
                        phase_diff = abs(lattice.oscillators[idx]['phase'] - 
                                       lattice.oscillators[j]['phase'])
                        stress += coupling * np.sin(phase_diff)
                        flux += abs(coupling)
            
            comfort_feedback.append({
                'memory_id': mem_id,
                'oscillator_idx': idx,
                'stress': stress,
                'flux': flux
            })
        
        # Adjust couplings for high-stress memories
        for feedback in comfort_feedback:
            if feedback['stress'] > 0.8:  # High stress threshold
                idx = feedback['oscillator_idx']
                
                # Reduce couplings by 10%
                if hasattr(lattice, 'K'):
                    for j in range(len(lattice.oscillators)):
                        if lattice.K[idx, j] > 0:
                            lattice.K[idx, j] *= 0.9
                            lattice.K[j, idx] *= 0.9
                            results['adjustments'] += 1
                
                logger.debug(f"Reduced couplings for stressed memory {feedback['memory_id']}")
        
        return results
    
    def force_consolidation_now(self):
        """Force an immediate consolidation (for testing/manual trigger)"""
        logger.info("Forcing immediate consolidation")
        asyncio.create_task(self.run_consolidation())


# Integration with existing growth engine
class EnhancedNightlyGrowthEngine:
    """Enhanced version that integrates all consolidation features"""
    
    def __init__(
        self,
        memory_system: EnhancedSolitonMemory,
        hot_swap: Optional[HotSwapLaplacian] = None,
        policy: Optional[TopologyPolicy] = None
    ):
        self.memory_system = memory_system
        self.hot_swap = hot_swap or HotSwapLaplacian()
        self.policy = policy or TopologyPolicy()
        
        # Create consolidation engine
        self.consolidation = NightlyConsolidationEngine(
            memory_system=memory_system,
            hot_swap=hot_swap
        )
        
        # State
        self.is_running = False
        
    async def start(self):
        """Start all growth processes"""
        self.is_running = True
        
        # Start nightly consolidation
        await self.consolidation.start()
        
        # Start continuous optimization loop
        asyncio.create_task(self._continuous_optimization())
        
        logger.info("Enhanced nightly growth engine started")
    
    async def stop(self):
        """Stop all growth processes"""
        self.is_running = False
        await self.consolidation.stop()
        logger.info("Enhanced nightly growth engine stopped")
    
    async def _continuous_optimization(self):
        """Continuous background optimization (runs during the day)"""
        while self.is_running:
            try:
                # Run every 30 minutes during daytime
                await asyncio.sleep(1800)
                
                # Skip if it's consolidation hour
                current_hour = datetime.now().hour
                if current_hour == self.consolidation.consolidation_hour:
                    continue
                
                # Light optimization: adjust topology based on current load
                lattice = get_global_lattice()
                metrics = self.policy.update_metrics(lattice)
                
                # Check if topology change is needed
                new_topology = self.policy.decide_topology(
                    self.hot_swap.current_topology,
                    metrics
                )
                
                if new_topology and new_topology != self.hot_swap.current_topology:
                    logger.info(f"Daytime topology adjustment: "
                              f"{self.hot_swap.current_topology} -> {new_topology}")
                    
                    # Smooth morph to new topology
                    self.hot_swap.initiate_morph(new_topology, blend_rate=0.02)
                
            except Exception as e:
                logger.error(f"Continuous optimization error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
