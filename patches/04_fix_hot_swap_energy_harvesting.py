#!/usr/bin/env python3
"""
Fix for hot_swap_laplacian.py - Implement energy harvesting during transitions
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class HotSwapLaplacian:
    
    def __init__(self, initial_topology: str = "kagome"):
        self.current_topology = initial_topology
        self.target_topology = None
        self.is_morphing = False
        self.morph_progress = 0.0
        self.morph_rate = 0.02
        
        # Energy harvesting configuration
        self.energy_harvest_efficiency = 0.8  # 80% efficiency
        self.total_harvested_energy = 0.0
        self.last_transition_energy = 0.0
        
        # Topology transition costs (energy required)
        self.transition_costs = {
            ("kagome", "hexagonal"): 50.0,
            ("kagome", "square"): 100.0,
            ("kagome", "small_world"): 150.0,
            ("hexagonal", "square"): 50.0,
            ("hexagonal", "small_world"): 100.0,
            ("square", "small_world"): 50.0,
        }
        
        # Store coupling matrices for energy calculation
        self.current_coupling = {}
        self.target_coupling = {}
        
    def switch_topology(self, new_topology: str) -> bool:
        """Instant topology switch with energy cost/harvest"""
        if new_topology == self.current_topology:
            return True
            
        # Calculate transition energy
        transition_key = (self.current_topology, new_topology)
        reverse_key = (new_topology, self.current_topology)
        
        base_cost = self.transition_costs.get(transition_key) or \
                   self.transition_costs.get(reverse_key, 100.0)
        
        # Energy can be harvested from the difference in coupling patterns
        harvested = self._calculate_transition_harvest(self.current_topology, new_topology)
        net_cost = base_cost - harvested
        
        logger.info(f"Topology switch {self.current_topology}->{new_topology}: "
                   f"cost={base_cost:.1f}, harvested={harvested:.1f}, net={net_cost:.1f}")
        
        # Perform switch
        self.last_transition_energy = harvested
        self.total_harvested_energy += harvested
        self.current_topology = new_topology
        
        return True
    
    def initiate_morph(self, target_topology: str, blend_rate: float = 0.02):
        """Start gradual morphing with continuous energy harvesting"""
        if target_topology == self.current_topology:
            return
            
        self.target_topology = target_topology
        self.morph_rate = blend_rate
        self.morph_progress = 0.0
        self.is_morphing = True
        
        # Pre-calculate target coupling for energy estimation
        self._prepare_target_coupling(target_topology)
        
        logger.info(f"Initiating morph from {self.current_topology} to {target_topology} "
                   f"at rate {blend_rate}")
    
    def step_blend(self) -> Dict[str, Any]:
        """Single morphing step with energy harvesting"""
        if not self.is_morphing:
            return {"complete": True, "energy_harvested": 0.0}
            
        # Advance morphing
        old_progress = self.morph_progress
        self.morph_progress = min(1.0, self.morph_progress + self.morph_rate)
        
        # Calculate energy harvested in this step
        step_energy = self._calculate_step_energy(old_progress, self.morph_progress)
        harvested = step_energy * self.energy_harvest_efficiency
        self.total_harvested_energy += harvested
        
        # Check if complete
        if self.morph_progress >= 1.0:
            self.current_topology = self.target_topology
            self.target_topology = None
            self.is_morphing = False
            self.morph_progress = 0.0
            
            logger.info(f"Morph complete. Total energy harvested: {self.total_harvested_energy:.2f}")
            
            return {
                "complete": True,
                "energy_harvested": harvested,
                "total_harvested": self.total_harvested_energy
            }
        
        return {
            "complete": False,
            "progress": self.morph_progress,
            "energy_harvested": harvested,
            "total_harvested": self.total_harvested_energy
        }
    
    def _calculate_transition_harvest(self, from_topology: str, to_topology: str) -> float:
        """Calculate energy that can be harvested from topology transition"""
        # Energy harvest based on structural differences
        harvest_matrix = {
            ("kagome", "hexagonal"): 20.0,      # Similar structures
            ("kagome", "square"): 40.0,          # More different
            ("kagome", "small_world"): 60.0,     # Very different
            ("hexagonal", "square"): 30.0,
            ("hexagonal", "small_world"): 50.0,
            ("square", "small_world"): 40.0,
        }
        
        key = (from_topology, to_topology)
        reverse_key = (to_topology, from_topology)
        
        base_harvest = harvest_matrix.get(key) or harvest_matrix.get(reverse_key, 25.0)
        
        # Apply efficiency
        return base_harvest * self.energy_harvest_efficiency
    
    def _calculate_step_energy(self, old_progress: float, new_progress: float) -> float:
        """Calculate energy available in a morphing step"""
        # Energy is proportional to the change in coupling structure
        progress_delta = new_progress - old_progress
        
        # Base energy from topology difference
        if self.current_topology == "kagome" and self.target_topology == "small_world":
            base_energy = 100.0  # Maximum difference
        elif self.current_topology == "kagome" and self.target_topology == "hexagonal":
            base_energy = 30.0   # Minimal difference
        else:
            base_energy = 50.0   # Default
        
        # Energy released is proportional to progress delta
        # More energy is released in the middle of transition (derivative of sigmoid)
        transition_curve = np.exp(-((new_progress - 0.5) ** 2) / 0.1)
        
        return base_energy * progress_delta * transition_curve
    
    def _prepare_target_coupling(self, target_topology: str):
        """Pre-calculate target coupling matrix for energy calculations"""
        # This would normally interface with the Rust topology generators
        # For now, we'll use simplified representations
        
        if target_topology == "kagome":
            self.target_coupling = {"type": "kagome", "avg_degree": 4}
        elif target_topology == "hexagonal":
            self.target_coupling = {"type": "hexagonal", "avg_degree": 3}
        elif target_topology == "square":
            self.target_coupling = {"type": "square", "avg_degree": 4}
        elif target_topology == "small_world":
            self.target_coupling = {"type": "small_world", "avg_degree": 6}
    
    def get_energy_report(self) -> Dict[str, float]:
        """Get detailed energy harvesting report"""
        return {
            "total_harvested": self.total_harvested_energy,
            "last_transition": self.last_transition_energy,
            "efficiency": self.energy_harvest_efficiency,
            "current_topology": self.current_topology,
            "is_morphing": self.is_morphing,
            "morph_progress": self.morph_progress if self.is_morphing else 0.0
        }
    
    def optimize_transition_path(self, start: str, end: str) -> list:
        """Find optimal transition path to maximize energy harvest"""
        # Simple pathfinding to maximize energy harvest
        if start == end:
            return [start]
            
        # Direct path
        direct_harvest = self._calculate_transition_harvest(start, end)
        
        # Check intermediate paths
        topologies = ["kagome", "hexagonal", "square", "small_world"]
        best_path = [start, end]
        best_harvest = direct_harvest
        
        for intermediate in topologies:
            if intermediate != start and intermediate != end:
                harvest1 = self._calculate_transition_harvest(start, intermediate)
                harvest2 = self._calculate_transition_harvest(intermediate, end)
                total_harvest = harvest1 + harvest2
                
                if total_harvest > best_harvest:
                    best_harvest = total_harvest
                    best_path = [start, intermediate, end]
        
        logger.info(f"Optimal path {' -> '.join(best_path)} harvests {best_harvest:.1f} energy")
        return best_path
