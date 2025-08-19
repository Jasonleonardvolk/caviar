#!/usr/bin/env python3
"""
TORI Self-Transformation Demo
Demonstrates the phase-coherent cognition system with:
- Constitutional safety boundaries
- Critic consensus for mutations
- Sandboxed testing of modifications
- Energy-aware computation
- Cross-domain knowledge transfer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safety.constitution import Constitution
from meta_genome.critics.aggregation import aggregate
from meta.energy_budget import EnergyBudget
from goals.analogical_transfer import AnalogicalTransfer
import numpy as np

def demo_self_transformation():
    print("=== TORI Self-Transformation Demo ===\n")
    
    # 1. Initialize constitutional safety
    print("1. Loading Constitutional Safety Boundaries...")
    try:
        const = Constitution(path="safety/constitution.json")
        print("✓ Constitution loaded successfully")
        print(f"  - CPU limit: {const.doc['resource_budget']['cpu_core_seconds']}s")
        print(f"  - RAM limit: {const.doc['resource_budget']['ram_bytes'] / (1024**3):.1f}GB")
        print(f"  - Forbidden calls: {', '.join(const.doc['safety_rules']['forbidden_calls'][:3])}...")
    except Exception as e:
        print(f"✗ Constitution error: {e}")
    
    # 2. Demonstrate critic consensus
    print("\n2. Testing Critic Consensus System...")
    
    # Simulate critic scores for a proposed mutation
    critic_scores = {
        "safety_critic": 0.9,      # High safety score
        "performance_critic": 0.6,  # Moderate performance gain
        "coherence_critic": 0.8,    # Good coherence
        "novelty_critic": 0.4       # Low novelty (conservative)
    }
    
    # Historical reliabilities (how often each critic was right)
    critic_reliabilities = {
        "safety_critic": 0.95,      # Very reliable
        "performance_critic": 0.7,   # Moderately reliable
        "coherence_critic": 0.85,    # Quite reliable
        "novelty_critic": 0.6        # Less reliable
    }
    
    accepted, score = aggregate(critic_scores, critic_reliabilities)
    print(f"  Mutation score: {score:.3f}")
    print(f"  Decision: {'ACCEPTED' if accepted else 'REJECTED'} (threshold: 0.7)")
    
    # 3. Energy budget demonstration
    print("\n3. Energy Budget Management...")
    energy = EnergyBudget(max_energy=100.0)
    
    # Simulate some computations
    computations = [
        (2.0, 10.0),  # 2 CPU seconds, utility 10
        (5.0, 15.0),  # 5 CPU seconds, utility 15
        (1.0, 2.0),   # 1 CPU second, utility 2
    ]
    
    for cpu_s, utility in computations:
        allowed = energy.update(cpu_s, utility)
        print(f"  Computation: {cpu_s}s CPU, utility={utility}")
        print(f"    Energy: {energy.current_energy:.1f}/{energy.max_energy}")
        print(f"    Status: {'OK' if allowed else 'THROTTLED'}")
    
    print(f"  Current efficiency: {energy.get_efficiency():.2%}")
    
    # 4. Analogical transfer demonstration
    print("\n4. Cross-Domain Knowledge Transfer...")
    transfer = AnalogicalTransfer()
    
    # Add some knowledge domains
    domains = {
        "mathematics": np.array([1.0, 0.8, 0.2, 0.1]),
        "physics": np.array([0.8, 1.0, 0.5, 0.3]),
        "biology": np.array([0.2, 0.5, 1.0, 0.6]),
        "psychology": np.array([0.1, 0.3, 0.6, 1.0])
    }
    
    for domain, embedding in domains.items():
        transfer.add_knowledge_cluster(domain, [f"{domain}_concept_{i}" for i in range(3)], embedding)
    
    transfer.compute_transfer_kernels()
    
    # Find analogies for physics
    analogies = transfer.find_analogies("physics", n_analogies=2)
    print("  Analogies for physics domain:")
    for domain, weight in analogies:
        print(f"    - {domain}: {weight:.3f} transfer weight")
    
    # Transfer a strategy
    physics_strategy = {"precision": 0.95, "abstraction": 0.8, "empirical": True}
    bio_strategy = transfer.transfer_strategy("physics", "biology", physics_strategy)
    print("\n  Strategy transfer (physics → biology):")
    print(f"    Original: {physics_strategy}")
    print(f"    Transferred: {bio_strategy}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    demo_self_transformation()
