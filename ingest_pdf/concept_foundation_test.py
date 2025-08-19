"""concept_foundation_test.py - Test and demonstrate core concept foundation components.

This module provides demonstration code and integration tests for ALAN's
foundational components:

1. TimeContext - Temporal synchronization backbone
2. ConceptMetadata - Semantic spine with provenance tracking  
3. ConceptLogger - Reflexive historian for concept events

Together these components form the bedrock of ALAN's memory sculpting and
phase-coherent cognition systems.
"""

import time
import logging
import random
from datetime import datetime, timedelta
import uuid
import json
import numpy as np
from typing import List, Dict, Tuple, Any

try:
    # Try absolute import first
    from time_context import TimeContext, default_time_context
except ImportError:
    # Fallback to relative import
    from .time_context import TimeContext, default_time_context
try:
    # Try absolute import first
    from concept_metadata import ConceptMetadata
except ImportError:
    # Fallback to relative import
    from .concept_metadata import ConceptMetadata
try:
    # Try absolute import first
    from concept_logger import ConceptLogger, default_concept_logger
except ImportError:
    # Fallback to relative import
    from .concept_logger import ConceptLogger, default_concept_logger
try:
    # Try absolute import first
    from models import ConceptTuple
except ImportError:
    # Fallback to relative import
    from .models import ConceptTuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("concept_foundation_test")

def create_mock_concept(name: str, domain: str = "test") -> ConceptTuple:
    """Create a mock concept for testing."""
    # Create random embedding
    emb_dim = 64
    embedding = np.random.normal(0, 1, emb_dim)
    embedding = embedding / np.linalg.norm(embedding)  # Normalize
    
    # Generate eigenfunction ID based on embedding
    fingerprint = hash(embedding.tobytes()) % 10000000
    eigen_id = f"eigen-{fingerprint}"
    
    # Create source provenance
    source_provenance = {
        "source_id": f"test-{uuid.uuid4().hex[:8]}",
        "domain": domain,
        "creation_time": datetime.now().isoformat()
    }
    
    # Generate spectral lineage
    spectral_lineage = [
        (random.uniform(-0.9, 0.9), random.uniform(0, 0.5)) 
        for _ in range(3)
    ]
    
    # Create concept tuple
    concept = ConceptTuple(
        name=name,
        embedding=embedding,
        context=f"Context for {name}",
        passage_embedding=embedding * 0.9,  # Slightly different
        cluster_members=[i for i in range(1, random.randint(2, 10))],
        resonance_score=random.uniform(0.5, 1.0),
        narrative_centrality=random.uniform(0, 1.0),
        predictability_score=random.uniform(0.3, 0.8),
        eigenfunction_id=eigen_id,
        source_provenance=source_provenance,
        spectral_lineage=spectral_lineage,
        cluster_coherence=random.uniform(0.4, 0.9)
    )
    
    return concept

def demonstrate_time_context():
    """Demonstrate TimeContext functionality."""
    print("\n=== TimeContext Demonstration ===")
    
    # Create a TimeContext with accelerated clock
    ctx = TimeContext(base_clock_rate=2.0)
    print(f"Created TimeContext with UUID: {ctx.uuid[:8]}")
    
    # Register some decay processes
    ctx.register_decay_process("resonance_decay", rate_factor=1.0)
    ctx.register_decay_process("memory_prune", rate_factor=0.5)
    
    # Simulate time passing
    for i in range(3):
        # Sleep for a short time
        time.sleep(0.5)
        
        # Update the clock
        delta = ctx.update()
        print(f"Update {i+1}: Delta time = {delta:.3f}s (clock-adjusted)")
        
        # Complete a cognitive cycle and check decay factors
        ctx.complete_cognitive_cycle()
        r_decay = ctx.get_decay_factor("resonance_decay", half_life=10.0)
        m_decay = ctx.get_decay_factor("memory_prune", half_life=20.0)
        
        print(f"  Resonance decay factor: {r_decay:.4f}")
        print(f"  Memory prune decay factor: {m_decay:.4f}")
    
    # Get runtime stats
    stats = ctx.get_runtime_stats()
    print("\nTimeContext Stats:")
    print(f"  Runtime: {stats['runtime_seconds']:.2f} seconds")
    print(f"  Cognitive cycles: {stats['cognitive_cycles']}")
    print(f"  Active processes: {stats['active_processes']}")
    
    # Test age factor calculation with a past timestamp
    past_time = time.time() - 86400  # One day ago
    age_factor = ctx.get_age_factor(past_time, scale_factor=1.0)
    print(f"  Age factor for 1-day old timestamp: {age_factor:.4f}")
    
    return ctx

def demonstrate_concept_metadata():
    """Demonstrate ConceptMetadata functionality."""
    print("\n=== ConceptMetadata Demonstration ===")
    
    # Create TimeContext for temporal integration
    time_ctx = TimeContext()
    
    # Create a ConceptMetadata instance
    metadata = ConceptMetadata(
        ψ_id="eigen-1234567",
        source_hash="abc123def456",
        provenance="arxiv-paper-12345",
        domain="mathematics"
    )
    print(f"Created ConceptMetadata with ψ_id: {metadata.ψ_id}")
    
    # Add semantic tags
    metadata.add_tag("linear_algebra")
    metadata.add_tag("eigenvalues")
    metadata.add_tag("vector_space")
    print(f"Added tags: {', '.join(metadata.tags)}")
    
    # Record activations
    print("Recording activations...")
    # First activation
    metadata.record_activation(strength=0.7)
    
    # Simulate time passing and more activations
    time.sleep(0.5)
    metadata.record_activation(strength=0.9)
    
    time.sleep(0.5)
    metadata.record_activation(strength=0.8)
    
    # Calculate activation statistics
    freq = metadata.activation_frequency(time_window_hours=1.0)
    print(f"  Activation frequency: {freq:.2f} activations per hour")
    print(f"  Activation count: {metadata.activation_count}")
    
    # Update stability over time
    print("\nUpdating stability...")
    print(f"  Initial stability: {metadata.stability:.2f}")
    
    metadata.update_stability(resonance=0.85, time_factor=0.3)
    print(f"  Updated stability: {metadata.stability:.2f}")
    
    metadata.update_stability(resonance=0.65, time_factor=0.3)
    print(f"  Updated stability: {metadata.stability:.2f}")
    
    # Test age-related features
    age_sec = metadata.age(time_ctx)
    print(f"\nConcept age: {age_sec:.2f} seconds")
    
    age_factor = metadata.age_factor(time_ctx, scale_factor=2.0)
    print(f"Age factor: {age_factor:.4f}")
    
    recency = metadata.get_activation_recency_score()
    print(f"Activation recency score: {recency:.4f}")
    
    # Test serialization
    data_dict = metadata.to_dict()
    print(f"\nSerialized metadata contains {len(data_dict)} fields")
    
    # Test deserialization
    recreated = ConceptMetadata.from_dict(data_dict)
    print(f"Successfully recreated metadata with ψ_id: {recreated.ψ_id}")
    
    return metadata

def demonstrate_concept_logger():
    """Demonstrate ConceptLogger functionality."""
    print("\n=== ConceptLogger Demonstration ===")
    
    # Create a test-specific logger
    log_file = "concept_test_events.log"
    logger = ConceptLogger(log_file=log_file, console=True)
    print(f"Created ConceptLogger, logging to: {log_file}")
    
    # Log concept birth events
    print("\nLogging concept birth events...")
    
    # Create a set of test concepts
    concepts = [
        create_mock_concept(f"TestConcept-{i}", domain=random.choice(["math", "physics", "chemistry"]))
        for i in range(5)
    ]
    
    for concept in concepts:
        logger.log_concept_birth(
            concept=concept,
            source="test_creation",
            details={"domain": concept.source_provenance.get("domain", "")}
        )
    
    # Log concept activations
    print("\nLogging concept activations...")
    for concept in concepts:
        strength = random.uniform(0.5, 1.0)
        logger.log_concept_activation(
            concept_id=concept.eigenfunction_id,
            strength=strength,
            concept_name=concept.name,
            activation_count=1
        )
    
    # Log a merge event
    print("\nLogging concept merge event...")
    parent_ids = [c.eigenfunction_id for c in concepts[:2]]
    parent_names = [c.name for c in concepts[:2]]
    child_id = f"eigen-merged-{uuid.uuid4().hex[:8]}"
    child_name = f"{parent_names[0]} / {parent_names[1]}"
    
    logger.log_concept_merge(
        parent_ids=parent_ids,
        child_id=child_id,
        reason="redundancy",
        parent_names=parent_names,
        child_name=child_name,
        score=0.85
    )
    
    # Log a stability change
    print("\nLogging stability change...")
    concept = concepts[2]
    logger.log_stability_change(
        concept_id=concept.eigenfunction_id,
        old_value=0.75,
        new_value=0.45, 
        concept_name=concept.name
    )
    
    # Log a phase coherence event
    print("\nLogging phase coherence event...")
    coherent_concepts = concepts[1:4]
    logger.log_phase_alert(
        concept_ids=[c.eigenfunction_id for c in coherent_concepts],
        coherence=0.87,
        event="resonant_synchrony",
        concept_names=[c.name for c in coherent_concepts]
    )
    
    # Log a pruning event
    print("\nLogging concept pruning event...")
    concept_to_prune = concepts[4]
    logger.log_concept_pruning(
        concept_id=concept_to_prune.eigenfunction_id,
        reason="low_entropy",
        metrics={
            "entropy": 0.12,
            "threshold": 0.20,
            "age_days": 3.5
        },
        concept_name=concept_to_prune.name
    )
    
    # Log an error event
    print("\nLogging error event...")
    logger.log_error(
        operation="vector_calculation",
        error="Dimension mismatch in embedding space",
        concept_id=concepts[0].eigenfunction_id,
        severity="WARNING"
    )
    
    # Get event statistics
    stats = logger.get_stats()
    print("\nLogger Statistics:")
    for event_type, count in stats["events"].items():
        print(f"  {event_type}: {count} events")
    print(f"  Total: {stats['total_events']} events")
    
    return logger

def demonstrate_integration():
    """Demonstrate how all components work together."""
    print("\n=== Integration Demonstration ===")
    
    # Create time context
    time_ctx = TimeContext(base_clock_rate=1.5)
    print(f"Created TimeContext: {time_ctx.uuid[:8]}")
    
    # Create logger
    logger = ConceptLogger(log_file="integration_test.log")
    
    # Create a set of concepts with metadata
    concepts = []
    for i in range(3):
        # Create base concept
        concept = create_mock_concept(f"IntegratedConcept-{i+1}")
        
        # Create and attach metadata
        metadata = ConceptMetadata(
            ψ_id=concept.eigenfunction_id,
            source_hash=f"hash-{i}",
            provenance=concept.source_provenance.get("source_id", ""),
            domain=concept.source_provenance.get("domain", "unknown")
        )
        
        # Add tags
        metadata.add_tag(f"tag{i+1}")
        metadata.add_tag("integration")
        
        # Store concept with its metadata
        concepts.append((concept, metadata))
        
        # Log birth event
        logger.log_concept_birth(
            concept=concept,
            source="integration_test",
            details={"metadata_id": metadata.ψ_id}
        )
    
    print(f"Created {len(concepts)} integrated concepts")
    
    # Simulate cognitive cycles with activations and stability updates
    print("\nSimulating cognitive cycles...")
    for cycle in range(3):
        # Update timebase
        delta = time_ctx.update()
        time_ctx.complete_cognitive_cycle()
        
        print(f"Cycle {cycle+1}: delta={delta:.3f}s")
        
        # Activate some concepts
        for concept, metadata in concepts:
            # Random chance of activation
            if random.random() > 0.3:
                # Generate activation strength
                strength = random.uniform(0.6, 1.0)
                
                # Record in metadata
                metadata.record_activation(strength=strength)
                
                # Calculate new stability based on resonance
                old_stability = metadata.stability
                metadata.update_stability(
                    resonance=strength, 
                    time_factor=0.2
                )
                
                # Log activation
                logger.log_concept_activation(
                    concept_id=concept.eigenfunction_id,
                    strength=strength,
                    concept_name=concept.name,
                    activation_count=metadata.activation_count
                )
                
                # Log stability change
                if abs(metadata.stability - old_stability) > 0.1:
                    logger.log_stability_change(
                        concept_id=concept.eigenfunction_id,
                        old_value=old_stability,
                        new_value=metadata.stability,
                        concept_name=concept.name
                    )
        
        # Sleep to simulate processing time
        time.sleep(0.5)
    
    # Calculate age-related metrics for all concepts
    print("\nAge-related metrics:")
    for concept, metadata in concepts:
        age_sec = metadata.age(time_ctx)
        age_factor = metadata.age_factor(time_ctx)
        recency = metadata.get_activation_recency_score()
        
        print(f"Concept '{concept.name}':")
        print(f"  Age: {age_sec:.2f}s")
        print(f"  Age factor: {age_factor:.4f}")
        print(f"  Recency: {recency:.4f}")
        print(f"  Stability: {metadata.stability:.4f}")
    
    # Get statistics
    time_stats = time_ctx.get_runtime_stats()
    logger_stats = logger.get_stats()
    
    print("\nFinal Statistics:")
    print(f"  Cognitive cycles: {time_stats['cognitive_cycles']}")
    print(f"  Runtime: {time_stats['runtime_seconds']:.2f}s")
    print(f"  Logger events: {logger_stats['total_events']}")
    
    return time_ctx, logger, concepts

def run_all_demonstrations():
    """Run all demonstration functions."""
    print("===== ALAN Concept Foundation Component Test =====")
    print("Testing the foundational components for ALAN's concept management:")
    print("1. TimeContext - Temporal backbone")
    print("2. ConceptMetadata - Semantic spine")
    print("3. ConceptLogger - Reflexive historian")
    
    try:
        # Individual component demonstrations
        time_ctx = demonstrate_time_context()
        metadata = demonstrate_concept_metadata()
        logger = demonstrate_concept_logger()
        
        # Integrated demonstration
        demonstrate_integration()
        
        print("\n===== Test Complete =====")
        print("All components functioning correctly!")
        
    except Exception as e:
        print(f"\nTEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    run_all_demonstrations()
