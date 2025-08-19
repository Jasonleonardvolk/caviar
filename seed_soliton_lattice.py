#!/usr/bin/env python3
"""Seed the Soliton Lattice with Initial Concepts"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def seed_concepts():
    """Seed the lattice with some initial concepts"""
    
    # Import the global lattice
    from python.core.oscillator_lattice import get_global_lattice
    lattice = get_global_lattice()
    
    # Import event system
    try:
        from python.core.fractal_soliton_events import concept_event_bus, ConceptEvent
        
        # Emit some concept events
        concepts = [
            ("quantum_mechanics", 0.1),
            ("consciousness", 0.3),
            ("information_theory", 0.5),
            ("emergence", 0.7),
            ("complexity", 0.9)
        ]
        
        for concept_id, phase in concepts:
            event = ConceptEvent(concept_id=concept_id, phase=phase)
            concept_event_bus.emit('concept_added', event)
            logger.info(f"Emitted concept event: {concept_id} (phase={phase})")
            
        # Also add oscillators directly to the lattice
        for i, (concept_id, phase) in enumerate(concepts):
            lattice.add_oscillator(omega=1.0 + i * 0.1, phase=phase)
            logger.info(f"Added oscillator directly: {concept_id}")
            
    except ImportError as e:
        logger.error(f"Event system not available: {e}")
        
        # Fallback: Add oscillators directly
        logger.info("Adding oscillators directly to lattice...")
        for i in range(5):
            lattice.add_oscillator(omega=1.0 + i * 0.1, phase=i * 0.2)
        
    # Check lattice state
    logger.info(f"Lattice now has {len(lattice.oscillators)} oscillators")
    logger.info(f"Order parameter R = {lattice.order_parameter():.3f}")
    logger.info(f"Phase entropy H = {lattice.phase_entropy():.3f}")

async def test_soliton_store():
    """Test storing memories through the soliton API"""
    import httpx
    
    async with httpx.AsyncClient() as client:
        # Initialize user
        await client.post(
            "http://localhost:8002/api/soliton/init",
            json={"user_id": "concept_seeder"}
        )
        
        # Store some concept memories
        concepts = [
            "The wave function collapse in quantum mechanics",
            "Emergent properties of complex systems",
            "Information as fundamental reality",
            "Consciousness as integrated information",
            "Holographic principle in physics"
        ]
        
        for i, concept in enumerate(concepts):
            response = await client.post(
                "http://localhost:8002/api/soliton/store",
                json={
                    "user_id": "concept_seeder",
                    "concept_id": f"seed_{i}",
                    "content": {"text": concept},
                    "activation_strength": 0.8
                }
            )
            if response.status_code == 200:
                logger.info(f"✅ Stored concept: {concept[:30]}...")
            else:
                logger.error(f"❌ Failed to store concept: {response.text}")

async def main():
    logger.info("🌱 Seeding Soliton Lattice with Concepts")
    logger.info("=" * 50)
    
    # Seed concepts directly
    await seed_concepts()
    
    # Also try through API
    logger.info("\nTrying to seed through Soliton API...")
    try:
        await test_soliton_store()
    except Exception as e:
        logger.error(f"API seeding failed: {e}")
    
    logger.info("\n✅ Seeding complete!")

if __name__ == "__main__":
    asyncio.run(main())
