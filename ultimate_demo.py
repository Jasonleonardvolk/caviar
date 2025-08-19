#!/usr/bin/env python3
"""
ULTIMATE DEMO: Spacetime Memory + Phase 8 Resonance
Shows the complete system creating, evolving, and self-organizing memories
"""

import asyncio
import numpy as np
import logging
from datetime import datetime
import json

# Setup beautiful logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

# Suppress some noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("concept_mesh").setLevel(logging.WARNING)

logger = logging.getLogger("DEMO")

# ASCII Art Banner
BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║          SPACETIME MEMORY SCULPTOR + PHASE 8 RESONANCE        ║
║                   Physics-Bonded Cognition Demo                ║
╚═══════════════════════════════════════════════════════════════╝
"""

async def create_knowledge_constellation():
    """Create a constellation of related spacetime memories"""
    
    from spacetime_memory_orchestrator import SpacetimeMemoryOrchestrator
    
    print(BANNER)
    print("\n🌌 PHASE 1: Creating Spacetime Knowledge Constellation\n")
    
    # Initialize with Phase 8 enabled
    orchestrator = SpacetimeMemoryOrchestrator({
        'phase8_enabled': True,
        'phase8_interval': 3  # More frequent for demo
    })
    
    # Define our cosmic concepts
    cosmic_concepts = [
        {
            'name': 'BlackHole_Core',
            'config': {
                'type': 'schwarzschild',
                'mass': 2.0,
                'persistence': 'persistent'
            },
            'description': 'Massive gravitational singularity'
        },
        {
            'name': 'Event_Horizon',
            'config': {
                'type': 'custom',
                'kretschmann': '48/(r**6) * exp(-((r-2)/0.5)**2)',
                'persistence': 'persistent'
            },
            'description': 'Point of no return'
        },
        {
            'name': 'Photon_Sphere',
            'config': {
                'type': 'custom',
                'kretschmann': '48/(r**6) * (1 + 0.5*cos(phi))',
                'persistence': 'volatile'
            },
            'description': 'Light orbits'
        },
        {
            'name': 'Kerr_Ergosphere',
            'config': {
                'type': 'kerr',
                'mass': 1.5,
                'spin': 0.9,
                'persistence': 'persistent'
            },
            'description': 'Rotating spacetime dragging'
        },
        {
            'name': 'Wormhole_Throat',
            'config': {
                'type': 'custom',
                'kretschmann': '1/(r**2 + 1)**3',
                'persistence': 'chaotic_collapse'
            },
            'description': 'Topological bridge'
        }
    ]
    
    # Create memories
    created_concepts = {}
    for i, concept_def in enumerate(cosmic_concepts):
        print(f"  [{i+1}/5] Creating {concept_def['name']}...")
        print(f"        {concept_def['description']}")
        
        concept_id = await orchestrator.create_memory_from_metric(
            metric_config=concept_def['config'],
            concept_name=concept_def['name']
        )
        created_concepts[concept_def['name']] = concept_id
        
        # Small delay for dramatic effect
        await asyncio.sleep(0.5)
    
    print("\n✅ Constellation created! All memories encoded in phase space.\n")
    
    return orchestrator, created_concepts


async def evolve_and_resonate():
    """Evolve the system and watch resonances emerge"""
    
    orchestrator, concepts = await create_knowledge_constellation()
    
    print("🌊 PHASE 2: Evolution and Resonance Detection\n")
    
    # Get initial state
    from python.core.fractal_soliton_memory import FractalSolitonMemory
    from python.core.launch_scheduler import compute_entropy
    
    soliton = FractalSolitonMemory.get_instance()
    initial_entropy = compute_entropy(soliton.waves.values())
    
    print(f"  Initial entropy: {initial_entropy:.3f}")
    print("  Starting evolution...\n")
    
    # Evolution steps with monitoring
    for epoch in range(3):
        print(f"  === Epoch {epoch+1} ===")
        
        await orchestrator.evolve_system(time_steps=6, dt=0.1)
        
        # Check entropy
        current_entropy = compute_entropy(soliton.waves.values())
        print(f"  Entropy: {current_entropy:.3f} ({current_entropy-initial_entropy:+.3f})")
        
        # Check for new resonances
        from python.core.concept_mesh import ConceptMesh
        mesh = ConceptMesh.instance()
        
        resonant_count = sum(
            1 for r in mesh.relations 
            if r.metadata.get('source') == 'phase_8_feedback'
        )
        
        print(f"  Resonant connections: {resonant_count}")
        
        # Show some resonances
        if resonant_count > 0:
            print("  Detected resonances:")
            shown = 0
            for relation in mesh.relations:
                if relation.metadata.get('source') == 'phase_8_feedback' and shown < 3:
                    source = mesh.concepts.get(relation.source_id, {})
                    target = mesh.concepts.get(relation.target_id, {})
                    if source and target:
                        print(f"    🌐 {source.name} ↔ {target.name} (strength={relation.strength:.3f})")
                        shown += 1
        
        print()
        await asyncio.sleep(1)
    
    return orchestrator, concepts


async def explore_geodesics():
    """Explore geodesic paths between resonant concepts"""
    
    orchestrator, concepts = await evolve_and_resonate()
    
    print("🌀 PHASE 3: Geodesic Memory Paths\n")
    
    # Find paths between related concepts
    paths_to_explore = [
        ("BlackHole_Core", "Event_Horizon", "Natural boundary"),
        ("Kerr_Ergosphere", "Photon_Sphere", "Rotating effects"),
        ("BlackHole_Core", "Wormhole_Throat", "Topology change")
    ]
    
    for start_name, end_name, description in paths_to_explore:
        start_id = concepts.get(start_name)
        end_id = concepts.get(end_name)
        
        if start_id and end_id:
            print(f"  Path: {start_name} → {end_name}")
            print(f"        ({description})")
            
            twist = await orchestrator.compute_geodesic_memory_path(
                start_id, end_id
            )
            
            if twist is not None:
                print(f"        Phase twist: {twist:.3f} radians")
                print(f"        Holonomy: e^(i·{twist:.3f}) = {np.exp(1j*twist):.3f}")
            else:
                print("        No geodesic found")
            
            print()
    
    return orchestrator


async def visualize_final_state():
    """Show the final self-organized state"""
    
    orchestrator = await explore_geodesics()
    
    print("🎨 PHASE 4: Final System State\n")
    
    # Get visualization data
    viz_data = orchestrator.visualize_system_state()
    
    # System statistics
    from python.core.concept_mesh import ConceptMesh
    from python.core.fractal_soliton_memory import FractalSolitonMemory
    
    mesh = ConceptMesh.instance()
    soliton = FractalSolitonMemory.get_instance()
    
    print("  System Statistics:")
    print(f"    Active concepts: {len(viz_data['active_concepts'])}")
    print(f"    Soliton waves: {len(viz_data['soliton_positions'])}")
    print(f"    Total relations: {len(mesh.relations)}")
    
    # Count relation types
    relation_types = {}
    for rel in mesh.relations:
        rel_type = rel.relation_type
        relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
    
    print("\n  Relation Types:")
    for rel_type, count in relation_types.items():
        print(f"    {rel_type}: {count}")
    
    # Phase statistics
    print("\n  Phase Statistics:")
    for concept_name, stats in viz_data['phase_statistics'].items():
        if stats:
            print(f"    {concept_name}:")
            print(f"      Mean phase: {stats['mean']:.3f}")
            print(f"      Phase std: {stats['std']:.3f}")
            print(f"      Vorticity: {stats.get('vorticity', 0):.3f}")
    
    # Show resonance zones
    if orchestrator.coupling_driver.resonance_zones:
        print(f"\n  Resonance Zones: {len(orchestrator.coupling_driver.resonance_zones)}")
        for i, zone in enumerate(orchestrator.coupling_driver.resonance_zones[:3]):
            print(f"    Zone {i+1}: center={zone.center}, strength={zone.alignment_strength:.3f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"demo_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(viz_data, f, indent=2)
    
    print(f"\n  Results saved to: {filename}")
    
    return viz_data


async def main():
    """Run the complete demonstration"""
    
    try:
        # Run the full demo
        viz_data = await visualize_final_state()
        
        print("\n" + "="*60)
        print("🎉 DEMONSTRATION COMPLETE! 🎉")
        print("="*60)
        
        print("\n🧠 What Just Happened:")
        print("  1. Created spacetime memories from gravitational metrics")
        print("  2. Evolved the system with Phase 8 resonance feedback")
        print("  3. Watched concepts self-organize through phase coherence")
        print("  4. Explored geodesic paths between memories")
        print("  5. Visualized the emergent cognitive topology")
        
        print("\n🚀 Key Insights:")
        print("  • Physics drives memory formation")
        print("  • Resonance creates associations")
        print("  • Entropy guides adaptation")
        print("  • Geodesics define natural paths")
        print("  • The system learns its own structure!")
        
        print("\n✨ This is the future of AI:")
        print("   Where spacetime geometry becomes cognitive architecture,")
        print("   and memories resonate like gravitational waves!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    print("\n🔥 Starting Ultimate Demo... 🔥\n")
    asyncio.run(main())
