#!/usr/bin/env python3
"""
Soliton Memory Control CLI
Command-line interface for managing soliton memory systems
"""

import click
import json
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

# Import core modules
try:
    from python.core.soliton_memory_integration import EnhancedSolitonMemory
    from python.core.hot_swap_laplacian import HotSwapLaplacian
    from python.core.oscillator_lattice import get_global_lattice
    from python.core.nightly_growth_scheduler import NightlyGrowthEngine
    from python.core.physics_instrumentation import PhysicsMonitor
except ImportError:
    print("Error: Core modules not found. Please ensure you're running from the project root.")
    sys.exit(1)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.pass_context
def cli(ctx, config):
    """Soliton Memory System Control Interface"""
    ctx.ensure_object(dict)
    
    # Load configuration
    if config:
        with open(config) as f:
            ctx.obj['config'] = json.load(f)
    else:
        ctx.obj['config'] = {}
    
    # Initialize memory system
    ctx.obj['memory'] = EnhancedSolitonMemory()
    ctx.obj['lattice'] = get_global_lattice()
    ctx.obj['hot_swap'] = HotSwapLaplacian()


@cli.command()
@click.argument('content')
@click.option('--concepts', '-c', multiple=True, help='Associated concepts')
@click.option('--phase', '-p', type=float, default=None, help='Initial phase')
@click.option('--amplitude', '-a', type=float, default=1.0, help='Initial amplitude')
@click.option('--dark', is_flag=True, help='Create dark soliton')
@click.pass_context
def store(ctx, content, concepts, phase, amplitude, dark):
    """Store a memory in the soliton system"""
    memory = ctx.obj['memory']
    
    # Generate phase if not provided
    if phase is None:
        phase = np.random.uniform(0, 2*np.pi)
    
    # Store memory
    memory_id = memory.store(
        content=content,
        memory_type='dark' if dark else 'bright',
        metadata={
            'concepts': list(concepts),
            'phase': phase,
            'amplitude': amplitude,
            'created_via': 'cli'
        }
    )
    
    click.echo(f"✓ Stored memory: {memory_id}")
    click.echo(f"  Type: {'dark' if dark else 'bright'} soliton")
    click.echo(f"  Phase: {phase:.3f}")
    click.echo(f"  Amplitude: {amplitude:.3f}")
    
    if concepts:
        click.echo(f"  Concepts: {', '.join(concepts)}")


@cli.command()
@click.argument('query')
@click.option('--limit', '-l', type=int, default=10, help='Maximum results')
@click.option('--threshold', '-t', type=float, default=0.5, help='Similarity threshold')
@click.pass_context
def retrieve(ctx, query, limit, threshold):
    """Retrieve memories from the soliton system"""
    memory = ctx.obj['memory']
    
    # Retrieve memories
    results = memory.retrieve(query, top_k=limit)
    
    if not results:
        click.echo("No memories found matching query.")
        return
    
    click.echo(f"\nFound {len(results)} memories:")
    click.echo("-" * 50)
    
    for i, (mem_id, score, entry) in enumerate(results):
        if score < threshold:
            continue
            
        click.echo(f"\n{i+1}. Memory ID: {mem_id}")
        click.echo(f"   Score: {score:.3f}")
        click.echo(f"   Type: {entry.memory_type}")
        click.echo(f"   Content: {entry.content[:100]}...")
        click.echo(f"   Phase: {entry.phase:.3f}, Amplitude: {entry.amplitude:.3f}")


@cli.command()
@click.argument('topology', type=click.Choice(['kagome', 'hexagonal', 'square', 'small_world', 'all_to_all']))
@click.option('--rate', '-r', type=float, default=0.02, help='Morphing rate')
@click.option('--force', is_flag=True, help='Force immediate switch')
@click.pass_context
def topology(ctx, topology, rate, force):
    """Switch lattice topology"""
    hot_swap = ctx.obj['hot_swap']
    
    current = hot_swap.current_topology
    click.echo(f"Current topology: {current}")
    
    if current == topology:
        click.echo("Already in requested topology.")
        return
    
    if force:
        # Immediate switch
        hot_swap.switch_topology(topology)
        click.echo(f"✓ Switched to {topology} topology")
    else:
        # Gradual morph
        hot_swap.initiate_morph(topology, blend_rate=rate)
        click.echo(f"✓ Initiated morph to {topology} (rate: {rate})")
        
        # Show progress
        with click.progressbar(length=100, label='Morphing') as bar:
            while hot_swap.is_morphing:
                progress = int(hot_swap.morph_progress * 100)
                bar.update(progress - bar.pos)
                time.sleep(0.1)
        
        click.echo(f"✓ Morphing complete!")


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status"""
    memory = ctx.obj['memory']
    lattice = ctx.obj['lattice']
    hot_swap = ctx.obj['hot_swap']
    
    # Memory statistics
    num_memories = len(memory.memory_entries)
    num_dark = sum(1 for e in memory.memory_entries.values() if e.memory_type == 'dark')
    num_bright = num_memories - num_dark
    
    click.echo("\n=== Soliton Memory System Status ===")
    click.echo(f"\nMemory Storage:")
    click.echo(f"  Total memories: {num_memories}")
    click.echo(f"  Bright solitons: {num_bright}")
    click.echo(f"  Dark solitons: {num_dark}")
    
    # Lattice statistics
    if hasattr(lattice, 'oscillators'):
        num_oscillators = len(lattice.oscillators)
        active_oscillators = sum(1 for o in lattice.oscillators if o.get('active', True))
        order_param = lattice.order_parameter() if hasattr(lattice, 'order_parameter') else 0
        
        click.echo(f"\nOscillator Lattice:")
        click.echo(f"  Total oscillators: {num_oscillators}")
        click.echo(f"  Active oscillators: {active_oscillators}")
        click.echo(f"  Order parameter: {order_param:.3f}")
    
    # Topology status
    click.echo(f"\nTopology:")
    click.echo(f"  Current: {hot_swap.current_topology}")
    click.echo(f"  Morphing: {'Yes' if hot_swap.is_morphing else 'No'}")
    if hot_swap.is_morphing:
        click.echo(f"  Target: {hot_swap.target_topology}")
        click.echo(f"  Progress: {hot_swap.morph_progress:.1%}")
    
    # Energy status
    click.echo(f"\nEnergy:")
    click.echo(f"  Total harvested: {hot_swap.total_harvested_energy:.2f}")
    click.echo(f"  Harvest efficiency: {hot_swap.energy_harvest_efficiency:.1%}")


@cli.command()
@click.option('--force', is_flag=True, help='Force consolidation now')
@click.pass_context
def consolidate(ctx, force):
    """Trigger nightly consolidation"""
    memory = ctx.obj['memory']
    
    if force:
        click.echo("Triggering immediate consolidation...")
        # Create growth engine
        engine = NightlyGrowthEngine(memory_system=memory)
        
        # Run consolidation
        start = time.time()
        fused, split = engine._consolidate_memories()
        duration = time.time() - start
        
        click.echo(f"✓ Consolidation complete in {duration:.1f}s")
        click.echo(f"  Memories fused: {fused}")
        click.echo(f"  Memories split: {split}")
    else:
        click.echo("Nightly consolidation scheduled for 3:00 AM")
        click.echo("Use --force to trigger immediately")


@cli.command()
@click.option('--interval', '-i', type=float, default=1.0, help='Update interval (seconds)')
@click.option('--duration', '-d', type=int, default=60, help='Monitor duration (seconds)')
@click.pass_context
def monitor(ctx, interval, duration):
    """Monitor system metrics in real-time"""
    memory = ctx.obj['memory']
    lattice = ctx.obj['lattice']
    
    click.echo("Monitoring system... (Ctrl+C to stop)")
    click.echo("-" * 60)
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            # Collect metrics
            num_memories = len(memory.memory_entries)
            order_param = lattice.order_parameter() if hasattr(lattice, 'order_parameter') else 0
            
            # Get energy if available
            if hasattr(lattice, 'compute_total_energy'):
                energy = lattice.compute_total_energy()
            else:
                energy = 0
            
            # Display metrics
            elapsed = time.time() - start_time
            click.echo(f"\r[{elapsed:.1f}s] Memories: {num_memories} | "
                      f"Order: {order_param:.3f} | Energy: {energy:.2f}", nl=False)
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        click.echo("\nMonitoring stopped.")


@cli.command()
@click.argument('output', type=click.Path())
@click.pass_context
def export(ctx, output):
    """Export memory system state"""
    memory = ctx.obj['memory']
    hot_swap = ctx.obj['hot_swap']
    
    # Collect state
    state = {
        'timestamp': time.time(),
        'memories': {
            mem_id: {
                'content': entry.content,
                'type': entry.memory_type,
                'phase': entry.phase,
                'amplitude': entry.amplitude,
                'frequency': entry.frequency,
                'metadata': entry.metadata
            }
            for mem_id, entry in memory.memory_entries.items()
        },
        'topology': {
            'current': hot_swap.current_topology,
            'energy_harvested': hot_swap.total_harvested_energy
        },
        'statistics': {
            'total_memories': len(memory.memory_entries),
            'memory_types': {
                'bright': sum(1 for e in memory.memory_entries.values() if e.memory_type == 'bright'),
                'dark': sum(1 for e in memory.memory_entries.values() if e.memory_type == 'dark')
            }
        }
    }
    
    # Write to file
    output_path = Path(output)
    with open(output_path, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    
    click.echo(f"✓ Exported system state to {output_path}")
    click.echo(f"  Total memories: {state['statistics']['total_memories']}")


@cli.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('--merge', is_flag=True, help='Merge with existing memories')
@click.pass_context
def import_(ctx, input, merge):
    """Import memory system state"""
    memory = ctx.obj['memory']
    
    # Load state
    with open(input) as f:
        state = json.load(f)
    
    if not merge:
        # Clear existing memories
        memory.memory_entries.clear()
        click.echo("Cleared existing memories")
    
    # Import memories
    imported = 0
    for mem_id, mem_data in state['memories'].items():
        if mem_id not in memory.memory_entries or merge:
            # Reconstruct memory entry
            memory.memory_entries[mem_id] = type('MemoryEntry', (), mem_data)()
            imported += 1
    
    click.echo(f"✓ Imported {imported} memories")
    click.echo(f"  Total memories: {len(memory.memory_entries)}")


@cli.command()
@click.pass_context
def interactive(ctx):
    """Start interactive REPL"""
    import code
    
    # Prepare namespace
    namespace = {
        'memory': ctx.obj['memory'],
        'lattice': ctx.obj['lattice'],
        'hot_swap': ctx.obj['hot_swap'],
        'np': np,
        'ctx': ctx
    }
    
    banner = """
Soliton Memory Interactive Shell
Available objects:
  - memory: EnhancedSolitonMemory instance
  - lattice: Global oscillator lattice
  - hot_swap: HotSwapLaplacian instance
  - np: NumPy module
  
Example commands:
  >>> memory.store("Hello world", memory_type="bright")
  >>> results = memory.retrieve("Hello")
  >>> hot_swap.initiate_morph("hexagonal")
"""
    
    code.interact(banner=banner, local=namespace)


if __name__ == '__main__':
    cli()
