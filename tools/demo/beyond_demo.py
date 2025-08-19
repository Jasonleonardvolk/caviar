#!/usr/bin/env python3
"""
Beyond Metacognition Demo - Interactive demonstration of self-transforming cognition
Shows how TORI can detect and respond to dimensional emergence
"""

import numpy as np
import time
import json
import asyncio
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import sys

# Add paths
sys.path.append(str(Path(__file__).parent))

from alan_backend.origin_sentry import OriginSentry
from python.core.braid_buffers import get_braiding_engine, TimeScale
from alan_backend.braid_aggregator import BraidAggregator
from python.core.observer_synthesis import get_observer_synthesis
from python.core.creative_feedback import get_creative_feedback

class BeyondMetacognitionDemo:
    """Interactive demo of self-transforming cognition"""
    
    def __init__(self):
        print("🌌 Initializing Beyond Metacognition Demo...")
        
        # Initialize all components
        self.origin_sentry = OriginSentry()
        self.braiding_engine = get_braiding_engine()
        self.observer_synthesis = get_observer_synthesis()
        self.creative_feedback = get_creative_feedback()
        self.braid_aggregator = BraidAggregator(
            self.braiding_engine,
            self.origin_sentry
        )
        
        # Tracking
        self.history = {
            'eigenvalues': [],
            'novelty': [],
            'coherence': [],
            'dimensions': [],
            'creative_mode': [],
            'self_measurements': []
        }
        
        print("✅ All components initialized")
        
    async def run_scenario(self, scenario: str = "emergence"):
        """Run a specific demonstration scenario"""
        
        scenarios = {
            "emergence": self._dimensional_emergence_scenario,
            "creative": self._creative_exploration_scenario,
            "reflexive": self._reflexive_measurement_scenario,
            "temporal": self._temporal_braiding_scenario
        }
        
        if scenario not in scenarios:
            print(f"❌ Unknown scenario: {scenario}")
            print(f"Available: {list(scenarios.keys())}")
            return
        
        print(f"\n🎭 Running scenario: {scenario}")
        print("=" * 60)
        
        # Start aggregator
        aggregator_task = asyncio.create_task(self.braid_aggregator.start())
        
        try:
            # Run selected scenario
            await scenarios[scenario]()
        finally:
            # Stop aggregator
            await self.braid_aggregator.stop()
            
    async def _dimensional_emergence_scenario(self):
        """Demonstrate dimensional emergence detection"""
        print("📊 Dimensional Emergence Scenario")
        print("Simulating the birth of new cognitive dimensions...")
        print()
        
        # Phase 1: Stable low-dimensional state
        print("Phase 1: Stable state (5 dimensions)")
        for i in range(20):
            eigenvalues = self._generate_stable_spectrum(5)
            await self._process_state(eigenvalues, f"stable_{i}")
            await asyncio.sleep(0.05)
        
        # Phase 2: Dimensional expansion
        print("\nPhase 2: Dimensional expansion begins...")
        for i in range(20):
            # Gradually introduce new modes
            base_eigenvalues = self._generate_stable_spectrum(5)
            new_modes = np.random.randn(i // 4) * 0.03
            eigenvalues = np.concatenate([base_eigenvalues, new_modes])
            await self._process_state(eigenvalues, f"expansion_{i}")
            await asyncio.sleep(0.05)
        
        # Phase 3: New stable state
        print("\nPhase 3: New stable high-dimensional state")
        for i in range(20):
            eigenvalues = self._generate_stable_spectrum(10)
            await self._process_state(eigenvalues, f"new_stable_{i}")
            await asyncio.sleep(0.05)
        
        # Report
        self._print_emergence_report()
        
    async def _creative_exploration_scenario(self):
        """Demonstrate creative feedback loop"""
        print("🎨 Creative Exploration Scenario")
        print("System will inject entropy when novelty is high...")
        print()
        
        # Generate novelty wave
        for i in range(100):
            # Create novelty pattern
            phase = i / 25.0
            if phase < 1:  # Low novelty
                base_novelty = 0.1
            elif phase < 2:  # Rising novelty
                base_novelty = 0.1 + (phase - 1) * 0.7
            elif phase < 3:  # High novelty
                base_novelty = 0.8
            else:  # Declining
                base_novelty = 0.8 - (phase - 3) * 0.6
            
            # Generate eigenvalues based on novelty
            n_modes = 5 + int(base_novelty * 5)
            eigenvalues = np.random.randn(n_modes) * (0.02 + base_novelty * 0.03)
            
            await self._process_state(eigenvalues, f"creative_{i}")
            
            # Show creative actions
            if self.creative_feedback.mode.value != 'stable':
                print(f"  Step {i}: Mode={self.creative_feedback.mode.value}, "
                      f"Novelty={base_novelty:.2f}")
            
            await asyncio.sleep(0.02)
        
        # Report
        self._print_creative_report()
        
    async def _reflexive_measurement_scenario(self):
        """Demonstrate self-measurement and metacognition"""
        print("🔍 Reflexive Self-Measurement Scenario")
        print("System observes its own spectral state...")
        print()
        
        measurement_count = 0
        
        for i in range(50):
            # Generate varying states
            coherence_cycle = ['local', 'global', 'critical'][i % 3]
            eigenvalues = self._generate_spectrum_for_coherence(coherence_cycle)
            
            # Process with higher measurement probability
            result = await self._process_state(
                eigenvalues, 
                f"reflexive_{i}",
                measurement_probability=0.5
            )
            
            if result.get('measurement'):
                measurement_count += 1
                print(f"  Measurement {measurement_count}: "
                      f"Hash={result['measurement'].spectral_hash[:8]}, "
                      f"Tokens={result['measurement'].metacognitive_tokens}")
            
            await asyncio.sleep(0.1)
        
        # Generate metacognitive context
        context = self.observer_synthesis.generate_metacognitive_context()
        print("\n📊 Metacognitive Context:")
        print(f"  Total measurements: {len(self.observer_synthesis.measurements)}")
        print(f"  Reflex budget remaining: {context['reflex_budget_remaining']}")
        print(f"  Token frequencies: {json.dumps(context['token_frequencies'], indent=2)}")
        print(f"  Reflexive patterns: {context['reflexive_patterns']}")
        
    async def _temporal_braiding_scenario(self):
        """Demonstrate multi-scale temporal braiding"""
        print("🕰️ Temporal Braiding Scenario")
        print("Recording events across multiple timescales...")
        print()
        
        # Generate events with different temporal patterns
        
        # Micro-scale bursts
        print("Generating micro-scale bursts...")
        for burst in range(5):
            for i in range(10):
                eigenvalues = np.random.randn(5) * 0.05
                await self._process_state(eigenvalues, f"micro_burst_{burst}_{i}")
                await asyncio.sleep(0.001)  # 1ms
            await asyncio.sleep(0.1)  # Gap between bursts
        
        # Meso-scale cycles
        print("\nGenerating meso-scale cycles...")
        for cycle in range(3):
            for i in range(20):
                eigenvalues = self._generate_stable_spectrum(5 + cycle)
                await self._process_state(eigenvalues, f"meso_cycle_{cycle}_{i}")
                await asyncio.sleep(0.5)  # 500ms
        
        # Check multi-scale context
        context = self.braiding_engine.get_multi_scale_context()
        
        print("\n📊 Multi-Scale Context:")
        for scale in TimeScale:
            scale_data = context[scale.value]
            print(f"\n{scale.value.upper()} scale:")
            print(f"  Events: {scale_data['summary']['count']}")
            print(f"  Fill ratio: {scale_data['summary']['fill_ratio']:.1%}")
            
            if scale_data['summary']['lambda_stats']['max']:
                print(f"  λ_max: {scale_data['summary']['lambda_stats']['max']:.3f}")
                print(f"  λ_mean: {scale_data['summary']['lambda_stats']['mean']:.3f}")
        
        # Show cross-scale coherence
        coherence = self.braid_aggregator.get_cross_scale_coherence()
        if coherence:
            print("\n🔗 Cross-Scale Coherence:")
            for pair, value in coherence.items():
                print(f"  {pair}: {value:.3f}")
    
    async def _process_state(self, eigenvalues: np.ndarray, 
                           event_id: str,
                           measurement_probability: float = 0.1) -> Dict[str, Any]:
        """Process a single state through all components"""
        
        # 1. Origin classification
        classification = self.origin_sentry.classify(eigenvalues)
        
        # 2. Record in temporal braid
        self.braiding_engine.record_event(
            kind='demo_state',
            lambda_max=float(np.max(np.abs(eigenvalues))),
            data={
                'event_id': event_id,
                'classification': classification
            }
        )
        
        # 3. Self-measurement (stochastic)
        measurement = self.observer_synthesis.apply_stochastic_measurement(
            eigenvalues,
            classification['coherence'],
            classification['novelty_score'],
            base_probability=measurement_probability
        )
        
        # 4. Creative feedback
        creative_action = self.creative_feedback.update({
            'novelty_score': classification['novelty_score'],
            'lambda_max': float(np.max(np.abs(eigenvalues))),
            'coherence_state': classification['coherence']
        })
        
        # 5. Track history
        self.history['eigenvalues'].append(eigenvalues.tolist())
        self.history['novelty'].append(classification['novelty_score'])
        self.history['coherence'].append(classification['coherence'])
        self.history['dimensions'].append(len(eigenvalues))
        self.history['creative_mode'].append(self.creative_feedback.mode.value)
        self.history['self_measurements'].append(measurement is not None)
        
        return {
            'classification': classification,
            'creative_action': creative_action,
            'measurement': measurement
        }
    
    def _generate_stable_spectrum(self, n_modes: int) -> np.ndarray:
        """Generate stable eigenvalue spectrum"""
        # Exponentially decaying spectrum
        base = 0.05
        eigenvalues = base * np.exp(-np.arange(n_modes) * 0.5)
        # Add small noise
        eigenvalues += np.random.randn(n_modes) * 0.001
        return eigenvalues
    
    def _generate_spectrum_for_coherence(self, coherence: str) -> np.ndarray:
        """Generate spectrum for specific coherence state"""
        if coherence == 'local':
            # Small eigenvalues
            return np.random.randn(5) * 0.005
        elif coherence == 'global':
            # Medium eigenvalues
            return np.random.randn(7) * 0.02
        else:  # critical
            # Large eigenvalues
            return np.random.randn(10) * 0.04
    
    def _print_emergence_report(self):
        """Print dimensional emergence report"""
        print("\n" + "="*60)
        print("📊 DIMENSIONAL EMERGENCE REPORT")
        print("="*60)
        
        report = self.origin_sentry.get_emergence_report()
        
        print(f"\nDimension births: {report['metrics']['dimension_expansions']}")
        print(f"Gap births: {report['metrics']['gap_births']}")
        print(f"Current dimension: {report['metrics']['current_dimension']}")
        
        if report['dimension_births']:
            print("\nRecent dimensional births:")
            for birth in report['dimension_births'][-3:]:
                print(f"  - {birth['timestamp']}: {birth['new_modes']} new modes")
        
        if report['coherence_transitions']:
            print("\nCoherence transitions:")
            for trans in report['coherence_transitions'][-3:]:
                print(f"  - {trans['from']} → {trans['to']} (λ_max={trans['lambda_max']:.3f})")
    
    def _print_creative_report(self):
        """Print creative exploration report"""
        print("\n" + "="*60)
        print("🎨 CREATIVE EXPLORATION REPORT")
        print("="*60)
        
        report = self.creative_feedback.get_exploration_report()
        metrics = report['metrics']
        
        print(f"\nTotal entropy injections: {metrics['total_injections']}")
        print(f"Successful explorations: {metrics['successful_explorations']}")
        print(f"Success rate: {metrics.get('success_rate', 0.0):.1%}")
        print(f"Total creative gain: {metrics['total_creative_gain']:.3f}")
        
        if report['recent_injections']:
            print("\nRecent injections:")
            for inj in report['recent_injections'][-3:]:
                print(f"  - Mode: {inj['mode']}, λ_factor: {inj['lambda_factor']:.2f}, "
                      f"Gain: {inj['outcomes'].get('creative_gain', 0.0):.3f}")
        
        print(f"\nPerformance trend: {report['performance_trend']['interpretation']}")
    
    def plot_history(self):
        """Plot system evolution"""
        if not self.history['novelty']:
            print("No history to plot")
            return
        
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))
        
        # Novelty score
        axes[0].plot(self.history['novelty'])
        axes[0].set_ylabel('Novelty Score')
        axes[0].set_title('System Evolution')
        axes[0].grid(True)
        
        # Dimensions
        axes[1].plot(self.history['dimensions'])
        axes[1].set_ylabel('Dimensions')
        axes[1].grid(True)
        
        # Creative mode
        mode_map = {'stable': 0, 'exploring': 1, 'consolidating': 0.5, 'emergency': -1}
        mode_values = [mode_map.get(m, 0) for m in self.history['creative_mode']]
        axes[2].plot(mode_values)
        axes[2].set_ylabel('Creative Mode')
        axes[2].set_yticks([-1, 0, 0.5, 1])
        axes[2].set_yticklabels(['Emergency', 'Stable', 'Consolidating', 'Exploring'])
        axes[2].grid(True)
        
        # Self-measurements
        measurement_points = [i for i, m in enumerate(self.history['self_measurements']) if m]
        axes[3].scatter(measurement_points, [1]*len(measurement_points), alpha=0.5)
        axes[3].set_ylabel('Self-Measurements')
        axes[3].set_ylim(0, 2)
        axes[3].set_xlabel('Time Step')
        axes[3].grid(True)
        
        plt.tight_layout()
        plt.savefig('beyond_metacognition_demo.png')
        print("\n📈 Plot saved to: beyond_metacognition_demo.png")

async def interactive_demo():
    """Run interactive demonstration"""
    demo = BeyondMetacognitionDemo()
    
    print("\n🌌 BEYOND METACOGNITION INTERACTIVE DEMO")
    print("="*60)
    print("\nAvailable scenarios:")
    print("1. emergence  - Dimensional emergence detection")
    print("2. creative   - Creative exploration with entropy injection")
    print("3. reflexive  - Self-measurement and metacognition")
    print("4. temporal   - Multi-scale temporal braiding")
    print("5. all        - Run all scenarios")
    print("6. quit       - Exit demo")
    print()
    
    while True:
        choice = input("Select scenario (1-6): ").strip()
        
        if choice == '6' or choice == 'quit':
            break
        elif choice == '1':
            await demo.run_scenario('emergence')
        elif choice == '2':
            await demo.run_scenario('creative')
        elif choice == '3':
            await demo.run_scenario('reflexive')
        elif choice == '4':
            await demo.run_scenario('temporal')
        elif choice == '5' or choice == 'all':
            for scenario in ['emergence', 'creative', 'reflexive', 'temporal']:
                await demo.run_scenario(scenario)
                print("\n" + "-"*60 + "\n")
        else:
            print("Invalid choice. Please try again.")
            continue
        
        # Offer to plot history
        if demo.history['novelty']:
            plot_choice = input("\nPlot system evolution? (y/n): ").strip().lower()
            if plot_choice == 'y':
                try:
                    demo.plot_history()
                except Exception as e:
                    print(f"Could not create plot: {e}")
        
        print("\n" + "-"*60 + "\n")
    
    print("\n👋 Thank you for exploring Beyond Metacognition!")

if __name__ == "__main__":
    # Check if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("⚠️ matplotlib not available - plotting disabled")
        print("Install with: pip install matplotlib")
    
    # Run demo
    asyncio.run(interactive_demo())
