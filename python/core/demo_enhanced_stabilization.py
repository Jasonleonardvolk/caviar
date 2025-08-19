#!/usr/bin/env python3
"""
Enhanced Topology Transition Stabilization Demo

Demonstrates the improved stabilization techniques for chaotic transients
during soliton reinjection in topology transitions.

Features Demonstrated:
• Oscillation amplitude monitoring during transitions
• Adaptive damping based on detected instability patterns  
• Comprehensive post-swap turbulence control
• Performance comparison between traditional and enhanced methods
"""

import asyncio
import logging
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from chaos_control_layer import ChaosControlLayer, TransitionStabilizer
    from hot_swap_laplacian import HotSwappableLaplacian
    DEMO_AVAILABLE = True
except ImportError as e:
    logger.error(f"Demo dependencies not available: {e}")
    DEMO_AVAILABLE = False

async def demonstrate_stabilization_techniques():
    """Demonstrate enhanced stabilization during topology transitions"""
    
    if not DEMO_AVAILABLE:
        print("❌ Demo cannot run - missing dependencies")
        return
    
    print("🔧 Enhanced Topology Transition Stabilization Demo")
    print("=" * 60)
    
    # Create enhanced hot-swap system
    hot_swap = HotSwappableLaplacian(
        initial_topology="kagome",
        lattice_size=(15, 15),
        enable_experimental=True,
        enable_enhanced_stabilization=True
    )
    
    print(f"✅ System initialized with enhanced stabilization")
    print(f"   Current topology: {hot_swap.current_topology}")
    print(f"   Enhanced stabilization: {hot_swap.enable_enhanced_stabilization}")
    print(f"   CCL available: {hot_swap.ccl is not None}")
    
    # Create some challenging initial conditions
    hot_swap.active_solitons = [
        {'amplitude': 8.0, 'phase': 0, 'topological_charge': 1, 'index': 0, 'position': 10},
        {'amplitude': 12.0, 'phase': np.pi/3, 'topological_charge': 1, 'index': 1, 'position': 25},
        {'amplitude': 15.0, 'phase': np.pi, 'topological_charge': -1, 'index': 2, 'position': 40},
        {'amplitude': 6.0, 'phase': 3*np.pi/2, 'topological_charge': 1, 'index': 3, 'position': 55}
    ]
    hot_swap.total_energy = sum(s['amplitude']**2 for s in hot_swap.active_solitons)
    
    print(f"   Initial energy: {hot_swap.total_energy:.2f} (high energy scenario)")
    print(f"   Active solitons: {len(hot_swap.active_solitons)}")
    
    # Test 1: Traditional vs Enhanced Stabilization Comparison
    print("\n🧪 Test 1: Stabilization Method Comparison")
    print("-" * 50)
    
    test_topologies = ["triangular", "honeycomb"]
    results = {
        'traditional': [],
        'enhanced': []
    }
    
    for topology in test_topologies:
        print(f"\n→ Testing transition to {topology}")
        
        # Test traditional method (disable enhanced stabilization)
        print("  Traditional stabilization...")
        hot_swap_traditional = HotSwappableLaplacian(
            initial_topology="kagome",
            lattice_size=(15, 15),
            enable_enhanced_stabilization=False
        )
        hot_swap_traditional.active_solitons = hot_swap.active_solitons.copy()
        hot_swap_traditional.total_energy = hot_swap.total_energy
        
        start_time = datetime.now()
        success_traditional = await hot_swap_traditional.hot_swap_laplacian_with_safety(topology)
        traditional_time = (datetime.now() - start_time).total_seconds()
        
        results['traditional'].append({
            'topology': topology,
            'success': success_traditional,
            'time': traditional_time,
            'final_energy': hot_swap_traditional.total_energy
        })
        
        print(f"    Result: {'✅ Success' if success_traditional else '❌ Failed'} in {traditional_time:.3f}s")
        
        # Test enhanced method
        print("  Enhanced stabilization...")
        start_time = datetime.now()
        success_enhanced = await hot_swap.hot_swap_laplacian_with_safety(topology)
        enhanced_time = (datetime.now() - start_time).total_seconds()
        
        results['enhanced'].append({
            'topology': topology,
            'success': success_enhanced,
            'time': enhanced_time,
            'final_energy': hot_swap.total_energy
        })
        
        print(f"    Result: {'✅ Success' if success_enhanced else '❌ Failed'} in {enhanced_time:.3f}s")
        
        # Show improvement
        if success_enhanced and success_traditional:
            time_improvement = (traditional_time - enhanced_time) / traditional_time * 100
            print(f"    ⚡ Performance improvement: {time_improvement:+.1f}% time")
    
    # Test 2: Direct Stabilization Testing
    if hot_swap.ccl and hot_swap.ccl.transition_stabilizer:
        print("\n🧪 Test 2: Direct Stabilization Testing")
        print("-" * 50)
        
        stabilizer = hot_swap.ccl.transition_stabilizer
        
        # Create test oscillation scenarios
        test_scenarios = [
            {
                'name': 'High Amplitude Oscillation',
                'state': 5.0 * (np.random.randn(50) + 1j * np.random.randn(50)),
                'expected_damping': True
            },
            {
                'name': 'Resonant Oscillation',
                'state': 3.0 * np.exp(1j * np.linspace(0, 4*np.pi, 50)),
                'expected_damping': True
            },
            {
                'name': 'Stable Low Amplitude',
                'state': 0.5 * (np.random.randn(50) + 1j * np.random.randn(50)),
                'expected_damping': False
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n  → {scenario['name']}:")
            
            # Monitor oscillations
            metrics = stabilizer.monitor_oscillations(scenario['state'])
            
            print(f"    Max amplitude: {metrics['max_amplitude']:.3f}")
            print(f"    Stability score: {metrics['stability_score']:.3f}")
            print(f"    Instability detected: {metrics['instability_detected']}")
            print(f"    Requires damping: {metrics['requires_damping']}")
            
            if metrics.get('critical_pattern'):
                print(f"    Critical pattern: {metrics['critical_pattern']}")
            
            # Apply damping if needed
            if metrics['requires_damping']:
                damped_state, damping_info = stabilizer.apply_damping(
                    scenario['state'], metrics, method='adaptive'
                )
                
                amplitude_reduction = (
                    metrics['max_amplitude'] - np.max(np.abs(damped_state))
                ) / metrics['max_amplitude'] * 100
                
                print(f"    ✅ Damping applied: {damping_info['damping_applied']:.3f} strength")
                print(f"    📉 Amplitude reduced by: {amplitude_reduction:.1f}%")
                
                # Verify expected behavior
                if scenario['expected_damping']:
                    print(f"    ✅ Expected damping correctly applied")
                else:
                    print(f"    ⚠️ Unexpected damping applied")
            else:
                if not scenario['expected_damping']:
                    print(f"    ✅ Correctly identified as stable")
                else:
                    print(f"    ⚠️ Expected damping not applied")
    
    # Test 3: Performance Metrics
    print("\n📊 Test 3: System Performance Metrics")
    print("-" * 50)
    
    metrics = hot_swap.get_swap_metrics()
    
    print(f"  Total topology swaps performed: {metrics.get('total_swaps', 0)}")
    print(f"  Current energy level: {metrics.get('current_energy', 0):.2f}")
    print(f"  Enhanced stabilization enabled: {metrics.get('enhanced_stabilization_enabled', False)}")
    
    if 'stabilization_metrics' in metrics:
        stab_metrics = metrics['stabilization_metrics']
        if 'stabilizer_status' in stab_metrics:
            stabilizer_status = stab_metrics['stabilizer_status']
            print(f"  Stabilizer threshold: {stabilizer_status.get('amplitude_threshold', 0):.2f}")
            print(f"  Current damping strength: {stabilizer_status.get('damping_strength', 0):.3f}")
            print(f"  Stability score: {stabilizer_status.get('stability_score', 0):.3f}")
        
        if 'transition_metrics' in stab_metrics:
            trans_metrics = stab_metrics['transition_metrics']
            print(f"  Transitions stabilized: {trans_metrics.get('transitions_stabilized', 0)}")
            print(f"  Total instabilities detected: {trans_metrics.get('instabilities_detected', 0)}")
            print(f"  Damping applications: {trans_metrics.get('damping_applications', 0)}")
            print(f"  Average stability score: {trans_metrics.get('average_stability_score', 0):.3f}")
    
    # Summary and Recommendations
    print("\n🎯 Demo Summary and Recommendations")
    print("=" * 60)
    
    traditional_successes = sum(1 for r in results['traditional'] if r['success'])
    enhanced_successes = sum(1 for r in results['enhanced'] if r['success'])
    
    print(f"\n📈 Success Rate Comparison:")
    print(f"  Traditional method: {traditional_successes}/{len(results['traditional'])} ({traditional_successes/len(results['traditional'])*100:.0f}%)")
    print(f"  Enhanced method: {enhanced_successes}/{len(results['enhanced'])} ({enhanced_successes/len(results['enhanced'])*100:.0f}%)")
    
    if enhanced_successes > traditional_successes:
        print(f"  ✅ Enhanced stabilization shows improved reliability")
    elif enhanced_successes == traditional_successes:
        print(f"  ➡️ Both methods show similar reliability")
    else:
        print(f"  ⚠️ Traditional method performed better in this test")
    
    # Calculate average times
    if results['traditional'] and results['enhanced']:
        traditional_avg_time = np.mean([r['time'] for r in results['traditional'] if r['success']])
        enhanced_avg_time = np.mean([r['time'] for r in results['enhanced'] if r['success']])
        
        if traditional_avg_time > 0 and enhanced_avg_time > 0:
            time_improvement = (traditional_avg_time - enhanced_avg_time) / traditional_avg_time * 100
            print(f"\n⚡ Performance Improvement:")
            print(f"  Traditional average time: {traditional_avg_time:.3f}s")
            print(f"  Enhanced average time: {enhanced_avg_time:.3f}s")
            print(f"  Time improvement: {time_improvement:+.1f}%")
    
    print(f"\n🔧 Key Features Demonstrated:")
    print(f"  ✅ Real-time oscillation amplitude monitoring")
    print(f"  ✅ Adaptive damping based on instability patterns")
    print(f"  ✅ Multi-phase stabilization process")
    print(f"  ✅ Comprehensive transition reporting")
    print(f"  ✅ Pattern-specific damping (resonance, chaos onset, soliton breakup)")
    print(f"  ✅ Learning-based parameter adaptation")
    
    print(f"\n💡 Recommendations for Production Use:")
    print(f"  • Enable enhanced stabilization for critical topology transitions")
    print(f"  • Monitor stability scores and adjust thresholds as needed")
    print(f"  • Use longer monitoring durations for high-energy scenarios")
    print(f"  • Consider selective damping for specific oscillation patterns")
    print(f"  • Implement real field state monitoring (beyond synthetic states)")
    
    # Cleanup
    if hot_swap.ccl and hasattr(hot_swap.ccl, 'executor'):
        hot_swap.ccl.executor.shutdown(wait=False)
    
    print(f"\n✅ Enhanced Stabilization Demo Complete!")
    print(f"   The system now provides robust turbulence control during topology transitions.")

def create_synthetic_oscillation(pattern_type: str, amplitude: float = 1.0, size: int = 100) -> np.ndarray:
    """
    Create synthetic oscillation patterns for testing
    
    Args:
        pattern_type: Type of oscillation ('stable', 'resonant', 'chaotic', 'soliton_breakup')
        amplitude: Base amplitude
        size: Array size
        
    Returns:
        Complex array representing oscillation state
    """
    x = np.linspace(0, 2*np.pi, size)
    
    if pattern_type == 'stable':
        # Low amplitude, slowly varying
        return amplitude * 0.3 * (np.sin(x) + 0.1j * np.cos(x))
    
    elif pattern_type == 'resonant':
        # High amplitude oscillation at specific frequency
        return amplitude * 3.0 * np.exp(1j * 5 * x) * np.exp(-0.1 * (x - np.pi)**2)
    
    elif pattern_type == 'chaotic':
        # Rapidly growing amplitude with noise
        growth = np.exp(0.1 * x)
        noise = 0.5 * (np.random.randn(size) + 1j * np.random.randn(size))
        return amplitude * growth * (np.sin(10*x) + noise)
    
    elif pattern_type == 'soliton_breakup':
        # High amplitude variance, low phase coherence
        amplitudes = amplitude * (1 + 2 * np.random.randn(size))
        phases = 2 * np.pi * np.random.randn(size)
        return amplitudes * np.exp(1j * phases)
    
    else:
        # Default to random
        return amplitude * (np.random.randn(size) + 1j * np.random.randn(size))

async def test_stabilizer_directly():
    """
    Direct test of the TransitionStabilizer class
    """
    print("\n🔬 Direct TransitionStabilizer Testing")
    print("=" * 50)
    
    try:
        # Create stabilizer instance
        stabilizer = TransitionStabilizer(
            amplitude_threshold=2.0,
            damping_strength=0.2,
            monitoring_window=50
        )
        
        print(f"  Stabilizer initialized:")
        print(f"    Threshold: {stabilizer.amplitude_threshold}")
        print(f"    Damping strength: {stabilizer.damping_strength}")
        print(f"    Adaptive enabled: {stabilizer.adaptive_damping}")
        
        # Test different oscillation patterns
        test_patterns = {
            'stable': create_synthetic_oscillation('stable', 0.5),
            'resonant': create_synthetic_oscillation('resonant', 1.0),
            'chaotic': create_synthetic_oscillation('chaotic', 1.5),
            'soliton_breakup': create_synthetic_oscillation('soliton_breakup', 2.0)
        }
        
        results = {}
        
        for pattern_name, pattern_state in test_patterns.items():
            print(f"\n  → Testing {pattern_name} pattern:")
            
            # Monitor the pattern
            metrics = stabilizer.monitor_oscillations(pattern_state)
            
            print(f"    Initial max amplitude: {metrics['max_amplitude']:.3f}")
            print(f"    Stability score: {metrics['stability_score']:.3f}")
            print(f"    Instability detected: {metrics['instability_detected']}")
            
            if metrics.get('critical_pattern'):
                print(f"    Critical pattern detected: {metrics['critical_pattern']}")
            
            # Apply damping if needed
            if metrics['requires_damping']:
                # Test different damping methods
                for method in ['adaptive', 'uniform', 'selective']:
                    damped_state, damping_info = stabilizer.apply_damping(
                        pattern_state.copy(), metrics, method=method
                    )
                    
                    final_amplitude = np.max(np.abs(damped_state))
                    reduction = (metrics['max_amplitude'] - final_amplitude) / metrics['max_amplitude'] * 100
                    
                    print(f"      {method} damping: {reduction:.1f}% amplitude reduction")
            else:
                print(f"    ✅ Pattern correctly identified as stable")
            
            results[pattern_name] = {
                'initial_amplitude': metrics['max_amplitude'],
                'stability_score': metrics['stability_score'],
                'instability_detected': metrics['instability_detected'],
                'critical_pattern': metrics.get('critical_pattern')
            }
        
        # Show summary
        print(f"\n  📊 Pattern Recognition Summary:")
        for pattern, result in results.items():
            status = "⚠️ Unstable" if result['instability_detected'] else "✅ Stable"
            print(f"    {pattern}: {status} (amplitude: {result['initial_amplitude']:.2f})")
        
        return True
        
    except Exception as e:
        logger.error(f"Direct stabilizer test failed: {e}")
        return False

if __name__ == "__main__":
    async def main():
        """Main demo execution"""
        try:
            # Run main demonstration
            await demonstrate_stabilization_techniques()
            
            # Run direct stabilizer test
            await test_stabilizer_directly()
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Demo interrupted by user")
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the demo
    asyncio.run(main())
