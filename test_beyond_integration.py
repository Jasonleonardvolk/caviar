#!/usr/bin/env python3
"""
test_beyond_integration.py - Comprehensive integration test for Beyond Metacognition

Tests the complete flow:
1. Spectral evolution → OriginSentry classification
2. Temporal braiding across scales
3. Self-measurement and metacognitive token generation
4. Creative feedback and entropy injection
5. Cross-component communication
"""

import asyncio
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
import sys

# Add paths
sys.path.append(str(Path(__file__).parent))

class BeyondIntegrationTest:
    """Comprehensive integration test suite"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "errors": [],
            "warnings": []
        }
        
    async def test_full_cognitive_cycle(self):
        """Test a complete cognitive cycle through all components"""
        print("\n🔄 Testing Full Cognitive Cycle")
        print("=" * 60)
        
        try:
            from alan_backend.origin_sentry import OriginSentry
            from python.core.braid_buffers import get_braiding_engine
            from alan_backend.braid_aggregator import BraidAggregator
            from python.core.observer_synthesis import get_observer_synthesis
            from python.core.creative_feedback import get_creative_feedback
            
            # Initialize all components
            origin = OriginSentry()
            braiding = get_braiding_engine()
            observer = get_observer_synthesis()
            creative = get_creative_feedback()
            aggregator = BraidAggregator(braiding, origin)
            
            # Start aggregator
            await aggregator.start()
            
            # Simulate 100 cognitive steps
            results = []
            for i in range(100):
                # Generate eigenvalues with evolving complexity
                phase = i / 25.0
                n_dims = 5 + int(phase)
                noise_level = 0.01 + phase * 0.02
                
                eigenvalues = np.random.randn(n_dims) * noise_level
                eigenvalues[0] = 0.03 + phase * 0.01  # Leading eigenvalue
                
                # 1. Classify with OriginSentry
                classification = origin.classify(eigenvalues)
                
                # 2. Record in temporal braid
                braiding.record_event(
                    kind='test_cycle',
                    lambda_max=float(np.max(np.abs(eigenvalues))),
                    data={
                        'step': i,
                        'classification': classification
                    }
                )
                
                # 3. Self-measurement (increased probability for testing)
                measurement = observer.apply_stochastic_measurement(
                    eigenvalues,
                    classification['coherence'],
                    classification['novelty_score'],
                    base_probability=0.3
                )
                
                # 4. Creative feedback
                creative_action = creative.update({
                    'novelty_score': classification['novelty_score'],
                    'lambda_max': float(np.max(np.abs(eigenvalues))),
                    'coherence_state': classification['coherence']
                })
                
                # Record results
                results.append({
                    'step': i,
                    'dims': n_dims,
                    'novelty': classification['novelty_score'],
                    'coherence': classification['coherence'],
                    'creative_mode': creative.mode.value,
                    'measured': measurement is not None
                })
                
                # Small delay for aggregator
                await asyncio.sleep(0.01)
            
            # Stop aggregator
            await aggregator.stop()
            
            # Analyze results
            dim_expansions = origin.metrics['dimension_expansions']
            measurements = len(observer.measurements)
            creative_injections = creative.metrics['total_injections']
            
            print(f"\n📊 Cycle Results:")
            print(f"  Dimensional expansions: {dim_expansions}")
            print(f"  Self-measurements: {measurements}")
            print(f"  Creative injections: {creative_injections}")
            print(f"  Final coherence: {results[-1]['coherence']}")
            print(f"  Final creative mode: {results[-1]['creative_mode']}")
            
            # Check success criteria
            success = (
                dim_expansions > 0 and
                measurements > 10 and
                creative_injections > 0
            )
            
            self.results['tests']['full_cycle'] = {
                'success': success,
                'dim_expansions': dim_expansions,
                'measurements': measurements,
                'creative_injections': creative_injections
            }
            
            return success
            
        except Exception as e:
            self.results['errors'].append(f"Full cycle test error: {str(e)}")
            print(f"❌ Error: {e}")
            return False
    
    async def test_temporal_coherence(self):
        """Test temporal braiding coherence across scales"""
        print("\n🕰️ Testing Temporal Coherence")
        print("=" * 60)
        
        try:
            from python.core.braid_buffers import get_braiding_engine, TimeScale
            from alan_backend.braid_aggregator import BraidAggregator
            
            braiding = get_braiding_engine()
            aggregator = BraidAggregator(braiding)
            
            # Start aggregator
            await aggregator.start()
            
            # Generate multi-scale pattern
            print("Generating multi-scale events...")
            
            # Micro-scale burst
            for i in range(50):
                braiding.record_event(
                    kind='micro_test',
                    lambda_max=0.01 + np.random.random() * 0.01
                )
                await asyncio.sleep(0.001)
            
            # Wait for aggregation
            await asyncio.sleep(0.5)
            
            # Meso-scale event
            for i in range(10):
                braiding.record_event(
                    kind='meso_test',
                    lambda_max=0.03 + np.random.random() * 0.02
                )
                await asyncio.sleep(0.1)
            
            # Check cross-scale coherence
            coherence = aggregator.get_cross_scale_coherence()
            context = braiding.get_multi_scale_context()
            
            # Stop aggregator
            await aggregator.stop()
            
            print(f"\n📊 Temporal Coherence Results:")
            print(f"  Cross-scale correlations: {len(coherence)}")
            
            for scale in TimeScale:
                buffer_data = context[scale.value]['summary']
                print(f"  {scale.value}: {buffer_data['count']} events")
            
            # Check for cross-scale coherence
            success = len(coherence) > 0
            
            self.results['tests']['temporal_coherence'] = {
                'success': success,
                'correlations': coherence,
                'event_counts': {
                    scale.value: context[scale.value]['summary']['count']
                    for scale in TimeScale
                }
            }
            
            return success
            
        except Exception as e:
            self.results['errors'].append(f"Temporal coherence test error: {str(e)}")
            print(f"❌ Error: {e}")
            return False
    
    async def test_reflexive_stability(self):
        """Test reflexive self-measurement doesn't cause oscillation"""
        print("\n🔍 Testing Reflexive Stability")
        print("=" * 60)
        
        try:
            from python.core.observer_synthesis import get_observer_synthesis
            
            observer = get_observer_synthesis()
            
            # Force rapid measurements to test oscillation detection
            oscillation_detected = False
            
            for i in range(20):
                # Alternate between two states
                if i % 2 == 0:
                    eigenvalues = np.array([0.03, 0.02, 0.01])
                    coherence = 'global'
                else:
                    eigenvalues = np.array([0.02, 0.03, 0.01])
                    coherence = 'local'
                
                # Force measurement
                measurement = observer.measure(
                    eigenvalues,
                    coherence,
                    0.5,
                    force=True
                )
                
                if observer.reflexive_mode:
                    oscillation_detected = True
                    break
                
                await asyncio.sleep(0.05)
            
            # Generate metacognitive context
            context = observer.generate_metacognitive_context()
            
            print(f"\n📊 Reflexive Stability Results:")
            print(f"  Total measurements: {len(observer.measurements)}")
            print(f"  Oscillation detected: {oscillation_detected}")
            print(f"  Reflexive patterns: {context.get('reflexive_patterns', [])}")
            print(f"  Warning: {context.get('warning', 'None')}")
            
            # Success = oscillation was detected and handled
            success = oscillation_detected and 'STATE_CYCLING' in context.get('reflexive_patterns', [])
            
            self.results['tests']['reflexive_stability'] = {
                'success': success,
                'oscillation_detected': oscillation_detected,
                'patterns': context.get('reflexive_patterns', [])
            }
            
            return success
            
        except Exception as e:
            self.results['errors'].append(f"Reflexive stability test error: {str(e)}")
            print(f"❌ Error: {e}")
            return False
    
    async def test_creative_exploration(self):
        """Test creative feedback loop with entropy injection"""
        print("\n🎨 Testing Creative Exploration")
        print("=" * 60)
        
        try:
            from python.core.creative_feedback import get_creative_feedback
            
            creative = get_creative_feedback()
            
            # Simulate novelty wave to trigger exploration
            injection_triggered = False
            modes_visited = set()
            
            for i in range(150):
                # Create novelty pattern
                phase = i / 50.0
                if phase < 1:
                    novelty = 0.1
                elif phase < 2:
                    novelty = 0.1 + (phase - 1) * 0.7
                else:
                    novelty = 0.8 - (phase - 2) * 0.3
                
                state = {
                    'novelty_score': novelty,
                    'lambda_max': 0.02 + novelty * 0.03,
                    'coherence_state': 'global' if novelty < 0.5 else 'critical'
                }
                
                action = creative.update(state)
                modes_visited.add(creative.mode.value)
                
                if action['action'] == 'inject_entropy':
                    injection_triggered = True
                    print(f"  Step {i}: Entropy injection triggered (λ_factor={action['lambda_factor']:.2f})")
                
                await asyncio.sleep(0.01)
            
            # Get final metrics
            metrics = creative.get_creative_metrics()
            report = creative.get_exploration_report()
            
            print(f"\n📊 Creative Exploration Results:")
            print(f"  Modes visited: {sorted(modes_visited)}")
            print(f"  Total injections: {metrics['total_injections']}")
            print(f"  Success rate: {metrics.get('success_rate', 0.0):.1%}")
            print(f"  Performance trend: {report['performance_trend']['interpretation']}")
            
            # Success = visited multiple modes and triggered injection
            success = len(modes_visited) >= 3 and injection_triggered
            
            self.results['tests']['creative_exploration'] = {
                'success': success,
                'modes_visited': list(modes_visited),
                'injections': metrics['total_injections']
            }
            
            return success
            
        except Exception as e:
            self.results['errors'].append(f"Creative exploration test error: {str(e)}")
            print(f"❌ Error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all integration tests"""
        print("\n🌌 BEYOND METACOGNITION INTEGRATION TESTS")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run tests
        test_results = []
        
        # Test 1: Full cognitive cycle
        result = await self.test_full_cognitive_cycle()
        test_results.append(("Full Cognitive Cycle", result))
        
        # Test 2: Temporal coherence
        result = await self.test_temporal_coherence()
        test_results.append(("Temporal Coherence", result))
        
        # Test 3: Reflexive stability
        result = await self.test_reflexive_stability()
        test_results.append(("Reflexive Stability", result))
        
        # Test 4: Creative exploration
        result = await self.test_creative_exploration()
        test_results.append(("Creative Exploration", result))
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for _, result in test_results if result)
        total = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<30} {status}")
        
        print(f"\nTotal: {passed}/{total} passed")
        
        # Save results
        self.results['summary'] = {
            'total_tests': total,
            'passed': passed,
            'failed': total - passed,
            'success_rate': passed / total if total > 0 else 0
        }
        
        results_file = Path("beyond_integration_test_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📝 Detailed results saved to: {results_file}")
        
        return passed == total

async def main():
    """Run integration tests"""
    tester = BeyondIntegrationTest()
    success = await tester.run_all_tests()
    
    if success:
        print("\n✅ ALL INTEGRATION TESTS PASSED!")
        print("The Beyond Metacognition system is fully operational.")
    else:
        print("\n⚠️  Some tests failed. Check the results file for details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
