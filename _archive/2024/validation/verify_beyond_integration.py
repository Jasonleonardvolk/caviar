#!/usr/bin/env python3
"""
Beyond Metacognition Integration Verification
Comprehensive verification of all components
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import time

# Add paths
sys.path.append(str(Path(__file__).parent))

def verify_integration():
    print("🔍 Verifying Beyond Metacognition Integration...")
    print("=" * 60)
    
    errors = []
    warnings = []
    successes = []
    
    # 1. Check OriginSentry
    print("\n1. Testing OriginSentry...")
    try:
        from alan_backend.origin_sentry import OriginSentry, SpectralDB
        origin = OriginSentry()
        
        # Test classification
        test_eigenvalues = np.array([0.05, 0.03, 0.02, 0.01, 0.005])
        result = origin.classify(test_eigenvalues)
        
        if 'dim_expansion' in result and 'novelty_score' in result:
            successes.append("✅ OriginSentry classification working")
            print(f"   - Novelty score: {result['novelty_score']:.3f}")
            print(f"   - Coherence state: {result['coherence']}")
            print(f"   - Dimension: {result['metrics']['current_dimension']}")
        else:
            errors.append("❌ OriginSentry classification incomplete")
            
    except Exception as e:
        errors.append(f"❌ OriginSentry error: {e}")
    
    # 2. Check Temporal Braiding
    print("\n2. Testing Temporal Braiding Engine...")
    try:
        from python.core.braid_buffers import TemporalBraidingEngine, TimeScale, get_braiding_engine
        engine = get_braiding_engine()
        
        # Record test events
        for i in range(5):
            engine.record_event(
                kind='test',
                lambda_max=0.01 + i * 0.01,
                betti=[1.0, float(i % 2)]
            )
        
        # Check buffers
        micro_count = len(engine.buffers[TimeScale.MICRO].buffer)
        if micro_count > 0:
            successes.append("✅ Temporal Braiding recording events")
            print(f"   - Micro buffer: {micro_count} events")
            
            # Test multi-scale context
            context = engine.get_multi_scale_context()
            print(f"   - Scales active: {list(context.keys())}")
        else:
            errors.append("❌ Temporal Braiding not recording")
            
    except Exception as e:
        errors.append(f"❌ Temporal Braiding error: {e}")
    
    # 3. Check Braid Aggregator
    print("\n3. Testing Braid Aggregator...")
    try:
        from alan_backend.braid_aggregator import BraidAggregator
        aggregator = BraidAggregator()
        
        # Get status
        status = aggregator.get_status()
        if 'spectral_cache_sizes' in status:
            successes.append("✅ Braid Aggregator initialized")
            print(f"   - Aggregations performed: {status['metrics']['aggregations_performed']}")
            print(f"   - Cross-scale coherence: {len(status['cross_scale_coherence'])} metrics")
        else:
            errors.append("❌ Braid Aggregator status incomplete")
            
    except Exception as e:
        errors.append(f"❌ Braid Aggregator error: {e}")
    
    # 4. Check Observer Synthesis
    print("\n4. Testing Observer-Observed Synthesis...")
    try:
        from python.core.observer_synthesis import ObserverObservedSynthesis, get_observer_synthesis
        synthesis = get_observer_synthesis()
        
        # Test measurement
        test_measurement = synthesis.measure(
            np.array([0.03, 0.02, 0.01]),
            'global',
            0.5,
            force=True
        )
        
        if test_measurement:
            successes.append("✅ Observer Synthesis measuring")
            print(f"   - Measurement hash: {test_measurement.spectral_hash[:8]}")
            print(f"   - Metacognitive tokens: {test_measurement.metacognitive_tokens}")
            print(f"   - Reflex budget: {synthesis._get_reflex_budget_remaining()}/{synthesis.reflex_budget}")
        else:
            errors.append("❌ Observer Synthesis measurement failed")
            
        # Test metacognitive context
        context = synthesis.generate_metacognitive_context()
        if context['has_self_observations']:
            print(f"   - Context generated with {len(context['metacognitive_tokens'])} tokens")
            
    except Exception as e:
        errors.append(f"❌ Observer Synthesis error: {e}")
    
    # 5. Check Creative Feedback
    print("\n5. Testing Creative Feedback Loop...")
    try:
        from python.core.creative_feedback import CreativeSingularityFeedback, get_creative_feedback
        feedback = get_creative_feedback()
        
        # Test update cycle
        test_states = [
            {'novelty_score': 0.8, 'lambda_max': 0.04, 'coherence_state': 'global'},
            {'novelty_score': 0.9, 'lambda_max': 0.05, 'coherence_state': 'critical'},
            {'novelty_score': 0.7, 'lambda_max': 0.03, 'coherence_state': 'global'}
        ]
        
        actions = []
        for state in test_states:
            action = feedback.update(state)
            actions.append(action)
        
        if any(a['action'] != 'none' for a in actions):
            successes.append("✅ Creative Feedback responding")
            print(f"   - Current mode: {feedback.mode.value}")
            print(f"   - Total injections: {feedback.metrics['total_injections']}")
            
            # Get metrics
            metrics = feedback.get_creative_metrics()
            print(f"   - Success rate: {metrics.get('success_rate', 0.0):.1%}")
        else:
            warnings.append("⚠️ Creative Feedback not triggering (may need higher novelty)")
            
    except Exception as e:
        errors.append(f"❌ Creative Feedback error: {e}")
    
    # 6. Check Topology Tracker
    print("\n6. Testing Topology Tracking...")
    try:
        from python.core.topology_tracker import compute_betti_numbers
        
        # Test Betti computation
        test_data = np.random.randn(10, 10)
        betti = compute_betti_numbers(test_data)
        
        if len(betti) > 0:
            successes.append("✅ Topology tracking available")
            print(f"   - Betti numbers: {betti}")
            warnings.append("⚠️ Using stub implementation (install gudhi/ripser for full support)")
        else:
            errors.append("❌ Betti computation failed")
            
    except Exception as e:
        warnings.append(f"⚠️ Topology tracking not available: {e}")
    
    # 7. Integration Test
    print("\n7. Running Integration Test...")
    try:
        # Simulate integrated flow
        from alan_backend.origin_sentry import OriginSentry
        from python.core.braid_buffers import get_braiding_engine
        from python.core.observer_synthesis import get_observer_synthesis
        from python.core.creative_feedback import get_creative_feedback
        
        origin = OriginSentry()
        braiding = get_braiding_engine()
        observer = get_observer_synthesis()
        creative = get_creative_feedback()
        
        # Generate test sequence
        print("   Running 10-step simulation...")
        for i in range(10):
            # Generate eigenvalues with increasing novelty
            eigenvalues = np.random.randn(5) * (0.02 + i * 0.005)
            
            # Classify
            classification = origin.classify(eigenvalues)
            
            # Record in braid
            braiding.record_event(
                kind='integration_test',
                lambda_max=float(np.max(np.abs(eigenvalues))),
                data={'step': i, 'classification': classification}
            )
            
            # Self-measurement (stochastic)
            measurement = observer.apply_stochastic_measurement(
                eigenvalues,
                classification['coherence'],
                classification['novelty_score']
            )
            
            # Creative feedback
            creative_action = creative.update({
                'novelty_score': classification['novelty_score'],
                'lambda_max': float(np.max(np.abs(eigenvalues))),
                'coherence_state': classification['coherence']
            })
            
            if creative_action['action'] != 'none':
                print(f"   Step {i}: {creative_action['action']} triggered")
        
        successes.append("✅ Integration flow completed")
        
    except Exception as e:
        errors.append(f"❌ Integration test failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    print(f"\n✅ Successes: {len(successes)}")
    for success in successes:
        print(f"  {success}")
    
    if warnings:
        print(f"\n⚠️ Warnings: {len(warnings)}")
        for warning in warnings:
            print(f"  {warning}")
    
    if errors:
        print(f"\n❌ Errors: {len(errors)}")
        for error in errors:
            print(f"  {error}")
    
    # Overall status
    print("\n" + "=" * 60)
    if not errors:
        print("✅ BEYOND METACOGNITION VERIFIED!")
        print("\nAll core components are working correctly.")
        print("The system is ready for self-transforming cognition.")
    else:
        print("❌ VERIFICATION INCOMPLETE")
        print(f"\n{len(errors)} errors must be resolved before the system is fully operational.")
    
    # Save verification report
    report = {
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'successes': successes,
        'warnings': warnings,
        'errors': errors,
        'status': 'PASSED' if not errors else 'FAILED'
    }
    
    report_path = Path('beyond_verification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Verification report saved to: {report_path}")
    
    return len(errors) == 0

if __name__ == "__main__":
    success = verify_integration()
    sys.exit(0 if success else 1)
