#!/usr/bin/env python3
"""
Comprehensive demo of Creative Feedback optional enhancements.
Shows entropy profiles, quality model, and metric streaming in action.
"""

import asyncio
import numpy as np
import json
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# Assume enhanced creative_feedback is available
from python.core.creative_feedback import (
    CreativeFeedbackLoop,
    configure_logging
)

# Configure logging
configure_logging()

class CreativityDashboard:
    """Simple dashboard to visualize streaming metrics"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=200)
        self.fig = None
        self.axes = None
        
    async def handle_metric(self, metric):
        """Handle streamed metric"""
        self.metrics_history.append(metric)
        print(f"\n📊 Metric #{metric.get('stream_sequence', 0)}")
        print(f"  Mode: {metric['mode']}")
        print(f"  Cumulative Gain: {metric['cumulative_gain']:.3f}")
        
        if 'current_injection' in metric and metric['current_injection']:
            inj = metric['current_injection']
            print(f"  Injection: factor={inj['entropy_factor']:.2f}, "
                  f"profile={inj.get('entropy_profile', 'unknown')}")
    
    def plot_history(self):
        """Plot metrics history"""
        if len(self.metrics_history) < 2:
            return
            
        # Extract data
        gains = [m['cumulative_gain'] for m in self.metrics_history]
        steps = list(range(len(gains)))
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Cumulative gain
        plt.subplot(2, 1, 1)
        plt.plot(steps, gains, 'b-', linewidth=2)
        plt.ylabel('Cumulative Creative Gain')
        plt.title('Creative Feedback Performance')
        plt.grid(True, alpha=0.3)
        
        # Mode indicators
        plt.subplot(2, 1, 2)
        modes = [m['mode'] for m in self.metrics_history]
        mode_values = {'stable': 0, 'exploring': 1, 'recovering': -1}
        mode_nums = [mode_values.get(m, 0) for m in modes]
        plt.plot(steps, mode_nums, 'g-', linewidth=2)
        plt.ylabel('Mode')
        plt.xlabel('Time Step')
        plt.yticks([-1, 0, 1], ['Recovering', 'Stable', 'Exploring'])
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('creative_feedback_demo.png')
        print("\n📈 Saved plot to creative_feedback_demo.png")


class SimulatedTORISystem:
    """Simulates a TORI-like system for testing"""
    
    def __init__(self):
        self.base_novelty = 0.4
        self.base_aesthetic = 0.6
        self.volatility = 0.2
        self.injection_response = 0.0
        
    def get_state(self):
        """Get current system state"""
        # Add some natural variation
        novelty = self.base_novelty + 0.1 * np.sin(self.injection_response) + np.random.normal(0, 0.05)
        aesthetic = self.base_aesthetic + 0.05 * np.cos(self.injection_response * 0.7) + np.random.normal(0, 0.03)
        
        return {
            'novelty': np.clip(novelty, 0, 1),
            'aesthetic_score': np.clip(aesthetic, 0, 1),
            'volatility': self.volatility,
            'timestamp': datetime.utcnow()
        }
    
    def apply_entropy(self, factor):
        """Simulate system response to entropy injection"""
        # System responds to entropy
        self.injection_response += factor * 0.5
        self.base_novelty += factor * 0.1 * np.random.random()
        self.volatility = min(0.9, self.volatility + factor * 0.2)
        
        # Aesthetic might improve or degrade
        if np.random.random() < 0.6:  # 60% chance of improvement
            self.base_aesthetic += factor * 0.05
        else:
            self.base_aesthetic -= factor * 0.03
        
        # Keep in bounds
        self.base_novelty = np.clip(self.base_novelty, 0.1, 0.9)
        self.base_aesthetic = np.clip(self.base_aesthetic, 0.1, 0.9)
    
    def stabilize(self):
        """Natural stabilization when not injecting"""
        self.volatility *= 0.95
        self.injection_response *= 0.9


async def demonstrate_entropy_profiles():
    """Demonstrate different entropy profiles in action"""
    print("\n" + "="*60)
    print("🌊 ENTROPY PROFILES DEMONSTRATION")
    print("="*60)
    
    feedback = CreativeFeedbackLoop()
    system = SimulatedTORISystem()
    
    # Test each profile type
    profiles_to_test = [
        ('constant', {'novelty': 0.5, 'volatility': 0.3}),
        ('cosine_ramp', {'novelty': 0.6, 'volatility': 0.2}),
        ('exponential_decay', {'novelty': 0.4, 'volatility': 0.8}),
        ('pulse', {'novelty': 0.2, 'volatility': 0.1})
    ]
    
    for profile_name, test_state in profiles_to_test:
        print(f"\n📍 Testing {profile_name} profile:")
        print(f"   State: novelty={test_state['novelty']:.2f}, volatility={test_state['volatility']:.2f}")
        
        # Get profile parameters
        params = feedback._get_profile_params(profile_name, test_state)
        print(f"   Parameters: {params}")
        
        # Show factor evolution
        factors = []
        steps = 50
        for step in range(steps):
            factor = feedback._get_profile_factor(0.5, step, steps, profile_name, params)
            factors.append(factor)
        
        # Show characteristic points
        print(f"   Factor evolution: start={factors[0]:.3f}, "
              f"mid={factors[steps//2]:.3f}, end={factors[-1]:.3f}")
        
        # Mini ASCII visualization
        print("   Pattern: ", end="")
        for i in range(0, steps, 2):
            level = int(factors[i] * 10)
            print("▁▂▃▄▅▆▇█"[min(7, level)], end="")
        print()


async def demonstrate_quality_model():
    """Demonstrate quality model learning and prediction"""
    print("\n" + "="*60)
    print("🧠 QUALITY MODEL DEMONSTRATION")
    print("="*60)
    
    feedback = CreativeFeedbackLoop()
    system = SimulatedTORISystem()
    
    print("\n📚 Training quality model with synthetic data...")
    
    # Generate training data
    training_samples = []
    
    for i in range(30):
        # Vary conditions
        state = {
            'novelty': 0.2 + (i % 7) * 0.1,
            'aesthetic_score': 0.4 + (i % 5) * 0.1,
            'volatility': (i % 4) * 0.2
        }
        
        factor = 0.3 + (i % 4) * 0.2
        duration = 50 + (i % 6) * 20
        
        # Simulate outcome (creative gain correlates with good conditions)
        # High gain when: medium novelty, high aesthetic, low volatility
        novelty_score = 1 - abs(state['novelty'] - 0.6) * 2  # Peak at 0.6
        aesthetic_score = state['aesthetic_score']
        volatility_penalty = 1 - state['volatility']
        
        gain = (novelty_score * 0.4 + aesthetic_score * 0.4 + volatility_penalty * 0.2) * factor
        gain += np.random.normal(0, 0.05)  # Add noise
        gain = np.clip(gain, 0, 1)
        
        # Record sample
        injection = feedback._create_injection(state, factor, duration)
        injection.performance_after = injection.performance_before + gain
        injection.creative_gain = gain
        
        feedback._update_quality_model(injection)
        
        training_samples.append({
            'state': state,
            'factor': factor,
            'duration': duration,
            'gain': gain
        })
    
    print(f"✅ Trained on {len(training_samples)} samples")
    
    # Test predictions
    print("\n🔮 Testing predictions on new states:")
    
    test_cases = [
        {
            'name': 'Ideal conditions',
            'state': {'novelty': 0.6, 'aesthetic_score': 0.8, 'volatility': 0.1},
            'factor': 0.6,
            'duration': 100
        },
        {
            'name': 'Poor conditions',
            'state': {'novelty': 0.9, 'aesthetic_score': 0.3, 'volatility': 0.8},
            'factor': 0.8,
            'duration': 150
        },
        {
            'name': 'Medium conditions',
            'state': {'novelty': 0.4, 'aesthetic_score': 0.6, 'volatility': 0.4},
            'factor': 0.5,
            'duration': 80
        }
    ]
    
    for test in test_cases:
        prediction = feedback._predict_creative_gain(
            test['state'], test['factor'], test['duration']
        )
        print(f"\n   {test['name']}:")
        print(f"   State: {test['state']}")
        print(f"   Predicted gain: {prediction:.3f}")
        
        # Show how this affects factor adjustment
        adjusted_factor = test['factor']
        if prediction < 0.1:
            adjusted_factor *= 0.7
        elif prediction > 0.5:
            adjusted_factor *= 1.2
        adjusted_factor = np.clip(adjusted_factor, 0.1, 0.9)
        
        print(f"   Factor adjustment: {test['factor']:.2f} → {adjusted_factor:.2f}")


async def demonstrate_full_integration():
    """Demonstrate all features working together"""
    print("\n" + "="*60)
    print("🚀 FULL INTEGRATION DEMONSTRATION")
    print("="*60)
    
    # Setup
    feedback = CreativeFeedbackLoop()
    system = SimulatedTORISystem()
    dashboard = CreativityDashboard()
    
    # Enable streaming
    feedback.enable_metric_streaming(dashboard.handle_metric, interval_steps=5)
    
    print("\n🎯 Running integrated creative feedback loop...")
    print("   - Automatic profile selection")
    print("   - Quality model predictions")
    print("   - Real-time metric streaming")
    print("\n" + "-"*40)
    
    # Run for multiple cycles
    for cycle in range(100):
        # Get system state
        state = system.get_state()
        
        # Update feedback system
        action = feedback.update(state)
        
        # Handle action
        if action['action'] == 'inject_entropy':
            # New injection started
            print(f"\n🎨 Step {cycle}: Starting new exploration")
            print(f"   Profile: {feedback.current_injection.entropy_profile}")
            print(f"   Factor: {action['entropy_factor']:.3f}")
            print(f"   Duration: {action['duration']} steps")
            
            # Apply to system
            system.apply_entropy(action['entropy_factor'])
            
        elif action['action'] == 'continue_exploration':
            # Ongoing injection - apply time-varying factor
            if feedback.current_injection:
                step = feedback.steps_in_mode
                current_factor = feedback._get_profile_factor(
                    feedback.current_injection.entropy_factor,
                    step,
                    feedback.current_injection.duration_steps,
                    feedback.current_injection.entropy_profile,
                    feedback.current_injection.profile_params
                )
                system.apply_entropy(current_factor * 0.1)  # Smaller continuous injection
                
        elif action['action'] == 'end_exploration':
            # Exploration completed
            print(f"\n✅ Step {cycle}: Exploration completed")
            if feedback.injection_history:
                last = feedback.injection_history[-1]
                print(f"   Creative gain: {last.creative_gain:.3f}")
                
        else:
            # Stable or recovering
            system.stabilize()
        
        # Small delay for realism
        await asyncio.sleep(0.05)
    
    print("\n" + "-"*40)
    print("📊 Final Statistics:")
    
    # Get final metrics
    final_metrics = feedback.get_creative_metrics()
    print(f"   Total explorations: {len(feedback.injection_history)}")
    print(f"   Cumulative gain: {final_metrics['cumulative_gain']:.3f}")
    print(f"   Quality model trained: {feedback.quality_model['trained']}")
    print(f"   Metrics streamed: {len(dashboard.metrics_history)}")
    
    # Plot results
    dashboard.plot_history()
    
    # Show exploration summary
    if feedback.injection_history:
        print("\n📋 Exploration Summary:")
        gains = [inj.creative_gain for inj in feedback.injection_history if inj.creative_gain]
        profiles = [inj.entropy_profile for inj in feedback.injection_history]
        
        print(f"   Average gain: {np.mean(gains):.3f}")
        print(f"   Best gain: {np.max(gains):.3f}")
        print(f"   Profile distribution:")
        for profile in set(profiles):
            count = profiles.count(profile)
            print(f"      {profile}: {count} times")


async def main():
    """Run all demonstrations"""
    print("🎨 CREATIVE FEEDBACK OPTIONAL ENHANCEMENTS DEMO")
    print("="*60)
    print("This demo shows:")
    print("1. Entropy Profiles - Time-varying injection patterns")
    print("2. Quality Model - Learning from exploration history")
    print("3. Metric Streaming - Real-time monitoring")
    print("="*60)
    
    # Run each demonstration
    await demonstrate_entropy_profiles()
    await asyncio.sleep(1)
    
    await demonstrate_quality_model()
    await asyncio.sleep(1)
    
    await demonstrate_full_integration()
    
    print("\n✨ Demo complete! Check creative_feedback_demo.png for visualization.")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
