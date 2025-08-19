#!/usr/bin/env python3
"""
Optional enhancements for creative_feedback.py
Implements future tweaks from the sharp review:
1. Entropy profiles (time-varying schedules)
2. Exploration quality model
3. Metric streaming
"""

import re
import sys
import shutil
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CreativeFeedbackEnhancer:
    """Applies optional enhancements to creative_feedback.py"""
    
    def __init__(self):
        self.file_path = Path(__file__).parent / ".." / "python" / "core" / "creative_feedback.py"
        self.backup_path = self.file_path.with_suffix('.py.enhanced_bak')
        self.patches = []
        
    def create_patches(self):
        """Create patches for optional enhancements"""
        
        # Enhancement 1: Entropy Profiles
        self.patches.append({
            'name': 'Add entropy profile support',
            'old': """@dataclass
class EntropyInjection:
    \"\"\"Record of an entropy injection event.\"\"\"
    start_time: datetime
    initial_state: Dict[str, Any]
    entropy_factor: float
    duration_steps: int
    performance_before: float
    performance_after: Optional[float] = None
    creative_gain: Optional[float] = None""",
            'new': """@dataclass
class EntropyInjection:
    \"\"\"Record of an entropy injection event.\"\"\"
    start_time: datetime
    initial_state: Dict[str, Any]
    entropy_factor: float
    duration_steps: int
    performance_before: float
    performance_after: Optional[float] = None
    creative_gain: Optional[float] = None
    entropy_profile: Optional[str] = 'constant'  # Profile type
    profile_params: Optional[Dict[str, float]] = None  # Profile parameters"""
        })
        
        self.patches.append({
            'name': 'Add entropy profile functions',
            'old': """    def _calculate_entropy_factor(self, state: Dict[str, Any]) -> float:
        \"\"\"Calculate appropriate entropy injection factor.\"\"\"
        novelty = state.get('novelty', 0.5)
        aesthetic = state.get('aesthetic_score', 0.5)
        
        # Base factor from novelty
        base_factor = 0.3 + (novelty * 0.4)
        
        # Modulate by aesthetic quality
        quality_mod = 1.0 + (aesthetic - 0.5) * 0.5
        
        # Sweet spot distance from target novelty
        novelty_target = 0.6
        diversity_score = max(0.0, 1 - abs(novelty_target - novelty))
        
        # Final factor
        factor = base_factor * quality_mod * (0.5 + diversity_score * 0.5)
        
        return np.clip(factor, 0.1, 0.9)""",
            'new': """    def _calculate_entropy_factor(self, state: Dict[str, Any]) -> float:
        \"\"\"Calculate appropriate entropy injection factor.\"\"\"
        novelty = state.get('novelty', 0.5)
        aesthetic = state.get('aesthetic_score', 0.5)
        
        # Base factor from novelty
        base_factor = 0.3 + (novelty * 0.4)
        
        # Modulate by aesthetic quality
        quality_mod = 1.0 + (aesthetic - 0.5) * 0.5
        
        # Sweet spot distance from target novelty
        novelty_target = 0.6
        diversity_score = max(0.0, 1 - abs(novelty_target - novelty))
        
        # Final factor
        factor = base_factor * quality_mod * (0.5 + diversity_score * 0.5)
        
        return np.clip(factor, 0.1, 0.9)
    
    def _get_profile_factor(self, base_factor: float, step: int, total_steps: int, 
                           profile: str = 'constant', params: Dict[str, float] = None) -> float:
        \"\"\"Apply time-varying profile to entropy factor.\"\"\"
        if params is None:
            params = {}
            
        progress = step / max(total_steps, 1)
        
        if profile == 'constant':
            return base_factor
            
        elif profile == 'cosine_ramp':
            # Smooth ramp up and down
            ramp_fraction = params.get('ramp_fraction', 0.2)
            if progress < ramp_fraction:
                # Ramp up
                t = progress / ramp_fraction
                scale = 0.5 * (1 - np.cos(np.pi * t))
            elif progress > (1 - ramp_fraction):
                # Ramp down
                t = (progress - (1 - ramp_fraction)) / ramp_fraction
                scale = 0.5 * (1 + np.cos(np.pi * t))
            else:
                scale = 1.0
            return base_factor * scale
            
        elif profile == 'exponential_decay':
            # Start high, decay over time
            decay_rate = params.get('decay_rate', 2.0)
            scale = np.exp(-decay_rate * progress)
            return base_factor * (0.5 + 0.5 * scale)
            
        elif profile == 'pulse':
            # Periodic pulses
            frequency = params.get('frequency', 3.0)
            scale = 0.5 * (1 + np.sin(2 * np.pi * frequency * progress))
            return base_factor * scale
            
        else:
            return base_factor"""
        })
        
        # Update inject_entropy to support profiles
        self.patches.append({
            'name': 'Add profile selection to inject_entropy',
            'old': """        # Create injection record
        self.current_injection = self._create_injection(
            base_state, 
            entropy_factor, 
            duration
        )""",
            'new': """        # Select entropy profile based on state
        profile = self._select_entropy_profile(base_state)
        profile_params = self._get_profile_params(profile, base_state)
        
        # Create injection record
        self.current_injection = self._create_injection(
            base_state, 
            entropy_factor, 
            duration,
            profile=profile,
            profile_params=profile_params
        )"""
        })
        
        # Add profile selection logic
        self.patches.append({
            'name': 'Add profile selection methods',
            'old': """    def _create_injection(self, state: Dict[str, Any], 
                        factor: float, duration: int) -> EntropyInjection:
        \"\"\"Create a new entropy injection record.\"\"\"
        return EntropyInjection(
            start_time=datetime.now(timezone.utc),
            initial_state=state.copy(),
            entropy_factor=factor,
            duration_steps=duration,
            performance_before=self._calculate_performance(state)
        )""",
            'new': """    def _select_entropy_profile(self, state: Dict[str, Any]) -> str:
        \"\"\"Select appropriate entropy profile based on state.\"\"\"
        novelty = state.get('novelty', 0.5)
        volatility = state.get('volatility', 0.0)
        
        # High volatility → exponential decay for stability
        if volatility > 0.7:
            return 'exponential_decay'
        # Low novelty → pulse to shake things up
        elif novelty < 0.3:
            return 'pulse'
        # Medium states → smooth cosine ramp
        elif 0.4 < novelty < 0.7:
            return 'cosine_ramp'
        # Default
        else:
            return 'constant'
    
    def _get_profile_params(self, profile: str, state: Dict[str, Any]) -> Dict[str, float]:
        \"\"\"Get profile-specific parameters.\"\"\"
        if profile == 'cosine_ramp':
            return {'ramp_fraction': 0.25}
        elif profile == 'exponential_decay':
            # Faster decay for higher volatility
            volatility = state.get('volatility', 0.5)
            return {'decay_rate': 1.0 + volatility * 3.0}
        elif profile == 'pulse':
            # More pulses for lower novelty
            novelty = state.get('novelty', 0.5)
            return {'frequency': 2.0 + (1 - novelty) * 3.0}
        else:
            return {}
    
    def _create_injection(self, state: Dict[str, Any], 
                        factor: float, duration: int,
                        profile: str = 'constant',
                        profile_params: Dict[str, float] = None) -> EntropyInjection:
        \"\"\"Create a new entropy injection record.\"\"\"
        return EntropyInjection(
            start_time=datetime.now(timezone.utc),
            initial_state=state.copy(),
            entropy_factor=factor,
            duration_steps=duration,
            performance_before=self._calculate_performance(state),
            entropy_profile=profile,
            profile_params=profile_params or {}
        )"""
        })
        
        # Enhancement 2: Exploration Quality Model
        self.patches.append({
            'name': 'Add quality model initialization',
            'old': """        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.injection_history = deque(maxlen=100)""",
            'new': """        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.injection_history = deque(maxlen=100)
        
        # Quality prediction model
        self.quality_model = {
            'trained': False,
            'feature_weights': None,
            'feature_history': deque(maxlen=500),
            'min_samples': 20
        }"""
        })
        
        self.patches.append({
            'name': 'Add quality model methods',
            'old': """        return {
            'entropy_factor': entropy_factor,
            'lambda_factor': entropy_factor * 1.5,
            'duration': duration,
            'action': 'inject_entropy'
        }""",
            'new': """        # Use quality model if available
        if self.quality_model['trained']:
            predicted_gain = self._predict_creative_gain(base_state, entropy_factor, duration)
            logger.debug(f"Quality model predicts gain: {predicted_gain:.3f}")
            
            # Adjust factor based on prediction
            if predicted_gain < 0.1:
                entropy_factor *= 0.7  # Reduce if low gain expected
            elif predicted_gain > 0.5:
                entropy_factor *= 1.2  # Boost if high gain expected
                
            entropy_factor = np.clip(entropy_factor, 0.1, 0.9)
        
        return {
            'entropy_factor': entropy_factor,
            'lambda_factor': entropy_factor * 1.5,
            'duration': duration,
            'action': 'inject_entropy'
        }"""
        })
        
        self.patches.append({
            'name': 'Add quality model training',
            'old': """        # Update history
        self.injection_history.append(self.current_injection)""",
            'new': """        # Update history
        self.injection_history.append(self.current_injection)
        
        # Update quality model
        self._update_quality_model(self.current_injection)"""
        })
        
        self.patches.append({
            'name': 'Add quality model implementation',
            'old': """    def get_creative_metrics(self) -> Dict[str, Any]:""",
            'new': """    def _extract_features(self, state: Dict[str, Any], factor: float, duration: int) -> np.ndarray:
        \"\"\"Extract features for quality prediction.\"\"\"
        return np.array([
            state.get('novelty', 0.5),
            state.get('aesthetic_score', 0.5),
            state.get('volatility', 0.0),
            factor,
            duration / self.max_exploration_steps,
            len(self.injection_history) / 100.0,  # Exploration count
        ])
    
    def _predict_creative_gain(self, state: Dict[str, Any], factor: float, duration: int) -> float:
        \"\"\"Predict creative gain using simple linear model.\"\"\"
        if not self.quality_model['trained'] or self.quality_model['feature_weights'] is None:
            return 0.3  # Default prediction
            
        features = self._extract_features(state, factor, duration)
        weights = self.quality_model['feature_weights']
        
        # Simple linear prediction
        prediction = np.dot(features, weights)
        return np.clip(prediction, 0.0, 1.0)
    
    def _update_quality_model(self, injection: EntropyInjection) -> None:
        \"\"\"Update quality model with completed injection.\"\"\"
        if injection.creative_gain is None:
            return
            
        # Store feature-outcome pair
        features = self._extract_features(
            injection.initial_state,
            injection.entropy_factor,
            injection.duration_steps
        )
        
        self.quality_model['feature_history'].append({
            'features': features,
            'gain': injection.creative_gain
        })
        
        # Train when enough samples
        if len(self.quality_model['feature_history']) >= self.quality_model['min_samples']:
            self._train_quality_model()
    
    def _train_quality_model(self) -> None:
        \"\"\"Train simple linear model on historical data.\"\"\"
        history = list(self.quality_model['feature_history'])
        
        X = np.array([h['features'] for h in history])
        y = np.array([h['gain'] for h in history])
        
        # Simple least squares (add small regularization)
        XtX = X.T @ X + 0.01 * np.eye(X.shape[1])
        Xty = X.T @ y
        
        try:
            weights = np.linalg.solve(XtX, Xty)
            self.quality_model['feature_weights'] = weights
            self.quality_model['trained'] = True
            logger.info("Quality model updated")
        except np.linalg.LinAlgError:
            logger.warning("Quality model training failed")
    
    def get_creative_metrics(self) -> Dict[str, Any]:"""
        })
        
        # Enhancement 3: Metric Streaming
        self.patches.append({
            'name': 'Add metric streaming support',
            'old': """        # Quality prediction model
        self.quality_model = {
            'trained': False,
            'feature_weights': None,
            'feature_history': deque(maxlen=500),
            'min_samples': 20
        }""",
            'new': """        # Quality prediction model
        self.quality_model = {
            'trained': False,
            'feature_weights': None,
            'feature_history': deque(maxlen=500),
            'min_samples': 20
        }
        
        # Metric streaming
        self.metric_stream = {
            'enabled': False,
            'callback': None,
            'interval_steps': 10,
            'last_stream_step': 0
        }"""
        })
        
        self.patches.append({
            'name': 'Add metric streaming methods',
            'old': """    def get_creative_metrics(self) -> Dict[str, Any]:""",
            'new': """    def enable_metric_streaming(self, callback: Callable, interval_steps: int = 10) -> None:
        \"\"\"Enable metric streaming to external system.\"\"\"
        self.metric_stream['enabled'] = True
        self.metric_stream['callback'] = callback
        self.metric_stream['interval_steps'] = interval_steps
        logger.info(f"Metric streaming enabled with interval {interval_steps}")
    
    def disable_metric_streaming(self) -> None:
        \"\"\"Disable metric streaming.\"\"\"
        self.metric_stream['enabled'] = False
        self.metric_stream['callback'] = None
        logger.info("Metric streaming disabled")
    
    async def _stream_metrics(self) -> None:
        \"\"\"Stream metrics if enabled and interval reached.\"\"\"
        if not self.metric_stream['enabled'] or not self.metric_stream['callback']:
            return
            
        if self.steps_in_mode - self.metric_stream['last_stream_step'] >= self.metric_stream['interval_steps']:
            metrics = self.get_creative_metrics()
            
            # Add real-time fields
            metrics['timestamp'] = datetime.now(timezone.utc).isoformat()
            metrics['stream_sequence'] = self.metric_stream.get('sequence', 0)
            self.metric_stream['sequence'] = metrics['stream_sequence'] + 1
            
            try:
                # Async callback for non-blocking
                if asyncio.iscoroutinefunction(self.metric_stream['callback']):
                    await self.metric_stream['callback'](metrics)
                else:
                    self.metric_stream['callback'](metrics)
                    
                self.metric_stream['last_stream_step'] = self.steps_in_mode
            except Exception as e:
                logger.error(f"Metric streaming error: {e}")
    
    def get_creative_metrics(self) -> Dict[str, Any]:"""
        })
        
        # Add streaming to update method
        self.patches.append({
            'name': 'Add streaming to update',
            'old': """        # Record performance
        self.performance_history.append({
            'novelty': current_state.get('novelty', 0.0),
            'aesthetic_score': current_state.get('aesthetic_score', 0.5)
        })""",
            'new': """        # Record performance
        self.performance_history.append({
            'novelty': current_state.get('novelty', 0.0),
            'aesthetic_score': current_state.get('aesthetic_score', 0.5)
        })
        
        # Stream metrics if enabled
        if self.metric_stream['enabled']:
            import asyncio
            if asyncio.iscoroutinefunction(self._stream_metrics):
                # If in async context
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self._stream_metrics())
                except RuntimeError:
                    # Fallback for sync context
                    pass"""
        })
        
        # Update inject_controlled_entropy to use profile
        self.patches.append({
            'name': 'Use entropy profile in inject_controlled_entropy',
            'old': """        # Apply entropy based on factor
        enhanced_state['entropy_level'] = entropy_level
        enhanced_state['lambda_factor'] = 1.0 + (entropy_level * 0.5)""",
            'new': """        # Apply entropy based on factor and profile
        if hasattr(self, 'current_injection') and self.current_injection:
            # Apply time-varying profile
            step = self.steps_in_mode
            profile_factor = self._get_profile_factor(
                entropy_level,
                step,
                self.current_injection.duration_steps,
                self.current_injection.entropy_profile,
                self.current_injection.profile_params
            )
            enhanced_state['entropy_level'] = profile_factor
            enhanced_state['lambda_factor'] = 1.0 + (profile_factor * 0.5)
        else:
            enhanced_state['entropy_level'] = entropy_level
            enhanced_state['lambda_factor'] = 1.0 + (entropy_level * 0.5)"""
        })
        
        # Add asyncio import
        self.patches.append({
            'name': 'Add asyncio import',
            'old': """import warnings

logger = logging.getLogger(__name__)""",
            'new': """import warnings
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Callable

logger = logging.getLogger(__name__)"""
        })
        
    def create_example(self):
        """Create example showing enhanced features"""
        example_content = '''#!/usr/bin/env python3
"""
Example of creative_feedback with optional enhancements:
- Entropy profiles (time-varying injection)
- Quality prediction model
- Metric streaming
"""

import asyncio
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from python.core.creative_feedback import (
    CreativeFeedbackLoop,
    get_creative_feedback,
    configure_logging
)

# Configure logging
configure_logging()

def demonstrate_entropy_profiles():
    """Show different entropy injection profiles"""
    print("\\n=== Entropy Profiles Demo ===")
    
    feedback = CreativeFeedbackLoop()
    
    # Test different profiles
    profiles = ['constant', 'cosine_ramp', 'exponential_decay', 'pulse']
    
    for profile in profiles:
        print(f"\\nProfile: {profile}")
        
        # Simulate steps with profile
        steps = 50
        factors = []
        
        for step in range(steps):
            factor = feedback._get_profile_factor(
                base_factor=0.5,
                step=step,
                total_steps=steps,
                profile=profile,
                params={'ramp_fraction': 0.2, 'decay_rate': 2.0, 'frequency': 3.0}
            )
            factors.append(factor)
        
        # Show pattern
        print(f"  Start: {factors[0]:.3f}")
        print(f"  Mid:   {factors[steps//2]:.3f}")
        print(f"  End:   {factors[-1]:.3f}")
        print(f"  Avg:   {np.mean(factors):.3f}")


def demonstrate_quality_model():
    """Show quality prediction model in action"""
    print("\\n=== Quality Model Demo ===")
    
    feedback = CreativeFeedbackLoop()
    
    # Simulate historical injections to train model
    print("Training quality model...")
    
    for i in range(25):
        # Create varied states
        state = {
            'novelty': 0.3 + (i % 5) * 0.1,
            'aesthetic_score': 0.5 + (i % 3) * 0.2,
            'volatility': (i % 4) * 0.25
        }
        
        # Simulate injection with outcome
        factor = 0.4 + (i % 3) * 0.2
        duration = 50 + (i % 5) * 20
        
        # Fake creative gain (higher for mid-range novelty)
        gain = 0.5 * (1 - abs(state['novelty'] - 0.6)) + 0.3 * state['aesthetic_score']
        
        # Create and complete injection
        injection = feedback._create_injection(state, factor, duration)
        injection.performance_after = injection.performance_before + gain
        injection.creative_gain = gain
        
        # Update model
        feedback._update_quality_model(injection)
    
    # Test predictions
    print("\\nTesting predictions:")
    
    test_states = [
        {'novelty': 0.3, 'aesthetic_score': 0.5, 'volatility': 0.2},
        {'novelty': 0.6, 'aesthetic_score': 0.8, 'volatility': 0.1},
        {'novelty': 0.9, 'aesthetic_score': 0.3, 'volatility': 0.8},
    ]
    
    for state in test_states:
        prediction = feedback._predict_creative_gain(state, 0.5, 100)
        print(f"  State: novelty={state['novelty']:.1f}, aesthetic={state['aesthetic_score']:.1f}")
        print(f"  Predicted gain: {prediction:.3f}")


async def demonstrate_metric_streaming():
    """Show real-time metric streaming"""
    print("\\n=== Metric Streaming Demo ===")
    
    feedback = CreativeFeedbackLoop()
    
    # Collected metrics
    streamed_metrics = []
    
    # Define streaming callback
    async def metric_callback(metrics):
        print(f"  Streamed metric #{metrics['stream_sequence']}")
        streamed_metrics.append(metrics)
    
    # Enable streaming
    feedback.enable_metric_streaming(metric_callback, interval_steps=5)
    
    # Simulate updates
    print("Simulating 20 update cycles...")
    
    for i in range(20):
        state = {
            'novelty': 0.5 + 0.1 * np.sin(i * 0.5),
            'aesthetic_score': 0.6 + 0.2 * np.cos(i * 0.3),
            'emergency_override': False
        }
        
        result = feedback.update(state)
        
        # Small delay
        await asyncio.sleep(0.1)
    
    print(f"\\nTotal metrics streamed: {len(streamed_metrics)}")
    
    # Show last metric
    if streamed_metrics:
        last = streamed_metrics[-1]
        print(f"Last metric: mode={last['mode']}, gain={last['cumulative_gain']:.3f}")
    
    # Disable streaming
    feedback.disable_metric_streaming()


async def main():
    """Run all demonstrations"""
    print("Creative Feedback Enhanced Features Demo")
    print("=" * 50)
    
    # Run demos
    demonstrate_entropy_profiles()
    demonstrate_quality_model()
    await demonstrate_metric_streaming()
    
    print("\\n✅ All enhanced features demonstrated!")


if __name__ == "__main__":
    asyncio.run(main())
'''
        
        example_path = self.file_path.parent / "example_creative_enhanced.py"
        with open(example_path, 'w', encoding='utf-8') as f:
            f.write(example_content)
        
        logger.info(f"Created example: {example_path}")
        return example_path
    
    def apply_patches(self, dry_run=False):
        """Apply enhancement patches"""
        if not self.file_path.exists():
            logger.error(f"File not found: {self.file_path}")
            return False
            
        # Create backup
        if not dry_run:
            shutil.copy2(self.file_path, self.backup_path)
            logger.info(f"Created backup: {self.backup_path}")
        
        # Read content
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        applied_patches = []
        failed_patches = []
        
        # Apply patches
        for patch in self.patches:
            if patch['old'] in content:
                if not dry_run:
                    content = content.replace(patch['old'], patch['new'])
                applied_patches.append(patch['name'])
                logger.info(f"✓ Applied: {patch['name']}")
            else:
                failed_patches.append(patch['name'])
                logger.warning(f"✗ Pattern not found: {patch['name']}")
        
        # Write back
        if not dry_run and content != original_content:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Wrote enhanced file: {self.file_path}")
        
        # Summary
        logger.info(f"\nEnhancement Summary:")
        logger.info(f"  Total patches: {len(self.patches)}")
        logger.info(f"  Applied: {len(applied_patches)}")
        logger.info(f"  Failed: {len(failed_patches)}")
        
        return len(failed_patches) == 0


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Apply optional enhancements to creative_feedback.py"
    )
    parser.add_argument('--dry-run', action='store_true', help='Preview changes')
    parser.add_argument('--create-example', action='store_true', help='Create example')
    parser.add_argument('--rollback', action='store_true', help='Restore backup')
    
    args = parser.parse_args()
    
    enhancer = CreativeFeedbackEnhancer()
    
    if args.rollback:
        if enhancer.backup_path.exists():
            shutil.copy2(enhancer.backup_path, enhancer.file_path)
            logger.info(f"Restored from backup: {enhancer.backup_path}")
        else:
            logger.error("No backup found")
        return
    
    logger.info("Note: Apply sharp review patches first!")
    
    # Create patches
    enhancer.create_patches()
    logger.info(f"Created {len(enhancer.patches)} enhancement patches")
    
    # Apply patches
    success = enhancer.apply_patches(dry_run=args.dry_run)
    
    # Create example
    if args.create_example and success:
        enhancer.create_example()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
