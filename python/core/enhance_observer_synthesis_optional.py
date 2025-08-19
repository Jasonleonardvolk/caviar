#!/usr/bin/env python3
"""
Optional enhancements for observer_synthesis.py
Implements future features from the walkthrough:
1. Enhanced thread safety
2. Advanced budget strategies (token bucket)
3. Persistent history with rotation
"""

import re
import sys
import shutil
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ObserverSynthesisEnhancer:
    """Applies optional enhancements to observer_synthesis.py"""
    
    def __init__(self):
        self.file_path = Path(__file__).parent / ".." / "python" / "core" / "observer_synthesis.py"
        self.backup_path = self.file_path.with_suffix('.py.enhanced_bak')
        self.patches = []
        
    def create_patches(self):
        """Create patches for optional enhancements"""
        
        # Enhancement 1: Full thread safety
        self.patches.append({
            'name': 'Add comprehensive thread locking',
            'old': """    def measure(self, eigenvalues: np.ndarray, 
               coherence_state: str,
               novelty_score: float,
               operator: str = 'spectral_hash',
               force: bool = False) -> Optional[SelfMeasurement]:
        \"\"\"
        Perform self-measurement with reflex budget enforcement.
        
        Args:
            eigenvalues: Current eigenvalue spectrum
            coherence_state: Current coherence state
            novelty_score: Current novelty score
            operator: Measurement operator to use
            force: Bypass budget and cooldown checks
            
        Returns:
            SelfMeasurement if performed, None if blocked
        \"\"\"
        # Check cooldown using monotonic time (jump-free)
        now_ms = int(time.monotonic() * 1000)
        if not force and (now_ms - self.last_measurement_time) < MEASUREMENT_COOLDOWN_MS:
            return None""",
            'new': """    def measure(self, eigenvalues: np.ndarray, 
               coherence_state: str,
               novelty_score: float,
               operator: str = 'spectral_hash',
               force: bool = False) -> Optional[SelfMeasurement]:
        \"\"\"
        Perform self-measurement with reflex budget enforcement.
        
        Args:
            eigenvalues: Current eigenvalue spectrum
            coherence_state: Current coherence state
            novelty_score: Current novelty score
            operator: Measurement operator to use
            force: Bypass budget and cooldown checks
            
        Returns:
            SelfMeasurement if performed, None if blocked
        \"\"\"
        with self._lock:
            # Check cooldown using monotonic time (jump-free)
            now_ms = int(time.monotonic() * 1000)
            if not force and (now_ms - self.last_measurement_time) < MEASUREMENT_COOLDOWN_MS:
                return None"""
        })
        
        # Add thread safety to other methods
        self.patches.append({
            'name': 'Thread-safe generate_metacognitive_context',
            'old': """    def generate_metacognitive_context(self, 
                                     recent_k: int = 5) -> Dict[str, Any]:
        \"\"\"
        Generate metacognitive context from recent measurements.
        
        This context gets fed back into the reasoning system.
        
        Args:
            recent_k: Number of recent measurements to consider
            
        Returns:
            Dictionary containing metacognitive context with tokens,
            patterns, and trajectory information
        \"\"\"
        recent = list(self.measurements)[-recent_k:]""",
            'new': """    def generate_metacognitive_context(self, 
                                     recent_k: int = 5) -> Dict[str, Any]:
        \"\"\"
        Generate metacognitive context from recent measurements.
        
        This context gets fed back into the reasoning system.
        
        Args:
            recent_k: Number of recent measurements to consider
            
        Returns:
            Dictionary containing metacognitive context with tokens,
            patterns, and trajectory information
        \"\"\"
        with self._lock:
            recent = list(self.measurements)[-recent_k:]"""
        })
        
        # Enhancement 2: Token bucket budget strategy
        self.patches.append({
            'name': 'Add token bucket implementation',
            'old': """from pathlib import Path

logger = logging.getLogger(__name__)""",
            'new': """from pathlib import Path
import math

logger = logging.getLogger(__name__)"""
        })
        
        self.patches.append({
            'name': 'Initialize token bucket',
            'old': """        # Track reflex usage
        self.reflex_window = deque()  # Timestamps within budget window
        self.last_measurement_time = 0.0  # Using monotonic time""",
            'new': """        # Track reflex usage
        self.reflex_window = deque()  # Timestamps within budget window
        self.last_measurement_time = 0.0  # Using monotonic time
        
        # Token bucket for smooth rate limiting
        self.token_bucket = {
            'tokens': float(reflex_budget),
            'max_tokens': float(reflex_budget),
            'refill_rate': reflex_budget / 3600.0,  # tokens per second
            'last_refill': time.monotonic()
        }"""
        })
        
        self.patches.append({
            'name': 'Add token bucket methods',
            'old': """        logger.info("Observer-Observed Synthesis initialized")""",
            'new': """        logger.info("Observer-Observed Synthesis initialized")
    
    def _refill_token_bucket(self) -> None:
        \"\"\"Refill token bucket based on elapsed time.\"\"\"
        now = time.monotonic()
        elapsed = now - self.token_bucket['last_refill']
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.token_bucket['refill_rate']
        self.token_bucket['tokens'] = min(
            self.token_bucket['max_tokens'],
            self.token_bucket['tokens'] + tokens_to_add
        )
        self.token_bucket['last_refill'] = now
    
    def _consume_token(self) -> bool:
        \"\"\"Try to consume a token from the bucket.\"\"\"
        self._refill_token_bucket()
        
        if self.token_bucket['tokens'] >= 1.0:
            self.token_bucket['tokens'] -= 1.0
            return True
        return False"""
        })
        
        # Replace budget check with token bucket
        self.patches.append({
            'name': 'Use token bucket for budget',
            'old': """            # Check reflex budget
            if not force and not self._check_reflex_budget():
                logger.debug("Reflex budget exhausted")
                return None""",
            'new': """            # Check reflex budget (token bucket)
            if not force and not self._consume_token():
                logger.debug("Token bucket exhausted")
                return None"""
        })
        
        # Enhancement 3: Persistent history with rotation
        self.patches.append({
            'name': 'Add persistent history configuration',
            'old': """        # Token bucket for smooth rate limiting
        self.token_bucket = {
            'tokens': float(reflex_budget),
            'max_tokens': float(reflex_budget),
            'refill_rate': reflex_budget / 3600.0,  # tokens per second
            'last_refill': time.monotonic()
        }""",
            'new': """        # Token bucket for smooth rate limiting
        self.token_bucket = {
            'tokens': float(reflex_budget),
            'max_tokens': float(reflex_budget),
            'refill_rate': reflex_budget / 3600.0,  # tokens per second
            'last_refill': time.monotonic()
        }
        
        # Persistent history configuration
        self.persistence_config = {
            'enabled': False,
            'path': Path('observer_measurements'),
            'rotation_size': 100_000,  # Rotate after 100k measurements
            'rotation_age': 86400,  # Rotate daily (seconds)
            'last_rotation': time.time()
        }"""
        })
        
        self.patches.append({
            'name': 'Add persistence methods',
            'old': """    def _consume_token(self) -> bool:
        \"\"\"Try to consume a token from the bucket.\"\"\"
        self._refill_token_bucket()
        
        if self.token_bucket['tokens'] >= 1.0:
            self.token_bucket['tokens'] -= 1.0
            return True
        return False""",
            'new': """    def _consume_token(self) -> bool:
        \"\"\"Try to consume a token from the bucket.\"\"\"
        self._refill_token_bucket()
        
        if self.token_bucket['tokens'] >= 1.0:
            self.token_bucket['tokens'] -= 1.0
            return True
        return False
    
    def enable_persistence(self, path: Path, rotation_size: int = 100_000) -> None:
        \"\"\"Enable persistent history with rotation.\"\"\"
        self.persistence_config['enabled'] = True
        self.persistence_config['path'] = path
        self.persistence_config['rotation_size'] = rotation_size
        
        # Ensure directory exists
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Persistence enabled at {path}")
    
    def _should_rotate(self) -> bool:
        \"\"\"Check if history should be rotated.\"\"\"
        if not self.persistence_config['enabled']:
            return False
            
        # Check size threshold
        if len(self.measurement_history) >= self.persistence_config['rotation_size']:
            return True
            
        # Check age threshold
        now = time.time()
        if now - self.persistence_config['last_rotation'] > self.persistence_config['rotation_age']:
            return True
            
        return False
    
    def _rotate_history(self) -> None:
        \"\"\"Rotate measurement history to disk.\"\"\"
        if not self.persistence_config['enabled']:
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.persistence_config['path'] / f'measurements_{timestamp}.ndjson.gz'
        
        # Write compressed NDJSON
        import gzip
        with gzip.open(filename, 'wt') as f:
            for measurement in self.measurement_history:
                f.write(json.dumps(measurement.to_dict()) + '\\n')
        
        # Clear history
        self.measurement_history.clear()
        self.persistence_config['last_rotation'] = time.time()
        
        logger.info(f"Rotated history to {filename}")"""
        })
        
        # Add rotation check to measure
        self.patches.append({
            'name': 'Check rotation after measurement',
            'old': """            # Check for reflexive oscillation
            self._check_oscillation(measurement)
            
            logger.debug(f"Self-measurement performed: {measurement.spectral_hash[:8]}")
            
            return measurement""",
            'new': """            # Check for reflexive oscillation
            self._check_oscillation(measurement)
            
            # Check if history needs rotation
            if self._should_rotate():
                self._rotate_history()
            
            logger.debug(f"Self-measurement performed: {measurement.spectral_hash[:8]}")
            
            return measurement"""
        })
        
        # Add exponential backoff for errors
        self.patches.append({
            'name': 'Add exponential backoff state',
            'old': """        # Persistent history configuration
        self.persistence_config = {
            'enabled': False,
            'path': Path('observer_measurements'),
            'rotation_size': 100_000,  # Rotate after 100k measurements
            'rotation_age': 86400,  # Rotate daily (seconds)
            'last_rotation': time.time()
        }""",
            'new': """        # Persistent history configuration
        self.persistence_config = {
            'enabled': False,
            'path': Path('observer_measurements'),
            'rotation_size': 100_000,  # Rotate after 100k measurements
            'rotation_age': 86400,  # Rotate daily (seconds)
            'last_rotation': time.time()
        }
        
        # Exponential backoff for repeated failures
        self.error_backoff = {
            'consecutive_errors': 0,
            'backoff_seconds': 0.1,
            'max_backoff': 60.0,
            'last_error_time': 0.0
        }"""
        })
        
    def create_example(self):
        """Create example showing enhanced features"""
        example_content = '''#!/usr/bin/env python3
"""
Example of observer_synthesis with optional enhancements:
- Token bucket rate limiting
- Persistent history with rotation
- Thread-safe operations
"""

import time
import threading
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from observer_synthesis import ObserverObservedSynthesis, configure_logging

def test_token_bucket():
    """Test token bucket rate limiting"""
    print("\\n=== Testing Token Bucket ===")
    
    # Create with small bucket for testing
    synthesis = ObserverObservedSynthesis(reflex_budget=10)
    
    # Burst of measurements
    successes = 0
    for i in range(15):
        result = synthesis.measure(
            np.array([0.1 * i]),
            'local',
            0.5
        )
        if result:
            successes += 1
        time.sleep(0.01)
    
    print(f"Burst test: {successes}/15 succeeded (bucket size: 10)")
    
    # Wait for refill
    print("Waiting 2 seconds for token refill...")
    time.sleep(2)
    
    # Try again
    result = synthesis.measure(np.array([1.0]), 'local', 0.5)
    print(f"After refill: {'Success' if result else 'Failed'}")
    
    # Check token state
    synthesis._refill_token_bucket()
    print(f"Current tokens: {synthesis.token_bucket['tokens']:.2f}")


def test_persistent_history():
    """Test persistent history with rotation"""
    print("\\n=== Testing Persistent History ===")
    
    synthesis = ObserverObservedSynthesis()
    
    # Enable persistence
    history_path = Path("test_measurements")
    synthesis.enable_persistence(history_path, rotation_size=100)
    
    # Generate measurements to trigger rotation
    print("Generating 150 measurements...")
    for i in range(150):
        synthesis.measure(
            np.array([i * 0.001]),
            ['local', 'global', 'critical'][i % 3],
            (i % 10) / 10.0,
            force=True
        )
    
    # Check for rotation files
    rotation_files = list(history_path.glob("measurements_*.ndjson.gz"))
    print(f"Rotation files created: {len(rotation_files)}")
    
    if rotation_files:
        import gzip
        with gzip.open(rotation_files[0], 'rt') as f:
            lines = f.readlines()
        print(f"First rotation file contains {len(lines)} measurements")
    
    # Cleanup
    import shutil
    shutil.rmtree(history_path)


def test_thread_safety():
    """Test thread-safe concurrent operations"""
    print("\\n=== Testing Thread Safety ===")
    
    synthesis = ObserverObservedSynthesis()
    results = []
    errors = []
    
    def worker(thread_id):
        try:
            # Each thread does measurements and context generation
            for i in range(20):
                # Measurement
                measurement = synthesis.measure(
                    np.array([thread_id * 0.1, i * 0.01]),
                    ['local', 'global'][i % 2],
                    (thread_id + i) / 30.0,
                    force=True
                )
                
                # Context generation
                context = synthesis.generate_metacognitive_context()
                
                if measurement:
                    results.append((thread_id, measurement.spectral_hash[:8]))
                
                time.sleep(0.001)
                
        except Exception as e:
            errors.append((thread_id, str(e)))
    
    # Run concurrent threads
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, i) for i in range(5)]
        for future in futures:
            future.result()
    
    print(f"Concurrent results: {len(results)} measurements")
    print(f"Errors: {len(errors)}")
    print(f"Unique hashes: {len(set(h for _, h in results))}")


if __name__ == "__main__":
    configure_logging()
    
    print("Observer Synthesis Enhanced Features Demo")
    print("=" * 50)
    
    # Test each enhancement
    test_token_bucket()
    test_persistent_history()
    test_thread_safety()
    
    print("\\n✅ All enhanced features demonstrated!")
'''
        
        example_path = self.file_path.parent / "example_observer_enhanced.py"
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
        description="Apply optional enhancements to observer_synthesis.py"
    )
    parser.add_argument('--dry-run', action='store_true', help='Preview changes')
    parser.add_argument('--create-example', action='store_true', help='Create example')
    parser.add_argument('--rollback', action='store_true', help='Restore backup')
    
    args = parser.parse_args()
    
    enhancer = ObserverSynthesisEnhancer()
    
    if args.rollback:
        if enhancer.backup_path.exists():
            shutil.copy2(enhancer.backup_path, enhancer.file_path)
            logger.info(f"Restored from backup: {enhancer.backup_path}")
        else:
            logger.error("No backup found")
        return
    
    logger.info("Note: Apply walkthrough patches first!")
    
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
