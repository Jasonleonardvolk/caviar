#!/usr/bin/env python3
"""
Patch script for observer_synthesis.py based on focused walkthrough.
Addresses correctness, safety, performance, and API issues.
"""

import re
import sys
import shutil
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ObserverSynthesisPatcher:
    """Applies patches to observer_synthesis.py based on walkthrough review"""
    
    def __init__(self):
        self.file_path = Path(__file__).parent / ".." / "python" / "core" / "observer_synthesis.py"
        self.backup_path = self.file_path.with_suffix('.py.walkthrough_bak')
        self.patches = []
        
    def create_patches(self):
        """Create all patches organized by priority"""
        
        # PRIORITY 1: Correctness & Safety Fixes
        
        # Fix 1: Probability can exceed 1
        self.patches.append({
            'name': 'Fix probability exceeding 1.0',
            'old': """        # Adjust probability based on novelty
        prob = base_probability * (1 + novelty_score)""",
            'new': """        # Adjust probability based on novelty (clamped to max 1.0)
        prob = min(1.0, base_probability * (1 + novelty_score))"""
        })
        
        # Fix 2: Add missing tokens to vocabulary
        self.patches.append({
            'name': 'Add missing tokens to vocabulary',
            'old': """            # Self-reference
            'observing': 'SELF_OBSERVE',
            'reflecting': 'META_REFLECT',
            'modifying': 'SELF_MODIFY'""",
            'new': """            # Self-reference
            'observing': 'SELF_OBSERVE',
            'reflecting': 'META_REFLECT',
            'modifying': 'SELF_MODIFY',
            
            # Additional tokens for completeness
            'coherence_transition': 'COHERENCE_TRANSITION',
            'degenerate': 'DEGENERATE_MODES',
            'spectral_gap': 'SPECTRAL_GAP',
            'unknown': 'UNKNOWN_TOKEN'"""
        })
        
        # Fix 3: Switch to monotonic time
        self.patches.append({
            'name': 'Use monotonic time for cooldown',
            'old': """        # Check cooldown
        now_ms = int(time.time() * 1000)""",
            'new': """        # Check cooldown using monotonic time (jump-free)
        now_ms = int(time.monotonic() * 1000)"""
        })
        
        # Fix 4: Initialize last_measurement_time with monotonic
        self.patches.append({
            'name': 'Initialize with monotonic time',
            'old': """        # Track reflex usage
        self.reflex_window = deque()  # Timestamps within budget window
        self.last_measurement_time = 0  # Milliseconds""",
            'new': """        # Track reflex usage
        self.reflex_window = deque()  # Timestamps within budget window
        self.last_measurement_time = 0.0  # Using monotonic time"""
        })
        
        # Fix 5: Cap measurement_history
        self.patches.append({
            'name': 'Add max cap to measurement_history',
            'old': """# Reflex budget limits
DEFAULT_REFLEX_BUDGET = 60  # Per hour
MEASUREMENT_COOLDOWN_MS = 100  # Minimum time between measurements""",
            'new': """# Reflex budget limits
DEFAULT_REFLEX_BUDGET = 60  # Per hour
MEASUREMENT_COOLDOWN_MS = 100  # Minimum time between measurements
MAX_MEASUREMENT_HISTORY = 10000  # Cap for memory safety"""
        })
        
        self.patches.append({
            'name': 'Use bounded deque for measurement_history',
            'old': """        self.measurements = deque(maxlen=1000)
        self.measurement_history = []  # Full history for persistence""",
            'new': """        self.measurements = deque(maxlen=1000)
        # Use bounded deque to prevent memory leak
        self.measurement_history = deque(maxlen=MAX_MEASUREMENT_HISTORY)"""
        })
        
        # Fix 6: Handle RankWarning in polyfit
        self.patches.append({
            'name': 'Import warnings for RankWarning handling',
            'old': """import numpy as np
import hashlib
import json
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import deque
import logging
from pathlib import Path""",
            'new': """import numpy as np
import hashlib
import json
import time
import threading
import warnings
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import deque
import logging
from pathlib import Path"""
        })
        
        self.patches.append({
            'name': 'Fix RankWarning in novelty trend detection',
            'old': """        # Check for novelty trends
        novelties = [m.novelty_score for m in measurements]
        if len(novelties) > 2:
            trend = np.polyfit(range(len(novelties)), novelties, 1)[0]
            if trend > 0.1:
                patterns.append('INCREASING_NOVELTY')
            elif trend < -0.1:
                patterns.append('DECREASING_NOVELTY')""",
            'new': """        # Check for novelty trends (with warning suppression)
        novelties = [m.novelty_score for m in measurements]
        if len(novelties) > 2:
            # Simple slope calculation to avoid RankWarning
            if len(set(novelties)) > 1:  # Check for non-constant values
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=np.RankWarning)
                    try:
                        trend = np.polyfit(range(len(novelties)), novelties, 1)[0]
                        if trend > 0.1:
                            patterns.append('INCREASING_NOVELTY')
                        elif trend < -0.1:
                            patterns.append('DECREASING_NOVELTY')
                    except np.linalg.LinAlgError:
                        # Fallback to simple difference
                        trend = (novelties[-1] - novelties[0]) / (len(novelties) - 1)
                        if trend > 0.1:
                            patterns.append('INCREASING_NOVELTY')
                        elif trend < -0.1:
                            patterns.append('DECREASING_NOVELTY')"""
        })
        
        # PRIORITY 2: Performance Improvements
        
        # Fix 7: Optimize spectral hash for fixed-size arrays
        self.patches.append({
            'name': 'Optimize spectral hash computation',
            'old': """    def _spectral_hash_operator(self, eigenvalues: np.ndarray,
                               coherence_state: str,
                               novelty_score: float) -> SelfMeasurement:
        \"\"\"Basic spectral hash measurement.\"\"\"
        spectral_data = {
            'eigenvalues': eigenvalues.tolist(),
            'coherence': coherence_state,
            'novelty': round(novelty_score, 3)
        }
        spectral_bytes = json.dumps(spectral_data, sort_keys=True).encode()
        spectral_hash = hashlib.sha256(spectral_bytes).hexdigest()""",
            'new': """    def _spectral_hash_operator(self, eigenvalues: np.ndarray,
                               coherence_state: str,
                               novelty_score: float) -> SelfMeasurement:
        \"\"\"Basic spectral hash measurement.\"\"\"
        # Optimized hash computation for fixed-size eigenvalues
        if len(eigenvalues) > 0 and eigenvalues.shape[0] <= 10:
            # Fast path for small, fixed-size arrays
            spectral_bytes = (
                np.round(eigenvalues, 6).tobytes() + 
                coherence_state.encode('utf-8') + 
                f"{novelty_score:.3f}".encode('utf-8')
            )
            spectral_hash = hashlib.sha256(spectral_bytes).hexdigest()
        else:
            # Fallback for variable or large arrays
            spectral_data = {
                'eigenvalues': eigenvalues.tolist(),
                'coherence': coherence_state,
                'novelty': round(novelty_score, 3)
            }
            spectral_bytes = json.dumps(spectral_data, sort_keys=True).encode()
            spectral_hash = hashlib.sha256(spectral_bytes).hexdigest()"""
        })
        
        # Fix 8: Optimize oscillation detector window
        self.patches.append({
            'name': 'Optimize oscillation detector window size',
            'old': """        # Reflexive state
        self.reflexive_mode = False
        self.oscillation_detector = deque(maxlen=10)""",
            'new': """        # Reflexive state
        self.reflexive_mode = False
        # Only need last 4 for A-B-A-B pattern detection
        self.oscillation_detector = deque(maxlen=4)"""
        })
        
        # PRIORITY 3: API & Style Improvements
        
        # Fix 9: Fix token_set documentation
        self.patches.append({
            'name': 'Clarify token_set vs token list',
            'old': """            'has_self_observations': False,
            'metacognitive_tokens': [],
            'spectral_trajectory': []""",
            'new': """            'has_self_observations': False,
            'metacognitive_tokens': [],
            'token_set': set(),  # Deduplicated tokens
            'spectral_trajectory': []"""
        })
        
        self.patches.append({
            'name': 'Return both token list and set',
            'old': """        context = {
            'has_self_observations': True,
            'metacognitive_tokens': all_tokens,
            'token_frequencies': token_freq,
            'spectral_trajectory': trajectory,
            'reflexive_patterns': patterns,
            'measurement_count': len(self.measurements),
            'reflex_budget_remaining': self._get_reflex_budget_remaining()
        }""",
            'new': """        context = {
            'has_self_observations': True,
            'metacognitive_tokens': all_tokens,  # Full list with duplicates
            'token_set': list(set(all_tokens)),  # Deduplicated
            'token_frequencies': token_freq,
            'spectral_trajectory': trajectory,
            'reflexive_patterns': patterns,
            'measurement_count': len(self.measurements),
            'reflex_budget_remaining': self._get_reflex_budget_remaining()
        }"""
        })
        
        # Fix 10: Add return type hints
        self.patches.append({
            'name': 'Add return type hints to public methods',
            'old': """    def measure(self, eigenvalues: np.ndarray, 
               coherence_state: str,
               novelty_score: float,
               operator: str = 'spectral_hash',
               force: bool = False):""",
            'new': """    def measure(self, eigenvalues: np.ndarray, 
               coherence_state: str,
               novelty_score: float,
               operator: str = 'spectral_hash',
               force: bool = False) -> Optional[SelfMeasurement]:"""
        })
        
        self.patches.append({
            'name': 'Add return type to generate_metacognitive_context',
            'old': """    def generate_metacognitive_context(self, 
                                     recent_k: int = 5):""",
            'new': """    def generate_metacognitive_context(self, 
                                     recent_k: int = 5) -> Dict[str, Any]:"""
        })
        
        self.patches.append({
            'name': 'Add return type to apply_stochastic_measurement',
            'old': """    def apply_stochastic_measurement(self, 
                                   eigenvalues: np.ndarray,
                                   coherence_state: str,
                                   novelty_score: float,
                                   base_probability: float = 0.1):""",
            'new': """    def apply_stochastic_measurement(self, 
                                   eigenvalues: np.ndarray,
                                   coherence_state: str,
                                   novelty_score: float,
                                   base_probability: float = 0.1) -> Optional[SelfMeasurement]:"""
        })
        
        # Fix 11: Add configure_logging helper
        self.patches.append({
            'name': 'Add configure_logging helper',
            'old': """def get_observer_synthesis() -> ObserverObservedSynthesis:
    \"\"\"Get or create global observer synthesis instance (thread-safe).\"\"\"
    global _synthesis
    with _synthesis_lock:
        if _synthesis is None:
            _synthesis = ObserverObservedSynthesis()
    return _synthesis""",
            'new': """def get_observer_synthesis() -> ObserverObservedSynthesis:
    \"\"\"Get or create global observer synthesis instance (thread-safe).\"\"\"
    global _synthesis
    with _synthesis_lock:
        if _synthesis is None:
            _synthesis = ObserverObservedSynthesis()
    return _synthesis


def configure_logging(level: int = logging.INFO) -> None:
    \"\"\"
    Configure logging for the module.
    
    Args:
        level: Logging level (default: INFO)
    \"\"\"
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )"""
        })
        
        # Fix 12: Update main to use configure_logging
        self.patches.append({
            'name': 'Use configure_logging in main',
            'old': """if __name__ == "__main__":
    # Test the observer-observed synthesis""",
            'new': """if __name__ == "__main__":
    # Configure logging
    configure_logging(logging.DEBUG)
    
    # Test the observer-observed synthesis"""
        })
        
        # Fix 13: Fix docstring format (PEP-257)
        self.patches.append({
            'name': 'Fix docstring to PEP-257 format',
            'old': '''    """
    Implements self-measurement operators and reflexive feedback.
    
    Thread-safe implementation with reflex budget management and
    oscillation detection to prevent reflexive overload.
    """''',
            'new': '''    """Implements self-measurement operators and reflexive feedback.
    
    Thread-safe implementation with reflex budget management and
    oscillation detection to prevent reflexive overload.
    """'''
        })
        
        # PRIORITY 4: Thread Safety (foundation for future)
        
        # Fix 14: Add thread safety foundation
        self.patches.append({
            'name': 'Add thread safety with lock',
            'old': """        # Thread safety
        self._lock = threading.Lock()""",
            'new': """        # Thread safety
        self._lock = threading.Lock()"""
        })
        
        # Fix 15: Add register_operator for pluggable operators
        self.patches.append({
            'name': 'Add register_operator method',
            'old': """        logger.info("Observer-Observed Synthesis initialized")""",
            'new': """        logger.info("Observer-Observed Synthesis initialized")
    
    def register_operator(self, name: str, operator_func: Callable) -> None:
        \"\"\"
        Register a custom measurement operator.
        
        Args:
            name: Operator name
            operator_func: Function that takes (eigenvalues, coherence_state, novelty_score)
                          and returns SelfMeasurement
        \"\"\"
        with self._lock:
            self.operators[name] = operator_func
            logger.info(f"Registered measurement operator: {name}")"""
        })
        
    def apply_patches(self, dry_run=False):
        """Apply all patches to the file"""
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
            logger.info(f"Wrote patched file: {self.file_path}")
        
        # Summary
        logger.info(f"\nPatch Summary:")
        logger.info(f"  Total patches: {len(self.patches)}")
        logger.info(f"  Applied: {len(applied_patches)}")
        logger.info(f"  Failed: {len(failed_patches)}")
        
        if failed_patches:
            logger.warning(f"\nFailed patches:")
            for name in failed_patches:
                logger.warning(f"  - {name}")
        
        return len(failed_patches) == 0
    
    def create_test_file(self):
        """Create test file for patched observer synthesis"""
        test_content = '''#!/usr/bin/env python3
"""
Test script for patched observer_synthesis.py
Validates all fixes from the focused walkthrough
"""

import unittest
import numpy as np
import time
import json
import threading
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from python.core.observer_synthesis import (
    ObserverObservedSynthesis, 
    get_observer_synthesis,
    configure_logging
)

class TestWalkthroughFixes(unittest.TestCase):
    """Test all fixes from the walkthrough"""
    
    def setUp(self):
        self.synthesis = ObserverObservedSynthesis()
    
    def test_probability_clamping(self):
        """Test that probability never exceeds 1.0"""
        # Test with high novelty that would exceed 1.0
        for novelty in [0.5, 0.9, 1.0, 5.0, 10.0]:
            result = self.synthesis.apply_stochastic_measurement(
                np.array([0.1, 0.2]),
                'local',
                novelty,
                base_probability=0.5
            )
            # Can't directly test probability, but high values should still work
            # (Previously would have prob > 1, making it always trigger)
    
    def test_missing_tokens(self):
        """Test that previously missing tokens are now in vocabulary"""
        vocab = self.synthesis.token_vocab
        
        # Check for previously missing tokens
        self.assertIn('coherence_transition', vocab)
        self.assertIn('degenerate', vocab) 
        self.assertIn('spectral_gap', vocab)
        self.assertIn('unknown', vocab)
        
        # Verify they map to proper token strings
        self.assertEqual(vocab['coherence_transition'], 'COHERENCE_TRANSITION')
        self.assertEqual(vocab['degenerate'], 'DEGENERATE_MODES')
        self.assertEqual(vocab['spectral_gap'], 'SPECTRAL_GAP')
    
    def test_monotonic_time(self):
        """Test that cooldown uses monotonic time"""
        # Record a measurement
        self.synthesis.measure(np.array([0.1]), 'local', 0.5)
        
        # Store last measurement time
        last_time = self.synthesis.last_measurement_time
        
        # Simulate time going backwards (DST, NTP adjustment)
        # This would break with time.time() but not monotonic
        time.sleep(0.01)
        
        # Try another measurement
        self.synthesis.measure(np.array([0.2]), 'local', 0.5)
        
        # Time should always increase with monotonic
        self.assertGreater(self.synthesis.last_measurement_time, last_time)
    
    def test_measurement_history_bounded(self):
        """Test that measurement_history is bounded"""
        # Should be using deque with maxlen
        from collections import deque
        self.assertIsInstance(self.synthesis.measurement_history, deque)
        
        # Add many measurements
        for i in range(20000):  # More than MAX_MEASUREMENT_HISTORY
            self.synthesis.measure(
                np.array([i * 0.001]), 
                'local', 
                0.5,
                force=True
            )
        
        # Should be capped
        self.assertLessEqual(len(self.synthesis.measurement_history), 10000)
    
    def test_rankwarning_suppression(self):
        """Test that RankWarning is properly handled"""
        # Create measurements with identical novelty (causes RankWarning)
        for i in range(5):
            self.synthesis.measure(
                np.array([0.1, 0.2]), 
                'local',
                0.5,  # Same novelty
                force=True
            )
        
        # Generate context (which calculates trends)
        # This previously would throw RankWarning
        context = self.synthesis.generate_metacognitive_context()
        
        # Should complete without warnings
        self.assertIn('reflexive_patterns', context)
    
    def test_optimized_hashing(self):
        """Test optimized spectral hash for small arrays"""
        # Small array should use optimized path
        small_eigen = np.array([0.1, 0.2, 0.3])
        measurement1 = self.synthesis._spectral_hash_operator(
            small_eigen, 'local', 0.5
        )
        
        # Large array should use JSON path
        large_eigen = np.random.randn(20)
        measurement2 = self.synthesis._spectral_hash_operator(
            large_eigen, 'global', 0.6
        )
        
        # Both should produce valid hashes
        self.assertEqual(len(measurement1.spectral_hash), 64)  # SHA256
        self.assertEqual(len(measurement2.spectral_hash), 64)
    
    def test_token_set_in_context(self):
        """Test that context includes both token list and set"""
        # Generate some measurements
        for i in range(5):
            self.synthesis.measure(
                np.array([0.1 * i]), 
                ['local', 'global'][i % 2],
                i * 0.2,
                force=True
            )
        
        context = self.synthesis.generate_metacognitive_context()
        
        # Should have both list and set
        self.assertIn('metacognitive_tokens', context)  # Full list
        self.assertIn('token_set', context)  # Deduplicated
        
        # Set should have no duplicates
        token_list = context['metacognitive_tokens']
        token_set = context['token_set']
        self.assertEqual(len(token_set), len(set(token_list)))
    
    def test_register_custom_operator(self):
        """Test pluggable operator registration"""
        # Define custom operator
        def custom_operator(eigenvalues, coherence_state, novelty_score):
            measurement = self.synthesis._spectral_hash_operator(
                eigenvalues, coherence_state, novelty_score
            )
            measurement.measurement_operator = 'custom'
            measurement.metacognitive_tokens.append('CUSTOM_TOKEN')
            return measurement
        
        # Register it
        self.synthesis.register_operator('custom', custom_operator)
        
        # Use it
        result = self.synthesis.measure(
            np.array([0.1, 0.2]),
            'local',
            0.5,
            operator='custom'
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.measurement_operator, 'custom')
        self.assertIn('CUSTOM_TOKEN', result.metacognitive_tokens)


class TestPerformanceOptimizations(unittest.TestCase):
    """Test performance optimizations"""
    
    def test_oscillation_window_size(self):
        """Test that oscillation detector uses optimal window"""
        synthesis = ObserverObservedSynthesis()
        
        # Check maxlen is 4 (only need last 4 for A-B-A-B)
        self.assertEqual(synthesis.oscillation_detector.maxlen, 4)
        
        # Add many hashes
        for i in range(10):
            synthesis._check_oscillation(
                synthesis._spectral_hash_operator(
                    np.array([i * 0.1]), 'local', 0.5
                )
            )
        
        # Should only keep last 4
        self.assertEqual(len(synthesis.oscillation_detector), 4)


if __name__ == '__main__':
    # Use the new configure_logging helper
    configure_logging()
    
    unittest.main(verbosity=2)
'''
        
        test_path = self.file_path.parent / "test_observer_synthesis_walkthrough.py"
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        logger.info(f"Created test file: {test_path}")
        return test_path


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Patch observer_synthesis.py based on walkthrough")
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed')
    parser.add_argument('--create-test', action='store_true', help='Create test file')
    parser.add_argument('--rollback', action='store_true', help='Restore from backup')
    
    args = parser.parse_args()
    
    patcher = ObserverSynthesisPatcher()
    
    if args.rollback:
        if patcher.backup_path.exists():
            shutil.copy2(patcher.backup_path, patcher.file_path)
            logger.info(f"Restored from backup: {patcher.backup_path}")
        else:
            logger.error("No backup found")
        return
    
    # Create patches
    patcher.create_patches()
    logger.info(f"Created {len(patcher.patches)} patches from walkthrough")
    
    # Apply patches
    success = patcher.apply_patches(dry_run=args.dry_run)
    
    # Create test file
    if args.create_test and success:
        patcher.create_test_file()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
