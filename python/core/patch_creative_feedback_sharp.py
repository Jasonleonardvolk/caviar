#!/usr/bin/env python3
"""
Patch script for creative_feedback.py based on sharp-edged review.
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

class CreativeFeedbackPatcher:
    """Applies patches to creative_feedback.py based on sharp review"""
    
    def __init__(self):
        self.file_path = Path(__file__).parent / ".." / "python" / "core" / "creative_feedback.py"
        self.backup_path = self.file_path.with_suffix('.py.sharp_bak')
        self.patches = []
        
    def create_patches(self):
        """Create all patches organized by priority"""
        
        # PRIORITY 1: Correctness & Safety Fixes
        
        # Fix 1: steps_in_mode increment logic
        self.patches.append({
            'name': 'Fix steps_in_mode increment logic',
            'old': """    def update(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Update creative feedback state and determine next action.\"\"\"
        self.steps_in_mode += 1""",
            'new': """    def update(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Update creative feedback state and determine next action.\"\"\""""
        })
        
        # Add increment in correct places
        self.patches.append({
            'name': 'Add steps_in_mode increment in STABLE branch',
            'old': """        if self.mode == CreativeMode.STABLE:
            if self._should_inject_entropy(current_state):""",
            'new': """        if self.mode == CreativeMode.STABLE:
            self.steps_in_mode += 1
            if self._should_inject_entropy(current_state):"""
        })
        
        self.patches.append({
            'name': 'Add steps_in_mode increment in EXPLORING branch',
            'old': """        elif self.mode == CreativeMode.EXPLORING:
            # Check if exploration should end""",
            'new': """        elif self.mode == CreativeMode.EXPLORING:
            self.steps_in_mode += 1
            # Check if exploration should end"""
        })
        
        self.patches.append({
            'name': 'Add steps_in_mode increment in RECOVERING branch',
            'old': """        elif self.mode == CreativeMode.RECOVERING:
            # Simple recovery: just wait""",
            'new': """        elif self.mode == CreativeMode.RECOVERING:
            self.steps_in_mode += 1
            # Simple recovery: just wait"""
        })
        
        # Fix 2: Duration steps bounds checking
        self.patches.append({
            'name': 'Fix duration_steps bounds',
            'old': """            duration = min(
                self.max_exploration_steps,
                100 + int(novelty * 200)
            )""",
            'new': """            duration = max(10, min(
                self.max_exploration_steps,
                100 + int(novelty * 200)
            ))"""
        })
        
        # Fix 3: Emergency override cancels exploration
        self.patches.append({
            'name': 'Cancel exploration on emergency',
            'old': """        # Emergency override
        if current_state.get('emergency_override', False):
            self.mode = CreativeMode.EMERGENCY
            self.steps_in_mode = 0
            return {'action': 'emergency_halt'}""",
            'new': """        # Emergency override
        if current_state.get('emergency_override', False):
            # End current exploration if active
            if self.mode == CreativeMode.EXPLORING and self.current_injection:
                self._end_exploration({})
            
            self.mode = CreativeMode.EMERGENCY
            self.steps_in_mode = 0
            return {'action': 'emergency_halt'}"""
        })
        
        # Fix 4: Set baseline aesthetic score
        self.patches.append({
            'name': 'Set baseline aesthetic score',
            'old': """        if self.mode == CreativeMode.STABLE:
            self.steps_in_mode += 1
            if self._should_inject_entropy(current_state):
                # Prepare for injection
                self.mode = CreativeMode.EXPLORING
                self.steps_in_mode = 0
                return self.inject_entropy(current_state)
            else:
                return {'action': 'maintain'}""",
            'new': """        if self.mode == CreativeMode.STABLE:
            self.steps_in_mode += 1
            
            # Set baseline aesthetic score on first stable evaluation
            if 'aesthetic_score' in current_state and self.regularizer.baseline_performance.get('score', 0.5) == 0.5:
                self.regularizer.baseline_performance['score'] = current_state['aesthetic_score']
            
            if self._should_inject_entropy(current_state):
                # Prepare for injection
                self.mode = CreativeMode.EXPLORING
                self.steps_in_mode = 0
                return self.inject_entropy(current_state)
            else:
                return {'action': 'maintain'}"""
        })
        
        # Fix 5: Clamp diversity score
        self.patches.append({
            'name': 'Clamp diversity score to valid range',
            'old': """        # Sweet spot distance from target novelty
        novelty_target = 0.6
        diversity_score = 1 - abs(novelty_target - novelty)""",
            'new': """        # Sweet spot distance from target novelty
        novelty_target = 0.6
        diversity_score = max(0.0, 1 - abs(novelty_target - novelty))"""
        })
        
        # Fix 6: Add JSON serialization helper
        self.patches.append({
            'name': 'Add JSON serialization for datetime',
            'old': """        return {
            'mode': self.mode.value,
            'steps_in_mode': self.steps_in_mode,
            'performance_history': list(self.performance_history),
            'current_injection': self.current_injection.to_dict() if self.current_injection else None,
            'cumulative_gain': self._calculate_cumulative_gain()
        }""",
            'new': """        # Convert datetime objects for JSON serialization
        history_serializable = []
        for entry in self.performance_history:
            entry_copy = entry.copy()
            if 'timestamp' in entry_copy and hasattr(entry_copy['timestamp'], 'isoformat'):
                entry_copy['timestamp'] = entry_copy['timestamp'].isoformat()
            history_serializable.append(entry_copy)
        
        return {
            'mode': self.mode.value,
            'steps_in_mode': self.steps_in_mode,
            'performance_history': history_serializable,
            'current_injection': self.current_injection.to_dict() if self.current_injection else None,
            'cumulative_gain': self._calculate_cumulative_gain()
        }"""
        })
        
        # PRIORITY 2: Performance Improvements
        
        # Fix 7: Replace polyfit with simple slope
        self.patches.append({
            'name': 'Import warnings for polyfit handling',
            'old': """import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import deque
from enum import Enum
import logging""",
            'new': """import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import deque
from enum import Enum
import logging
import warnings"""
        })
        
        self.patches.append({
            'name': 'Replace polyfit with simple slope calculation',
            'old': """        # Trend analysis
        if len(recent_novelties) > 1:
            novelty_trend = np.polyfit(range(len(recent_novelties)), recent_novelties, 1)[0]
        else:
            novelty_trend = 0.0
            
        if len(recent_aesthetics) > 1:
            aesthetic_trend = np.polyfit(range(len(recent_aesthetics)), recent_aesthetics, 1)[0]
        else:
            aesthetic_trend = 0.0""",
            'new': """        # Trend analysis using simple slope to avoid RankWarning
        if len(recent_novelties) > 1:
            novelty_trend = (recent_novelties[-1] - recent_novelties[0]) / (len(recent_novelties) - 1)
        else:
            novelty_trend = 0.0
            
        if len(recent_aesthetics) > 1:
            aesthetic_trend = (recent_aesthetics[-1] - recent_aesthetics[0]) / (len(recent_aesthetics) - 1)
        else:
            aesthetic_trend = 0.0"""
        })
        
        # PRIORITY 3: API & Style Improvements
        
        # Fix 8: Add Enum for actions
        self.patches.append({
            'name': 'Add Action enum',
            'old': """class CreativeMode(Enum):
    STABLE = "stable"
    EXPLORING = "exploring"
    RECOVERING = "recovering"
    EMERGENCY = "emergency\"""",
            'new': """class CreativeMode(Enum):
    STABLE = "stable"
    EXPLORING = "exploring"
    RECOVERING = "recovering"
    EMERGENCY = "emergency"


class CreativeAction(Enum):
    MAINTAIN = "maintain"
    INJECT_ENTROPY = "inject_entropy"
    CONTINUE_EXPLORATION = "continue_exploration"
    END_EXPLORATION = "end_exploration"
    RECOVER = "recover"
    EMERGENCY_HALT = "emergency_halt\""""
        })
        
        # Update return statements to use enum
        self.patches.append({
            'name': 'Use Action enum in returns',
            'old': """                return {'action': 'maintain'}""",
            'new': """                return {'action': CreativeAction.MAINTAIN.value}"""
        })
        
        self.patches.append({
            'name': 'Use Action enum for emergency',
            'old': """            return {'action': 'emergency_halt'}""",
            'new': """            return {'action': CreativeAction.EMERGENCY_HALT.value}"""
        })
        
        self.patches.append({
            'name': 'Use Action enum for continue',
            'old': """                    return {'action': 'continue_exploration'}""",
            'new': """                    return {'action': CreativeAction.CONTINUE_EXPLORATION.value}"""
        })
        
        self.patches.append({
            'name': 'Use Action enum for recover',
            'old': """                return {'action': 'recover'}""",
            'new': """                return {'action': CreativeAction.RECOVER.value}"""
        })
        
        # Fix 9: Add configure_logging helper
        self.patches.append({
            'name': 'Add configure_logging helper',
            'old': """# Global instance
_feedback_loop = None


def get_creative_feedback() -> CreativeFeedbackLoop:
    \"\"\"Get or create global creative feedback loop.\"\"\"
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = CreativeFeedbackLoop()
    return _feedback_loop""",
            'new': """# Global instance
_feedback_loop = None


def get_creative_feedback() -> CreativeFeedbackLoop:
    \"\"\"Get or create global creative feedback loop.\"\"\"
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = CreativeFeedbackLoop()
    return _feedback_loop


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
        
        # Fix 10: Add type hints
        self.patches.append({
            'name': 'Add return type to inject_entropy',
            'old': """    def inject_entropy(self, base_state: Dict[str, Any]):""",
            'new': """    def inject_entropy(self, base_state: Dict[str, Any]) -> Dict[str, Any]:"""
        })
        
        self.patches.append({
            'name': 'Add return type to _end_exploration',
            'old': """    def _end_exploration(self, final_state: Dict[str, Any]):""",
            'new': """    def _end_exploration(self, final_state: Dict[str, Any]) -> None:"""
        })
        
        # Fix 11: Fix docstring format (PEP-257)
        self.patches.append({
            'name': 'Fix docstring to PEP-257 format',
            'old': '''    """
    Manages creative entropy injection cycles.
    
    Monitors system aesthetics and injects controlled randomness
    when creative potential is detected.
    """''',
            'new': '''    """Manages creative entropy injection cycles.
    
    Monitors system aesthetics and injects controlled randomness
    when creative potential is detected.
    """'''
        })
        
        # Fix 12: Update main to use configure_logging
        self.patches.append({
            'name': 'Use configure_logging in main',
            'old': """if __name__ == "__main__":
    # Test the creative feedback loop""",
            'new': """if __name__ == "__main__":
    # Configure logging
    configure_logging(logging.DEBUG)
    
    # Test the creative feedback loop"""
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
        """Create test file for patched creative_feedback"""
        test_content = '''#!/usr/bin/env python3
"""
Test script for patched creative_feedback.py
Validates all fixes from the sharp-edged review
"""

import unittest
import numpy as np
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from python.core.creative_feedback import (
    CreativeFeedbackLoop,
    CreativeMode,
    CreativeAction,
    get_creative_feedback,
    configure_logging
)

class TestCreativeFeedbackFixes(unittest.TestCase):
    """Test all fixes from the sharp review"""
    
    def setUp(self):
        self.feedback = CreativeFeedbackLoop()
    
    def test_steps_in_mode_increment(self):
        """Test that steps_in_mode increments correctly"""
        # Start in STABLE
        self.assertEqual(self.feedback.mode, CreativeMode.STABLE)
        self.assertEqual(self.feedback.steps_in_mode, 0)
        
        # First update in STABLE
        state = {'novelty': 0.3, 'aesthetic_score': 0.7}
        self.feedback.update(state)
        self.assertEqual(self.feedback.steps_in_mode, 1)
        
        # Another update
        self.feedback.update(state)
        self.assertEqual(self.feedback.steps_in_mode, 2)
        
        # Force mode change to EXPLORING
        state['novelty'] = 0.8
        result = self.feedback.update(state)
        
        # Should reset to 0 when changing modes
        if self.feedback.mode == CreativeMode.EXPLORING:
            self.assertEqual(self.feedback.steps_in_mode, 0)
    
    def test_duration_bounds(self):
        """Test that duration_steps has minimum bound"""
        # Mock _should_inject_entropy to return True
        self.feedback._should_inject_entropy = lambda x: True
        
        # Very low novelty should still have min duration
        state = {'novelty': 0.0, 'aesthetic_score': 0.8}
        result = self.feedback.inject_entropy(state)
        
        # Check injection was created with min duration
        self.assertIsNotNone(self.feedback.current_injection)
        self.assertGreaterEqual(self.feedback.current_injection.duration_steps, 10)
    
    def test_emergency_cancels_exploration(self):
        """Test that emergency override cancels active exploration"""
        # Start exploration
        self.feedback.mode = CreativeMode.EXPLORING
        self.feedback.current_injection = self.feedback._create_injection(
            {'novelty': 0.5}, 0.8, 100
        )
        
        # Trigger emergency
        state = {'emergency_override': True}
        result = self.feedback.update(state)
        
        # Should cancel exploration
        self.assertEqual(self.feedback.mode, CreativeMode.EMERGENCY)
        self.assertIsNone(self.feedback.current_injection)
        self.assertEqual(result['action'], CreativeAction.EMERGENCY_HALT.value)
    
    def test_baseline_aesthetic_set(self):
        """Test that baseline aesthetic score is set"""
        # Initial baseline should be default
        self.assertEqual(
            self.feedback.regularizer.baseline_performance.get('score', 0.5),
            0.5
        )
        
        # Update with aesthetic score
        state = {'novelty': 0.3, 'aesthetic_score': 0.75}
        self.feedback.update(state)
        
        # Baseline should be updated
        self.assertEqual(
            self.feedback.regularizer.baseline_performance['score'],
            0.75
        )
    
    def test_diversity_score_clamping(self):
        """Test that diversity score is clamped to [0, 1]"""
        # Test with novelty > 1.0 (would cause negative diversity)
        factor = self.feedback._calculate_entropy_factor({'novelty': 1.5})
        
        # Factor should still be valid (uses clamped diversity)
        self.assertGreaterEqual(factor, 0.0)
        self.assertLessEqual(factor, 1.0)
        
        # Test with novelty < 0 (also would cause issues)
        factor = self.feedback._calculate_entropy_factor({'novelty': -0.5})
        self.assertGreaterEqual(factor, 0.0)
        self.assertLessEqual(factor, 1.0)
    
    def test_json_serialization(self):
        """Test that creative metrics are JSON serializable"""
        # Add some performance history with datetime
        self.feedback.performance_history.append({
            'timestamp': datetime.now(timezone.utc),
            'novelty': 0.5,
            'aesthetic_score': 0.7
        })
        
        # Get metrics
        metrics = self.feedback.get_creative_metrics()
        
        # Should be JSON serializable
        try:
            json_str = json.dumps(metrics)
            self.assertIsInstance(json_str, str)
        except TypeError:
            self.fail("Metrics should be JSON serializable")
    
    def test_action_enum_values(self):
        """Test that actions use enum values"""
        # Test various states and check action values
        states_and_expected = [
            ({'novelty': 0.2, 'aesthetic_score': 0.5}, CreativeAction.MAINTAIN.value),
            ({'emergency_override': True}, CreativeAction.EMERGENCY_HALT.value),
        ]
        
        for state, expected_action in states_and_expected:
            result = self.feedback.update(state)
            self.assertIn(result['action'], [e.value for e in CreativeAction])
    
    def test_no_polyfit_warnings(self):
        """Test that trend analysis doesn't raise warnings"""
        # Add identical values (would cause RankWarning with polyfit)
        for i in range(10):
            self.feedback.performance_history.append({
                'novelty': 0.5,  # Constant
                'aesthetic_score': 0.7  # Constant
            })
        
        # This should not raise warnings
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trends = self.feedback._analyze_performance_trend()
            
            # Check no RankWarning
            rank_warnings = [warning for warning in w 
                           if issubclass(warning.category, np.RankWarning)]
            self.assertEqual(len(rank_warnings), 0)


class TestConfigureLogging(unittest.TestCase):
    """Test configure_logging helper"""
    
    def test_configure_logging_exists(self):
        """Test that configure_logging is available"""
        from python.core.creative_feedback import configure_logging
        self.assertTrue(callable(configure_logging))
    
    def test_configure_logging_usage(self):
        """Test using configure_logging"""
        import logging
        configure_logging(logging.WARNING)
        # Should complete without error


if __name__ == '__main__':
    # Use the new configure_logging helper
    configure_logging()
    
    unittest.main(verbosity=2)
'''
        
        test_path = self.file_path.parent / "test_creative_feedback_sharp.py"
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        logger.info(f"Created test file: {test_path}")
        return test_path


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Patch creative_feedback.py based on sharp review")
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed')
    parser.add_argument('--create-test', action='store_true', help='Create test file')
    parser.add_argument('--rollback', action='store_true', help='Restore from backup')
    
    args = parser.parse_args()
    
    patcher = CreativeFeedbackPatcher()
    
    if args.rollback:
        if patcher.backup_path.exists():
            shutil.copy2(patcher.backup_path, patcher.file_path)
            logger.info(f"Restored from backup: {patcher.backup_path}")
        else:
            logger.error("No backup found")
        return
    
    # Create patches
    patcher.create_patches()
    logger.info(f"Created {len(patcher.patches)} patches from sharp review")
    
    # Apply patches
    success = patcher.apply_patches(dry_run=args.dry_run)
    
    # Create test file
    if args.create_test and success:
        patcher.create_test_file()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
