#!/usr/bin/env python3
"""
Quick fixes for code review issues in No-DB migration
Apply these patches after the main migration
"""

import re
from pathlib import Path
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeReviewFixes:
    """Apply specific fixes from code review"""
    
    def __init__(self, backend_path: Path):
        self.backend_path = backend_path
        self.fixes_applied = []
        
    def fix_origin_sentry(self):
        """Fix issues in origin_sentry.py"""
        file_path = self.backend_path / "origin_sentry.py"
        if not file_path.exists():
            file_path = self.backend_path / "origin_sentry_modified.py"
            
        if not file_path.exists():
            logger.warning(f"origin_sentry.py not found")
            return
            
        content = file_path.read_text()
        original = content
        
        # Fix 1: Initialize _last_betti in __init__
        if "__init__" in content and "_last_betti" not in content.split("__init__")[1].split("def ")[0]:
            content = re.sub(
                r'(def __init__.*?\n(?:.*?\n)*?)([ ]+)(logger\.info.*?initialized.*?".*?\))',
                r'\1\2self._last_betti = []  # Initialize for novelty computation\n\2\3',
                content,
                flags=re.MULTILINE | re.DOTALL
            )
            self.fixes_applied.append("origin_sentry: Added _last_betti initialization")
            
        # Fix 2: Add EPS to capsule.yml parameters
        if "EPS = 0.01" in content:
            content = re.sub(
                r'EPS = 0\.01  # Novelty threshold',
                'EPS = float(os.getenv("TORI_NOVELTY_THRESHOLD", "0.01"))  # Novelty threshold from env/config',
                content
            )
            # Add import if needed
            if "import os" not in content:
                content = "import os\n" + content
            self.fixes_applied.append("origin_sentry: Made EPS configurable")
            
        # Fix 3: Better distance metric with PCA hint
        distance_method = content.find("def distance(self")
        if distance_method > 0:
            # Add comment about PCA enhancement
            comment = '''        # TODO: Store compressed PCA vectors in registry metadata for richer comparison
        # Example: pca_vector = PCA(n_components=5).fit_transform(eigenvalues.reshape(1, -1))
        '''
            if "TODO: Store compressed PCA" not in content:
                content = content[:distance_method] + comment + content[distance_method:]
                self.fixes_applied.append("origin_sentry: Added PCA enhancement comment")
                
        if content != original:
            file_path.write_text(content)
            logger.info(f"✅ Fixed origin_sentry.py")
            
    def fix_eigensentry_guard(self):
        """Fix issues in eigensentry_guard.py"""
        file_path = self.backend_path / "eigensentry_guard.py"
        if not file_path.exists():
            file_path = self.backend_path / "eigensentry_guard_modified.py"
            
        if not file_path.exists():
            logger.warning(f"eigensentry_guard.py not found")
            return
            
        content = file_path.read_text()
        original = content
        
        # Fix 1: Make websockets optional
        if "import websockets" in content and "try:" not in content.split("import websockets")[0][-50:]:
            content = re.sub(
                r'import websockets',
                '''try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None''',
                content
            )
            
            # Guard WebSocket usage
            content = re.sub(
                r'(\s+)(self\.ws_clients = set\(\))',
                r'\1\2 if WEBSOCKETS_AVAILABLE else None',
                content
            )
            
            self.fixes_applied.append("eigensentry_guard: Made websockets optional")
            
        # Fix 2: Add observer synthesis attribute check
        if "add_to_context" in content:
            content = re.sub(
                r'(if is_significant and self\.observer_synthesis:)',
                r'if is_significant and hasattr(self, "observer_synthesis") and self.observer_synthesis:',
                content
            )
            self.fixes_applied.append("eigensentry_guard: Added observer_synthesis attribute check")
            
        # Fix 3: Document poll_counter behavior
        poll_method = content.find("def poll_spectral_stability")
        if poll_method > 0 and "tie to guard steps not wall-time" not in content:
            docstring_addition = '''        """
        Poll spectral stability every N steps
        
        Note: Polling is tied to guard steps, not wall-time. This ensures
        consistent sampling regardless of processing speed.
        """
'''
            # Insert after method definition
            content = content[:poll_method] + content[poll_method:].replace(
                'def poll_spectral_stability(self, soliton_state: np.ndarray):',
                f'def poll_spectral_stability(self, soliton_state: np.ndarray):\n{docstring_addition}',
                1
            )
            self.fixes_applied.append("eigensentry_guard: Documented poll_counter behavior")
            
        if content != original:
            file_path.write_text(content)
            logger.info(f"✅ Fixed eigensentry_guard.py")
            
    def fix_chaos_controller(self):
        """Fix issues in chaos_channel_controller.py"""
        file_path = self.backend_path / "chaos_channel_controller.py"
        if not file_path.exists():
            file_path = self.backend_path / "chaos_channel_controller_modified.py"
            
        if not file_path.exists():
            logger.warning(f"chaos_channel_controller.py not found")
            return
            
        content = file_path.read_text()
        original = content
        
        # Fix 1: Import path consistency
        content = re.sub(
            r'from python\.core\.torus_registry import',
            'from kha.python.core.torus_registry import',
            content
        )
        self.fixes_applied.append("chaos_controller: Fixed import path")
        
        # Fix 2: Cap energy_history
        if "self.energy_history = deque(maxlen=1000)" not in content:
            content = re.sub(
                r'self\.energy_history = deque\(maxlen=1000\)',
                'self.energy_history = deque(maxlen=1000)  # Capped to prevent unbounded growth',
                content
            )
            # If no maxlen specified
            content = re.sub(
                r'self\.energy_history = deque\(\)',
                'self.energy_history = deque(maxlen=1000)  # Capped to prevent unbounded growth',
                content
            )
            self.fixes_applied.append("chaos_controller: Capped energy_history")
            
        # Fix 3: Better lattice sampling
        inject_method = content.find("def _inject_chaos_energy")
        if inject_method > 0:
            # Add stratified sampling comment
            if "stratified sampling" not in content:
                comment = '''        # TODO: Implement stratified sampling for better topology representation
        # Currently sampling first 100 points; consider:
        # - Stratified grid sampling
        # - Random sampling with fixed seed
        # - Checksum of full state
        '''
                content = content[:inject_method] + content[inject_method:].replace(
                    "vertices=self.lattice_state.flatten()[:100],  # Sample for storage",
                    f"vertices=self.lattice_state.flatten()[:100],  # Sample for storage\n{comment}",
                    1
                )
                self.fixes_applied.append("chaos_controller: Added sampling improvement note")
                
        if content != original:
            file_path.write_text(content)
            logger.info(f"✅ Fixed chaos_channel_controller.py")
            
    def fix_braid_aggregator(self):
        """Fix issues in braid_aggregator.py"""
        file_path = self.backend_path / "braid_aggregator.py"
        if not file_path.exists():
            file_path = self.backend_path / "braid_aggregator_modified.py"
            
        if not file_path.exists():
            logger.warning(f"braid_aggregator.py not found")
            return
            
        content = file_path.read_text()
        original = content
        
        # Fix 1: Remove sys.path.append
        if "sys.path.append" in content:
            # Replace with proper import
            content = re.sub(
                r'sys\.path\.append\(.*?\)\n',
                '# Imports handled by PYTHONPATH or package installation\n',
                content
            )
            self.fixes_applied.append("braid_aggregator: Removed sys.path.append")
            
        # Fix 2: Add Betti caching check
        compute_method = content.find("def _compute_spectral_summary")
        if compute_method > 0 and "Check cache first" not in content:
            cache_code = '''                # Check if we've already computed Betti for similar configuration
                cache_key = f"{scale.value}_{len(lambda_values)}_{hash(tuple(lambda_values[-5:]))}"
                if hasattr(self, '_betti_cache') and cache_key in self._betti_cache:
                    cached = self._betti_cache[cache_key]
                    summary['betti_max'] = cached['betti']
                    summary['betti_computed'] = True
                    summary['betti_cached'] = True
                else:
'''
            # Find where to insert
            betti_section = content.find("# Compute Betti numbers using TorusCells")
            if betti_section > 0:
                # Add cache initialization in __init__ if not present
                if "_betti_cache" not in content:
                    init_section = content.find("def __init__")
                    if init_section > 0:
                        content = content[:init_section] + content[init_section:].replace(
                            "self.metrics = {",
                            "self._betti_cache = {}  # Cache for Betti computations\n        self.metrics = {",
                            1
                        )
                
                # Add cache check
                content = content[:betti_section] + cache_code + content[betti_section:]
                self.fixes_applied.append("braid_aggregator: Added Betti computation caching")
                
        # Fix 3: Fix correlation division by zero
        if "np.corrcoef" in content:
            content = re.sub(
                r'(corr = np\.corrcoef\(.*?\)\[0, 1\])',
                r'# Check for zero variance to avoid warnings\n                        if np.std(arr1) > 0 and np.std(arr2) > 0:\n                            \1\n                        else:\n                            corr = 0.0',
                content
            )
            self.fixes_applied.append("braid_aggregator: Fixed correlation zero variance")
            
        # Fix 4: Change JSON dumps to debug logging
        content = re.sub(
            r'self\.logger\.info\(f"Novelty event: \{json\.dumps\(event_data, indent=2\)\}"\)',
            'self.logger.debug(f"Novelty event details: {event_data}")',
            content
        )
        self.fixes_applied.append("braid_aggregator: Changed JSON dumps to debug level")
        
        if content != original:
            file_path.write_text(content)
            logger.info(f"✅ Fixed braid_aggregator.py")
            
    def add_import_normalization(self):
        """Create import normalization helper"""
        helper_content = '''#!/usr/bin/env python3
"""
Import path normalizer for TORI components
Ensures consistent imports across the codebase
"""

import os
import sys
from pathlib import Path

def setup_tori_paths():
    """Setup Python paths for TORI imports"""
    # Get TORI root (parent of alan_backend)
    tori_root = Path(__file__).parent.parent
    
    # Add to Python path if not already there
    paths_to_add = [
        tori_root,  # For 'alan_backend.x' imports
        tori_root / "kha",  # For 'python.core.x' imports
    ]
    
    for path in paths_to_add:
        path_str = str(path.absolute())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            
    # Set environment variable if not set
    if not os.getenv('TORI_STATE_ROOT'):
        state_root = tori_root / 'data' / 'tori_state'
        state_root.mkdir(parents=True, exist_ok=True)
        os.environ['TORI_STATE_ROOT'] = str(state_root)
        
    return tori_root

# Auto-setup when imported
TORI_ROOT = setup_tori_paths()

# Convenience imports that work from anywhere
def get_torus_registry():
    """Get TorusRegistry with proper imports"""
    try:
        from python.core.torus_registry import get_torus_registry as _get_reg
    except ImportError:
        from kha.python.core.torus_registry import get_torus_registry as _get_reg
    return _get_reg()

def get_torus_cells():
    """Get TorusCells with proper imports"""
    try:
        from python.core.torus_cells import get_torus_cells as _get_cells
    except ImportError:
        from kha.python.core.torus_cells import get_torus_cells as _get_cells
    return _get_cells()

def emit_token(data):
    """Emit observer token with proper imports"""
    try:
        from python.core.observer_synthesis import emit_token as _emit
    except ImportError:
        from kha.python.core.observer_synthesis import emit_token as _emit
    return _emit(data)
'''
        
        helper_path = self.backend_path / "tori_imports.py"
        helper_path.write_text(helper_content)
        self.fixes_applied.append("Created tori_imports.py for import normalization")
        logger.info(f"✅ Created import helper: {helper_path}")
        
    def add_rate_limiter(self):
        """Add rate limiter for observer tokens"""
        rate_limiter_content = '''#!/usr/bin/env python3
"""
Rate limiter for observer token emission
Prevents token storm in high-frequency loops
"""

import time
from collections import deque
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class TokenRateLimiter:
    """Rate limiter for observer tokens"""
    
    def __init__(self, max_tokens_per_minute: int = 200):
        self.max_tokens = max_tokens_per_minute
        self.window = 60.0  # seconds
        self.emissions = deque()
        self.dropped_count = 0
        
    def check_rate(self) -> bool:
        """Check if emission is allowed under rate limit"""
        now = time.time()
        
        # Remove old emissions outside window
        cutoff = now - self.window
        while self.emissions and self.emissions[0] < cutoff:
            self.emissions.popleft()
            
        # Check rate
        if len(self.emissions) >= self.max_tokens:
            self.dropped_count += 1
            if self.dropped_count % 100 == 1:  # Log every 100 drops
                logger.warning(f"Token rate limit exceeded: {self.dropped_count} tokens dropped")
            return False
            
        # Allow emission
        self.emissions.append(now)
        return True
        
    def get_stats(self) -> dict:
        """Get rate limiter statistics"""
        return {
            'current_rate': len(self.emissions),
            'max_rate': self.max_tokens,
            'dropped_total': self.dropped_count,
            'window_seconds': self.window
        }

# Global rate limiter
_rate_limiter = TokenRateLimiter()

def rate_limited_emit_token(original_emit_token):
    """Decorator to add rate limiting to emit_token"""
    @wraps(original_emit_token)
    def wrapper(data):
        if _rate_limiter.check_rate():
            return original_emit_token(data)
        else:
            # Return a dummy token when rate limited
            return "RATE_LIMITED_" + str(int(time.time()))
    return wrapper

# Monkey-patch observer_synthesis.emit_token with rate limiting
def apply_rate_limiting():
    """Apply rate limiting to observer token emission"""
    try:
        from python.core import observer_synthesis
        if hasattr(observer_synthesis, 'emit_token'):
            original = observer_synthesis.emit_token
            observer_synthesis.emit_token = rate_limited_emit_token(original)
            logger.info("Applied token rate limiting (200/min)")
    except ImportError:
        logger.warning("Could not apply token rate limiting - observer_synthesis not found")
'''
        
        limiter_path = self.backend_path / "token_rate_limiter.py"
        limiter_path.write_text(rate_limiter_content)
        self.fixes_applied.append("Created token_rate_limiter.py")
        logger.info(f"✅ Created rate limiter: {limiter_path}")
        
    def run_all_fixes(self):
        """Run all code review fixes"""
        logger.info("🔧 Applying code review fixes...")
        
        self.fix_origin_sentry()
        self.fix_eigensentry_guard()
        self.fix_chaos_controller()
        self.fix_braid_aggregator()
        self.add_import_normalization()
        self.add_rate_limiter()
        
        logger.info(f"\n✅ Applied {len(self.fixes_applied)} fixes:")
        for fix in self.fixes_applied:
            logger.info(f"  - {fix}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply code review fixes")
    parser.add_argument("--path", type=Path, default=Path("alan_backend"),
                       help="Path to alan_backend directory")
    
    args = parser.parse_args()
    
    if not args.path.exists():
        logger.error(f"Path not found: {args.path}")
        return 1
        
    fixer = CodeReviewFixes(args.path)
    fixer.run_all_fixes()
    
    return 0

if __name__ == "__main__":
    exit(main())
