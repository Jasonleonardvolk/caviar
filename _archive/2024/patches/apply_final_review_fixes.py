#!/usr/bin/env python3
"""
Final fixes based on targeted review feedback
Addresses import paths, rollover, token limits, and other issues
"""

import os
import re
from pathlib import Path
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalReviewFixes:
    """Apply final review fixes to No-DB migration"""
    
    def __init__(self, kha_path: Path):
        self.kha_path = kha_path
        self.fixes_applied = []
        
    def fix_torus_registry_rollover(self):
        """Add max_rows parameter for Parquet rollover"""
        file_path = self.kha_path / "python" / "core" / "torus_registry.py"
        if not file_path.exists():
            logger.warning("torus_registry.py not found")
            return
            
        content = file_path.read_text()
        original = content
        
        # Add max_rows parameter to __init__
        content = re.sub(
            r'def __init__\(self, path: Path\):',
            'def __init__(self, path: Path, max_rows: int = 1_000_000):',
            content
        )
        
        # Add rollover logic in flush method
        flush_method = content.find("def flush(self):")
        if flush_method > 0 and "rollover" not in content:
            rollover_code = '''        
        # Check if rollover needed
        if hasattr(self, 'max_rows') and len(self.df) > self.max_rows:
            # Create rollover file
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            rollover_path = self.path.with_name(f"{self.path.stem}_{timestamp}{self.path.suffix}")
            
            # Save current data to rollover file
            self.df.to_parquet(rollover_path, index=False, compression='snappy')
            logger.info(f"Created rollover file: {rollover_path}")
            
            # Reset current dataframe
            self.df = self.df.iloc[0:0].copy()  # Keep schema, drop rows
            self._pending_writes.clear()
'''
            # Insert before atomic write
            content = content[:flush_method] + content[flush_method:].replace(
                "# Atomic write with temp file",
                rollover_code + "\n        # Atomic write with temp file",
                1
            )
            
        # Add max_rows attribute
        content = re.sub(
            r'(self\.path = path\n)',
            r'\1        self.max_rows = max_rows\n',
            content
        )
        
        if content != original:
            file_path.write_text(content)
            self.fixes_applied.append("torus_registry: Added max_rows parameter and rollover logic")
            logger.info("✅ Fixed torus_registry.py rollover")
            
    def standardize_imports(self):
        """Create import standardization script"""
        script_content = '''#!/usr/bin/env python3
"""
Import path standardization for TORI No-DB migration
Run this to fix all import paths to use consistent root
"""

import os
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the canonical import root
CANONICAL_ROOT = "python.core"  # Change to "kha.python.core" if preferred

IMPORT_MAPPINGS = {
    # Map all variations to canonical form
    r'from kha\.python\.core\.': f'from {CANONICAL_ROOT}.',
    r'from python\.core\.': f'from {CANONICAL_ROOT}.',
    r'import kha\.python\.core\.': f'import {CANONICAL_ROOT}.',
    r'import python\.core\.': f'import {CANONICAL_ROOT}.',
}

def standardize_file(file_path: Path) -> bool:
    """Standardize imports in a single file"""
    try:
        content = file_path.read_text()
        original = content
        
        for pattern, replacement in IMPORT_MAPPINGS.items():
            content = re.sub(pattern, replacement, content)
            
        if content != original:
            file_path.write_text(content)
            logger.info(f"✅ Standardized imports in {file_path.name}")
            return True
            
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        
    return False

def main():
    # Files to standardize
    files_to_fix = [
        "alan_backend/origin_sentry_modified.py",
        "alan_backend/eigensentry_guard_modified.py",
        "alan_backend/braid_aggregator_modified.py",
        "alan_backend/chaos_channel_controller_modified.py",
        "alan_backend/migrate_to_nodb.py",
        "alan_backend/test_nodb_migration.py",
        "python/core/torus_cells.py",
        "python/core/observer_synthesis.py",
    ]
    
    kha_root = Path(__file__).parent
    fixed_count = 0
    
    for file_path in files_to_fix:
        full_path = kha_root / file_path
        if full_path.exists():
            if standardize_file(full_path):
                fixed_count += 1
        else:
            logger.warning(f"File not found: {file_path}")
            
    logger.info(f"\\nStandardized imports in {fixed_count} files")
    logger.info(f"All imports now use: {CANONICAL_ROOT}")

if __name__ == "__main__":
    main()
'''
        
        script_path = self.kha_path / "standardize_imports.py"
        script_path.write_text(script_content)
        self.fixes_applied.append("Created standardize_imports.py")
        logger.info("✅ Created import standardization script")
        
    def fix_observer_token_limits(self):
        """Add configurable rate limits to observer_synthesis.py"""
        file_path = self.kha_path / "python" / "core" / "observer_synthesis.py"
        if not file_path.exists():
            logger.warning("observer_synthesis.py not found")
            return
            
        content = file_path.read_text()
        original = content
        
        # Add environment variable support for max tokens
        if "MAX_TOKENS_PER_MIN" not in content:
            content = re.sub(
                r'def __init__\(self, max_tokens: int = 1000\):',
                '''def __init__(self, max_tokens: int = 1000):
        # Get rate limit from environment or use default
        self.max_tokens_per_minute = int(os.getenv('MAX_TOKENS_PER_MIN', '200'))''',
                content
            )
            
            # Update TokenRateLimiter initialization
            content = re.sub(
                r'class TokenRateLimiter:\n\s+"""Rate limiter for observer tokens"""\n\s+\n\s+def __init__\(self, max_tokens_per_minute: int = 200\):',
                '''class TokenRateLimiter:
    """Rate limiter for observer tokens"""
    
    def __init__(self, max_tokens_per_minute: int = None):
        if max_tokens_per_minute is None:
            max_tokens_per_minute = int(os.getenv('MAX_TOKENS_PER_MIN', '200'))''',
                content
            )
            
        if content != original:
            file_path.write_text(content)
            self.fixes_applied.append("observer_synthesis: Added MAX_TOKENS_PER_MIN env var")
            logger.info("✅ Fixed observer_synthesis.py token limits")
            
    def fix_eigensentry_cleanup(self):
        """Add proper cleanup to eigensentry_guard.py"""
        file_path = self.kha_path / "alan_backend" / "eigensentry_guard_modified.py"
        if not file_path.exists():
            logger.warning("eigensentry_guard_modified.py not found")
            return
            
        content = file_path.read_text()
        original = content
        
        # Add __aexit__ method if not present
        if "__aexit__" not in content:
            # Find class definition
            class_def = content.find("class CurvatureAwareGuard:")
            if class_def > 0:
                # Add context manager methods
                cleanup_code = '''
    async def __aenter__(self):
        """Async context manager entry"""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup WebSocket clients"""
        if WEBSOCKETS_AVAILABLE and self.ws_clients:
            # Close all WebSocket connections
            for client in list(self.ws_clients):
                try:
                    await client.close()
                except Exception:
                    pass
            self.ws_clients.clear()
            logger.info("Closed all WebSocket connections")
        return False
'''
                # Insert after last method
                last_method = content.rfind("\n    def ")
                if last_method > 0:
                    # Find end of last method
                    next_class = content.find("\nclass ", last_method)
                    insert_pos = next_class if next_class > 0 else len(content)
                    content = content[:insert_pos] + cleanup_code + content[insert_pos:]
                    
        if content != original:
            file_path.write_text(content)
            self.fixes_applied.append("eigensentry_guard: Added WebSocket cleanup in __aexit__")
            logger.info("✅ Fixed eigensentry_guard.py cleanup")
            
    def fix_origin_sentry_betti(self):
        """Fix _last_betti type consistency in origin_sentry.py"""
        file_path = self.kha_path / "alan_backend" / "origin_sentry_modified.py"
        if not file_path.exists():
            logger.warning("origin_sentry_modified.py not found")
            return
            
        content = file_path.read_text()
        original = content
        
        # Ensure _last_betti is always treated as list
        content = re.sub(
            r'if betti_numbers:\n\s+self\._last_betti = betti_numbers',
            '''if betti_numbers:
            self._last_betti = list(betti_numbers) if not isinstance(betti_numbers, list) else betti_numbers''',
            content
        )
        
        if content != original:
            file_path.write_text(content)
            self.fixes_applied.append("origin_sentry: Fixed _last_betti type consistency")
            logger.info("✅ Fixed origin_sentry.py betti types")
            
    def add_ast_diff_option(self):
        """Add --print-diff option to AST migration script"""
        file_path = self.kha_path / "alan_backend" / "migrate_to_nodb_ast.py"
        if not file_path.exists():
            logger.warning("migrate_to_nodb_ast.py not found")
            return
            
        content = file_path.read_text()
        original = content
        
        # Add diff printing capability
        if "--print-diff" not in content:
            # Add import
            if "import difflib" not in content:
                content = "import difflib\n" + content
                
            # Add argument
            content = re.sub(
                r'(parser\.add_argument\("--dry-run".*?\))',
                r'''\1
    parser.add_argument("--print-diff", action="store_true",
                       help="Print unified diff of changes")''',
                content,
                flags=re.DOTALL
            )
            
            # Add diff logic in migrate_file
            diff_code = '''
                if self.print_diff:
                    # Generate diff
                    diff = difflib.unified_diff(
                        content.splitlines(keepends=True),
                        new_content.splitlines(keepends=True),
                        fromfile=f"{file_path.name} (original)",
                        tofile=f"{file_path.name} (modified)",
                        lineterm=''
                    )
                    print(''.join(diff))
'''
            content = re.sub(
                r'(new_content = astor\.to_source\(tree\))',
                r'\1' + diff_code,
                content
            )
            
            # Add print_diff attribute
            content = re.sub(
                r'def __init__\(self, backend_path: Path, dry_run: bool = True\):',
                'def __init__(self, backend_path: Path, dry_run: bool = True, print_diff: bool = False):',
                content
            )
            
            content = re.sub(
                r'(self\.dry_run = dry_run)',
                r'\1\n        self.print_diff = print_diff',
                content
            )
            
            # Pass print_diff from args
            content = re.sub(
                r'migrator = ASTMigrator\(args\.path, dry_run=args\.dry_run\)',
                'migrator = ASTMigrator(args.path, dry_run=args.dry_run, print_diff=args.print_diff)',
                content
            )
            
        if content != original:
            file_path.write_text(content)
            self.fixes_applied.append("migrate_to_nodb_ast: Added --print-diff option")
            logger.info("✅ Added --print-diff to AST migration")
            
    def update_powershell_examples(self):
        """Update documentation with proper PowerShell syntax"""
        doc_content = '''# PowerShell Environment Setup for TORI No-DB

## Setting Environment Variables

```powershell
# Windows PowerShell
$env:TORI_STATE_ROOT = "C:\\tori_state"
$env:MAX_TOKENS_PER_MIN = "200"

# Create state directory
New-Item -ItemType Directory -Force -Path $env:TORI_STATE_ROOT

# Verify
Write-Host "TORI_STATE_ROOT: $env:TORI_STATE_ROOT"
```

## Running Migration Scripts

```powershell
# Set Python path for imports
$env:PYTHONPATH = "$PWD;$PWD\\kha"

# Run migrations
python alan_backend\\remove_alan_backend_db.py --dry-run
python alan_backend\\migrate_to_nodb_ast.py --print-diff
python alan_backend\\apply_code_review_fixes.py

# Run tests
python alan_backend\\test_nodb_migration.py
```

## Creating Distribution ZIP

```powershell
# Using environment variable for paths
$stateRoot = $env:TORI_STATE_ROOT
if (-not $stateRoot) {
    $stateRoot = "C:\\tori_state"
    Write-Warning "TORI_STATE_ROOT not set, using default: $stateRoot"
}

# Create ZIP with verification
$files = Get-ChildItem -Path "alan_backend\\*_modified.py", "python\\core\\torus*.py", "python\\core\\observer*.py"
$zipPath = "$env:USERPROFILE\\Desktop\\tori_nodb_final.zip"
Compress-Archive -Path $files -DestinationPath $zipPath -Force
Write-Host "Created: $zipPath ($(Get-Item $zipPath).Length bytes)"
```
'''
        
        doc_path = self.kha_path / "POWERSHELL_SETUP.md"
        doc_path.write_text(doc_content)
        self.fixes_applied.append("Created POWERSHELL_SETUP.md")
        logger.info("✅ Created PowerShell setup documentation")
        
    def create_final_validation_script(self):
        """Create comprehensive validation script"""
        validation_content = '''#!/usr/bin/env python3
"""
Final validation script for No-DB migration
Checks all requirements from the review
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalValidator:
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        
    def check(self, name: str, condition: bool, details: str = ""):
        """Run a single check"""
        if condition:
            logger.info(f"✅ {name}")
            self.checks_passed += 1
        else:
            logger.error(f"❌ {name}: {details}")
            self.checks_failed += 1
            
    def run_all_checks(self):
        """Run all validation checks"""
        logger.info("🔍 Running final No-DB validation...\\n")
        
        # 1. Check environment
        self.check(
            "TORI_STATE_ROOT configured",
            bool(os.getenv('TORI_STATE_ROOT')),
            "Set with: export TORI_STATE_ROOT=/var/lib/tori"
        )
        
        # 2. Check no database imports
        try:
            result = subprocess.run(
                ['python', '-m', 'flake8', '--select=FORBID', 'alan_backend/'],
                capture_output=True,
                text=True
            )
            self.check(
                "No database imports found",
                result.returncode == 0,
                result.stderr
            )
        except Exception as e:
            self.check("No database imports found", False, str(e))
            
        # 3. Check imports are standardized
        import_variations = set()
        for module in ['torus_registry', 'torus_cells', 'observer_synthesis']:
            for prefix in ['python.core', 'kha.python.core']:
                try:
                    importlib.import_module(f'{prefix}.{module}')
                    import_variations.add(prefix)
                except ImportError:
                    pass
                    
        self.check(
            "Import paths standardized",
            len(import_variations) == 1,
            f"Found multiple import roots: {import_variations}"
        )
        
        # 4. Check Parquet files created
        state_root = Path(os.getenv('TORI_STATE_ROOT', '/var/lib/tori'))
        if state_root.exists():
            parquet_files = list(state_root.glob('*.parquet'))
            self.check(
                "Parquet persistence working",
                len(parquet_files) > 0,
                f"No .parquet files in {state_root}"
            )
        else:
            self.check("Parquet persistence working", False, f"State root not found: {state_root}")
            
        # 5. Check bounded collections
        try:
            from alan_backend.chaos_channel_controller_modified import ChaosChannelController
            controller = ChaosChannelController()
            self.check(
                "ChaosController has bounded history",
                hasattr(controller.energy_history, 'maxlen'),
                "energy_history is not bounded"
            )
        except Exception as e:
            self.check("ChaosController has bounded history", False, str(e))
            
        # 6. Check WebSocket optional
        try:
            # Temporarily hide websockets
            sys.modules['websockets'] = None
            from alan_backend.eigensentry_guard_modified import CurvatureAwareGuard
            guard = CurvatureAwareGuard()
            self.check("WebSockets are optional", True)
            del sys.modules['websockets']
        except Exception as e:
            self.check("WebSockets are optional", False, str(e))
            
        # Summary
        logger.info(f"\\n📊 Validation Summary:")
        logger.info(f"   Passed: {self.checks_passed}")
        logger.info(f"   Failed: {self.checks_failed}")
        
        if self.checks_failed == 0:
            logger.info("\\n🎉 All validation checks passed! Ready for deployment.")
            return 0
        else:
            logger.error("\\n⚠️  Some checks failed. Please fix before deploying.")
            return 1

if __name__ == "__main__":
    validator = FinalValidator()
    sys.exit(validator.run_all_checks())
'''
        
        script_path = self.kha_path / "alan_backend" / "validate_nodb_final.py"
        script_path.write_text(validation_content)
        self.fixes_applied.append("Created validate_nodb_final.py")
        logger.info("✅ Created final validation script")
        
    def run_all_fixes(self):
        """Apply all final review fixes"""
        logger.info("🔧 Applying final review fixes...\\n")
        
        self.fix_torus_registry_rollover()
        self.standardize_imports()
        self.fix_observer_token_limits()
        self.fix_eigensentry_cleanup()
        self.fix_origin_sentry_betti()
        self.add_ast_diff_option()
        self.update_powershell_examples()
        self.create_final_validation_script()
        
        logger.info(f"\\n✅ Applied {len(self.fixes_applied)} fixes:")
        for fix in self.fixes_applied:
            logger.info(f"  - {fix}")
            
        logger.info("\\n📋 Next steps:")
        logger.info("  1. Run: python standardize_imports.py")
        logger.info("  2. Run: python alan_backend/validate_nodb_final.py")
        logger.info("  3. Set: export TORI_STATE_ROOT=/var/lib/tori")
        logger.info("  4. Set: export MAX_TOKENS_PER_MIN=200")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply final review fixes")
    parser.add_argument("--path", type=Path, default=Path("."),
                       help="Path to kha directory")
    
    args = parser.parse_args()
    
    if not args.path.exists():
        logger.error(f"Path not found: {args.path}")
        return 1
        
    fixer = FinalReviewFixes(args.path)
    fixer.run_all_fixes()
    
    return 0

if __name__ == "__main__":
    exit(main())
