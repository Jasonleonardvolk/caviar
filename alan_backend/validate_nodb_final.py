#!/usr/bin/env python3
"""
Final validation script for No-DB migration
Cross-platform compatible without Unix dependencies
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path
import logging
import re

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
            
    def check_pd_io_json_usage(self):
        """Check for pd.io.json usage using pure Python"""
        found_files = []
        patterns = ['alan_backend', 'python/core']
        
        for pattern in patterns:
            base_path = Path(pattern)
            if base_path.exists():
                for py_file in base_path.rglob('*.py'):
                    try:
                        content = py_file.read_text(encoding='utf-8')
                        if 'pd.io.json' in content:
                            found_files.append(str(py_file))
                    except Exception as e:
                        logger.debug(f"Could not read {py_file}: {e}")
                        
        self.check(
            "No pd.io.json usage",
            len(found_files) == 0,
            f"Found in: {', '.join(found_files)}" if found_files else ""
        )
        
    def check_database_imports(self):
        """Check for database imports using pure Python"""
        db_patterns = [
            r'import\s+sqlite3',
            r'from\s+sqlite3',
            r'import\s+sqlalchemy',
            r'from\s+sqlalchemy',
            r'SpectralDB',
            r'\.db["\']',  # .db file references
        ]
        
        found_issues = []
        for pattern in ['alan_backend', 'python/core']:
            base_path = Path(pattern)
            if base_path.exists():
                for py_file in base_path.rglob('*.py'):
                    # Skip test files and backups
                    if 'test' in py_file.name or py_file.suffix == '.backup':
                        continue
                        
                    try:
                        content = py_file.read_text(encoding='utf-8')
                        for db_pattern in db_patterns:
                            if re.search(db_pattern, content):
                                found_issues.append(f"{py_file.name}: {db_pattern}")
                                break
                    except Exception as e:
                        logger.debug(f"Could not read {py_file}: {e}")
                        
        self.check(
            "No database imports found",
            len(found_issues) == 0,
            '; '.join(found_issues[:3]) + '...' if len(found_issues) > 3 else '; '.join(found_issues)
        )
            
    def run_all_checks(self):
        """Run all validation checks"""
        logger.info("🔍 Running final No-DB validation...\n")
        
        # 1. Check environment
        self.check(
            "TORI_STATE_ROOT configured",
            bool(os.getenv('TORI_STATE_ROOT')),
            "Set with: $env:TORI_STATE_ROOT = 'C:\\tori_state'"
        )
        
        self.check(
            "MAX_TOKENS_PER_MIN configured",
            bool(os.getenv('MAX_TOKENS_PER_MIN')),
            "Set with: $env:MAX_TOKENS_PER_MIN = '200'"
        )
        
        # 2. Check no database imports
        self.check_database_imports()
        
        # 3. Check no pd.io.json usage
        self.check_pd_io_json_usage()
        
        # 4. Check imports are standardized
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
            len(import_variations) <= 1,
            f"Found multiple import roots: {import_variations}"
        )
        
        # 5. Check Parquet files/directory
        state_root = Path(os.getenv('TORI_STATE_ROOT', 'C:\\tori_state'))
        if state_root.exists():
            parquet_files = list(state_root.glob('*.parquet'))
            self.check(
                "State directory exists",
                True,
                f"Ready for Parquet files at {state_root}"
            )
        else:
            self.check("State directory exists", False, f"Create dir: {state_root}")
            
        # 6. Check bounded collections with better error handling
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            
            # Try different module names
            for module_name in ['chaos_channel_controller_modified', 'chaos_channel_controller']:
                try:
                    module = importlib.import_module(module_name)
                    # Handle potential constructor signature changes
                    try:
                        controller = module.ChaosChannelController()
                        self.check(
                            "ChaosController has bounded history",
                            hasattr(controller.energy_history, 'maxlen'),
                            "energy_history is not bounded"
                        )
                        break
                    except TypeError as e:
                        self.check(
                            "ChaosController import",
                            False,
                            f"Constructor signature changed: {e}"
                        )
                        break
                except ImportError:
                    continue
            else:
                self.check("ChaosController has bounded history", False, "Module not found")
                
        except Exception as e:
            self.check("ChaosController has bounded history", False, str(e))
            
        # 7. Check WebSocket optional with better error handling
        try:
            # Temporarily hide websockets
            websockets_backup = sys.modules.get('websockets')
            sys.modules['websockets'] = None
            
            for module_name in ['eigensentry_guard_modified', 'eigensentry_guard']:
                try:
                    module = importlib.import_module(module_name)
                    try:
                        guard = module.CurvatureAwareGuard()
                        self.check("WebSockets are optional", True)
                        break
                    except TypeError as e:
                        self.check(
                            "EigenSentry import",
                            False,
                            f"Constructor signature changed: {e}"
                        )
                        break
                except ImportError:
                    continue
            else:
                self.check("WebSockets are optional", False, "Module not found")
                
            # Restore websockets
            if websockets_backup is None:
                del sys.modules['websockets']
            else:
                sys.modules['websockets'] = websockets_backup
                
        except Exception as e:
            self.check("WebSockets are optional", False, str(e))
            
        # Summary
        logger.info(f"\n📊 Validation Summary:")
        logger.info(f"   Passed: {self.checks_passed}")
        logger.info(f"   Failed: {self.checks_failed}")
        
        if self.checks_failed == 0:
            logger.info("\n🎉 All validation checks passed! Ready for deployment.")
            logger.info("\n📋 Next steps:")
            logger.info("  1. Ensure environment variables are set")
            logger.info("  2. Run: python alan_backend\\migrate_to_nodb_ast.py")
            logger.info("  3. Start system: python alan_backend\\start_true_metacognition.bat")
            return 0
        else:
            logger.error("\n⚠️  Some checks failed. Please fix before deploying.")
            return 1

if __name__ == "__main__":
    validator = FinalValidator()
    sys.exit(validator.run_all_checks())
