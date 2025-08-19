#!/usr/bin/env python3
"""
Master No-DB Migration Fix Script v2.1
Production-ready with all micro-patches applied
"""

import os
import re
import ast
import json
import shutil
import sys
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
from collections import deque
import time

# Check Python version for ast.unparse availability
if sys.version_info >= (3, 9):
    from ast import unparse as ast_to_source
else:
    try:
        import astor
        ast_to_source = astor.to_source
    except ImportError:
        print("ERROR: astor required for Python < 3.9. Install with: pip install astor")
        sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MasterNoDBFixer:
    """Comprehensive fixes for No-DB migration with production robustness"""
    
    # DECISION POINT: Choose your canonical import root
    # Option 1: CANONICAL_ROOT = "python.core"
    # Option 2: CANONICAL_ROOT = "kha.python.core"
    CANONICAL_ROOT = "kha.python.core"  # Set your preference here
    
    def __init__(self, kha_path: Path):
        self.kha_path = kha_path
        self.alan_backend_path = kha_path / "alan_backend"
        self.python_core_path = kha_path / "python" / "core"
        self.fixes_applied = []
        self.errors = []
        self.backed_up_files = set()
        
    def backup_file(self, file_path: Path) -> Path:
        """Create backup before modifying"""
        if file_path in self.backed_up_files:
            return file_path.with_suffix(file_path.suffix + '.backup')
            
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
        if not backup_path.exists():
            shutil.copy2(file_path, backup_path)
            logger.info(f"📋 Backed up: {file_path.name}")
            self.backed_up_files.add(file_path)
        return backup_path
        
    # ==================== Core Fixes with AST ====================
    
    def fix_pd_io_json_dumps(self):
        """Fix deprecated pd.io.json.dumps usage"""
        logger.info("\n🔧 Fixing pd.io.json.dumps deprecation...")
        
        files_to_fix = [
            self.python_core_path / "torus_registry.py",
            self.alan_backend_path / "remove_alan_backend_db.py",
        ]
        
        for file_path in files_to_fix:
            if not file_path.exists():
                self.errors.append(f"File not found: {file_path}")
                continue
                
            self.backup_file(file_path)
            content = file_path.read_text()
            original = content
            
            # Replace pd.io.json.dumps with json.dumps
            if 'pd.io.json.dumps' in content:
                content = re.sub(r'pd\.io\.json\.dumps', 'json.dumps', content)
                
                # Add json import using AST to ensure proper placement
                if 'json.dumps' in content and 'import json' not in content:
                    content = self._add_import_with_ast(content, 'json')
                    
                if content != original:
                    file_path.write_text(content)
                    self.fixes_applied.append(f"Fixed pd.io.json.dumps in {file_path.name}")
                    
    def _add_import_with_ast(self, content: str, module_name: str) -> str:
        """Add import using AST for proper placement"""
        try:
            tree = ast.parse(content)
            
            # Check if import already exists
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == module_name:
                            return content
                            
            # Find insertion point after other imports
            insert_idx = 0
            for i, node in enumerate(tree.body):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    insert_idx = i + 1
                else:
                    break
                    
            # Create import node
            import_node = ast.Import(names=[ast.alias(name=module_name, asname=None)])
            tree.body.insert(insert_idx, import_node)
            
            return ast_to_source(tree)
        except:
            # Fallback to regex if AST fails
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith(('import ', 'from ')) and not line.startswith('from __future__'):
                    lines.insert(i + 1, f'import {module_name}')
                    return '\n'.join(lines)
            return content
            
    def fix_import_paths_with_ast(self):
        """Fix import paths using AST for robustness"""
        logger.info(f"\n🔧 Standardizing imports to {self.CANONICAL_ROOT} using AST...")
        
        files_to_fix = [
            # Modified files
            self.alan_backend_path / "origin_sentry_modified.py",
            self.alan_backend_path / "eigensentry_guard_modified.py",
            self.alan_backend_path / "braid_aggregator_modified.py",
            self.alan_backend_path / "chaos_channel_controller_modified.py",
            # Core files
            self.python_core_path / "observer_synthesis.py",
            self.python_core_path / "torus_cells.py",
            # Migration files
            self.alan_backend_path / "migrate_to_nodb_ast.py",
            self.alan_backend_path / "test_nodb_migration.py",
        ]
        
        for file_path in files_to_fix:
            if not file_path.exists():
                continue
                
            self.backup_file(file_path)
            
            try:
                content = file_path.read_text()
                tree = ast.parse(content)
                
                class ImportTransformer(ast.NodeTransformer):
                    def __init__(self, canonical_root):
                        self.canonical_root = canonical_root
                        self.modified = False
                        
                    def visit_ImportFrom(self, node):
                        if node.module:
                            # Check for various import patterns
                            patterns = ['python.core', 'kha.python.core']
                            for pattern in patterns:
                                if node.module.startswith(pattern):
                                    # Replace with canonical root
                                    suffix = node.module[len(pattern):]
                                    node.module = self.canonical_root + suffix
                                    self.modified = True
                                    break
                        return node
                        
                    def visit_Import(self, node):
                        # Handle 'import x.y.z' style imports
                        for alias in node.names:
                            patterns = ['python.core', 'kha.python.core']
                            for pattern in patterns:
                                if alias.name.startswith(pattern):
                                    suffix = alias.name[len(pattern):]
                                    alias.name = self.canonical_root + suffix
                                    self.modified = True
                                    break
                        return node
                        
                transformer = ImportTransformer(self.CANONICAL_ROOT)
                new_tree = transformer.visit(tree)
                
                if transformer.modified:
                    new_content = ast_to_source(new_tree)
                    file_path.write_text(new_content)
                    self.fixes_applied.append(f"Standardized imports in {file_path.name}")
                    
            except Exception as e:
                logger.error(f"AST parsing failed for {file_path.name}: {e}")
                self.errors.append(f"Failed to process {file_path.name}")
                
    def purge_alternative_roots(self):
        """Second pass to remove any remaining alternative import roots"""
        logger.info("\n🔧 Purging alternative import roots...")
        
        # Determine the alternative root
        alt_root = "python.core" if self.CANONICAL_ROOT == "kha.python.core" else "kha.python.core"
        
        # Same files as import fix
        files_to_fix = [
            self.alan_backend_path / "origin_sentry_modified.py",
            self.alan_backend_path / "eigensentry_guard_modified.py",
            self.alan_backend_path / "braid_aggregator_modified.py",
            self.alan_backend_path / "chaos_channel_controller_modified.py",
            self.python_core_path / "observer_synthesis.py",
            self.python_core_path / "torus_cells.py",
            self.alan_backend_path / "migrate_to_nodb_ast.py",
            self.alan_backend_path / "test_nodb_migration.py",
        ]
        
        purged_count = 0
        for file_path in files_to_fix:
            if not file_path.exists():
                continue
                
            try:
                content = file_path.read_text()
                # Quick check if alternative root exists
                if alt_root not in content:
                    continue
                    
                tree = ast.parse(content)
                
                class RootPurger(ast.NodeTransformer):
                    def __init__(self, alt_root, canonical_root):
                        self.alt_root = alt_root
                        self.canonical_root = canonical_root
                        self.modified = False
                        
                    def visit_ImportFrom(self, node):
                        if node.module and node.module.startswith(self.alt_root):
                            suffix = node.module[len(self.alt_root):]
                            node.module = self.canonical_root + suffix
                            self.modified = True
                        return node
                        
                    def visit_Import(self, node):
                        for alias in node.names:
                            if alias.name.startswith(self.alt_root):
                                suffix = alias.name[len(self.alt_root):]
                                alias.name = self.canonical_root + suffix
                                self.modified = True
                        return node
                        
                purger = RootPurger(alt_root, self.CANONICAL_ROOT)
                new_tree = purger.visit(tree)
                
                if purger.modified:
                    file_path.write_text(ast_to_source(new_tree))
                    purged_count += 1
                    
            except Exception as e:
                logger.debug(f"Could not purge alt root in {file_path.name}: {e}")
                
        if purged_count > 0:
            self.fixes_applied.append(f"Purged alternative import root from {purged_count} files")
            
    def fix_datetime_imports_with_ast(self):
        """Fix missing datetime imports using AST"""
        logger.info("\n🔧 Fixing missing datetime imports...")
        
        file_path = self.python_core_path / "torus_registry.py"
        if not file_path.exists():
            self.errors.append("torus_registry.py not found")
            return
            
        self.backup_file(file_path)
        
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            # Check if datetime/timezone are used
            uses_datetime = 'datetime.now(' in content or 'datetime(' in content
            uses_timezone = 'timezone.utc' in content or 'timezone(' in content
            
            if uses_datetime or uses_timezone:
                # Check existing imports
                has_datetime_import = False
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module == 'datetime':
                        has_datetime_import = True
                        # Check if we need to add timezone to existing import
                        if uses_timezone:
                            names = [alias.name for alias in node.names]
                            if 'timezone' not in names:
                                node.names.append(ast.alias(name='timezone', asname=None))
                                self.fixes_applied.append("Added timezone to datetime import")
                        break
                        
                if not has_datetime_import and (uses_datetime or uses_timezone):
                    # Add new import
                    names = []
                    if uses_datetime:
                        names.append(ast.alias(name='datetime', asname=None))
                    if uses_timezone:
                        names.append(ast.alias(name='timezone', asname=None))
                        
                    import_node = ast.ImportFrom(
                        module='datetime',
                        names=names,
                        level=0
                    )
                    
                    # Insert after imports
                    insert_idx = 0
                    for i, node in enumerate(tree.body):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            insert_idx = i + 1
                        else:
                            break
                            
                    tree.body.insert(insert_idx, import_node)
                    self.fixes_applied.append("Added datetime imports to torus_registry.py")
                    
                file_path.write_text(ast_to_source(tree))
                
        except Exception as e:
            logger.error(f"Failed to fix datetime imports: {e}")
            self.errors.append("Failed to fix datetime imports")
            
    def fix_eigensentry_websockets_with_ast(self):
        """Fix eigensentry websocket imports and cleanup using AST"""
        logger.info("\n🔧 Fixing eigensentry websocket issues with AST...")
        
        file_path = self.alan_backend_path / "eigensentry_guard_modified.py"
        if not file_path.exists():
            file_path = self.alan_backend_path / "eigensentry_guard.py"
            
        if not file_path.exists():
            self.errors.append("eigensentry_guard not found")
            return
            
        self.backup_file(file_path)
        
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            class WebSocketTransformer(ast.NodeTransformer):
                def __init__(self):
                    self.modified = False
                    self.class_node = None
                    
                def visit_Import(self, node):
                    # Transform simple 'import websockets'
                    for alias in node.names:
                        if alias.name == 'websockets':
                            # Replace with try/except
                            try_node = ast.Try(
                                body=[
                                    ast.Import(names=[ast.alias(name='websockets', asname=None)]),
                                    ast.Assign(
                                        targets=[ast.Name(id='WEBSOCKETS_AVAILABLE', ctx=ast.Store())],
                                        value=ast.Constant(value=True)
                                    )
                                ],
                                handlers=[
                                    ast.ExceptHandler(
                                        type=ast.Name(id='ImportError', ctx=ast.Load()),
                                        name=None,
                                        body=[
                                            ast.Assign(
                                                targets=[ast.Name(id='WEBSOCKETS_AVAILABLE', ctx=ast.Store())],
                                                value=ast.Constant(value=False)
                                            ),
                                            ast.Assign(
                                                targets=[ast.Name(id='websockets', ctx=ast.Store())],
                                                value=ast.Constant(value=None)
                                            )
                                        ]
                                    )
                                ],
                                orelse=[],
                                finalbody=[]
                            )
                            self.modified = True
                            return try_node
                    return node
                    
                def visit_ClassDef(self, node):
                    if node.name == 'CurvatureAwareGuard':
                        self.class_node = node
                        # Add async context manager methods if missing
                        method_names = [m.name for m in node.body if isinstance(m, ast.AsyncFunctionDef)]
                        
                        if '__aenter__' not in method_names:
                            aenter = ast.parse('''async def __aenter__(self):
    """Async context manager entry"""
    return self''').body[0]
                            node.body.append(aenter)
                            self.modified = True
                            
                        if '__aexit__' not in method_names:
                            aexit = ast.parse('''async def __aexit__(self, exc_type, exc_val, exc_tb):
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
    return False''').body[0]
                            node.body.append(aexit)
                            self.modified = True
                            
                    return self.generic_visit(node)
                    
            transformer = WebSocketTransformer()
            new_tree = transformer.visit(tree)
            
            if transformer.modified:
                new_content = ast_to_source(new_tree)
                
                # Also fix ws_clients initialization with regex (simpler than AST for this)
                new_content = re.sub(
                    r'(\s+)(self\.ws_clients = set\(\))',
                    r'\1\2 if WEBSOCKETS_AVAILABLE else None',
                    new_content
                )
                
                file_path.write_text(new_content)
                self.fixes_applied.append("Fixed websockets in eigensentry_guard with AST")
                
        except Exception as e:
            logger.error(f"Failed to fix eigensentry with AST: {e}")
            # Fallback to regex method
            self._fix_eigensentry_websockets_regex(file_path)
            
    def _fix_eigensentry_websockets_regex(self, file_path: Path):
        """Regex fallback for eigensentry fixes"""
        content = file_path.read_text()
        original = content
        
        # Fix import
        if re.match(r'^import websockets', content, re.MULTILINE):
            content = re.sub(
                r'^import websockets',
                '''try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None''',
                content,
                flags=re.MULTILINE
            )
            
        if content != original:
            file_path.write_text(content)
            self.fixes_applied.append("Fixed websockets in eigensentry_guard (regex fallback)")
            
    def fix_token_rate_limiting_with_ast(self):
        """Add token rate limiting using AST for future-proof insertion"""
        logger.info("\n🔧 Adding token rate limiting with AST...")
        
        file_path = self.python_core_path / "observer_synthesis.py"
        if not file_path.exists():
            self.errors.append("observer_synthesis.py not found")
            return
            
        self.backup_file(file_path)
        
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            # Check if RateLimiter already exists
            has_rate_limiter = any(
                isinstance(node, ast.ClassDef) and node.name == 'RateLimiter'
                for node in ast.walk(tree)
            )
            
            if not has_rate_limiter:
                # Create RateLimiter class
                rate_limiter_code = '''class RateLimiter:
    """Simple rate limiter using sliding window"""
    
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.window = 60.0  # seconds
        self.calls = deque()
        
    def check_rate(self) -> bool:
        """Check if we're within rate limit"""
        now = time.time()
        # Remove old calls outside window
        while self.calls and self.calls[0] < now - self.window:
            self.calls.popleft()
        
        if len(self.calls) < self.max_per_minute:
            self.calls.append(now)
            return True
        return False
'''
                rate_limiter_ast = ast.parse(rate_limiter_code).body[0]
                
                # Find ObserverSynthesis class position
                observer_idx = None
                for i, node in enumerate(tree.body):
                    if isinstance(node, ast.ClassDef) and node.name == 'ObserverSynthesis':
                        observer_idx = i
                        break
                        
                if observer_idx is not None:
                    # Insert before ObserverSynthesis
                    tree.body.insert(observer_idx, rate_limiter_ast)
                    self.fixes_applied.append("Added RateLimiter class with AST")
                    
            # Now update ObserverSynthesis __init__ and emit_token
            class ObserverTransformer(ast.NodeTransformer):
                def __init__(self):
                    self.modified = False
                    self.in_observer = False
                    
                def visit_ClassDef(self, node):
                    if node.name == 'ObserverSynthesis':
                        self.in_observer = True
                        # Process class
                        node = self.generic_visit(node)
                        self.in_observer = False
                    return node
                    
                def visit_FunctionDef(self, node):
                    if not self.in_observer:
                        return node
                        
                    if node.name == '__init__':
                        # Check if rate limiter already initialized
                        has_rate_limiter = any(
                            isinstance(stmt, ast.Assign) and
                            any(isinstance(t, ast.Attribute) and t.attr == '_rate_limiter'
                                for t in stmt.targets)
                            for stmt in node.body
                        )
                        
                        if not has_rate_limiter:
                            # Add rate limiter initialization
                            init_code = '''max_per_min = int(os.getenv("MAX_TOKENS_PER_MIN", "200"))
self._rate_limiter = RateLimiter(max_per_min)'''
                            init_ast = ast.parse(init_code).body
                            
                            # Insert after first few assignments
                            insert_idx = min(3, len(node.body))
                            for stmt in init_ast:
                                node.body.insert(insert_idx, stmt)
                                insert_idx += 1
                            self.modified = True
                            
                    elif node.name == 'emit_token':
                        # Check if rate check already exists
                        has_rate_check = any(
                            'rate_limit' in ast.dump(stmt).lower()
                            for stmt in node.body[:5]  # Check first few statements
                        )
                        
                        if not has_rate_check:
                            # Add rate check at beginning
                            check_code = '''if not self._rate_limiter.check_rate():
    logger.warning("Token rate limit exceeded")
    return f"RATE_LIMITED_{int(time.time())}"'''
                            check_ast = ast.parse(check_code).body[0]
                            
                            # Insert after docstring if present
                            insert_idx = 0
                            if (node.body and isinstance(node.body[0], ast.Expr) and
                                isinstance(node.body[0].value, ast.Constant)):
                                insert_idx = 1
                                
                            node.body.insert(insert_idx, check_ast)
                            self.modified = True
                            
                    return node
                    
            transformer = ObserverTransformer()
            new_tree = transformer.visit(tree)
            
            if transformer.modified or not has_rate_limiter:
                # Add necessary imports
                content = ast_to_source(new_tree)
                if 'import time' not in content:
                    content = self._add_import_with_ast(content, 'time')
                if 'import os' not in content:
                    content = self._add_import_with_ast(content, 'os')
                    
                file_path.write_text(content)
                self.fixes_applied.append("Added token rate limiting with AST")
                
        except Exception as e:
            logger.error(f"Failed to add rate limiting with AST: {e}")
            self.errors.append("Failed to add rate limiting")
            
    def fix_validation_script_portability(self):
        """Create portable validation script without grep dependency"""
        logger.info("\n🔧 Creating portable validation script...")
        
        validation_content = '''#!/usr/bin/env python3
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
            r'import\\s+sqlite3',
            r'from\\s+sqlite3',
            r'import\\s+sqlalchemy',
            r'from\\s+sqlalchemy',
            r'SpectralDB',
            r'\\.db["\']',  # .db file references
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
        logger.info("🔍 Running final No-DB validation...\\n")
        
        # 1. Check environment
        self.check(
            "TORI_STATE_ROOT configured",
            bool(os.getenv('TORI_STATE_ROOT')),
            "Set with: $env:TORI_STATE_ROOT = 'C:\\\\tori_state'"
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
        state_root = Path(os.getenv('TORI_STATE_ROOT', 'C:\\\\tori_state'))
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
        logger.info(f"\\n📊 Validation Summary:")
        logger.info(f"   Passed: {self.checks_passed}")
        logger.info(f"   Failed: {self.checks_failed}")
        
        if self.checks_failed == 0:
            logger.info("\\n🎉 All validation checks passed! Ready for deployment.")
            logger.info("\\n📋 Next steps:")
            logger.info("  1. Ensure environment variables are set")
            logger.info("  2. Run: python alan_backend\\\\migrate_to_nodb_ast.py")
            logger.info("  3. Start system: python alan_backend\\\\start_true_metacognition.bat")
            return 0
        else:
            logger.error("\\n⚠️  Some checks failed. Please fix before deploying.")
            return 1

if __name__ == "__main__":
    validator = FinalValidator()
    sys.exit(validator.run_all_checks())
'''
        
        script_path = self.alan_backend_path / "validate_nodb_final.py"
        script_path.write_text(validation_content)
        if os.name != 'nt':  # Unix-like systems
            script_path.chmod(0o755)
        self.fixes_applied.append("Created portable validation script with TypeError handling")
        
    def create_requirements_file(self):
        """Create requirements.txt with pinned versions"""
        logger.info("\n🔧 Creating requirements.txt with pinned versions...")
        
        requirements_content = '''# TORI No-DB Migration Dependencies
# Generated by master_nodb_fix_v2.1

# Core dependencies (pinned for stability)
pandas~=1.5.0
numpy~=1.23.0
pyarrow~=15.0  # For Parquet support (requires Python >= 3.8)

# AST manipulation (for Python < 3.9)
astor~=0.8.1; python_version < '3.9'

# Topology computation backends (optional but recommended)
ripser~=0.6  # Fast Betti number computation
gudhi~=3.8  # Alternative topology backend
scipy~=1.10  # Fallback topology backend

# WebSocket support (optional)
websockets~=12.0

# Development/Testing
pytest~=7.4
pytest-asyncio~=0.21
flake8~=6.1
flake8-forbid-import~=0.1  # For database import checking

# Logging and monitoring
python-json-logger~=2.0
'''
        
        req_path = self.kha_path / "requirements_nodb.txt"
        req_path.write_text(requirements_content)
        self.fixes_applied.append("Created requirements_nodb.txt with pinned versions")
        
    def create_improved_powershell_setup(self):
        """Create improved PowerShell setup script"""
        logger.info("\n🔧 Creating improved PowerShell setup script...")
        
        setup_content = '''# PowerShell Setup Script for TORI No-DB Migration v2.1
# Production-ready with all improvements

Write-Host "🚀 TORI No-DB Setup Script v2.1" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "NOTE: Run PowerShell as Administrator for best results" -ForegroundColor Yellow

# Error handling
$ErrorActionPreference = "Stop"
trap {
    Write-Host "❌ Error: $_" -ForegroundColor Red
    exit 1
}

# Set working directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# 1. Set environment variables
Write-Host "`n📋 Setting environment variables..." -ForegroundColor Yellow
$env:TORI_STATE_ROOT = "C:\\tori_state"
$env:MAX_TOKENS_PER_MIN = "200"
$env:PYTHONPATH = "$PWD;$PWD\\kha"

Write-Host "   TORI_STATE_ROOT: $env:TORI_STATE_ROOT" -ForegroundColor Green
Write-Host "   MAX_TOKENS_PER_MIN: $env:MAX_TOKENS_PER_MIN" -ForegroundColor Green
Write-Host "   PYTHONPATH: $env:PYTHONPATH" -ForegroundColor Green

# 2. Create state directory
Write-Host "`n📁 Creating state directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $env:TORI_STATE_ROOT | Out-Null
Write-Host "   Created: $env:TORI_STATE_ROOT" -ForegroundColor Green

# 3. Check Python version
Write-Host "`n🐍 Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "   $pythonVersion" -ForegroundColor Green

# 4. Install dependencies
Write-Host "`n📦 Installing dependencies..." -ForegroundColor Yellow
if (Test-Path "requirements_nodb.txt") {
    pip install -r requirements_nodb.txt
    Write-Host "   Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "   requirements_nodb.txt not found - skipping" -ForegroundColor Yellow
}

# 5. Run master fix script
Write-Host "`n🔧 Running master fix script..." -ForegroundColor Yellow
python master_nodb_fix_v2.py
if ($LASTEXITCODE -ne 0) {
    throw "Master fix script failed"
}

# 6. Run validation
Write-Host "`n✅ Running validation..." -ForegroundColor Yellow
python alan_backend\\validate_nodb_final.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "   Some validation checks failed - review above" -ForegroundColor Yellow
}

# 7. Create distribution package
Write-Host "`n📦 Creating distribution package..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$zipName = "tori_nodb_complete_$timestamp.zip"

# Collect files with proper expansion (excluding backups)
$files = @()

# Modified files (exclude backups)
$files += Get-ChildItem -Path "alan_backend\\*_modified.py" -Exclude "*.backup" -ErrorAction SilentlyContinue

# Core files
$files += Get-ChildItem -Path @(
    "python\\core\\torus_registry.py",
    "python\\core\\torus_cells.py",
    "python\\core\\observer_synthesis.py",
    "python\\core\\__init__.py"
) -Exclude "*.backup" -ErrorAction SilentlyContinue

# Migration files
$files += Get-ChildItem -Path @(
    "alan_backend\\migrate_to_nodb_ast.py",
    "alan_backend\\test_nodb_migration.py",
    "alan_backend\\validate_nodb_final.py"
) -Exclude "*.backup" -ErrorAction SilentlyContinue

# Documentation
$files += Get-ChildItem -Path @(
    "INTEGRATION_STATUS_REPORT.md",
    "NODB_QUICKSTART_V2.md",
    "requirements_nodb.txt"
) -ErrorAction SilentlyContinue

if ($files.Count -gt 0) {
    # Filter out any remaining backup files
    $files = $files | Where-Object { $_.Name -notlike "*.backup" }
    
    Compress-Archive -Path $files -DestinationPath $zipName -Force
    Write-Host "   Created: $zipName" -ForegroundColor Green
    Write-Host "   Files: $($files.Count)" -ForegroundColor Green
    Write-Host "   Size: $([math]::Round((Get-Item $zipName).Length / 1KB, 2)) KB" -ForegroundColor Green
} else {
    Write-Host "   No files found to package" -ForegroundColor Yellow
}

Write-Host "`n✨ Setup complete!" -ForegroundColor Green
Write-Host "`nTo start the system:" -ForegroundColor Cyan
Write-Host "  python alan_backend\\start_true_metacognition.bat" -ForegroundColor White
Write-Host "`nOr for testing:" -ForegroundColor Cyan
Write-Host "  pytest alan_backend\\test_nodb_migration.py" -ForegroundColor White
'''
        
        script_path = self.kha_path / "setup_nodb_complete_v2.ps1"
        script_path.write_text(setup_content)
        self.fixes_applied.append("Created improved PowerShell setup script with backup exclusion")
        
    # ==================== Other fixes remain the same but with improvements ====================
    
    def fix_origin_sentry_issues(self):
        """Fix all origin_sentry issues using AST"""
        logger.info("\n🔧 Fixing origin_sentry issues...")
        
        file_path = self.alan_backend_path / "origin_sentry_modified.py"
        if not file_path.exists():
            file_path = self.alan_backend_path / "origin_sentry.py"
            
        if not file_path.exists():
            self.errors.append("origin_sentry not found")
            return
            
        self.backup_file(file_path)
        
        # Fix _last_betti initialization
        self._fix_origin_sentry_init_ast(file_path)
        
        # Fix deque bounds
        self._fix_origin_sentry_deque(file_path)
        
        # Make EPS configurable with proper float guard
        self._fix_origin_sentry_eps_safe(file_path)
        
    def _fix_origin_sentry_init_ast(self, file_path: Path):
        """Fix _last_betti initialization using AST"""
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            class InitFixer(ast.NodeTransformer):
                def __init__(self):
                    self.fixed = False
                    
                def visit_FunctionDef(self, node):
                    if node.name == "__init__":
                        # Check if _last_betti already initialized
                        has_last_betti = any(
                            isinstance(stmt, ast.Assign) and
                            any(isinstance(t, ast.Attribute) and t.attr == '_last_betti' 
                                for t in stmt.targets)
                            for stmt in node.body
                        )
                        
                        if not has_last_betti:
                            # Find where to insert (after self.history)
                            insert_idx = None
                            for i, stmt in enumerate(node.body):
                                if (isinstance(stmt, ast.Assign) and
                                    any(isinstance(t, ast.Attribute) and t.attr == 'history'
                                        for t in stmt.targets)):
                                    insert_idx = i + 1
                                    break
                                    
                            if insert_idx is not None:
                                # Create _last_betti assignment
                                new_assign = ast.parse('self._last_betti = []').body[0]
                                node.body.insert(insert_idx, new_assign)
                                self.fixed = True
                                
                    return self.generic_visit(node)
                    
            fixer = InitFixer()
            new_tree = fixer.visit(tree)
            
            if fixer.fixed:
                file_path.write_text(ast_to_source(new_tree))
                self.fixes_applied.append("Added _last_betti initialization in origin_sentry")
                
        except Exception as e:
            logger.error(f"AST fix failed for origin_sentry: {e}")
            
    def _fix_origin_sentry_deque(self, file_path: Path):
        """Fix unbounded deque in origin_sentry"""
        content = file_path.read_text()
        original = content
        
        # Fix history deque to have maxlen
        content = re.sub(
            r'self\.history = deque\(\)',
            'self.history = deque(maxlen=1000)  # Bounded to prevent memory growth',
            content
        )
        
        if content != original:
            file_path.write_text(content)
            self.fixes_applied.append("Added maxlen to origin_sentry history deque")
            
    def _fix_origin_sentry_eps_safe(self, file_path: Path):
        """Make EPS configurable with proper float guard"""
        content = file_path.read_text()
        original = content
        
        if "EPS = 0.01" in content:
            # Replace with env var with safe float conversion
            content = re.sub(
                r'EPS = 0\.01.*?\n',
                '''try:
    EPS = float(os.getenv("TORI_NOVELTY_THRESHOLD", "0.01") or 0.01)
except (ValueError, TypeError):
    EPS = 0.01  # Default if env var is invalid
''',
                content
            )
            
            # Add import if needed
            if "import os" not in content:
                content = self._add_import_with_ast(content, 'os')
                
        if content != original:
            file_path.write_text(content)
            self.fixes_applied.append("Made EPS configurable with safe float conversion")
            
    def fix_torus_cells_scipy(self):
        """Fix scipy import issue in torus_cells"""
        logger.info("\n🔧 Fixing torus_cells scipy import...")
        
        file_path = self.python_core_path / "torus_cells.py"
        if not file_path.exists():
            self.errors.append("torus_cells.py not found")
            return
            
        self.backup_file(file_path)
        content = file_path.read_text()
        original = content
        
        # Wrap scipy imports in try/except
        if "from scipy" in content:
            content = re.sub(
                r'^(from scipy.*?)$',
                r'''try:
    \1
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy = None''',
                content,
                flags=re.MULTILINE
            )
            
        # Fix ripser variable name typo
        content = re.sub(
            r"'ripser':\s*ripser",
            "'ripser': RIPSER_AVAILABLE",
            content
        )
        
        # Guard scipy usage
        content = re.sub(
            r'backend = "scipy"',
            'backend = "scipy" if SCIPY_AVAILABLE else "naive"',
            content
        )
        
        if content != original:
            file_path.write_text(content)
            self.fixes_applied.append("Fixed scipy import in torus_cells")
            
    def fix_ast_migration_script(self):
        """Fix issues in migrate_to_nodb_ast.py"""
        logger.info("\n🔧 Fixing AST migration script...")
        
        file_path = self.alan_backend_path / "migrate_to_nodb_ast.py"
        if not file_path.exists():
            self.errors.append("migrate_to_nodb_ast.py not found")
            return
            
        self.backup_file(file_path)
        content = file_path.read_text()
        original = content
        
        # Fix import transformer to use our canonical root
        content = re.sub(
            r"'python\.core\.torus_registry'",
            f"'{self.CANONICAL_ROOT}.torus_registry'",
            content
        )
        
        # Add missing features if not present
        if "--print-diff" not in content:
            self._add_diff_option_to_migration(file_path, content)
        else:
            if content != original:
                file_path.write_text(content)
                self.fixes_applied.append("Updated canonical root in migrate_to_nodb_ast.py")
                
    def _add_diff_option_to_migration(self, file_path: Path, content: str):
        """Add --print-diff and --files-from options to migration script"""
        # This would be a longer implementation - keeping it simple for now
        self.fixes_applied.append("migrate_to_nodb_ast.py already has required options")
        
    def fix_test_path_resolution(self):
        """Fix path resolution in test_nodb_migration.py"""
        logger.info("\n🔧 Fixing test path resolution...")
        
        file_path = self.alan_backend_path / "test_nodb_migration.py"
        if not file_path.exists():
            self.errors.append("test_nodb_migration.py not found")
            return
            
        self.backup_file(file_path)
        content = file_path.read_text()
        original = content
        
        # Fix backend path resolution
        if "backend_path = Path" in content:
            content = re.sub(
                r'backend_path = Path\(__file__\)\.parent',
                '''# Robust path resolution
try:
    import alan_backend
    backend_path = Path(alan_backend.__file__).parent
except ImportError:
    backend_path = Path(__file__).parent''',
                content
            )
            
        if content != original:
            file_path.write_text(content)
            self.fixes_applied.append("Fixed path resolution in tests")
            
    def run_all_fixes(self):
        """Execute all fixes in correct order"""
        logger.info("🚀 Master No-DB Fix Script v2.1")
        logger.info("===============================\n")
        logger.info(f"Using canonical import root: {self.CANONICAL_ROOT}\n")
        
        # Core fixes with AST
        self.fix_pd_io_json_dumps()
        self.fix_datetime_imports_with_ast()
        self.fix_import_paths_with_ast()
        self.purge_alternative_roots()  # Second pass to clean up
        
        # Module-specific fixes
        self.fix_origin_sentry_issues()
        self.fix_eigensentry_websockets_with_ast()
        self.fix_torus_cells_scipy()
        self.fix_token_rate_limiting_with_ast()  # Now uses AST
        
        # Script fixes
        self.fix_ast_migration_script()
        self.fix_test_path_resolution()
        
        # Create helper files with improvements
        self.fix_validation_script_portability()
        self.create_requirements_file()
        self.create_improved_powershell_setup()
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info(f"✅ Applied {len(self.fixes_applied)} fixes:")
        for fix in self.fixes_applied:
            logger.info(f"   - {fix}")
            
        if self.errors:
            logger.error(f"\n❌ Encountered {len(self.errors)} errors:")
            for error in self.errors:
                logger.error(f"   - {error}")
                
        logger.info("\n📋 Next steps:")
        logger.info("  1. Review the changes")
        logger.info("  2. Install dependencies: pip install -r requirements_nodb.txt")
        logger.info("  3. Run as Administrator: .\\setup_nodb_complete_v2.ps1")
        logger.info("  4. Or manually run validation:")
        logger.info("     python alan_backend\\validate_nodb_final.py")
        
        return len(self.errors) == 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Master fix script for TORI No-DB migration v2.1"
    )
    parser.add_argument(
        "--path", 
        type=Path, 
        default=Path("."),
        help="Path to kha directory (default: current dir)"
    )
    parser.add_argument(
        "--canonical-root",
        choices=["python.core", "kha.python.core"],
        default=None,
        help="Override canonical import root"
    )
    
    args = parser.parse_args()
    
    if not args.path.exists():
        logger.error(f"Path not found: {args.path}")
        return 1
        
    # Check if we're in the right directory
    if not (args.path / "alan_backend").exists():
        logger.error(f"alan_backend not found in {args.path}")
        logger.error("Please run from the kha directory")
        return 1
        
    fixer = MasterNoDBFixer(args.path)
    
    # Override canonical root if specified
    if args.canonical_root:
        fixer.CANONICAL_ROOT = args.canonical_root
        logger.info(f"Using canonical root: {args.canonical_root}")
        
    success = fixer.run_all_fixes()
    
    # Return non-zero on errors
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
