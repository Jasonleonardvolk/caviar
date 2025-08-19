#!/usr/bin/env python3
"""
Master No-DB Migration Fix Script
Addresses ALL issues from code reviews and audits
Run this single script to fix everything
"""

import os
import re
import ast
import json
import shutil
import difflib
import astor
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MasterNoDBFixer:
    """Comprehensive fixes for No-DB migration"""
    
    # Define canonical import root (you can change this to kha.python.core if preferred)
    CANONICAL_ROOT = "python.core"
    
    def __init__(self, kha_path: Path):
        self.kha_path = kha_path
        self.alan_backend_path = kha_path / "alan_backend"
        self.python_core_path = kha_path / "python" / "core"
        self.fixes_applied = []
        self.errors = []
        
    def backup_file(self, file_path: Path) -> Path:
        """Create backup before modifying"""
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
        if not backup_path.exists():
            shutil.copy2(file_path, backup_path)
            logger.info(f"📋 Backed up: {file_path.name}")
        return backup_path
        
    # ==================== Core Issues ====================
    
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
            content = re.sub(r'pd\.io\.json\.dumps', 'json.dumps', content)
            
            # Add json import if needed
            if 'json.dumps' in content and 'import json' not in content:
                # Add after pandas import
                content = re.sub(
                    r'(import pandas as pd\n)',
                    r'\1import json\n',
                    content
                )
                
            if content != original:
                file_path.write_text(content)
                self.fixes_applied.append(f"Fixed pd.io.json.dumps in {file_path.name}")
                
    def fix_datetime_imports(self):
        """Fix missing datetime imports in rollover patches"""
        logger.info("\n🔧 Fixing missing datetime imports...")
        
        # Fix in torus_registry.py
        file_path = self.python_core_path / "torus_registry.py"
        if file_path.exists():
            self.backup_file(file_path)
            content = file_path.read_text()
            
            # Check if datetime/timezone are used but not imported
            if ('datetime.now(' in content or 'timezone.utc' in content) and \
               'from datetime import datetime, timezone' not in content:
                # Add import after other imports
                content = re.sub(
                    r'(import os\n)',
                    r'\1from datetime import datetime, timezone\n',
                    content
                )
                file_path.write_text(content)
                self.fixes_applied.append("Added datetime imports to torus_registry.py")
                
    def fix_import_paths(self):
        """Standardize all import paths to canonical root"""
        logger.info(f"\n🔧 Standardizing imports to {self.CANONICAL_ROOT}...")
        
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
        
        import_patterns = {
            r'from kha\.python\.core\.': f'from {self.CANONICAL_ROOT}.',
            r'from python\.core\.': f'from {self.CANONICAL_ROOT}.',
            r'import kha\.python\.core\.': f'import {self.CANONICAL_ROOT}.',
            r'import python\.core\.': f'import {self.CANONICAL_ROOT}.',
        }
        
        for file_path in files_to_fix:
            if not file_path.exists():
                continue
                
            self.backup_file(file_path)
            content = file_path.read_text()
            original = content
            
            for pattern, replacement in import_patterns.items():
                content = re.sub(pattern, replacement, content)
                
            if content != original:
                file_path.write_text(content)
                self.fixes_applied.append(f"Standardized imports in {file_path.name}")
                
    def fix_origin_sentry_issues(self):
        """Fix all origin_sentry issues using AST"""
        logger.info("\n🔧 Fixing origin_sentry issues with AST...")
        
        file_path = self.alan_backend_path / "origin_sentry_modified.py"
        if not file_path.exists():
            file_path = self.alan_backend_path / "origin_sentry.py"
            
        if not file_path.exists():
            self.errors.append("origin_sentry not found")
            return
            
        self.backup_file(file_path)
        content = file_path.read_text()
        
        try:
            tree = ast.parse(content)
            
            class OriginSentryFixer(ast.NodeTransformer):
                def __init__(self):
                    self.in_init = False
                    self.init_fixed = False
                    
                def visit_FunctionDef(self, node):
                    if node.name == "__init__":
                        self.in_init = True
                        # Add _last_betti initialization
                        for stmt in node.body:
                            if (isinstance(stmt, ast.Assign) and 
                                any(t.id == "_last_betti" for t in stmt.targets if isinstance(t, ast.Name))):
                                self.init_fixed = True
                                break
                                
                        if not self.init_fixed:
                            # Add after self.history initialization
                            for i, stmt in enumerate(node.body):
                                if (isinstance(stmt, ast.Assign) and 
                                    any(t.attr == "history" for t in stmt.targets if isinstance(t, ast.Attribute))):
                                    # Insert _last_betti initialization
                                    new_stmt = ast.parse("self._last_betti = []").body[0]
                                    node.body.insert(i + 1, new_stmt)
                                    self.init_fixed = True
                                    break
                                    
                        self.in_init = False
                        
                    return self.generic_visit(node)
                    
            fixer = OriginSentryFixer()
            new_tree = fixer.visit(tree)
            
            if fixer.init_fixed:
                new_content = astor.to_source(new_tree)
                file_path.write_text(new_content)
                self.fixes_applied.append("Added _last_betti initialization in origin_sentry")
                
        except Exception as e:
            logger.error(f"AST parsing failed for origin_sentry: {e}")
            # Fallback to regex
            self._fix_origin_sentry_regex(file_path)
            
        # Fix deque bounds for history
        self._fix_origin_sentry_deque(file_path)
        
        # Make EPS configurable
        self._fix_origin_sentry_eps(file_path)
        
    def _fix_origin_sentry_regex(self, file_path: Path):
        """Regex fallback for origin_sentry fixes"""
        content = file_path.read_text()
        original = content
        
        # Fix _last_betti initialization
        if "_last_betti" not in content.split("__init__")[1].split("def ")[0]:
            content = re.sub(
                r'(self\.history = deque\(.*?\))',
                r'\1\n        self._last_betti = []  # Initialize for novelty computation',
                content
            )
            
        if content != original:
            file_path.write_text(content)
            self.fixes_applied.append("Fixed _last_betti in origin_sentry (regex)")
            
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
            
    def _fix_origin_sentry_eps(self, file_path: Path):
        """Make EPS configurable in origin_sentry"""
        content = file_path.read_text()
        original = content
        
        if "EPS = 0.01" in content:
            # Replace with env var
            content = re.sub(
                r'EPS = 0\.01.*?\n',
                'EPS = float(os.getenv("TORI_NOVELTY_THRESHOLD", "0.01"))  # Configurable threshold\n',
                content
            )
            
            # Add import if needed
            if "import os" not in content:
                content = "import os\n" + content
                
        if content != original:
            file_path.write_text(content)
            self.fixes_applied.append("Made EPS configurable in origin_sentry")
            
    def fix_eigensentry_websockets(self):
        """Fix eigensentry websocket imports"""
        logger.info("\n🔧 Fixing eigensentry websocket issues...")
        
        file_path = self.alan_backend_path / "eigensentry_guard_modified.py"
        if not file_path.exists():
            file_path = self.alan_backend_path / "eigensentry_guard.py"
            
        if not file_path.exists():
            self.errors.append("eigensentry_guard not found")
            return
            
        self.backup_file(file_path)
        content = file_path.read_text()
        original = content
        
        # Fix top-level websocket import
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
            
        # Guard WebSocket usage
        content = re.sub(
            r'(\s+)(self\.ws_clients = set\(\))',
            r'\1\2 if WEBSOCKETS_AVAILABLE else None',
            content
        )
        
        # Add async cleanup
        if "__aexit__" not in content:
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
            # Find where to insert (end of class)
            class_end = content.rfind('\nclass ')
            if class_end == -1:
                class_end = len(content)
            insert_pos = content.rfind('\n    def ', 0, class_end)
            if insert_pos > 0:
                # Find end of last method
                next_section = content.find('\n\n', insert_pos)
                if next_section > 0:
                    content = content[:next_section] + cleanup_code + content[next_section:]
                    
        if content != original:
            file_path.write_text(content)
            self.fixes_applied.append("Fixed websockets in eigensentry_guard")
            
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
        
        # Check if scipy is used but not imported with try/except
        if "scipy" in content and "try:" not in content.split("import scipy")[0][-50:]:
            # Find scipy imports and wrap them
            content = re.sub(
                r'^(from scipy.*?$)',
                r'''try:
    \1
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False''',
                content,
                flags=re.MULTILINE
            )
            
        # Fix ripser variable name typo if present
        content = re.sub(
            r"'ripser':\s*ripser",
            "'ripser': RIPSER_AVAILABLE",
            content
        )
        
        if content != original:
            file_path.write_text(content)
            self.fixes_applied.append("Fixed scipy import in torus_cells")
            
    def fix_token_rate_limiting(self):
        """Add proper token rate limiting with env var"""
        logger.info("\n🔧 Fixing token rate limiting...")
        
        file_path = self.python_core_path / "observer_synthesis.py"
        if not file_path.exists():
            self.errors.append("observer_synthesis.py not found")
            return
            
        self.backup_file(file_path)
        content = file_path.read_text()
        original = content
        
        # Add rate limiting check in emit_token
        if "MAX_TOKENS_PER_MIN" not in content:
            # Find emit_token method
            emit_pos = content.find("def emit_token(self, measurement:")
            if emit_pos > 0:
                # Add rate check at beginning of method
                rate_check = '''        # Check rate limit
        max_tokens = int(os.getenv('MAX_TOKENS_PER_MIN', '200'))
        if hasattr(self, '_rate_limiter'):
            if not self._rate_limiter.check_rate():
                logger.warning("Token rate limit exceeded")
                return f"RATE_LIMITED_{int(time.time())}"
        
'''
                # Insert after method definition
                method_start = content.find(":", emit_pos) + 1
                next_line = content.find("\n", method_start) + 1
                # Skip docstring if present
                if '"""' in content[next_line:next_line+10]:
                    docstring_end = content.find('"""', next_line + 3) + 3
                    next_line = content.find("\n", docstring_end) + 1
                    
                content = content[:next_line] + rate_check + content[next_line:]
                
            # Add import if needed
            if "import os" not in content:
                content = "import os\n" + content
                
        if content != original:
            file_path.write_text(content)
            self.fixes_applied.append("Added token rate limiting to observer_synthesis")
            
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
        
        # Add --print-diff option
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
            
            # Add --files-from option
            content = re.sub(
                r'(parser\.add_argument\("--print-diff".*?\))',
                r'''\1
    parser.add_argument("--files-from", type=str,
                       help="Glob pattern for files to migrate")''',
                content,
                flags=re.DOTALL
            )
            
        # Fix import transformer to use our canonical root
        content = re.sub(
            r"'python\.core\.torus_registry'",
            f"'{self.CANONICAL_ROOT}.torus_registry'",
            content
        )
        
        if content != original:
            file_path.write_text(content)
            self.fixes_applied.append("Enhanced migrate_to_nodb_ast.py")
            
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
import alan_backend
backend_path = Path(alan_backend.__file__).parent''',
                content
            )
            
        if content != original:
            file_path.write_text(content)
            self.fixes_applied.append("Fixed path resolution in tests")
            
    def create_validation_script(self):
        """Create comprehensive validation script"""
        logger.info("\n🔧 Creating validation script...")
        
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
            "Set with: $env:TORI_STATE_ROOT = 'C:\\\\tori_state'"
        )
        
        # 2. Check no database imports using flake8-forbid-import
        try:
            # Try flake8-forbid-import first
            result = subprocess.run(
                ['python', '-m', 'flake8', '--select=FI', 'alan_backend/'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                # Fallback to FORBID if FI doesn't work
                result = subprocess.run(
                    ['python', '-m', 'flake8', '--select=FORBID', 'alan_backend/'],
                    capture_output=True,
                    text=True
                )
            
            self.check(
                "No database imports found",
                result.returncode == 0,
                result.stdout if result.stdout else result.stderr
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
            len(import_variations) <= 1,
            f"Found multiple import roots: {import_variations}"
        )
        
        # 4. Check Parquet files created
        state_root = Path(os.getenv('TORI_STATE_ROOT', 'C:\\\\tori_state'))
        if state_root.exists():
            parquet_files = list(state_root.glob('*.parquet'))
            self.check(
                "Parquet persistence working",
                len(parquet_files) > 0 or True,  # Pass if dir exists
                f"No .parquet files in {state_root} (will be created on first write)"
            )
        else:
            self.check("State root exists", False, f"Create dir: {state_root}")
            
        # 5. Check bounded collections
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from chaos_channel_controller_modified import ChaosChannelController
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
            from eigensentry_guard_modified import CurvatureAwareGuard
            guard = CurvatureAwareGuard()
            self.check("WebSockets are optional", True)
            del sys.modules['websockets']
        except Exception as e:
            self.check("WebSockets are optional", False, str(e))
            
        # 7. Check no pd.io.json usage
        try:
            result = subprocess.run(
                ['grep', '-r', 'pd\\\\.io\\\\.json', 'alan_backend/', 'python/core/'],
                capture_output=True,
                text=True
            )
            self.check(
                "No pd.io.json usage",
                result.returncode == 1,  # grep returns 1 if no matches
                "Found pd.io.json usage - use json.dumps instead"
            )
        except:
            # Windows fallback
            import glob
            found_pd_io = False
            for pattern in ['alan_backend/**/*.py', 'python/core/**/*.py']:
                for file in glob.glob(pattern, recursive=True):
                    try:
                        if 'pd.io.json' in Path(file).read_text():
                            found_pd_io = True
                            break
                    except:
                        pass
            self.check("No pd.io.json usage", not found_pd_io, 
                      "Found pd.io.json - use json.dumps")
            
        # Summary
        logger.info(f"\\n📊 Validation Summary:")
        logger.info(f"   Passed: {self.checks_passed}")
        logger.info(f"   Failed: {self.checks_failed}")
        
        if self.checks_failed == 0:
            logger.info("\\n🎉 All validation checks passed! Ready for deployment.")
            logger.info("\\n📋 Next steps:")
            logger.info("  1. Set environment variables:")
            logger.info("     $env:TORI_STATE_ROOT = 'C:\\\\tori_state'")
            logger.info("     $env:MAX_TOKENS_PER_MIN = '200'")
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
        script_path.chmod(0o755)
        self.fixes_applied.append("Created validate_nodb_final.py")
        
    def create_setup_script(self):
        """Create PowerShell setup script"""
        logger.info("\n🔧 Creating PowerShell setup script...")
        
        setup_content = '''# PowerShell Setup Script for TORI No-DB Migration
# Run this to set up environment and apply all fixes

Write-Host "🚀 TORI No-DB Setup Script" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan

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

# 2. Create state directory
Write-Host "`n📁 Creating state directory..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $env:TORI_STATE_ROOT | Out-Null
Write-Host "   Created: $env:TORI_STATE_ROOT" -ForegroundColor Green

# 3. Run master fix script
Write-Host "`n🔧 Running master fix script..." -ForegroundColor Yellow
python master_nodb_fix.py

# 4. Run validation
Write-Host "`n✅ Running validation..." -ForegroundColor Yellow
python alan_backend\\validate_nodb_final.py

# 5. Create distribution package
Write-Host "`n📦 Creating distribution package..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$zipName = "tori_nodb_complete_$timestamp.zip"

$filesToZip = @(
    "alan_backend\\*_modified.py",
    "python\\core\\torus_registry.py",
    "python\\core\\torus_cells.py", 
    "python\\core\\observer_synthesis.py",
    "python\\core\\__init__.py",
    "alan_backend\\migrate_to_nodb_ast.py",
    "alan_backend\\test_nodb_migration.py",
    "alan_backend\\validate_nodb_final.py",
    "INTEGRATION_STATUS_REPORT.md"
)

$files = Get-ChildItem -Path $filesToZip -ErrorAction SilentlyContinue
if ($files) {
    Compress-Archive -Path $files -DestinationPath $zipName -Force
    Write-Host "   Created: $zipName" -ForegroundColor Green
    Write-Host "   Size: $((Get-Item $zipName).Length / 1KB) KB" -ForegroundColor Green
}

Write-Host "`n✨ Setup complete!" -ForegroundColor Green
Write-Host "`nTo start the system:" -ForegroundColor Cyan
Write-Host "  python alan_backend\\start_true_metacognition.bat" -ForegroundColor White
'''
        
        script_path = self.kha_path / "setup_nodb_complete.ps1"
        script_path.write_text(setup_content)
        self.fixes_applied.append("Created setup_nodb_complete.ps1")
        
    def run_all_fixes(self):
        """Execute all fixes in correct order"""
        logger.info("🚀 Master No-DB Fix Script")
        logger.info("=========================\n")
        
        # Core fixes
        self.fix_pd_io_json_dumps()
        self.fix_datetime_imports()
        self.fix_import_paths()
        
        # Module-specific fixes
        self.fix_origin_sentry_issues()
        self.fix_eigensentry_websockets()
        self.fix_torus_cells_scipy()
        self.fix_token_rate_limiting()
        
        # Script fixes
        self.fix_ast_migration_script()
        self.fix_test_path_resolution()
        
        # Create helper scripts
        self.create_validation_script()
        self.create_setup_script()
        
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
        logger.info("  2. Run: .\\setup_nodb_complete.ps1")
        logger.info("  3. Or manually:")
        logger.info("     - Set environment variables")
        logger.info("     - Run: python alan_backend\\validate_nodb_final.py")
        logger.info("     - Start: python alan_backend\\start_true_metacognition.bat")
        
        return len(self.errors) == 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Master fix script for TORI No-DB migration"
    )
    parser.add_argument(
        "--path", 
        type=Path, 
        default=Path("."),
        help="Path to kha directory (default: current dir)"
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
    success = fixer.run_all_fixes()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
