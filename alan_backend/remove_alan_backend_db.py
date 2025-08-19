#!/usr/bin/env python3
"""
Complete alan_backend database removal script
Removes ALL database code from alan_backend directory
"""

import os
import shutil
from pathlib import Path
import re
import ast
import logging
from typing import List, Set, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlanBackendDBCleaner:
    """Remove all database code from alan_backend"""
    
    def __init__(self, alan_backend_path: Path, dry_run: bool = True):
        self.backend_path = alan_backend_path
        self.dry_run = dry_run
        self.backup_dir = self.backend_path / "backup_db_removal"
        self.files_with_db = []
        self.files_removed = []
        self.files_modified = []
        
    def scan_for_db_usage(self) -> List[Tuple[Path, List[str]]]:
        """Scan all Python files for database imports and usage"""
        db_patterns = [
            r'import\s+sqlite3',
            r'from\s+sqlite3',
            r'import\s+psycopg2', 
            r'from\s+psycopg2',
            r'import\s+sqlalchemy',
            r'from\s+sqlalchemy',
            r'import\s+pymongo',
            r'from\s+pymongo',
            r'import\s+redis',
            r'from\s+redis',
            r'\.connect\(',
            r'\.cursor\(',
            r'\.execute\(',
            r'\.commit\(',
            r'\.rollback\(',
            r'CREATE TABLE',
            r'INSERT INTO',
            r'SELECT.*FROM',
            r'UPDATE.*SET',
            r'DELETE FROM',
        ]
        
        combined_pattern = '|'.join(f'({p})' for p in db_patterns)
        
        results = []
        for py_file in self.backend_path.rglob("*.py"):
            if "backup" in str(py_file) or "__pycache__" in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                matches = re.findall(combined_pattern, content, re.IGNORECASE | re.MULTILINE)
                
                if matches:
                    # Get line numbers for each match
                    lines_with_db = []
                    for line_num, line in enumerate(content.splitlines(), 1):
                        if re.search(combined_pattern, line, re.IGNORECASE):
                            lines_with_db.append(f"Line {line_num}: {line.strip()}")
                    
                    results.append((py_file, lines_with_db))
                    self.files_with_db.append(py_file)
                    
            except Exception as e:
                logger.error(f"Error scanning {py_file}: {e}")
                
        return results
    
    def identify_db_helper_files(self) -> List[Path]:
        """Identify dedicated database helper files to remove entirely"""
        db_file_patterns = [
            "**/db_*.py",
            "**/*_db.py",
            "**/database*.py",
            "**/sql*.py",
            "**/sqlite*.py",
            "**/spectral_db.py",  # Specific to ALAN
            "**/persistence/db/*.py",
        ]
        
        db_files = []
        for pattern in db_file_patterns:
            db_files.extend(self.backend_path.glob(pattern))
            
        # Filter out backups and cache
        db_files = [f for f in db_files if "backup" not in str(f) and "__pycache__" not in str(f)]
        
        return db_files
    
    def create_torus_shim(self, original_file: Path) -> str:
        """Create TorusRegistry shim to replace database class"""
        shim_template = '''#!/usr/bin/env python3
"""
{filename} - Migrated to use TorusRegistry (No-DB)
Original database functionality replaced with Parquet-based persistence
"""

from pathlib import Path
import logging
from python.core.torus_registry import TorusRegistry, get_torus_registry

logger = logging.getLogger(__name__)

# Legacy class name for compatibility
class {classname}:
    """Compatibility wrapper using TorusRegistry"""
    
    def __init__(self, *args, **kwargs):
        self.registry = get_torus_registry()
        logger.info(f"{classname} initialized with TorusRegistry backend")
        
    def save(self, *args, **kwargs):
        """Save data using registry"""
        self.registry.flush()
        
    def load(self, *args, **kwargs):
        """Load handled automatically by registry"""
        return self.registry.query_recent(100)
        
    def close(self):
        """No-op for compatibility"""
        pass
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

# Convenience functions
def get_{varname}():
    """Get instance for legacy code"""
    return {classname}()
'''
        
        # Extract class name from file
        filename = original_file.stem
        classname = self._extract_main_class(original_file)
        varname = filename.lower().replace('_db', '').replace('db_', '')
        
        return shim_template.format(
            filename=original_file.name,
            classname=classname,
            varname=varname
        )
    
    def _extract_main_class(self, py_file: Path) -> str:
        """Extract main class name from Python file"""
        try:
            content = py_file.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Find first class definition
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    return node.name
                    
        except Exception as e:
            logger.warning(f"Could not parse {py_file}: {e}")
            
        # Fallback - guess from filename
        name = py_file.stem.replace('_', ' ').title().replace(' ', '')
        if name.endswith('Db'):
            return name
        return name + 'DB'
    
    def remove_db_files(self, db_files: List[Path]):
        """Remove or replace database files"""
        for db_file in db_files:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would process: {db_file}")
                
                # Check if this is a critical file that needs a shim
                content = db_file.read_text(encoding='utf-8')
                if 'class' in content and any(x in content.lower() for x in ['spectral', 'persist', 'store']):
                    logger.info(f"  -> Would create TorusRegistry shim")
                else:
                    logger.info(f"  -> Would remove entirely")
            else:
                # Backup first
                if not self.backup_dir.exists():
                    self.backup_dir.mkdir()
                    
                backup_path = self.backup_dir / db_file.name
                shutil.copy2(db_file, backup_path)
                
                # Decide: shim or remove
                content = db_file.read_text(encoding='utf-8')
                if 'class' in content and any(x in content.lower() for x in ['spectral', 'persist', 'store']):
                    # Create shim
                    shim_content = self.create_torus_shim(db_file)
                    db_file.write_text(shim_content)
                    self.files_modified.append(db_file)
                    logger.info(f"Created TorusRegistry shim: {db_file}")
                else:
                    # Remove entirely
                    db_file.unlink()
                    self.files_removed.append(db_file)
                    logger.info(f"Removed database file: {db_file}")
    
    def patch_imports_in_file(self, py_file: Path):
        """Replace database imports with TorusRegistry imports"""
        try:
            content = py_file.read_text(encoding='utf-8')
            original = content
            
            # Replacement patterns
            replacements = [
                # SQLite imports
                (r'import\s+sqlite3\s*\n', 'from python.core.torus_registry import get_torus_registry\n'),
                (r'from\s+sqlite3\s+import.*\n', 'from python.core.torus_registry import get_torus_registry\n'),
                
                # Local DB imports
                (r'from\s+\.(\w*db\w*)\s+import\s+(\w+)', r'from python.core.torus_registry import TorusRegistry as \2'),
                (r'from\s+alan_backend\.(\w*db\w*)\s+import\s+(\w+)', r'from python.core.torus_registry import TorusRegistry as \2'),
                (r'import\s+alan_backend\.(\w*db\w*)', 'from python.core.torus_registry import get_torus_registry'),
                
                # Connection patterns
                (r'(\w+)\.connect\([^)]*\)', r'get_torus_registry()'),
                (r'(\w+)\.cursor\(\)', r'\1.registry'),
                (r'(\w+)\.execute\(', r'\1.registry.record_shape(vertices=[], metadata='),
                (r'(\w+)\.commit\(\)', r'\1.registry.flush()'),
                
                # SQL statements
                (r'["\'](CREATE TABLE|INSERT INTO|SELECT.*FROM|UPDATE.*SET|DELETE FROM)[^"\']*["\']', '""'),
            ]
            
            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE | re.MULTILINE)
            
            if content != original:
                if not self.dry_run:
                    # Backup
                    if not self.backup_dir.exists():
                        self.backup_dir.mkdir()
                    backup_path = self.backup_dir / py_file.name
                    py_file.rename(backup_path)
                    
                    # Write patched version
                    py_file.write_text(content)
                    
                self.files_modified.append(py_file)
                logger.info(f"Patched imports in: {py_file}")
                
        except Exception as e:
            logger.error(f"Error patching {py_file}: {e}")
    
    def run(self):
        """Run the complete cleanup process"""
        logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Starting alan_backend database cleanup...")
        
        # Step 1: Scan for database usage
        logger.info("\n📡 Scanning for database usage...")
        db_usage = self.scan_for_db_usage()
        
        if db_usage:
            logger.info(f"\nFound {len(db_usage)} files with database code:")
            for file, lines in db_usage[:5]:  # Show first 5
                logger.info(f"\n{file.relative_to(self.backend_path)}:")
                for line in lines[:3]:  # Show first 3 lines
                    logger.info(f"  {line}")
                if len(lines) > 3:
                    logger.info(f"  ... and {len(lines) - 3} more")
        
        # Step 2: Identify dedicated DB files
        logger.info("\n🗃️ Identifying database helper files...")
        db_files = self.identify_db_helper_files()
        
        if db_files:
            logger.info(f"\nFound {len(db_files)} database files:")
            for file in db_files:
                logger.info(f"  {file.relative_to(self.backend_path)}")
        
        # Step 3: Remove/replace DB files
        if db_files:
            logger.info("\n🔧 Processing database files...")
            self.remove_db_files(db_files)
        
        # Step 4: Patch remaining imports
        if self.files_with_db:
            logger.info("\n📝 Patching database imports...")
            for file in self.files_with_db:
                if file not in db_files:  # Don't patch files we removed
                    self.patch_imports_in_file(file)
        
        # Summary
        logger.info("\n📊 Cleanup Summary:")
        logger.info(f"  Files scanned: {len(list(self.backend_path.rglob('*.py')))}")
        logger.info(f"  Files with DB code: {len(self.files_with_db)}")
        logger.info(f"  Files removed: {len(self.files_removed)}")
        logger.info(f"  Files modified: {len(self.files_modified)}")
        
        if self.dry_run:
            logger.info("\n✅ Dry run complete. Run without --dry-run to apply changes.")
        else:
            logger.info(f"\n✅ Cleanup complete! Backups saved to: {self.backup_dir}")
            
        # Write summary report
        self._write_summary_report()
    
    def _write_summary_report(self):
        """Write detailed summary of changes"""
        report_path = self.backend_path / "db_cleanup_report.md"
        
        report = f"""# Alan Backend Database Cleanup Report

## Summary
- **Date**: {Path.ctime(Path(__file__))}
- **Mode**: {'DRY RUN' if self.dry_run else 'APPLIED'}
- **Files with DB code**: {len(self.files_with_db)}
- **Files removed**: {len(self.files_removed)}
- **Files modified**: {len(self.files_modified)}

## Files Removed
"""
        
        for file in self.files_removed:
            report += f"- `{file.relative_to(self.backend_path)}`\n"
            
        report += "\n## Files Modified\n"
        
        for file in self.files_modified:
            report += f"- `{file.relative_to(self.backend_path)}`\n"
            
        report += "\n## Next Steps\n"
        report += "1. Run tests: `pytest alan_backend/tests/`\n"
        report += "2. Check imports: `python -m py_compile alan_backend/**/*.py`\n"
        report += "3. Verify no DB: `grep -r 'sqlite3' alan_backend/`\n"
        
        if not self.dry_run:
            report_path.write_text(report)
            logger.info(f"\n📄 Report written to: {report_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove all database code from alan_backend")
    parser.add_argument("--path", type=Path, default=Path("alan_backend"),
                       help="Path to alan_backend directory")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    parser.add_argument("--force", action="store_true",
                       help="Skip confirmation prompts")
    
    args = parser.parse_args()
    
    if not args.path.exists():
        logger.error(f"Path not found: {args.path}")
        return 1
        
    if not args.dry_run and not args.force:
        response = input(f"Remove all database code from {args.path}? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Cancelled")
            return 0
    
    cleaner = AlanBackendDBCleaner(args.path, dry_run=args.dry_run)
    cleaner.run()
    
    return 0

if __name__ == "__main__":
    exit(main())
