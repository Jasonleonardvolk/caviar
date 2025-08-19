#!/usr/bin/env python3
"""
Migration script: SpectralDB → TorusRegistry
Removes all database dependencies and migrates to Parquet-based storage
"""

import os
import sys
import shutil
from pathlib import Path
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class NoDBMigration:
    """Migrate ALAN backend to No-DB persistence"""
    
    def __init__(self, alan_backend_path: Path):
        self.backend_path = alan_backend_path
        self.backup_dir = self.backend_path / "backup_pre_nodb"
        self.files_modified = []
        self.files_deleted = []
        
    def run(self, dry_run: bool = False):
        """Execute the migration"""
        logger.info("🚀 Starting No-DB Migration")
        logger.info(f"Backend path: {self.backend_path}")
        logger.info(f"Dry run: {dry_run}\n")
        
        if not dry_run:
            self.create_backup()
            
        # Step 1: Update origin_sentry.py
        self.migrate_origin_sentry(dry_run)
        
        # Step 2: Update braid_aggregator.py
        self.migrate_braid_aggregator(dry_run)
        
        # Step 3: Update chaos_channel_controller.py
        self.migrate_chaos_controller(dry_run)
        
        # Step 4: Add observer synthesis integration
        self.add_observer_integration(dry_run)
        
        # Step 5: Update capsule.yml
        self.update_capsule_yml(dry_run)
        
        # Step 6: Remove SQLite files
        self.remove_sqlite_files(dry_run)
        
        # Step 7: Add CI guards
        self.add_ci_guards(dry_run)
        
        # Summary
        self.print_summary()
        
    def create_backup(self):
        """Create backup of files before modification"""
        if self.backup_dir.exists():
            logger.warning(f"Backup directory already exists: {self.backup_dir}")
            response = input("Overwrite? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
            shutil.rmtree(self.backup_dir)
            
        self.backup_dir.mkdir(exist_ok=True)
        logger.info(f"Created backup directory: {self.backup_dir}")
        
    def migrate_origin_sentry(self, dry_run: bool):
        """Update origin_sentry.py to use TorusRegistry"""
        logger.info("\n📝 Migrating origin_sentry.py...")
        
        origin_file = self.backend_path / "origin_sentry.py"
        if not origin_file.exists():
            logger.error(f"File not found: {origin_file}")
            return
            
        content = origin_file.read_text()
        original_content = content
        
        # Replace imports
        content = re.sub(
            r'from dataclasses import dataclass, field\n',
            'from dataclasses import dataclass, field\nfrom python.core.torus_registry import TorusRegistry, get_torus_registry\nfrom python.core.observer_synthesis import emit_token\n',
            content
        )
        
        # Replace SpectralDB class with wrapper
        spectral_db_class = re.search(
            r'class SpectralDB:.*?(?=\nclass|\Z)',
            content,
            re.DOTALL
        )
        
        if spectral_db_class:
            replacement = '''class SpectralDB:
    """Compatibility wrapper around TorusRegistry"""
    
    def __init__(self, max_entries: int = 10000, max_size_mb: int = 200):
        self.registry = get_torus_registry()
        self.max_entries = max_entries
        
    def add(self, signature: SpectralSignature):
        self.registry.record_shape(
            vertices=signature.eigenvalues,
            betti_numbers=signature.betti_numbers,
            coherence_band=signature.coherence_state,
            metadata={
                'hash_id': signature.hash_id,
                'gaps': signature.gaps
            }
        )
        
    def distance(self, eigenvalues: np.ndarray, top_k: int = 10) -> float:
        # Simple distance calculation
        recent = self.registry.query_recent(top_k)
        if recent.empty:
            return float('inf')
            
        min_dist = float('inf')
        for _, row in recent.iterrows():
            # Reconstruct approximate eigenvalues from Betti
            stored_betti = [row['betti0'], row['betti1']]
            dist = np.mean(np.abs(stored_betti[0] - len(eigenvalues)))
            min_dist = min(min_dist, dist)
            
        return min_dist
        
    def _save(self):
        self.registry.flush()
        
    def _load(self):
        pass  # Registry handles its own loading'''
            
            content = content[:spectral_db_class.start()] + replacement + content[spectral_db_class.end():]
            
        # Add observer token emission in classify method
        classify_method = re.search(
            r'def classify\(self.*?\n(\s+)return {',
            content,
            re.DOTALL
        )
        
        if classify_method:
            indent = classify_method.group(1)
            token_code = f'''
{indent}# Emit observer token
{indent}token = emit_token({{
{indent}    "type": "origin_spectral",
{indent}    "source": "origin_sentry",
{indent}    "lambda_max": float(lambda_max),
{indent}    "coherence": coherence,
{indent}    "novelty_score": float(novelty_score),
{indent}    "dim_expansion": dim_expansion
{indent}}})
{indent}logger.debug(f"Emitted token: {{token[:8]}}...")
{indent}
{indent}return {{'''
            
            content = content[:classify_method.start()] + \
                     content[classify_method.start():classify_method.end()].replace('return {', token_code) + \
                     content[classify_method.end():]
                     
        if content != original_content:
            if not dry_run:
                shutil.copy2(origin_file, self.backup_dir / origin_file.name)
                origin_file.write_text(content)
            self.files_modified.append(origin_file)
            logger.info("✅ Updated origin_sentry.py")
        else:
            logger.warning("⚠️  No changes needed in origin_sentry.py")
            
    def migrate_braid_aggregator(self, dry_run: bool):
        """Update braid_aggregator.py to use TorusCells"""
        logger.info("\n📝 Migrating braid_aggregator.py...")
        
        braid_file = self.backend_path / "braid_aggregator.py"
        if not braid_file.exists():
            logger.error(f"File not found: {braid_file}")
            return
            
        content = braid_file.read_text()
        original_content = content
        
        # Add imports
        import_section = re.search(r'(import.*?\n\n)', content, re.DOTALL)
        if import_section:
            new_imports = import_section.group(1).rstrip() + '\nfrom python.core.torus_cells import get_torus_cells, betti_update\n\n'
            content = content[:import_section.start()] + new_imports + content[import_section.end():]
            
        # Replace inline Betti computation with TorusCells call
        content = re.sub(
            r'# Get Origin classification\n(\s+)test_eigenvalues = self\._reconstruct_eigenvalues\(summary\)\n(\s+)classification = self\.origin_sentry\.classify\(',
            r'# Get Origin classification\n\1test_eigenvalues = self._reconstruct_eigenvalues(summary)\n\1# Use TorusCells for Betti computation\n\1torus_cells = get_torus_cells()\n\1if summary.get("event_count", 0) > 0:\n\1    betti_numbers = summary.get("betti_max", [])\n\1else:\n\1    betti_numbers = []\n\1classification = self.origin_sentry.classify(',
            content
        )
        
        if content != original_content:
            if not dry_run:
                shutil.copy2(braid_file, self.backup_dir / braid_file.name)
                braid_file.write_text(content)
            self.files_modified.append(braid_file)
            logger.info("✅ Updated braid_aggregator.py")
        else:
            logger.warning("⚠️  No changes needed in braid_aggregator.py")
            
    def migrate_chaos_controller(self, dry_run: bool):
        """Update chaos controller to use registry path"""
        logger.info("\n📝 Migrating chaos_channel_controller.py...")
        
        chaos_file = self.backend_path / "chaos_channel_controller.py"
        if not chaos_file.exists():
            logger.error(f"File not found: {chaos_file}")
            return
            
        content = chaos_file.read_text()
        original_content = content
        
        # Add imports
        content = re.sub(
            r'from datetime import datetime, timezone, timedelta\n',
            'from datetime import datetime, timezone, timedelta\nfrom python.core.torus_registry import get_torus_registry\nfrom python.core.observer_synthesis import emit_token\n',
            content
        )
        
        # Add token emission in _complete_burst
        complete_burst = re.search(
            r'def _complete_burst\(self\):.*?\n(\s+)# Notify callbacks',
            content,
            re.DOTALL
        )
        
        if complete_burst:
            indent = complete_burst.group(1)
            token_code = f'''
{indent}# Emit observer token for burst completion
{indent}token = emit_token({{
{indent}    "type": "chaos_burst",
{indent}    "source": "chaos_controller",
{indent}    "burst_id": self.current_burst.burst_id,
{indent}    "peak_energy": float(peak_energy),
{indent}    "discoveries": len(discoveries),
{indent}    "duration": self.burst_step
{indent}}})
{indent}logger.debug(f"Burst token: {{token[:8]}}...")
{indent}
{indent}# Notify callbacks'''
            
            content = content[:complete_burst.start()] + \
                     content[complete_burst.start():complete_burst.end()].replace('# Notify callbacks', token_code) + \
                     content[complete_burst.end():]
                     
        if content != original_content:
            if not dry_run:
                shutil.copy2(chaos_file, self.backup_dir / chaos_file.name)
                chaos_file.write_text(content)
            self.files_modified.append(chaos_file)
            logger.info("✅ Updated chaos_channel_controller.py")
            
    def add_observer_integration(self, dry_run: bool):
        """Add observer integration to eigensentry_guard.py"""
        logger.info("\n📝 Adding observer integration to eigensentry_guard.py...")
        
        eigen_file = self.backend_path / "eigensentry_guard.py"
        if not eigen_file.exists():
            logger.error(f"File not found: {eigen_file}")
            return
            
        content = eigen_file.read_text()
        original_content = content
        
        # Add import
        content = re.sub(
            r'from alan_backend\.lyap_exporter import LyapunovExporter\n',
            'from alan_backend.lyap_exporter import LyapunovExporter\nfrom python.core.observer_synthesis import emit_token, integrate_with_eigensentry\n',
            content
        )
        
        # Add token emission in check_eigenvalues
        check_method = re.search(
            r'# Broadcast to WebSocket clients\n(\s+)asyncio\.create_task',
            content
        )
        
        if check_method:
            indent = check_method.group(1)
            token_code = f'''
{indent}# Emit observer token
{indent}token = emit_token({{
{indent}    "type": "curvature",
{indent}    "source": "eigensentry",
{indent}    "lambda_max": float(max_eigenvalue),
{indent}    "mean_curvature": float(curvature_metrics.mean_curvature),
{indent}    "threshold": float(self.current_threshold),
{indent}    "damping_active": self.damping_active
{indent}}})
{indent}
{indent}# Broadcast to WebSocket clients'''
            
            content = content[:check_method.start()] + token_code + '\n' + content[check_method.start():]
            
        if content != original_content:
            if not dry_run:
                shutil.copy2(eigen_file, self.backup_dir / eigen_file.name)
                eigen_file.write_text(content)
            self.files_modified.append(eigen_file)
            logger.info("✅ Updated eigensentry_guard.py")
            
    def update_capsule_yml(self, dry_run: bool):
        """Update or create capsule.yml with state_path"""
        logger.info("\n📝 Updating capsule.yml...")
        
        capsule_file = self.backend_path.parent / "capsule.yml"
        
        capsule_content = '''# TORI ALAN Backend Configuration
name: alan_backend
version: 1.0.0

resources:
  state_path: /var/lib/tori
  
  # Override with environment variable:
  # export TORI_STATE_ROOT=/mnt/ramdisk
  
services:
  - name: origin_sentry
    type: spectral_monitor
    
  - name: eigensentry
    type: curvature_guard
    
  - name: braid_aggregator
    type: temporal_processor
    
  - name: chaos_controller
    type: creativity_engine
    
  - name: torus_cells
    type: topology_memory

# No database configuration needed!
# All persistence via Parquet shards
'''
        
        if not dry_run:
            if capsule_file.exists():
                shutil.copy2(capsule_file, self.backup_dir / capsule_file.name)
            capsule_file.write_text(capsule_content)
            
        self.files_modified.append(capsule_file)
        logger.info("✅ Created/Updated capsule.yml")
        
    def remove_sqlite_files(self, dry_run: bool):
        """Remove any SQLite files and imports"""
        logger.info("\n🗑️  Removing SQLite files...")
        
        # Find all .sqlite files
        sqlite_files = list(self.backend_path.glob("**/*.sqlite"))
        sqlite_files.extend(list(self.backend_path.glob("**/*.db")))
        
        for sqlite_file in sqlite_files:
            logger.info(f"  Deleting: {sqlite_file}")
            if not dry_run:
                sqlite_file.unlink()
            self.files_deleted.append(sqlite_file)
            
        # Remove any spectral_db.py if it exists as separate file
        spectral_db_file = self.backend_path / "spectral_db.py"
        if spectral_db_file.exists():
            logger.info(f"  Deleting: {spectral_db_file}")
            if not dry_run:
                shutil.copy2(spectral_db_file, self.backup_dir / spectral_db_file.name)
                spectral_db_file.unlink()
            self.files_deleted.append(spectral_db_file)
            
    def add_ci_guards(self, dry_run: bool):
        """Add CI configuration to prevent database imports"""
        logger.info("\n🛡️  Adding CI guards...")
        
        tox_file = self.backend_path / "tox.ini"
        tox_content = '''[tox]
envlist = py38,py39,py310,py311,flake8
skipsdist = True

[testenv]
deps = 
    pytest
    pytest-asyncio
    numpy
    pandas
    pyarrow
commands = pytest {posargs}

[testenv:flake8]
deps = 
    flake8
    flake8-forbid-import
commands = flake8 .

[flake8]
select = E,F,W
exclude = .git,__pycache__,build,backup_pre_nodb
max-line-length = 120
per-file-ignores =
    */tests/*: E402
    
# Forbid database imports
forbidden-imports = sqlite3,psycopg2,sqlalchemy,pymongo,redis

[forbid-import]
sqlite3 = Database imports are forbidden. Use TorusRegistry for persistence.
psycopg2 = Database imports are forbidden. Use TorusRegistry for persistence.
sqlalchemy = Database imports are forbidden. Use TorusRegistry for persistence.
'''
        
        if not dry_run:
            tox_file.write_text(tox_content)
            
        self.files_modified.append(tox_file)
        logger.info("✅ Created tox.ini with database import guards")
        
        # Add pre-commit hook
        precommit_file = self.backend_path / ".pre-commit-config.yaml"
        precommit_content = '''repos:
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        additional_dependencies: [flake8-forbid-import]
        args: ['--config=tox.ini']
        
  - repo: local
    hooks:
      - id: no-database-files
        name: Check for database files
        entry: bash -c 'find . -name "*.sqlite" -o -name "*.db" | grep -v backup_pre_nodb | head -1 && exit 1 || exit 0'
        language: system
        pass_filenames: false
'''
        
        if not dry_run:
            precommit_file.write_text(precommit_content)
            
        self.files_modified.append(precommit_file)
        logger.info("✅ Created .pre-commit-config.yaml")
        
    def print_summary(self):
        """Print migration summary"""
        logger.info("\n" + "="*60)
        logger.info("📊 Migration Summary")
        logger.info("="*60)
        
        logger.info(f"\n✏️  Files modified: {len(self.files_modified)}")
        for f in self.files_modified:
            logger.info(f"  - {f.relative_to(self.backend_path.parent)}")
            
        logger.info(f"\n🗑️  Files deleted: {len(self.files_deleted)}")
        for f in self.files_deleted:
            logger.info(f"  - {f.relative_to(self.backend_path.parent)}")
            
        logger.info("\n✅ Migration complete!")
        logger.info("\n📝 Next steps:")
        logger.info("  1. Set environment variable: export TORI_STATE_ROOT=/var/lib/tori")
        logger.info("  2. Run tests: pytest -k 'betti or torus'")
        logger.info("  3. Install pre-commit: pre-commit install")
        logger.info("  4. Run pre-commit: pre-commit run --all-files")
        logger.info("  5. Start TORI and check logs for: 'TorusRegistry loaded'")
        
        if self.backup_dir.exists():
            logger.info(f"\n💾 Backup saved to: {self.backup_dir}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate ALAN backend to No-DB persistence")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--backend-path", type=Path, default=Path(__file__).parent, 
                       help="Path to alan_backend directory")
    
    args = parser.parse_args()
    
    migration = NoDBMigration(args.backend_path)
    migration.run(dry_run=args.dry_run)

if __name__ == "__main__":
    main()
