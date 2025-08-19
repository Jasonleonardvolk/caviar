#!/usr/bin/env python3
"""
AST-based migration script: SpectralDB → TorusRegistry
Uses AST manipulation instead of regex for safer code transformation
"""

import ast
import astor  # pip install astor
import os
import sys
import shutil
from pathlib import Path
import logging
from typing import Set, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class ImportTransformer(ast.NodeTransformer):
    """Transform imports from database to TorusRegistry"""
    
    def __init__(self):
        self.imports_modified = False
        self.torus_imported = False
        self.observer_imported = False
        
    def visit_Import(self, node):
        """Transform import statements"""
        new_names = []
        
        for alias in node.names:
            if alias.name in ['sqlite3', 'psycopg2', 'sqlalchemy', 'pymongo', 'redis']:
                # Skip database imports
                self.imports_modified = True
                logger.debug(f"Removing import: {alias.name}")
            else:
                new_names.append(alias)
                
        if new_names:
            node.names = new_names
            return node
        else:
            # Replace with TorusRegistry import
            if not self.torus_imported:
                self.torus_imported = True
                return ast.ImportFrom(
                    module='python.core.torus_registry',
                    names=[
                        ast.alias(name='TorusRegistry', asname=None),
                        ast.alias(name='get_torus_registry', asname=None)
                    ],
                    level=0
                )
            return None
            
    def visit_ImportFrom(self, node):
        """Transform from imports"""
        if node.module and any(db in node.module for db in ['sqlite3', 'psycopg2', 'sqlalchemy']):
            self.imports_modified = True
            
            # Replace with TorusRegistry import
            if not self.torus_imported:
                self.torus_imported = True
                return ast.ImportFrom(
                    module='python.core.torus_registry',
                    names=[
                        ast.alias(name='get_torus_registry', asname=None)
                    ],
                    level=0
                )
            return None
            
        # Add observer imports where needed
        if node.module == 'datetime' and not self.observer_imported:
            # Good place to add observer import
            self.observer_imported = True
            return [
                node,  # Keep original datetime import
                ast.ImportFrom(
                    module='python.core.observer_synthesis',
                    names=[ast.alias(name='emit_token', asname=None)],
                    level=0
                )
            ]
            
        return self.generic_visit(node)

class SpectralDBTransformer(ast.NodeTransformer):
    """Transform SpectralDB usage to TorusRegistry"""
    
    def visit_ClassDef(self, node):
        """Transform SpectralDB class definition"""
        if node.name == 'SpectralDB':
            # Create wrapper class
            logger.info(f"Transforming class {node.name}")
            
            # Keep only __init__ and key methods
            new_body = []
            
            # Add docstring
            new_body.append(ast.Expr(value=ast.Str(s="""
    Compatibility wrapper around TorusRegistry
    Maintains API compatibility while using Parquet-based persistence
    """)))
            
            # Find and transform methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == '__init__':
                        # Transform init
                        new_body.append(self._create_init_method())
                    elif item.name in ['add', 'distance', '_save', '_load']:
                        # Keep these methods but transform their bodies
                        new_body.append(self._transform_method(item))
                        
            node.body = new_body
            
        return self.generic_visit(node)
    
    def _create_init_method(self):
        """Create new __init__ method"""
        return ast.parse("""
def __init__(self, max_entries: int = 10000, max_size_mb: int = 200):
    self.registry = get_torus_registry()
    self.max_entries = max_entries
    self.max_size_mb = max_size_mb
    self._recent_cache = deque(maxlen=100)
    self._cache_hits = 0
    self._cache_misses = 0
    logger.info(f"SpectralDB initialized with TorusRegistry backend at {self.registry.path}")
""").body[0]
    
    def _transform_method(self, method):
        """Transform method to use registry"""
        if method.name == 'add':
            return ast.parse("""
def add(self, signature):
    try:
        shape_id = self.registry.record_shape(
            vertices=signature.eigenvalues,
            betti_numbers=signature.betti_numbers if signature.betti_numbers else [],
            coherence_band=signature.coherence_state,
            metadata={
                'hash_id': signature.hash_id,
                'gaps': signature.gaps,
                'timestamp': signature.timestamp.isoformat()
            }
        )
        self._recent_cache.append(signature)
        if len(self._recent_cache) % 10 == 0:
            self.registry.flush()
        logger.debug(f"Added spectral signature {signature.hash_id} as {shape_id}")
    except Exception as e:
        logger.error(f"Failed to add spectral signature: {e}")
        raise
""").body[0]
        
        elif method.name == '_save':
            return ast.parse("""
def _save(self):
    try:
        self.registry.flush()
        logger.debug("SpectralDB flushed to registry")
    except Exception as e:
        logger.error(f"Failed to flush SpectralDB: {e}")
""").body[0]
        
        # Keep original for other methods
        return method

class ObserverTokenInjector(ast.NodeTransformer):
    """Inject observer token emissions at key points"""
    
    def __init__(self, source_name):
        self.source_name = source_name
        self.tokens_added = 0
        
    def visit_FunctionDef(self, node):
        """Add token emission to specific methods"""
        if node.name == 'classify' and self.source_name == 'origin_sentry':
            # Add token emission before return
            return self._add_token_to_classify(node)
            
        elif node.name == 'check_eigenvalues' and self.source_name == 'eigensentry':
            # Add token emission before return
            return self._add_token_to_check(node)
            
        elif node.name == '_complete_burst' and self.source_name == 'chaos':
            # Add token emission in burst completion
            return self._add_token_to_burst(node)
            
        return self.generic_visit(node)
    
    def _add_token_to_classify(self, node):
        """Add token emission to classify method"""
        # Find the return statement
        for i, stmt in enumerate(node.body):
            if isinstance(stmt, ast.Return):
                # Insert token emission before return
                token_code = ast.parse("""
try:
    token = emit_token({
        "type": "origin_spectral",
        "source": "origin_sentry",
        "lambda_max": lambda_max,
        "coherence": coherence,
        "novelty_score": float(novelty_score),
        "dim_expansion": dim_expansion,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    self.metrics['tokens_emitted'] = self.metrics.get('tokens_emitted', 0) + 1
    logger.debug(f"Emitted observer token: {token[:8]}...")
except Exception as e:
    logger.warning(f"Failed to emit observer token: {e}")
""").body[0]
                
                node.body.insert(i, token_code)
                self.tokens_added += 1
                break
                
        return node
    
    def _add_token_to_check(self, node):
        """Add token emission to check_eigenvalues"""
        # Similar pattern for eigensentry
        for i, stmt in enumerate(node.body):
            if isinstance(stmt, ast.Return):
                token_code = ast.parse("""
if self._token_emission_enabled:
    try:
        token = emit_token({
            "type": "curvature",
            "source": "eigensentry",
            "lambda_max": max_eigenvalue,
            "mean_curvature": float(curvature_metrics.mean_curvature),
            "threshold": float(self.current_threshold),
            "damping_active": self.damping_active,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        self.metrics['tokens_emitted'] += 1
        logger.debug(f"Emitted observer token: {token[:8]}...")
    except Exception as e:
        logger.warning(f"Failed to emit observer token: {e}")
""").body[0]
                
                node.body.insert(i, token_code)
                self.tokens_added += 1
                break
                
        return node
    
    def _add_token_to_burst(self, node):
        """Add token emission to chaos burst completion"""
        # Find where to insert (after metrics creation)
        for i, stmt in enumerate(node.body):
            if isinstance(stmt, ast.Assign) and any(
                target.id == 'metrics' for target in stmt.targets 
                if isinstance(target, ast.Name)
            ):
                token_code = ast.parse("""
try:
    token = emit_token({
        "type": "chaos_burst_complete",
        "source": "chaos_controller",
        "burst_id": self.current_burst.burst_id,
        "peak_energy": peak_energy,
        "discoveries": len(discoveries),
        "duration": self.burst_step,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    self.metrics['tokens_emitted'] += 1
    logger.debug(f"Emitted burst completion token: {token[:8]}...")
except Exception as e:
    logger.warning(f"Failed to emit burst token: {e}")
""").body[0]
                
                node.body.insert(i + 1, token_code)
                self.tokens_added += 1
                break
                
        return node

class ASTMigrator:
    """Main AST-based migration coordinator"""
    
    def __init__(self, backend_path: Path, dry_run: bool = True):
        self.backend_path = backend_path
        self.dry_run = dry_run
        self.backup_dir = backend_path / "backup_ast_migration"
        self.files_modified = []
        
    def migrate_file(self, file_path: Path) -> bool:
        """Migrate a single Python file using AST transformation"""
        try:
            logger.info(f"\n📝 Processing {file_path.name}...")
            
            # Read and parse
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Apply transformations
            transformers = [
                ImportTransformer(),
                SpectralDBTransformer(),
                ObserverTokenInjector(file_path.stem)
            ]
            
            modified = False
            for transformer in transformers:
                tree = transformer.visit(tree)
                if hasattr(transformer, 'imports_modified') and transformer.imports_modified:
                    modified = True
                if hasattr(transformer, 'tokens_added') and transformer.tokens_added > 0:
                    modified = True
                    logger.info(f"  Added {transformer.tokens_added} observer tokens")
                    
            if modified:
                # Generate new code
                new_content = astor.to_source(tree)
                
                if self.dry_run:
                    logger.info(f"  [DRY RUN] Would modify {file_path.name}")
                    # Show a snippet of changes
                    logger.debug("Sample of changes:")
                    logger.debug(new_content[:500])
                else:
                    # Backup original
                    if not self.backup_dir.exists():
                        self.backup_dir.mkdir()
                    backup_path = self.backup_dir / file_path.name
                    shutil.copy2(file_path, backup_path)
                    
                    # Write transformed code
                    file_path.write_text(new_content)
                    self.files_modified.append(file_path)
                    logger.info(f"  ✅ Modified {file_path.name}")
                    
                return True
            else:
                logger.info(f"  No changes needed in {file_path.name}")
                return False
                
        except Exception as e:
            logger.error(f"  ❌ Error processing {file_path.name}: {e}")
            return False
    
    def run(self, target_files: List[str]):
        """Run AST migration on target files"""
        logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Starting AST-based migration...")
        
        success_count = 0
        for filename in target_files:
            file_path = self.backend_path / filename
            if file_path.exists():
                if self.migrate_file(file_path):
                    success_count += 1
            else:
                logger.warning(f"File not found: {file_path}")
                
        logger.info(f"\n📊 Migration Summary:")
        logger.info(f"  Files processed: {len(target_files)}")
        logger.info(f"  Files modified: {success_count}")
        
        if self.dry_run:
            logger.info("\n✅ Dry run complete. Run without --dry-run to apply changes.")
        else:
            logger.info(f"\n✅ Migration complete! Backups in: {self.backup_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AST-based No-DB migration")
    parser.add_argument("--path", type=Path, default=Path("alan_backend"),
                       help="Path to alan_backend directory")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done")
    parser.add_argument("--files", nargs="+", default=[
        "origin_sentry.py",
        "eigensentry_guard.py", 
        "braid_aggregator.py",
        "chaos_channel_controller.py"
    ], help="Files to migrate")
    
    args = parser.parse_args()
    
    # Check for astor
    try:
        import astor
    except ImportError:
        logger.error("Please install astor: pip install astor")
        return 1
        
    migrator = ASTMigrator(args.path, dry_run=args.dry_run)
    migrator.run(args.files)
    
    return 0

if __name__ == "__main__":
    exit(main())
