#!/usr/bin/env python3
"""
caps_migrate.py - TORI Legacy to Capsule Migration Tool
Converts existing /opt/tori installations to capsule format
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LegacyMigrator:
    """Migrates legacy TORI deployments to capsule format"""
    
    def __init__(self, legacy_root: Path = Path('/opt/tori')):
        self.legacy_root = legacy_root
        self.work_dir = Path(tempfile.mkdtemp(prefix="caps_migrate_"))
        self.discovered_services = []
        self.discovered_configs = {}
        self.discovered_venvs = []
        
    def discover_legacy_installation(self) -> Dict:
        """Scan legacy installation and discover components"""
        logger.info(f"Discovering legacy installation at {self.legacy_root}")
        
        discovery = {
            'services': [],
            'python_apps': [],
            'rust_binaries': [],
            'configs': [],
            'venvs': [],
            'data_dirs': [],
            'systemd_units': []
        }
        
        # Find Python applications
        for py_file in self.legacy_root.rglob('*.py'):
            # Look for main entry points
            if any(marker in py_file.name for marker in ['main.py', 'start_', 'server.py', '__main__.py']):
                rel_path = py_file.relative_to(self.legacy_root)
                discovery['python_apps'].append({
                    'path': str(rel_path),
                    'name': py_file.stem,
                    'dir': str(rel_path.parent)
                })
                
        # Find Rust binaries
        for bin_path in [self.legacy_root / 'bin', self.legacy_root / 'target/release']:
            if bin_path.exists():
                for binary in bin_path.iterdir():
                    if binary.is_file() and os.access(binary, os.X_OK):
                        discovery['rust_binaries'].append({
                            'path': str(binary.relative_to(self.legacy_root)),
                            'name': binary.name
                        })
                        
        # Find virtual environments
        for venv_marker in ['pyvenv.cfg', 'pip-selfcheck.json']:
            for marker_file in self.legacy_root.rglob(venv_marker):
                venv_root = marker_file.parent
                if marker_file.name == 'pip-selfcheck.json':
                    venv_root = venv_root.parent
                    
                discovery['venvs'].append({
                    'path': str(venv_root.relative_to(self.legacy_root)),
                    'python_version': self._get_venv_python_version(venv_root)
                })
                
        # Find configuration files
        for config_pattern in ['*.yml', '*.yaml', '*.toml', '*.json', '*.conf', '*.ini']:
            for config_file in self.legacy_root.rglob(config_pattern):
                # Skip venv and git directories
                if any(skip in str(config_file) for skip in ['/venv/', '/.git/', '/node_modules/']):
                    continue
                    
                discovery['configs'].append({
                    'path': str(config_file.relative_to(self.legacy_root)),
                    'name': config_file.name
                })
                
        # Find data directories
        for data_marker in ['data', 'storage', 'state', 'db', 'logs']:
            data_path = self.legacy_root / data_marker
            if data_path.exists() and data_path.is_dir():
                discovery['data_dirs'].append(str(data_path.relative_to(self.legacy_root)))
                
        # Find systemd units
        systemd_paths = [
            Path('/etc/systemd/system'),
            Path('/usr/lib/systemd/system'),
            Path.home() / '.config/systemd/user'
        ]
        
        for systemd_path in systemd_paths:
            if systemd_path.exists():
                for unit_file in systemd_path.glob('tori*.service'):
                    discovery['systemd_units'].append({
                        'path': str(unit_file),
                        'name': unit_file.name
                    })
                    
        # Try to determine services from systemd units
        for unit in discovery['systemd_units']:
            service_name = unit['name'].replace('.service', '').replace('tori-', '')
            discovery['services'].append(service_name)
            
        return discovery
        
    def _get_venv_python_version(self, venv_path: Path) -> str:
        """Extract Python version from virtual environment"""
        try:
            python_bin = venv_path / 'bin' / 'python'
            if python_bin.exists():
                result = subprocess.run(
                    [str(python_bin), '--version'],
                    capture_output=True,
                    text=True
                )
                return result.stdout.strip().split()[-1]
        except:
            pass
        return "unknown"
        
    def generate_manifest(self, discovery: Dict, name: str, version: str) -> Dict:
        """Generate capsule.yml manifest from discovery"""
        logger.info("Generating capsule manifest from discovery")
        
        # Determine main entrypoint
        entrypoint = None
        if discovery['python_apps']:
            # Prefer main.py or start_* patterns
            for app in discovery['python_apps']:
                if 'main.py' in app['path'] or 'start_' in app['name']:
                    entrypoint = app['path']
                    break
            if not entrypoint:
                entrypoint = discovery['python_apps'][0]['path']
        elif discovery['rust_binaries']:
            entrypoint = discovery['rust_binaries'][0]['path']
            
        if not entrypoint:
            raise ValueError("No entrypoint found in legacy installation")
            
        # Build manifest
        manifest = {
            'name': name,
            'version': version,
            'entrypoint': entrypoint,
            'services': discovery['services'] or [name],
            'env': {
                'LEGACY_MIGRATION': 'true',
                'MIGRATION_DATE': datetime.utcnow().isoformat()
            }
        }
        
        # Add Python dependencies if found
        if discovery['venvs']:
            venv = discovery['venvs'][0]
            py_version = venv['python_version']
            if py_version != 'unknown':
                manifest['dependencies'] = {
                    'python': '.'.join(py_version.split('.')[:2])  # Major.minor only
                }
                
            # Look for requirements files
            for config in discovery['configs']:
                if 'requirements' in config['name']:
                    manifest['dependencies'] = manifest.get('dependencies', {})
                    manifest['dependencies']['pip_requirements'] = config['path']
                    break
                    
        # Add data volumes if found
        if discovery['data_dirs']:
            manifest['volumes'] = []
            for data_dir in discovery['data_dirs']:
                manifest['volumes'].append({
                    'source': f"/opt/tori/state/{data_dir}",
                    'target': f"/{data_dir}",
                    'mode': 'rw',
                    'description': f"Legacy {data_dir} directory"
                })
                
        # Add files to include
        manifest['files'] = []
        
        # Include all Python apps
        for app in discovery['python_apps']:
            manifest['files'].append({
                'src': app['dir'],
                'dst': app['dir']
            })
            
        # Include configs
        config_dirs = set()
        for config in discovery['configs']:
            config_dir = str(Path(config['path']).parent)
            if config_dir != '.':
                config_dirs.add(config_dir)
                
        for config_dir in config_dirs:
            manifest['files'].append({
                'src': config_dir,
                'dst': f"config/{Path(config_dir).name}"
            })
            
        # Add resource limits (conservative defaults)
        manifest['resources'] = {
            'cpu_quota': '50%',
            'memory_max': '4G',
            'memory_high': '3G'
        }
        
        # Add migration metadata
        manifest['metadata'] = {
            'migrated_from': str(self.legacy_root),
            'migration_tool': 'caps_migrate.py',
            'migration_date': datetime.utcnow().isoformat(),
            'discovery_summary': {
                'python_apps': len(discovery['python_apps']),
                'rust_binaries': len(discovery['rust_binaries']),
                'configs': len(discovery['configs']),
                'venvs': len(discovery['venvs'])
            }
        }
        
        return manifest
        
    def migrate_to_capsule(self, name: str, version: str, 
                          output_path: Optional[Path] = None,
                          dry_run: bool = False) -> Path:
        """Perform the migration to capsule format"""
        
        # Discover installation
        discovery = self.discover_legacy_installation()
        
        # Print discovery summary
        logger.info("\n📋 Discovery Summary:")
        logger.info(f"   Python apps: {len(discovery['python_apps'])}")
        logger.info(f"   Rust binaries: {len(discovery['rust_binaries'])}")
        logger.info(f"   Virtual envs: {len(discovery['venvs'])}")
        logger.info(f"   Config files: {len(discovery['configs'])}")
        logger.info(f"   Data dirs: {len(discovery['data_dirs'])}")
        logger.info(f"   Systemd units: {len(discovery['systemd_units'])}")
        
        # Generate manifest
        manifest = self.generate_manifest(discovery, name, version)
        
        # Create capsule structure
        capsule_dir = self.work_dir / "capsule"
        capsule_dir.mkdir(parents=True, exist_ok=True)
        
        # Write manifest
        manifest_path = capsule_dir / "capsule.yml"
        import yaml
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
            
        logger.info(f"\n✅ Generated manifest: {manifest_path}")
        
        if dry_run:
            logger.info("\n🔍 Dry run - would create capsule with:")
            print(yaml.dump(manifest, default_flow_style=False))
            return manifest_path
            
        # Copy files according to manifest
        for file_spec in manifest.get('files', []):
            src = self.legacy_root / file_spec['src']
            dst = capsule_dir / file_spec['dst']
            
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    logger.info(f"   Copied directory: {file_spec['src']}")
                else:
                    shutil.copy2(src, dst)
                    logger.info(f"   Copied file: {file_spec['src']}")
                    
        # Copy virtual environments
        if discovery['venvs']:
            venv_info = discovery['venvs'][0]
            src_venv = self.legacy_root / venv_info['path']
            dst_venv = capsule_dir / 'venv'
            
            if src_venv.exists():
                logger.info(f"   Copying virtual environment...")
                shutil.copytree(src_venv, dst_venv, dirs_exist_ok=True)
                self._fix_venv_paths(dst_venv)
                
        # Copy binaries
        if discovery['rust_binaries']:
            bin_dir = capsule_dir / 'bin'
            bin_dir.mkdir(exist_ok=True)
            
            for binary_info in discovery['rust_binaries']:
                src = self.legacy_root / binary_info['path']
                dst = bin_dir / binary_info['name']
                if src.exists():
                    shutil.copy2(src, dst)
                    dst.chmod(0o755)
                    logger.info(f"   Copied binary: {binary_info['name']}")
                    
        # Generate migration report
        report = {
            'migration_date': datetime.utcnow().isoformat(),
            'source': str(self.legacy_root),
            'manifest': manifest,
            'discovery': discovery
        }
        
        report_path = capsule_dir / '.migration_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Create capsule tarball
        output_path = output_path or Path(f"capsule-{name}-migrated-{version}.tar.gz")
        
        # Use capsbuild if available
        capsbuild_path = Path(__file__).parent / 'capsbuild.py'
        if capsbuild_path.exists():
            logger.info("\n🔨 Using capsbuild to create capsule...")
            subprocess.run([
                sys.executable,
                str(capsbuild_path),
                '--from-dir', str(capsule_dir),
                '--manifest', str(manifest_path),
                '--output', str(output_path)
            ], check=True)
        else:
            # Fallback to simple tarball
            logger.info("\n📦 Creating capsule tarball...")
            with tarfile.open(output_path, 'w:gz') as tar:
                for item in capsule_dir.iterdir():
                    tar.add(item, arcname=item.name)
                    
        logger.info(f"\n✅ Migration complete: {output_path}")
        logger.info(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return output_path
        
    def _fix_venv_paths(self, venv_path: Path):
        """Fix hardcoded paths in venv for relocation"""
        # Same as in capsbuild.py
        activate_path = venv_path / "bin" / "activate"
        if activate_path.exists():
            content = activate_path.read_text()
            content = content.replace(
                f'VIRTUAL_ENV="{venv_path}"',
                'VIRTUAL_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"'
            )
            activate_path.write_text(content)
            
    def cleanup(self):
        """Clean up temporary directory"""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate legacy TORI installations to capsule format"
    )
    
    parser.add_argument(
        '--legacy-root',
        type=Path,
        default=Path('/opt/tori'),
        help="Root of legacy installation (default: /opt/tori)"
    )
    
    parser.add_argument(
        '--name',
        required=True,
        help="Name for the migrated capsule"
    )
    
    parser.add_argument(
        '--version',
        required=True,
        help="Version for the migrated capsule"
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help="Output path for capsule"
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be migrated without creating capsule"
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force migration even if legacy root doesn't exist"
    )
    
    args = parser.parse_args()
    
    # Check legacy root
    if not args.legacy_root.exists() and not args.force:
        logger.error(f"Legacy root not found: {args.legacy_root}")
        logger.error("Use --force to proceed anyway")
        return 1
        
    # Windows compatibility - check common Windows paths
    if sys.platform == 'win32' and not args.legacy_root.exists():
        windows_paths = [
            Path(r'C:\tori'),
            Path(r'C:\opt\tori'),
            Path.home() / 'tori'
        ]
        for win_path in windows_paths:
            if win_path.exists():
                logger.info(f"Found legacy installation at: {win_path}")
                args.legacy_root = win_path
                break
                
    migrator = LegacyMigrator(args.legacy_root)
    
    try:
        output_path = migrator.migrate_to_capsule(
            args.name,
            args.version,
            args.output,
            args.dry_run
        )
        
        if not args.dry_run:
            print(f"\n✅ SUCCESS: Migrated capsule at {output_path}")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 1
        
    finally:
        migrator.cleanup()
        
    return 0
    

if __name__ == '__main__':
    sys.exit(main())
