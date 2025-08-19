#!/usr/bin/env python3
"""
capsbuild.py - TORI Capsule Builder
Packages applications into immutable, content-addressed capsules for Dickbox deployment
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zipfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CapsuleBuilder:
    """Builds immutable capsules for TORI services"""
    
    REQUIRED_MANIFEST_FIELDS = ['name', 'version', 'entrypoint']
    VALID_SECTIONS = ['bin', 'lib', 'venv', 'config', 'assets', 'data']
    
    def __init__(self, work_dir: Optional[Path] = None):
        self.work_dir = work_dir or Path(tempfile.mkdtemp(prefix="capsbuild_"))
        self.manifest = {}
        self.build_metadata = {
            'build_time': datetime.utcnow().isoformat(),
            'builder_version': '1.0.0',
            'host': os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        }
        
    def build_from_directory(self, source_dir: Path, manifest_path: Path,
                           output_path: Optional[Path] = None,
                           sign_key: Optional[str] = None) -> Path:
        """Build capsule from a directory structure"""
        logger.info(f"Building capsule from directory: {source_dir}")
        
        # Load and validate manifest
        self.manifest = self._load_manifest(manifest_path)
        self._validate_manifest()
        
        # Create capsule structure
        capsule_dir = self.work_dir / "capsule"
        capsule_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy manifest
        shutil.copy2(manifest_path, capsule_dir / "capsule.yml")
        
        # Process each section
        self._process_directory_structure(source_dir, capsule_dir)
        
        # Verify dependencies
        self._verify_dependencies(capsule_dir)
        
        # Generate BUILD_SHA
        build_sha = self._generate_build_sha(capsule_dir)
        
        # Create metadata file
        self._write_build_metadata(capsule_dir, build_sha)
        
        # Package into tarball
        output_path = output_path or Path(f"capsule-{self.manifest['name']}-{build_sha[:12]}.tar.gz")
        self._create_tarball(capsule_dir, output_path)
        
        # Optional: Sign the capsule
        if sign_key:
            self._sign_capsule(output_path, sign_key)
            
        logger.info(f"✅ Capsule built successfully: {output_path}")
        logger.info(f"   BUILD_SHA: {build_sha}")
        logger.info(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return output_path
        
    def build_from_zip(self, zip_path: Path, manifest_path: Path,
                      output_path: Optional[Path] = None,
                      sign_key: Optional[str] = None) -> Path:
        """Build capsule from a zip file"""
        logger.info(f"Building capsule from zip: {zip_path}")
        
        # Extract zip to temp directory
        extract_dir = self.work_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
            
        # Build from extracted directory
        return self.build_from_directory(extract_dir, manifest_path, output_path, sign_key)
        
    def _load_manifest(self, manifest_path: Path) -> Dict:
        """Load and parse capsule.yml manifest"""
        try:
            with open(manifest_path, 'r') as f:
                manifest = yaml.safe_load(f)
                
            if not isinstance(manifest, dict):
                raise ValueError("Manifest must be a YAML dictionary")
                
            return manifest
            
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            raise
            
    def _validate_manifest(self):
        """Validate manifest has required fields"""
        missing = []
        for field in self.REQUIRED_MANIFEST_FIELDS:
            if field not in self.manifest:
                missing.append(field)
                
        if missing:
            raise ValueError(f"Missing required manifest fields: {', '.join(missing)}")
            
        # Validate entrypoint format
        entrypoint = self.manifest['entrypoint']
        if not isinstance(entrypoint, str) or not entrypoint:
            raise ValueError("Entrypoint must be a non-empty string")
            
        # Validate services if present
        if 'services' in self.manifest:
            if not isinstance(self.manifest['services'], list):
                raise ValueError("Services must be a list")
                
        # Validate dependencies if present
        if 'dependencies' in self.manifest:
            deps = self.manifest['dependencies']
            if not isinstance(deps, dict):
                raise ValueError("Dependencies must be a dictionary")
                
            # Check Python version format
            if 'python' in deps:
                py_ver = deps['python']
                if not isinstance(py_ver, str) or not py_ver.replace('.', '').isdigit():
                    raise ValueError(f"Invalid Python version: {py_ver}")
                    
        logger.info("✅ Manifest validation passed")
        
    def _process_directory_structure(self, source_dir: Path, capsule_dir: Path):
        """Copy source files into capsule structure"""
        
        # Standard directory mapping
        dir_mapping = {
            'bin': 'bin',
            'scripts': 'bin',
            'lib': 'lib',
            'libs': 'lib',
            'venv': 'venv',
            '.venv': 'venv',
            'config': 'config',
            'conf': 'config',
            'assets': 'assets',
            'static': 'assets',
            'data': 'data'
        }
        
        # Process Python virtual environment specially
        venv_processed = False
        for venv_name in ['venv', '.venv', 'env']:
            venv_path = source_dir / venv_name
            if venv_path.exists() and venv_path.is_dir():
                logger.info(f"Processing Python virtual environment: {venv_name}")
                self._process_venv(venv_path, capsule_dir / 'venv')
                venv_processed = True
                break
                
        # Process requirements files
        for req_file in ['requirements.txt', 'requirements_nodb.txt', 'pyproject.toml', 'poetry.lock']:
            req_path = source_dir / req_file
            if req_path.exists():
                logger.info(f"Including dependency file: {req_file}")
                shutil.copy2(req_path, capsule_dir / req_file)
                
        # Process other directories
        for src_name, dst_name in dir_mapping.items():
            src_path = source_dir / src_name
            if src_path.exists() and src_path.is_dir() and src_name not in ['venv', '.venv']:
                logger.info(f"Copying {src_name} -> {dst_name}")
                shutil.copytree(src_path, capsule_dir / dst_name, dirs_exist_ok=True)
                
        # Copy specific files from manifest
        if 'files' in self.manifest:
            for file_spec in self.manifest['files']:
                src = source_dir / file_spec['src']
                dst = capsule_dir / file_spec.get('dst', file_spec['src'])
                
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if src.is_file():
                        shutil.copy2(src, dst)
                    else:
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    logger.warning(f"File not found: {src}")
                    
        # Handle entrypoint
        entrypoint_src = source_dir / self.manifest['entrypoint']
        entrypoint_dst = capsule_dir / self.manifest['entrypoint']
        
        if entrypoint_src.exists():
            entrypoint_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entrypoint_src, entrypoint_dst)
            # Make executable
            entrypoint_dst.chmod(entrypoint_dst.stat().st_mode | 0o111)
        else:
            # Check if entrypoint is in capsule already
            if not entrypoint_dst.exists():
                raise FileNotFoundError(f"Entrypoint not found: {self.manifest['entrypoint']}")
                
    def _process_venv(self, src_venv: Path, dst_venv: Path):
        """Process Python virtual environment for relocation"""
        logger.info("Processing virtual environment for relocation...")
        
        # Copy venv
        shutil.copytree(src_venv, dst_venv, dirs_exist_ok=True)
        
        # Fix shebangs and paths for relocation
        self._fix_venv_paths(dst_venv)
        
        # Create activation script wrapper
        activate_wrapper = dst_venv.parent / "activate.sh"
        wrapper_content = f'''#!/bin/bash
# Capsule venv activation wrapper
export VIRTUAL_ENV="$(cd "$(dirname "${{BASH_SOURCE[0]}})")/venv" && pwd)"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$(cd "$(dirname "${{BASH_SOURCE[0]}})")" && pwd):$PYTHONPATH"
echo "Activated capsule virtual environment at $VIRTUAL_ENV"
'''
        activate_wrapper.write_text(wrapper_content)
        activate_wrapper.chmod(0o755)
        
    def _fix_venv_paths(self, venv_path: Path):
        """Fix hardcoded paths in venv for relocation"""
        # Fix activate script
        activate_path = venv_path / "bin" / "activate"
        if activate_path.exists():
            content = activate_path.read_text()
            # Make VIRTUAL_ENV relative
            content = content.replace(
                f'VIRTUAL_ENV="{venv_path}"',
                'VIRTUAL_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"'
            )
            activate_path.write_text(content)
            
        # Fix pip/python shebangs to use #!/usr/bin/env python
        for script in (venv_path / "bin").glob("*"):
            if script.is_file():
                try:
                    with open(script, 'rb') as f:
                        first_line = f.readline()
                        
                    if first_line.startswith(b'#!') and b'python' in first_line:
                        # Replace with portable shebang
                        content = script.read_bytes()
                        content = b'#!/usr/bin/env python\n' + content.split(b'\n', 1)[1]
                        script.write_bytes(content)
                except:
                    pass  # Skip binary files
                    
    def _verify_dependencies(self, capsule_dir: Path):
        """Verify all dependencies are present"""
        deps = self.manifest.get('dependencies', {})
        
        # Check Python version if specified
        if 'python' in deps:
            required_py = deps['python']
            logger.info(f"Verifying Python {required_py} compatibility")
            
            # Check if venv exists
            venv_python = capsule_dir / 'venv' / 'bin' / 'python'
            if venv_python.exists():
                try:
                    result = subprocess.run(
                        [str(venv_python), '--version'],
                        capture_output=True,
                        text=True
                    )
                    actual_version = result.stdout.strip().split()[-1]
                    logger.info(f"Venv Python version: {actual_version}")
                except:
                    logger.warning("Could not determine venv Python version")
                    
        # Check pip requirements
        if 'pip_requirements' in deps:
            req_file = capsule_dir / deps['pip_requirements']
            if not req_file.exists():
                raise FileNotFoundError(f"Requirements file not found: {deps['pip_requirements']}")
                
            logger.info(f"✅ Found requirements file: {deps['pip_requirements']}")
            
        # Verify entrypoint is executable
        entrypoint_path = capsule_dir / self.manifest['entrypoint']
        if not entrypoint_path.exists():
            raise FileNotFoundError(f"Entrypoint not found in capsule: {self.manifest['entrypoint']}")
            
        # Check if it's a Python script that needs venv
        if entrypoint_path.suffix == '.py' and not (capsule_dir / 'venv').exists():
            logger.warning("Python entrypoint found but no venv included")
            
    def _generate_build_sha(self, capsule_dir: Path) -> str:
        """Generate SHA256 hash of all capsule contents"""
        logger.info("Generating BUILD_SHA...")
        
        sha256 = hashlib.sha256()
        file_count = 0
        
        # Sort files for deterministic hashing
        all_files = sorted(capsule_dir.rglob('*'))
        
        for file_path in all_files:
            if file_path.is_file():
                # Include relative path in hash for structure integrity
                rel_path = file_path.relative_to(capsule_dir)
                sha256.update(str(rel_path).encode('utf-8'))
                
                # Include file contents
                with open(file_path, 'rb') as f:
                    while chunk := f.read(8192):
                        sha256.update(chunk)
                        
                file_count += 1
                
        build_sha = sha256.hexdigest()
        logger.info(f"Generated BUILD_SHA from {file_count} files: {build_sha}")
        
        return build_sha
        
    def _write_build_metadata(self, capsule_dir: Path, build_sha: str):
        """Write build metadata file"""
        metadata = {
            **self.build_metadata,
            'build_sha': build_sha,
            'manifest': self.manifest,
            'file_count': sum(1 for _ in capsule_dir.rglob('*') if _.is_file())
        }
        
        metadata_path = capsule_dir / '.capsule_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        logger.info("Written build metadata")
        
    def _create_tarball(self, capsule_dir: Path, output_path: Path):
        """Create compressed tarball of capsule"""
        logger.info(f"Creating tarball: {output_path}")
        
        with tarfile.open(output_path, 'w:gz') as tar:
            # Add files with relative paths
            for item in capsule_dir.iterdir():
                tar.add(item, arcname=item.name)
                
        logger.info(f"Created tarball: {output_path.stat().st_size / 1024:.2f} KB")
        
    def _sign_capsule(self, capsule_path: Path, sign_key: str):
        """Sign capsule with minisign (if available)"""
        try:
            # Check if minisign is available
            result = subprocess.run(['minisign', '-v'], capture_output=True)
            if result.returncode != 0:
                logger.warning("minisign not found, skipping signature")
                return
                
            # Sign the capsule
            sig_path = capsule_path.with_suffix('.tar.gz.sig')
            subprocess.run([
                'minisign', '-Sm', str(capsule_path),
                '-s', sign_key
            ], check=True)
            
            logger.info(f"✅ Signed capsule: {sig_path}")
            
        except Exception as e:
            logger.warning(f"Failed to sign capsule: {e}")
            
    def cleanup(self):
        """Clean up temporary work directory"""
        if self.work_dir.exists() and str(self.work_dir).startswith('/tmp'):
            shutil.rmtree(self.work_dir)
            

def main():
    parser = argparse.ArgumentParser(
        description="TORI Capsule Builder - Package applications into immutable capsules"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--from-dir',
        type=Path,
        help="Build capsule from directory"
    )
    input_group.add_argument(
        '--from-zip',
        type=Path,
        help="Build capsule from zip file"
    )
    
    # Manifest
    parser.add_argument(
        '--manifest',
        type=Path,
        required=True,
        help="Path to capsule.yml manifest"
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=Path,
        help="Output path for capsule (default: auto-generated)"
    )
    
    # Optional features
    parser.add_argument(
        '--sign',
        help="Sign capsule with minisign key"
    )
    
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help="Keep temporary build directory"
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Build capsule
    builder = CapsuleBuilder()
    
    try:
        if args.from_dir:
            output_path = builder.build_from_directory(
                args.from_dir,
                args.manifest,
                args.output,
                args.sign
            )
        else:
            output_path = builder.build_from_zip(
                args.from_zip,
                args.manifest,
                args.output,
                args.sign
            )
            
        print(f"\n✅ SUCCESS: Capsule built at {output_path}")
        
        # Print BUILD_SHA for reference
        with tarfile.open(output_path, 'r:gz') as tar:
            metadata_file = tar.extractfile('.capsule_metadata.json')
            if metadata_file:
                metadata = json.load(metadata_file)
                print(f"   BUILD_SHA: {metadata['build_sha']}")
                
    except Exception as e:
        logger.error(f"❌ Build failed: {e}")
        return 1
        
    finally:
        if not args.keep_temp:
            builder.cleanup()
            
    return 0
    

if __name__ == '__main__':
    sys.exit(main())
