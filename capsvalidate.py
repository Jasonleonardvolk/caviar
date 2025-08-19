#!/usr/bin/env python3
"""
capsvalidate.py - TORI Capsule Manifest Validator
Validates capsule.yml files for correctness and best practices
"""

import argparse
import json
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CapsuleValidator:
    """Validates capsule manifests against schema and best practices"""
    
    # Required top-level fields
    REQUIRED_FIELDS = {
        'name': str,
        'version': str,
        'entrypoint': str
    }
    
    # Optional fields with expected types
    OPTIONAL_FIELDS = {
        'services': list,
        'env': dict,
        'dependencies': dict,
        'volumes': list,
        'resources': dict,
        'health': dict,
        'files': list,
        'hooks': dict,
        'metadata': dict
    }
    
    # Valid resource keys
    VALID_RESOURCES = [
        'cpu_quota', 'cpu_weight', 'cpu_shares',
        'memory_max', 'memory_high', 'memory_low',
        'io_weight', 'io_max', 'tasks_max'
    ]
    
    # Valid hook names
    VALID_HOOKS = ['pre_start', 'post_start', 'pre_stop', 'post_stop']
    
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.errors = []
        self.warnings = []
        
    def validate_file(self, manifest_path: Path) -> bool:
        """Validate a capsule.yml file"""
        logger.info(f"Validating manifest: {manifest_path}")
        
        # Reset errors/warnings
        self.errors = []
        self.warnings = []
        
        # Load manifest
        try:
            with open(manifest_path, 'r') as f:
                manifest = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML syntax: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Failed to read manifest: {e}")
            return False
            
        if not isinstance(manifest, dict):
            self.errors.append("Manifest must be a YAML dictionary")
            return False
            
        # Validate structure
        self._validate_required_fields(manifest)
        self._validate_optional_fields(manifest)
        self._validate_name_version(manifest)
        self._validate_entrypoint(manifest)
        self._validate_services(manifest)
        self._validate_environment(manifest)
        self._validate_dependencies(manifest)
        self._validate_volumes(manifest)
        self._validate_resources(manifest)
        self._validate_health_checks(manifest)
        self._validate_files(manifest)
        self._validate_hooks(manifest)
        
        # Check for unknown fields
        all_fields = set(self.REQUIRED_FIELDS.keys()) | set(self.OPTIONAL_FIELDS.keys())
        unknown = set(manifest.keys()) - all_fields
        if unknown:
            msg = f"Unknown fields: {', '.join(unknown)}"
            if self.strict:
                self.errors.append(msg)
            else:
                self.warnings.append(msg)
                
        # Print results
        self._print_results()
        
        return len(self.errors) == 0
        
    def _validate_required_fields(self, manifest: Dict):
        """Check all required fields are present with correct types"""
        for field, expected_type in self.REQUIRED_FIELDS.items():
            if field not in manifest:
                self.errors.append(f"Missing required field: {field}")
            elif not isinstance(manifest[field], expected_type):
                self.errors.append(
                    f"Field '{field}' must be {expected_type.__name__}, "
                    f"got {type(manifest[field]).__name__}"
                )
                
    def _validate_optional_fields(self, manifest: Dict):
        """Check optional fields have correct types"""
        for field, expected_type in self.OPTIONAL_FIELDS.items():
            if field in manifest and not isinstance(manifest[field], expected_type):
                self.errors.append(
                    f"Field '{field}' must be {expected_type.__name__}, "
                    f"got {type(manifest[field]).__name__}"
                )
                
    def _validate_name_version(self, manifest: Dict):
        """Validate name and version format"""
        if 'name' in manifest:
            name = manifest['name']
            if not name:
                self.errors.append("Name cannot be empty")
            elif not name.replace('-', '').replace('_', '').isalnum():
                self.errors.append("Name must contain only alphanumeric, dash, or underscore")
            elif len(name) > 63:
                self.errors.append("Name must be 63 characters or less")
                
        if 'version' in manifest:
            version = manifest['version']
            if not version:
                self.errors.append("Version cannot be empty")
            # Basic semver check
            elif not self._is_valid_version(version):
                self.warnings.append(f"Version '{version}' doesn't follow semver (X.Y.Z)")
                
    def _is_valid_version(self, version: str) -> bool:
        """Check if version follows basic semver pattern"""
        parts = version.split('.')
        if len(parts) != 3:
            return False
        try:
            return all(p.isdigit() for p in parts)
        except:
            return False
            
    def _validate_entrypoint(self, manifest: Dict):
        """Validate entrypoint format"""
        if 'entrypoint' not in manifest:
            return
            
        entrypoint = manifest['entrypoint']
        if not entrypoint:
            self.errors.append("Entrypoint cannot be empty")
        elif entrypoint.startswith('/'):
            self.errors.append("Entrypoint must be relative path, not absolute")
        elif '..' in entrypoint:
            self.errors.append("Entrypoint cannot contain '..'")
            
    def _validate_services(self, manifest: Dict):
        """Validate services list"""
        if 'services' not in manifest:
            return
            
        services = manifest['services']
        if not services:
            self.warnings.append("Services list is empty")
            
        for i, service in enumerate(services):
            if not isinstance(service, str):
                self.errors.append(f"Service [{i}] must be a string")
            elif not service:
                self.errors.append(f"Service [{i}] cannot be empty")
            elif not service.replace('-', '').replace('_', '').isalnum():
                self.errors.append(f"Service '{service}' contains invalid characters")
                
    def _validate_environment(self, manifest: Dict):
        """Validate environment variables"""
        if 'env' not in manifest:
            return
            
        env = manifest['env']
        for key, value in env.items():
            if not key:
                self.errors.append("Environment variable name cannot be empty")
            elif not key.replace('_', '').isalnum():
                self.errors.append(f"Environment variable '{key}' contains invalid characters")
            elif not isinstance(value, (str, int, float, bool)):
                self.errors.append(f"Environment variable '{key}' has invalid type")
                
            # Check for templating
            if isinstance(value, str) and '{' in value:
                # Check for valid template variables
                if '{CAPSULE_SHA}' in value:
                    pass  # Valid
                elif '{BUILD_SHA}' in value:
                    pass  # Valid
                else:
                    self.warnings.append(f"Environment variable '{key}' uses unknown template")
                    
    def _validate_dependencies(self, manifest: Dict):
        """Validate dependencies section"""
        if 'dependencies' not in manifest:
            return
            
        deps = manifest['dependencies']
        
        # Check Python version
        if 'python' in deps:
            py_ver = deps['python']
            if not isinstance(py_ver, str):
                self.errors.append("Python version must be a string")
            elif not py_ver.replace('.', '').isdigit():
                self.errors.append(f"Invalid Python version: {py_ver}")
            else:
                parts = py_ver.split('.')
                if len(parts) < 2:
                    self.errors.append("Python version must be X.Y or X.Y.Z")
                    
        # Check pip requirements
        if 'pip_requirements' in deps:
            req_file = deps['pip_requirements']
            if not isinstance(req_file, str):
                self.errors.append("pip_requirements must be a string")
            elif not req_file:
                self.errors.append("pip_requirements cannot be empty")
                
        # Check for pinned versions
        if 'pip_requirements' in deps and deps['pip_requirements'] == 'requirements.txt':
            self.warnings.append("Consider using specific requirements file like 'requirements_nodb.txt'")
            
    def _validate_volumes(self, manifest: Dict):
        """Validate volume mounts"""
        if 'volumes' not in manifest:
            return
            
        volumes = manifest['volumes']
        for i, vol in enumerate(volumes):
            if not isinstance(vol, dict):
                self.errors.append(f"Volume [{i}] must be a dictionary")
                continue
                
            # Check required fields
            if 'source' not in vol:
                self.errors.append(f"Volume [{i}] missing 'source'")
            if 'target' not in vol:
                self.errors.append(f"Volume [{i}] missing 'target'")
                
            # Check mode
            if 'mode' in vol:
                if vol['mode'] not in ['ro', 'rw']:
                    self.errors.append(f"Volume [{i}] mode must be 'ro' or 'rw'")
                    
    def _validate_resources(self, manifest: Dict):
        """Validate resource limits"""
        if 'resources' not in manifest:
            return
            
        resources = manifest['resources']
        for key, value in resources.items():
            if key not in self.VALID_RESOURCES:
                self.warnings.append(f"Unknown resource limit: {key}")
                
            # Validate CPU quota format
            if key == 'cpu_quota' and isinstance(value, str):
                if not value.endswith('%'):
                    self.errors.append("cpu_quota must end with %")
                else:
                    try:
                        pct = int(value[:-1])
                        if pct <= 0 or pct > 100:
                            self.errors.append("cpu_quota must be between 1% and 100%")
                    except:
                        self.errors.append(f"Invalid cpu_quota: {value}")
                        
            # Validate memory format
            if key in ['memory_max', 'memory_high', 'memory_low']:
                if isinstance(value, str):
                    if not any(value.endswith(u) for u in ['K', 'M', 'G', 'T']):
                        self.errors.append(f"{key} must end with K, M, G, or T")
                        
    def _validate_health_checks(self, manifest: Dict):
        """Validate health check configuration"""
        if 'health' not in manifest:
            return
            
        health = manifest['health']
        
        # HTTP health check
        if 'http' in health:
            http = health['http']
            if not isinstance(http, dict):
                self.errors.append("health.http must be a dictionary")
            else:
                if 'path' not in http:
                    self.errors.append("health.http missing 'path'")
                if 'port' not in http:
                    self.errors.append("health.http missing 'port'")
                elif not isinstance(http['port'], int) or http['port'] <= 0:
                    self.errors.append("health.http.port must be positive integer")
                    
        # Exec health check
        if 'exec' in health:
            exec_check = health['exec']
            if not isinstance(exec_check, dict):
                self.errors.append("health.exec must be a dictionary")
            elif 'command' not in exec_check:
                self.errors.append("health.exec missing 'command'")
            elif not isinstance(exec_check['command'], list):
                self.errors.append("health.exec.command must be a list")
                
    def _validate_files(self, manifest: Dict):
        """Validate files section"""
        if 'files' not in manifest:
            return
            
        files = manifest['files']
        for i, file_spec in enumerate(files):
            if not isinstance(file_spec, dict):
                self.errors.append(f"File spec [{i}] must be a dictionary")
                continue
                
            if 'src' not in file_spec:
                self.errors.append(f"File spec [{i}] missing 'src'")
                
            # Check for absolute paths
            for key in ['src', 'dst']:
                if key in file_spec and file_spec[key].startswith('/'):
                    self.errors.append(f"File spec [{i}].{key} must be relative path")
                    
    def _validate_hooks(self, manifest: Dict):
        """Validate lifecycle hooks"""
        if 'hooks' not in manifest:
            return
            
        hooks = manifest['hooks']
        for hook_name, script in hooks.items():
            if hook_name not in self.VALID_HOOKS:
                self.warnings.append(f"Unknown hook: {hook_name}")
                
            if not isinstance(script, str):
                self.errors.append(f"Hook '{hook_name}' must be a string (shell script)")
            elif not script.strip():
                self.warnings.append(f"Hook '{hook_name}' is empty")
                
    def _print_results(self):
        """Print validation results"""
        if self.errors:
            logger.error(f"\n❌ Validation failed with {len(self.errors)} errors:")
            for error in self.errors:
                logger.error(f"   - {error}")
                
        if self.warnings:
            logger.warning(f"\n⚠️  {len(self.warnings)} warnings:")
            for warning in self.warnings:
                logger.warning(f"   - {warning}")
                
        if not self.errors and not self.warnings:
            logger.info("\n✅ Manifest validation passed!")
            
    def lint(self, manifest_path: Path) -> List[str]:
        """Run additional linting checks for best practices"""
        suggestions = []
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = yaml.safe_load(f)
        except:
            return suggestions
            
        # Check for missing optional but recommended fields
        if 'resources' not in manifest:
            suggestions.append("Consider adding 'resources' section for systemd limits")
            
        if 'health' not in manifest:
            suggestions.append("Consider adding 'health' checks for monitoring")
            
        if 'metadata' not in manifest:
            suggestions.append("Consider adding 'metadata' for tracking")
            
        # Check for security best practices
        if 'env' in manifest:
            for key, value in manifest['env'].items():
                if 'PASSWORD' in key or 'SECRET' in key or 'KEY' in key:
                    suggestions.append(f"Avoid hardcoding secrets: {key}")
                    
        # Check for production readiness
        if 'version' in manifest and 'dev' in manifest['version'].lower():
            suggestions.append("Production capsules should not use 'dev' versions")
            
        return suggestions


def main():
    parser = argparse.ArgumentParser(
        description="TORI Capsule Manifest Validator"
    )
    
    parser.add_argument(
        'manifest',
        type=Path,
        help="Path to capsule.yml manifest"
    )
    
    parser.add_argument(
        '--strict',
        action='store_true',
        help="Treat warnings as errors"
    )
    
    parser.add_argument(
        '--lint',
        action='store_true',
        help="Show additional best practice suggestions"
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    if not args.manifest.exists():
        logger.error(f"Manifest not found: {args.manifest}")
        return 1
        
    validator = CapsuleValidator(strict=args.strict)
    is_valid = validator.validate_file(args.manifest)
    
    # Run linting if requested
    if args.lint:
        suggestions = validator.lint(args.manifest)
        if suggestions:
            logger.info(f"\n💡 {len(suggestions)} suggestions:")
            for suggestion in suggestions:
                logger.info(f"   - {suggestion}")
                
    # JSON output if requested
    if args.json:
        result = {
            'valid': is_valid,
            'errors': validator.errors,
            'warnings': validator.warnings
        }
        if args.lint:
            result['suggestions'] = validator.lint(args.manifest)
        print(json.dumps(result, indent=2))
        
    return 0 if is_valid else 1
    

if __name__ == '__main__':
    sys.exit(main())
