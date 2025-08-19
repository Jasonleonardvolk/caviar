"""
Manifest parser and validator for ELFIN packages.

This module provides tools for parsing and validating elfpkg.toml manifests.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import toml

# Import semver for version handling
try:
    import semver
except ImportError:
    raise ImportError("semver package is required. Install with: pip install semver")


class ManifestError(Exception):
    """Exception raised for manifest parsing and validation errors."""
    pass


@dataclass
class Dependency:
    """Represents a package dependency with version constraints."""
    name: str
    version_req: str
    optional: bool = False
    features: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Validate version requirement is a valid semver range
        try:
            # Check if the semver range is valid
            semver.VersionInfo.parse(self.version_req.lstrip("^~=<>"))
        except ValueError as e:
            raise ManifestError(f"Invalid version requirement for {self.name}: {e}")
    
    def matches(self, version: str) -> bool:
        """Check if a version matches this dependency's requirements."""
        try:
            version_info = semver.VersionInfo.parse(version)
            return version_info.match(self.version_req)
        except ValueError:
            return False


@dataclass
class Manifest:
    """
    Represents an ELFIN package manifest (elfpkg.toml).
    
    This is the primary configuration for an ELFIN package, similar to
    Cargo.toml in Rust or package.json in Node.js.
    """
    name: str
    version: str
    authors: List[str] = field(default_factory=list)
    edition: str = "elfin-1.0"
    description: Optional[str] = None
    license: Optional[str] = None
    dependencies: Dict[str, Dependency] = field(default_factory=dict)
    dev_dependencies: Dict[str, Dependency] = field(default_factory=dict)
    solver: Dict[str, str] = field(default_factory=dict)
    features: Dict[str, List[str]] = field(default_factory=dict)
    
    @classmethod
    def load(cls, path: Path = Path("elfpkg.toml")) -> "Manifest":
        """
        Load and parse a manifest file.
        
        Args:
            path: Path to the elfpkg.toml file
            
        Returns:
            Parsed Manifest object
            
        Raises:
            ManifestError: If the manifest is invalid or can't be parsed
            FileNotFoundError: If the manifest file doesn't exist
        """
        try:
            data = toml.load(path)
        except toml.TomlDecodeError as e:
            raise ManifestError(f"Failed to parse {path}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Manifest file not found: {path}")
        
        # Validate required sections
        if "package" not in data:
            raise ManifestError("Missing [package] section in manifest")
        
        pkg = data["package"]
        
        # Validate required fields
        for field in ["name", "version"]:
            if field not in pkg:
                raise ManifestError(f"Missing required field '{field}' in [package] section")
        
        # Validate package name format
        name = pkg["name"]
        if not re.match(r'^[a-zA-Z][\w\-]*$', name):
            raise ManifestError(f"Invalid package name: {name}. Must start with a letter and contain only letters, numbers, underscores, and hyphens.")
        
        # Parse dependencies
        dependencies = {}
        for dep_name, dep_req in data.get("dependencies", {}).items():
            if isinstance(dep_req, str):
                # Simple version requirement
                dependencies[dep_name] = Dependency(dep_name, dep_req)
            elif isinstance(dep_req, dict):
                # Complex dependency with features, etc.
                dependencies[dep_name] = Dependency(
                    dep_name,
                    dep_req.get("version", "*"),
                    optional=dep_req.get("optional", False),
                    features=dep_req.get("features", [])
                )
            else:
                raise ManifestError(f"Invalid dependency specification for {dep_name}")
        
        # Parse dev dependencies
        dev_dependencies = {}
        for dep_name, dep_req in data.get("dev_dependencies", {}).items():
            if isinstance(dep_req, str):
                # Simple version requirement
                dev_dependencies[dep_name] = Dependency(dep_name, dep_req)
            elif isinstance(dep_req, dict):
                # Complex dependency with features, etc.
                dev_dependencies[dep_name] = Dependency(
                    dep_name,
                    dep_req.get("version", "*"),
                    optional=dep_req.get("optional", False),
                    features=dep_req.get("features", [])
                )
            else:
                raise ManifestError(f"Invalid dev-dependency specification for {dep_name}")
        
        # Parse features
        features = data.get("features", {})
        
        # Create the manifest object
        return cls(
            name=name,
            version=pkg["version"],
            authors=pkg.get("authors", []),
            edition=pkg.get("edition", "elfin-1.0"),
            description=pkg.get("description"),
            license=pkg.get("license"),
            dependencies=dependencies,
            dev_dependencies=dev_dependencies,
            solver=data.get("solver", {}),
            features=features
        )
    
    def save(self, path: Path = Path("elfpkg.toml")) -> None:
        """
        Save the manifest to a file.
        
        Args:
            path: Path to save the elfpkg.toml file
            
        Raises:
            ManifestError: If the manifest can't be saved
        """
        # Convert the manifest to a dict structure
        data = {
            "package": {
                "name": self.name,
                "version": self.version,
                "edition": self.edition
            }
        }
        
        # Add optional package fields
        if self.authors:
            data["package"]["authors"] = self.authors
        if self.description:
            data["package"]["description"] = self.description
        if self.license:
            data["package"]["license"] = self.license
        
        # Add dependencies
        if self.dependencies:
            data["dependencies"] = {}
            for dep_name, dep in self.dependencies.items():
                if dep.features or dep.optional:
                    # Complex dependency
                    data["dependencies"][dep_name] = {
                        "version": dep.version_req
                    }
                    if dep.features:
                        data["dependencies"][dep_name]["features"] = dep.features
                    if dep.optional:
                        data["dependencies"][dep_name]["optional"] = True
                else:
                    # Simple dependency
                    data["dependencies"][dep_name] = dep.version_req
        
        # Add dev dependencies
        if self.dev_dependencies:
            data["dev_dependencies"] = {}
            for dep_name, dep in self.dev_dependencies.items():
                if dep.features or dep.optional:
                    # Complex dependency
                    data["dev_dependencies"][dep_name] = {
                        "version": dep.version_req
                    }
                    if dep.features:
                        data["dev_dependencies"][dep_name]["features"] = dep.features
                    if dep.optional:
                        data["dev_dependencies"][dep_name]["optional"] = True
                else:
                    # Simple dependency
                    data["dev_dependencies"][dep_name] = dep.version_req
        
        # Add solver configuration
        if self.solver:
            data["solver"] = self.solver
        
        # Add features
        if self.features:
            data["features"] = self.features
        
        # Write to file
        try:
            with open(path, 'w') as f:
                toml.dump(data, f)
        except Exception as e:
            raise ManifestError(f"Failed to save manifest to {path}: {e}")
    
    def check_semver(self, current: str) -> bool:
        """
        Check if current version is compatible with this manifest.
        
        Args:
            current: Current version string
            
        Returns:
            True if compatible, False otherwise
            
        Raises:
            ValueError: If versions can't be parsed
        """
        try:
            current_version = semver.VersionInfo.parse(current)
            package_version = semver.VersionInfo.parse(self.version)
            
            # Major version must match for versions >= 1.0.0
            if package_version.major >= 1:
                return current_version.major == package_version.major
            
            # For versions < 1.0.0, minor must match
            return (current_version.major == package_version.major and 
                    current_version.minor == package_version.minor)
        except ValueError as e:
            raise ValueError(f"Failed to parse version: {e}")
    
    def get_env_vars(self) -> Dict[str, str]:
        """
        Get environment variables used in the manifest.
        
        This extracts environment variables referenced in solver settings
        and other fields, allowing resolution of values like ${HOME}/mosek.lic
        
        Returns:
            Dictionary of environment variable names to their values
        """
        env_vars = {}
        
        # Check solver settings for environment variables
        for key, value in self.solver.items():
            if isinstance(value, str):
                env_var_matches = re.findall(r'\${([^}]+)}', value)
                for var_name in env_var_matches:
                    if var_name in os.environ:
                        env_vars[var_name] = os.environ[var_name]
        
        return env_vars
    
    def resolve_env_vars(self) -> Dict[str, str]:
        """
        Resolve all environment variables in the manifest.
        
        Returns:
            Dictionary with resolved solver settings
        """
        resolved = {}
        
        # Resolve solver settings
        for key, value in self.solver.items():
            if isinstance(value, str):
                # Replace ${VAR} with environment variable value
                for var_name, var_value in self.get_env_vars().items():
                    value = value.replace(f"${{{var_name}}}", var_value)
                resolved[key] = value
            else:
                resolved[key] = value
        
        return resolved
