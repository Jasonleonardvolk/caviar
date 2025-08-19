"""
Lockfile handling for ELFIN packages.

This module provides tools for generating and parsing elf.lock files,
which are used to ensure reproducible builds by pinning exact dependency versions.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import toml

from .manifest import Manifest


class LockfileError(Exception):
    """Exception raised for lockfile parsing and validation errors."""
    pass


@dataclass
class PackageId:
    """Unique identifier for a package in the dependency graph."""
    name: str
    version: str
    
    def __str__(self) -> str:
        return f"{self.name} {self.version}"
    
    def as_key(self) -> str:
        """Get a string key for use in dictionaries."""
        return f"{self.name}:{self.version}"


@dataclass
class ResolvedDependency:
    """A resolved dependency with exact version and other metadata."""
    name: str
    version: str
    checksum: Optional[str] = None
    source: Optional[str] = None
    features: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # List of PackageId strings
    
    def to_package_id(self) -> PackageId:
        """Convert to a PackageId."""
        return PackageId(self.name, self.version)
    
    def as_key(self) -> str:
        """Get a string key for use in dictionaries."""
        return f"{self.name}:{self.version}"


@dataclass
class Lockfile:
    """
    Represents an ELFIN lockfile (elf.lock).
    
    This file locks all dependencies to exact versions for reproducible builds.
    """
    version: str = "1"
    packages: Dict[str, ResolvedDependency] = field(default_factory=dict)
    root_dependencies: List[str] = field(default_factory=list)  # List of PackageId strings
    
    @classmethod
    def load(cls, path: Path = Path("elf.lock")) -> "Lockfile":
        """
        Load and parse a lockfile.
        
        Args:
            path: Path to the elf.lock file
            
        Returns:
            Parsed Lockfile object
            
        Raises:
            LockfileError: If the lockfile is invalid or can't be parsed
            FileNotFoundError: If the lockfile doesn't exist
        """
        try:
            data = toml.load(path)
        except toml.TomlDecodeError as e:
            raise LockfileError(f"Failed to parse {path}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Lockfile not found: {path}")
        
        # Validate lockfile version
        if "version" not in data:
            raise LockfileError("Missing 'version' field in lockfile")
        
        # Create new lockfile
        lockfile = cls(version=data["version"])
        
        # Parse packages
        for pkg_data in data.get("package", []):
            name = pkg_data.get("name")
            version = pkg_data.get("version")
            
            if not name or not version:
                raise LockfileError("Package entry missing name or version")
            
            # Create resolved dependency
            dependency = ResolvedDependency(
                name=name,
                version=version,
                checksum=pkg_data.get("checksum"),
                source=pkg_data.get("source"),
                features=pkg_data.get("features", []),
                dependencies=pkg_data.get("dependencies", [])
            )
            
            # Add to packages dict
            lockfile.packages[dependency.as_key()] = dependency
        
        # Parse root dependencies
        lockfile.root_dependencies = data.get("root", {}).get("dependencies", [])
        
        return lockfile
    
    def save(self, path: Path = Path("elf.lock")) -> None:
        """
        Save the lockfile to disk.
        
        Args:
            path: Path to save the elf.lock file
            
        Raises:
            LockfileError: If the lockfile can't be saved
        """
        # Convert to TOML-friendly format
        data = {
            "version": self.version,
            "package": []
        }
        
        # Add packages
        for pkg_id, dependency in self.packages.items():
            pkg_data = {
                "name": dependency.name,
                "version": dependency.version
            }
            
            # Add optional fields
            if dependency.checksum:
                pkg_data["checksum"] = dependency.checksum
            if dependency.source:
                pkg_data["source"] = dependency.source
            if dependency.features:
                pkg_data["features"] = dependency.features
            if dependency.dependencies:
                pkg_data["dependencies"] = dependency.dependencies
            
            data["package"].append(pkg_data)
        
        # Add root dependencies
        data["root"] = {
            "dependencies": self.root_dependencies
        }
        
        # Write to file
        try:
            with open(path, 'w') as f:
                toml.dump(data, f)
        except Exception as e:
            raise LockfileError(f"Failed to save lockfile to {path}: {e}")
    
    def from_manifest(self, manifest: Manifest, resolved_deps: Dict[str, ResolvedDependency]) -> "Lockfile":
        """
        Create a lockfile from a manifest and resolved dependencies.
        
        Args:
            manifest: The project manifest
            resolved_deps: Mapping of package IDs to resolved dependencies
            
        Returns:
            A new Lockfile instance
        """
        lockfile = Lockfile()
        
        # Add all resolved dependencies
        lockfile.packages = resolved_deps
        
        # Add root dependencies
        root_deps = []
        for dep_name, dep in manifest.dependencies.items():
            # Find the resolved version
            for pkg_id, resolved_dep in resolved_deps.items():
                if resolved_dep.name == dep_name:
                    root_deps.append(pkg_id)
                    break
        
        lockfile.root_dependencies = root_deps
        
        return lockfile
    
    def has_dependency(self, name: str) -> bool:
        """
        Check if a dependency is present in the lockfile.
        
        Args:
            name: Name of the dependency
            
        Returns:
            True if dependency is present, False otherwise
        """
        for pkg_id in self.packages:
            if pkg_id.startswith(f"{name}:"):
                return True
        return False
    
    def get_dependency(self, name: str) -> Optional[ResolvedDependency]:
        """
        Get a dependency by name.
        
        If multiple versions are present, returns the one used by root.
        
        Args:
            name: Name of the dependency
            
        Returns:
            ResolvedDependency if found, None otherwise
        """
        # First check root dependencies
        for pkg_id_str in self.root_dependencies:
            pkg_id = pkg_id_str.split(":")
            if len(pkg_id) == 2 and pkg_id[0] == name:
                return self.packages.get(pkg_id_str)
        
        # Otherwise check all packages
        for pkg_id, dep in self.packages.items():
            if dep.name == name:
                return dep
        
        return None
    
    def get_all_versions(self, name: str) -> List[str]:
        """
        Get all versions of a dependency in the lockfile.
        
        Args:
            name: Name of the dependency
            
        Returns:
            List of version strings
        """
        versions = []
        for _, dep in self.packages.items():
            if dep.name == name:
                versions.append(dep.version)
        return versions


def generate_lockfile(
    manifest: Manifest,
    resolved_deps: Dict[str, ResolvedDependency],
    path: Path = Path("elf.lock")
) -> Lockfile:
    """
    Generate a lockfile from a manifest and resolved dependencies.
    
    Args:
        manifest: Project manifest
        resolved_deps: Mapping of package IDs to resolved dependencies
        path: Path to save the lockfile
        
    Returns:
        The generated Lockfile
        
    Raises:
        LockfileError: If lockfile generation fails
    """
    # Create lockfile
    lockfile = Lockfile()
    
    # Add all resolved dependencies
    lockfile.packages = resolved_deps
    
    # Add root dependencies
    root_deps = []
    for dep_name in manifest.dependencies:
        # Find the resolved version
        for pkg_id, resolved_dep in resolved_deps.items():
            if resolved_dep.name == dep_name:
                root_deps.append(pkg_id)
                break
    
    lockfile.root_dependencies = root_deps
    
    # Save lockfile
    try:
        lockfile.save(path)
    except Exception as e:
        raise LockfileError(f"Failed to generate lockfile: {e}")
    
    return lockfile
