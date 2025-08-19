"""
ELFIN Package Management System.

This module provides package management functionality for ELFIN,
including package versioning, dependencies, and resolution.
"""

import os
import json
import re
import shutil
import tarfile
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass
import semver


@dataclass
class PackageVersion:
    """Representation of a package version."""
    
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None
    
    def __str__(self) -> str:
        """Convert to string representation."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version
    
    @staticmethod
    def parse(version_str: str) -> 'PackageVersion':
        """
        Parse a version string into a PackageVersion.
        
        Args:
            version_str: The version string (e.g., "1.2.3-alpha+build.1")
            
        Returns:
            A PackageVersion object
            
        Raises:
            ValueError: If the version string is invalid
        """
        try:
            version_info = semver.VersionInfo.parse(version_str)
            return PackageVersion(
                major=version_info.major,
                minor=version_info.minor,
                patch=version_info.patch,
                prerelease=version_info.prerelease,
                build=version_info.build
            )
        except ValueError:
            raise ValueError(f"Invalid version string: {version_str}")
    
    def satisfies(self, version_range: str) -> bool:
        """
        Check if this version satisfies a version range.
        
        Args:
            version_range: The version range (e.g., ">=1.0.0 <2.0.0")
            
        Returns:
            True if this version satisfies the range, False otherwise
        """
        try:
            return semver.match(str(self), version_range)
        except ValueError:
            return False


@dataclass
class PackageDependency:
    """Representation of a package dependency."""
    
    name: str
    version_range: str
    
    def matches(self, name: str, version: Union[str, PackageVersion]) -> bool:
        """
        Check if a package matches this dependency.
        
        Args:
            name: The package name
            version: The package version
            
        Returns:
            True if the package matches this dependency, False otherwise
        """
        if name != self.name:
            return False
        
        # Convert version to string if it's a PackageVersion
        if isinstance(version, PackageVersion):
            version = str(version)
        
        # Check if the version is in the range
        try:
            return semver.match(version, self.version_range)
        except ValueError:
            return False


class Package:
    """
    An ELFIN package.
    
    A package is a collection of ELFIN modules that can be distributed
    and reused across projects.
    """
    
    def __init__(
        self,
        name: str,
        version: PackageVersion,
        description: str = "",
        author: str = "",
        dependencies: Optional[List[PackageDependency]] = None,
        path: Optional[str] = None
    ):
        """
        Initialize a package.
        
        Args:
            name: The name of the package
            version: The version of the package
            description: A description of the package (optional)
            author: The author of the package (optional)
            dependencies: List of package dependencies (optional)
            path: Path to the package directory (optional)
        """
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.dependencies = dependencies or []
        self.path = path
    
    @property
    def full_name(self) -> str:
        """Get the full name of the package, including version."""
        return f"{self.name}@{self.version}"
    
    @staticmethod
    def from_json(json_data: Dict[str, Any], path: Optional[str] = None) -> 'Package':
        """
        Create a package from JSON data.
        
        Args:
            json_data: The JSON data
            path: Path to the package directory (optional)
            
        Returns:
            A Package object
        """
        name = json_data["name"]
        version = PackageVersion.parse(json_data["version"])
        description = json_data.get("description", "")
        author = json_data.get("author", "")
        
        # Parse dependencies
        dependencies = []
        deps_dict = json_data.get("dependencies", {})
        for dep_name, dep_version in deps_dict.items():
            dependencies.append(PackageDependency(dep_name, dep_version))
        
        return Package(name, version, description, author, dependencies, path)
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert the package to JSON data.
        
        Returns:
            A dictionary of JSON data
        """
        # Convert dependencies to dictionary
        deps_dict = {}
        for dep in self.dependencies:
            deps_dict[dep.name] = dep.version_range
        
        return {
            "name": self.name,
            "version": str(self.version),
            "description": self.description,
            "author": self.author,
            "dependencies": deps_dict
        }
    
    def save_manifest(self, path: Optional[str] = None) -> str:
        """
        Save the package manifest to a file.
        
        Args:
            path: Path to save the manifest (optional)
            
        Returns:
            The path to the saved manifest
        """
        path = path or self.path
        if path is None:
            raise ValueError("No path specified for saving manifest")
        
        manifest_path = os.path.join(path, "elfin.json")
        with open(manifest_path, "w") as f:
            json.dump(self.to_json(), f, indent=2)
        
        return manifest_path
    
    @staticmethod
    def load_manifest(path: str) -> 'Package':
        """
        Load a package from a manifest file.
        
        Args:
            path: Path to the manifest file or directory
            
        Returns:
            A Package object
            
        Raises:
            FileNotFoundError: If the manifest file is not found
            ValueError: If the manifest is invalid
        """
        # If path is a directory, look for elfin.json
        if os.path.isdir(path):
            manifest_path = os.path.join(path, "elfin.json")
        else:
            manifest_path = path
        
        # Check if the manifest file exists
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        
        # Load the manifest
        with open(manifest_path, "r") as f:
            try:
                json_data = json.load(f)
                return Package.from_json(json_data, os.path.dirname(manifest_path))
            except json.JSONDecodeError:
                raise ValueError(f"Invalid manifest file: {manifest_path}")
            except KeyError as e:
                raise ValueError(f"Missing required field in manifest: {e}")


class PackageRegistry:
    """
    A registry of ELFIN packages.
    
    This registry manages the installed packages in an ELFIN project.
    """
    
    def __init__(self, registry_dir: str):
        """
        Initialize a package registry.
        
        Args:
            registry_dir: Directory for the registry
        """
        self.registry_dir = registry_dir
        self.packages: Dict[str, List[Package]] = {}
        
        # Create the registry directory if it doesn't exist
        os.makedirs(registry_dir, exist_ok=True)
        
        # Load the installed packages
        self._load_packages()
    
    def _load_packages(self) -> None:
        """Load all installed packages from the registry directory."""
        # Reset the packages dictionary
        self.packages = {}
        
        # List all directories in the registry
        for package_dir in os.listdir(self.registry_dir):
            package_path = os.path.join(self.registry_dir, package_dir)
            
            # Skip files
            if not os.path.isdir(package_path):
                continue
            
            # Try to load the package manifest
            try:
                package = Package.load_manifest(package_path)
                
                # Add to the packages dictionary
                if package.name not in self.packages:
                    self.packages[package.name] = []
                self.packages[package.name].append(package)
            except (FileNotFoundError, ValueError):
                # Skip directories without a valid manifest
                continue
    
    def get_package(self, name: str, version: Optional[str] = None) -> Optional[Package]:
        """
        Get a package by name and optional version.
        
        Args:
            name: The name of the package
            version: The version or version range (optional)
            
        Returns:
            The package if found, None otherwise
        """
        # Check if the package exists
        if name not in self.packages:
            return None
        
        # If no version is specified, return the latest version
        if version is None:
            return max(self.packages[name], key=lambda p: p.version)
        
        # Try to find a package that matches the version
        for package in self.packages[name]:
            if package.version.satisfies(version):
                return package
        
        return None
    
    def install_package(self, package: Package) -> str:
        """
        Install a package into the registry.
        
        Args:
            package: The package to install
            
        Returns:
            The path to the installed package
            
        Raises:
            ValueError: If the package is already installed
        """
        # Check if the package is already installed
        if name in self.packages:
            for existing_package in self.packages[package.name]:
                if existing_package.version == package.version:
                    raise ValueError(f"Package {package.full_name} is already installed")
        
        # Create the package directory
        package_dir = os.path.join(self.registry_dir, f"{package.name}-{package.version}")
        os.makedirs(package_dir, exist_ok=True)
        
        # Copy the package files
        if package.path:
            for item in os.listdir(package.path):
                source = os.path.join(package.path, item)
                dest = os.path.join(package_dir, item)
                
                if os.path.isdir(source):
                    shutil.copytree(source, dest)
                else:
                    shutil.copy2(source, dest)
        
        # Save the package manifest
        package.save_manifest(package_dir)
        
        # Update the package's path
        package.path = package_dir
        
        # Add to the packages dictionary
        if package.name not in self.packages:
            self.packages[package.name] = []
        self.packages[package.name].append(package)
        
        return package_dir
    
    def uninstall_package(self, name: str, version: Optional[str] = None) -> bool:
        """
        Uninstall a package from the registry.
        
        Args:
            name: The name of the package
            version: The version or version range (optional)
            
        Returns:
            True if the package was uninstalled, False otherwise
        """
        # Check if the package exists
        if name not in self.packages:
            return False
        
        # If no version is specified, uninstall all versions
        if version is None:
            for package in self.packages[name]:
                self._remove_package_dir(package)
            
            del self.packages[name]
            return True
        
        # Find packages that match the version
        to_remove = []
        for package in self.packages[name]:
            if package.version.satisfies(version):
                to_remove.append(package)
        
        # Remove the packages
        for package in to_remove:
            self._remove_package_dir(package)
            self.packages[name].remove(package)
        
        # Remove the package name if no versions are left
        if not self.packages[name]:
            del self.packages[name]
        
        return bool(to_remove)
    
    def _remove_package_dir(self, package: Package) -> None:
        """
        Remove a package directory.
        
        Args:
            package: The package to remove
        """
        if package.path and os.path.exists(package.path):
            shutil.rmtree(package.path)
    
    def resolve_dependencies(self, package: Package) -> Dict[str, Package]:
        """
        Resolve the dependencies of a package.
        
        Args:
            package: The package to resolve dependencies for
            
        Returns:
            A dictionary of package names to packages
            
        Raises:
            ValueError: If a dependency cannot be resolved
        """
        resolved: Dict[str, Package] = {}
        unresolved: List[PackageDependency] = list(package.dependencies)
        
        while unresolved:
            dep = unresolved.pop(0)
            
            # Skip if already resolved
            if dep.name in resolved:
                continue
            
            # Try to find a package that matches the dependency
            dependency = self.get_package(dep.name, dep.version_range)
            if dependency is None:
                raise ValueError(f"Dependency {dep.name}@{dep.version_range} not found")
            
            # Add to resolved dependencies
            resolved[dep.name] = dependency
            
            # Add the dependency's dependencies to unresolved
            for transitive_dep in dependency.dependencies:
                if transitive_dep.name not in resolved:
                    unresolved.append(transitive_dep)
        
        return resolved


class PackageManager:
    """
    A manager for ELFIN packages.
    
    This manager provides high-level functionality for working with packages,
    including installing, uninstalling, and resolving dependencies.
    """
    
    def __init__(self, registry_dir: Optional[str] = None):
        """
        Initialize a package manager.
        
        Args:
            registry_dir: Directory for the registry (optional)
        """
        # Use default registry directory if none provided
        if registry_dir is None:
            home_dir = os.path.expanduser("~")
            registry_dir = os.path.join(home_dir, ".elfin", "packages")
        
        self.registry = PackageRegistry(registry_dir)
    
    def install(self, package_path: str) -> Package:
        """
        Install a package from a directory or archive.
        
        Args:
            package_path: Path to the package directory or archive
            
        Returns:
            The installed package
            
        Raises:
            ValueError: If the package cannot be installed
        """
        # Check if the path is a directory
        if os.path.isdir(package_path):
            # Load the package manifest
            package = Package.load_manifest(package_path)
            
            # Install the package
            self.registry.install_package(package)
            
            return package
        
        # Check if the path is an archive
        if os.path.isfile(package_path):
            # Extract the archive to a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Try to extract the archive
                try:
                    if package_path.endswith(".zip"):
                        with zipfile.ZipFile(package_path, "r") as zip_ref:
                            zip_ref.extractall(temp_dir)
                    elif package_path.endswith(".tar.gz") or package_path.endswith(".tgz"):
                        with tarfile.open(package_path, "r:gz") as tar_ref:
                            tar_ref.extractall(temp_dir)
                    else:
                        raise ValueError(f"Unsupported archive format: {package_path}")
                except (zipfile.BadZipFile, tarfile.ReadError):
                    raise ValueError(f"Invalid archive: {package_path}")
                
                # Load the package manifest
                package = Package.load_manifest(temp_dir)
                
                # Install the package
                self.registry.install_package(package)
                
                return package
        
        raise ValueError(f"Invalid package path: {package_path}")
    
    def uninstall(self, name: str, version: Optional[str] = None) -> bool:
        """
        Uninstall a package.
        
        Args:
            name: The name of the package
            version: The version or version range (optional)
            
        Returns:
            True if the package was uninstalled, False otherwise
        """
        return self.registry.uninstall_package(name, version)
    
    def create_package(self, name: str, version: str, description: str = "", author: str = "") -> Package:
        """
        Create a new package.
        
        Args:
            name: The name of the package
            version: The version of the package
            description: A description of the package (optional)
            author: The author of the package (optional)
            
        Returns:
            The created package
        """
        version_obj = PackageVersion.parse(version)
        return Package(name, version_obj, description, author)
    
    def get_package(self, name: str, version: Optional[str] = None) -> Optional[Package]:
        """
        Get a package by name and optional version.
        
        Args:
            name: The name of the package
            version: The version or version range (optional)
            
        Returns:
            The package if found, None otherwise
        """
        return self.registry.get_package(name, version)
    
    def list_packages(self) -> Dict[str, List[Package]]:
        """
        List all installed packages.
        
        Returns:
            A dictionary of package names to lists of packages
        """
        return self.registry.packages
    
    def resolve_dependencies(self, package: Package) -> Dict[str, Package]:
        """
        Resolve the dependencies of a package.
        
        Args:
            package: The package to resolve dependencies for
            
        Returns:
            A dictionary of package names to packages
            
        Raises:
            ValueError: If a dependency cannot be resolved
        """
        return self.registry.resolve_dependencies(package)
    
    def pack(self, package_dir: str, output_dir: Optional[str] = None, format: str = "zip") -> str:
        """
        Pack a package directory into an archive.
        
        Args:
            package_dir: Path to the package directory
            output_dir: Directory to save the archive (optional)
            format: Archive format ("zip" or "tar.gz")
            
        Returns:
            The path to the archive
            
        Raises:
            ValueError: If the package cannot be packed
        """
        # Load the package manifest
        package = Package.load_manifest(package_dir)
        
        # Use the package directory for output if none specified
        output_dir = output_dir or os.path.dirname(package_dir)
        
        # Create the output filename
        output_filename = f"{package.name}-{package.version}"
        
        # Create the archive
        if format == "zip":
            output_path = os.path.join(output_dir, f"{output_filename}.zip")
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zip_ref:
                for root, _, files in os.walk(package_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, package_dir)
                        zip_ref.write(file_path, arcname)
        elif format == "tar.gz":
            output_path = os.path.join(output_dir, f"{output_filename}.tar.gz")
            with tarfile.open(output_path, "w:gz") as tar_ref:
                for root, _, files in os.walk(package_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, package_dir)
                        tar_ref.add(file_path, arcname=arcname)
        else:
            raise ValueError(f"Unsupported archive format: {format}")
        
        return output_path
