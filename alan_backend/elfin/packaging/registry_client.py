"""
Registry client for ELFIN packages.

This module provides tools for interacting with the ELFIN package registry,
allowing packages to be published, downloaded, and updated.
"""

import json
import os
import hashlib
import logging
import tempfile
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import requests

try:
    import semver
except ImportError:
    raise ImportError("semver package is required. Install with: pip install semver")

from .manifest import Manifest, Dependency
from .resolver import PackageInfo


logger = logging.getLogger(__name__)


class RegistryError(Exception):
    """Exception raised for registry operations."""
    pass


class RegistryClient:
    """
    Client for interacting with the ELFIN package registry.
    
    The registry uses a git-backed index similar to crates.io, with package
    metadata stored in JSON files and package archives stored in a blob store.
    """
    
    def __init__(self, registry_url: str, api_token: Optional[str] = None):
        """
        Initialize registry client.
        
        Args:
            registry_url: URL of the registry
            api_token: API token for authenticated operations
        """
        self.registry_url = registry_url.rstrip('/')
        self.api_token = api_token
        self.index_url = f"{self.registry_url}/index"
        self.api_url = f"{self.registry_url}/api/v1"
        self.blob_url = f"{self.registry_url}/blobs"
    
    def get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers for API requests.
        
        Returns:
            Dictionary of headers
        """
        headers = {
            "Accept": "application/json",
            "User-Agent": "elfin-registry-client/0.1.0"
        }
        
        if self.api_token:
            headers["Authorization"] = f"Token {self.api_token}"
        
        return headers
    
    def get_package_info(self, name: str) -> Optional[PackageInfo]:
        """
        Get information about a package.
        
        Args:
            name: Package name
            
        Returns:
            PackageInfo or None if not found
            
        Raises:
            RegistryError: If request fails
        """
        # Compute package path in index
        path = self._get_package_index_path(name)
        url = f"{self.index_url}/{path}"
        
        try:
            response = requests.get(url, headers=self.get_headers())
            
            if response.status_code == 404:
                return None
            
            if response.status_code != 200:
                raise RegistryError(f"Failed to get package info: {response.text}")
            
            # Parse JSON
            data = response.json()
            
            # Extract versions
            versions = []
            yanked = set()
            
            for version_data in data.get("versions", []):
                version = version_data.get("version")
                if version:
                    versions.append(version)
                    if version_data.get("yanked", False):
                        yanked.add(version)
            
            return PackageInfo(
                name=name,
                versions=sorted(versions, key=lambda v: semver.VersionInfo.parse(v), reverse=True),
                yanked_versions=yanked
            )
            
        except requests.RequestException as e:
            raise RegistryError(f"Failed to get package info: {e}")
    
    def get_dependencies(self, name: str, version: str) -> List[Dependency]:
        """
        Get dependencies for a package version.
        
        Args:
            name: Package name
            version: Package version
            
        Returns:
            List of dependencies
            
        Raises:
            RegistryError: If request fails
        """
        # Compute package path in index
        path = self._get_package_index_path(name)
        url = f"{self.index_url}/{path}"
        
        try:
            response = requests.get(url, headers=self.get_headers())
            
            if response.status_code == 404:
                raise RegistryError(f"Package {name} not found")
            
            if response.status_code != 200:
                raise RegistryError(f"Failed to get package info: {response.text}")
            
            # Parse JSON
            data = response.json()
            
            # Find version
            for version_data in data.get("versions", []):
                if version_data.get("version") == version:
                    # Extract dependencies
                    dependencies = []
                    for dep_name, dep_req in version_data.get("dependencies", {}).items():
                        if isinstance(dep_req, str):
                            # Simple dependency
                            dependencies.append(Dependency(dep_name, dep_req))
                        elif isinstance(dep_req, dict):
                            # Complex dependency
                            dependencies.append(Dependency(
                                dep_name,
                                dep_req.get("version", "*"),
                                optional=dep_req.get("optional", False),
                                features=dep_req.get("features", [])
                            ))
                    
                    return dependencies
            
            raise RegistryError(f"Version {version} not found for package {name}")
            
        except requests.RequestException as e:
            raise RegistryError(f"Failed to get dependencies: {e}")
    
    def download_package(self, name: str, version: str, dest_dir: Path) -> Path:
        """
        Download a package from the registry.
        
        Args:
            name: Package name
            version: Package version
            dest_dir: Destination directory
            
        Returns:
            Path to downloaded package archive
            
        Raises:
            RegistryError: If download fails
        """
        # Get package info to verify version exists
        package_info = self.get_package_info(name)
        if not package_info:
            raise RegistryError(f"Package {name} not found")
        
        if version not in package_info.versions:
            raise RegistryError(f"Version {version} not found for package {name}")
        
        # Compute blob URL
        checksum = self._get_package_checksum(name, version)
        if not checksum:
            raise RegistryError(f"Checksum not found for {name}@{version}")
        
        url = f"{self.blob_url}/{checksum[:2]}/{checksum[2:4]}/{checksum}"
        
        # Download package
        try:
            response = requests.get(url, headers=self.get_headers(), stream=True)
            
            if response.status_code != 200:
                raise RegistryError(f"Failed to download package: {response.text}")
            
            # Save to file
            dest_file = dest_dir / f"{name}-{version}.tar.gz"
            with open(dest_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify checksum
            computed_checksum = self._compute_file_checksum(dest_file)
            if computed_checksum != checksum:
                os.unlink(dest_file)
                raise RegistryError(f"Checksum mismatch: expected {checksum}, got {computed_checksum}")
            
            return dest_file
            
        except requests.RequestException as e:
            raise RegistryError(f"Failed to download package: {e}")
    
    def extract_package(self, archive_path: Path, dest_dir: Path) -> Path:
        """
        Extract a package archive.
        
        Args:
            archive_path: Path to package archive
            dest_dir: Destination directory
            
        Returns:
            Path to extracted package directory
            
        Raises:
            RegistryError: If extraction fails
        """
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                # Extract to a temporary directory first
                with tempfile.TemporaryDirectory() as temp_dir:
                    tar.extractall(temp_dir)
                    
                    # Check extracted contents
                    temp_path = Path(temp_dir)
                    pkg_dir = next(temp_path.iterdir())
                    
                    # Check for manifest
                    manifest_path = pkg_dir / "elfpkg.toml"
                    if not manifest_path.exists():
                        raise RegistryError("Invalid package: missing elfpkg.toml")
                    
                    # Create destination directory
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    # Move contents to destination
                    for item in pkg_dir.iterdir():
                        shutil.move(str(item), str(dest_dir / item.name))
            
            return dest_dir
        
        except Exception as e:
            raise RegistryError(f"Failed to extract package: {e}")
    
    def publish_package(self, package_dir: Path) -> None:
        """
        Publish a package to the registry.
        
        Args:
            package_dir: Directory containing the package
            
        Raises:
            RegistryError: If publish fails
        """
        # Check for manifest
        manifest_path = package_dir / "elfpkg.toml"
        if not manifest_path.exists():
            raise RegistryError("Invalid package: missing elfpkg.toml")
        
        try:
            # Load manifest
            manifest = Manifest.load(manifest_path)
        except Exception as e:
            raise RegistryError(f"Failed to load manifest: {e}")
        
        # Create package archive
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = temp_path / f"{manifest.name}-{manifest.version}.tar.gz"
            
            try:
                # Create archive
                with tarfile.open(archive_path, 'w:gz') as tar:
                    tar.add(package_dir, arcname=f"{manifest.name}-{manifest.version}")
                
                # Compute checksum
                checksum = self._compute_file_checksum(archive_path)
                
                # Upload to blob store
                self._upload_blob(archive_path, checksum)
                
                # Update index
                self._update_index(manifest, checksum)
                
            except Exception as e:
                raise RegistryError(f"Failed to publish package: {e}")
    
    def _get_package_index_path(self, name: str) -> str:
        """
        Get path to package in the index.
        
        Args:
            name: Package name
            
        Returns:
            Path in the index
        """
        if len(name) <= 3:
            return f"{len(name)}/{name}"
        else:
            return f"{name[0]}/{name[1]}/{name[2]}/{name}"
    
    def _get_package_checksum(self, name: str, version: str) -> Optional[str]:
        """
        Get checksum for a package version.
        
        Args:
            name: Package name
            version: Package version
            
        Returns:
            Checksum or None if not found
            
        Raises:
            RegistryError: If request fails
        """
        # Compute package path in index
        path = self._get_package_index_path(name)
        url = f"{self.index_url}/{path}"
        
        try:
            response = requests.get(url, headers=self.get_headers())
            
            if response.status_code == 404:
                return None
            
            if response.status_code != 200:
                raise RegistryError(f"Failed to get package info: {response.text}")
            
            # Parse JSON
            data = response.json()
            
            # Find version
            for version_data in data.get("versions", []):
                if version_data.get("version") == version:
                    return version_data.get("checksum")
            
            return None
            
        except requests.RequestException as e:
            raise RegistryError(f"Failed to get package checksum: {e}")
    
    def _compute_file_checksum(self, file_path: Path) -> str:
        """
        Compute SHA-256 checksum of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Checksum as hex string
        """
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    
    def _upload_blob(self, file_path: Path, checksum: str) -> None:
        """
        Upload a file to the blob store.
        
        Args:
            file_path: Path to file
            checksum: File checksum
            
        Raises:
            RegistryError: If upload fails
        """
        # Compute blob URL
        url = f"{self.api_url}/blobs/{checksum}"
        
        try:
            with open(file_path, 'rb') as f:
                response = requests.put(
                    url,
                    headers=self.get_headers(),
                    data=f
                )
                
                if response.status_code not in (200, 201):
                    raise RegistryError(f"Failed to upload blob: {response.text}")
                
        except requests.RequestException as e:
            raise RegistryError(f"Failed to upload blob: {e}")
    
    def _update_index(self, manifest: Manifest, checksum: str) -> None:
        """
        Update package index with a new version.
        
        Args:
            manifest: Package manifest
            checksum: Package archive checksum
            
        Raises:
            RegistryError: If update fails
        """
        # Compute package path in index
        path = self._get_package_index_path(manifest.name)
        url = f"{self.api_url}/index/{path}"
        
        # Prepare version data
        version_data = {
            "name": manifest.name,
            "version": manifest.version,
            "checksum": checksum,
            "dependencies": {},
            "features": manifest.features
        }
        
        # Add dependencies
        for dep_name, dep in manifest.dependencies.items():
            if dep.features or dep.optional:
                # Complex dependency
                version_data["dependencies"][dep_name] = {
                    "version": dep.version_req
                }
                if dep.features:
                    version_data["dependencies"][dep_name]["features"] = dep.features
                if dep.optional:
                    version_data["dependencies"][dep_name]["optional"] = True
            else:
                # Simple dependency
                version_data["dependencies"][dep_name] = dep.version_req
        
        try:
            # Check if package already exists
            response = requests.get(f"{self.index_url}/{path}", headers=self.get_headers())
            
            data = {}
            if response.status_code == 200:
                # Package exists, update
                data = response.json()
                
                # Check if version already exists
                for i, ver in enumerate(data.get("versions", [])):
                    if ver.get("version") == manifest.version:
                        # Replace existing version
                        data["versions"][i] = version_data
                        break
                else:
                    # Add new version
                    if "versions" not in data:
                        data["versions"] = []
                    data["versions"].append(version_data)
            else:
                # Create new package
                data = {
                    "name": manifest.name,
                    "versions": [version_data]
                }
            
            # Update index
            response = requests.put(
                url,
                headers=self.get_headers(),
                json=data
            )
            
            if response.status_code not in (200, 201):
                raise RegistryError(f"Failed to update index: {response.text}")
            
        except requests.RequestException as e:
            raise RegistryError(f"Failed to update index: {e}")
