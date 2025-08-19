"""
Setup utilities for ELFIN package ecosystem.

This module provides helper functions for initial setup of the package registry,
workspace configuration, and other setup tasks.
"""

import os
import sys
import shutil
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import toml
import requests
import tarfile
import tempfile

from .manifest import Manifest, Dependency

logger = logging.getLogger(__name__)


def initialize_registry(registry_dir: Path) -> None:
    """
    Initialize a local package registry directory structure.
    
    This creates the directory structure needed for a Git-backed registry index
    and blob store, similar to crates.io's design.
    
    Args:
        registry_dir: Base directory for the registry
        
    Raises:
        FileExistsError: If registry directory already exists and is not empty
    """
    # Ensure registry directory exists
    if registry_dir.exists() and any(registry_dir.iterdir()):
        raise FileExistsError(f"Registry directory already exists and is not empty: {registry_dir}")
    
    registry_dir.mkdir(exist_ok=True)
    
    # Create index directory structure
    index_dir = registry_dir / "index"
    index_dir.mkdir(exist_ok=True)
    
    # Create directories for 1, 2, 3-letter packages
    for i in range(1, 4):
        (index_dir / str(i)).mkdir(exist_ok=True)
    
    # Create directories for longer packages
    for c in "abcdefghijklmnopqrstuvwxyz":
        c_dir = index_dir / c
        c_dir.mkdir(exist_ok=True)
        
        for c2 in "abcdefghijklmnopqrstuvwxyz":
            c2_dir = c_dir / c2
            c2_dir.mkdir(exist_ok=True)
            
            # Create 3rd level directories
            for c3 in "abcdefghijklmnopqrstuvwxyz":
                (c2_dir / c3).mkdir(exist_ok=True)
    
    # Create blobs directory with structure for efficient lookup
    blobs_dir = registry_dir / "blobs"
    blobs_dir.mkdir(exist_ok=True)
    
    # Create prefix directories for efficient blob lookup
    # Using first 4 chars of SHA-256 as prefix (2 levels of 2 chars each)
    for c1 in "0123456789abcdef":
        for c2 in "0123456789abcdef":
            prefix_dir = blobs_dir / f"{c1}{c2}"
            prefix_dir.mkdir(exist_ok=True)
            
            for c3 in "0123456789abcdef":
                for c4 in "0123456789abcdef":
                    (prefix_dir / f"{c3}{c4}").mkdir(exist_ok=True)
    
    # Create config file
    config = {
        "index_version": "1",
        "name": "ELFIN Registry",
        "description": "Local ELFIN package registry"
    }
    
    with open(index_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Initialized registry at {registry_dir}")


def seed_registry_with_core_packages(registry_dir: Path, packages: List[Dict[str, Any]]) -> None:
    """
    Seed the registry with core packages.
    
    Args:
        registry_dir: Base directory for the registry
        packages: List of package metadata dictionaries
            Each dictionary should have:
            - name: Package name
            - version: Package version
            - dependencies: Dictionary of dependencies
            - source_dir: Path to source directory (optional)
            - archive: Path to archive file (optional)
            
    Raises:
        ValueError: If package data is invalid
    """
    index_dir = registry_dir / "index"
    blobs_dir = registry_dir / "blobs"
    
    for pkg_data in packages:
        name = pkg_data.get("name")
        version = pkg_data.get("version")
        
        if not name or not version:
            raise ValueError("Package missing name or version")
        
        # Compute index path
        if len(name) <= 3:
            pkg_path = index_dir / str(len(name)) / name
        else:
            pkg_path = index_dir / name[0] / name[1] / name[2] / name
        
        # Create package directory
        pkg_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create or update package index entry
        if pkg_path.exists():
            # Update existing package
            with open(pkg_path, "r") as f:
                data = json.load(f)
        else:
            # Create new package entry
            data = {
                "name": name,
                "versions": []
            }
        
        # Check if version already exists
        for i, ver in enumerate(data.get("versions", [])):
            if ver.get("version") == version:
                # Remove existing version
                data["versions"].pop(i)
                break
        
        # Process source directory or archive
        checksum = None
        source_dir = pkg_data.get("source_dir")
        archive = pkg_data.get("archive")
        
        if source_dir:
            # Create temporary archive from source directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                archive_path = temp_path / f"{name}-{version}.tar.gz"
                
                with tarfile.open(archive_path, "w:gz") as tar:
                    tar.add(source_dir, arcname=f"{name}-{version}")
                
                # Compute checksum
                import hashlib
                with open(archive_path, "rb") as f:
                    checksum = hashlib.sha256(f.read()).hexdigest()
                
                # Copy to blobs directory
                blob_path = blobs_dir / checksum[:2] / checksum[2:4] / checksum
                shutil.copy2(archive_path, blob_path)
        
        elif archive:
            # Use existing archive
            if not Path(archive).exists():
                raise ValueError(f"Archive file not found: {archive}")
            
            # Compute checksum
            import hashlib
            with open(archive, "rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            
            # Copy to blobs directory
            blob_path = blobs_dir / checksum[:2] / checksum[2:4] / checksum
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive, blob_path)
        
        # Create version entry
        version_data = {
            "name": name,
            "version": version,
            "checksum": checksum
        }
        
        # Add dependencies
        deps = pkg_data.get("dependencies", {})
        if deps:
            version_data["dependencies"] = deps
        
        # Add features
        features = pkg_data.get("features", {})
        if features:
            version_data["features"] = features
        
        # Add to versions list
        if "versions" not in data:
            data["versions"] = []
        data["versions"].append(version_data)
        
        # Write to file
        with open(pkg_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Added package to registry: {name}@{version}")


def create_template_project(
    template_dir: Path, 
    output_dir: Path, 
    name: str,
    template_type: str = "basic",
    edition: str = "elfin-1.0",
    authors: Optional[List[str]] = None
) -> None:
    """
    Create a new project from a template.
    
    Args:
        template_dir: Directory containing templates
        output_dir: Directory to create the project in
        name: Project name
        template_type: Template type (basic, application, library)
        edition: ELFIN edition
        authors: List of author names
        
    Raises:
        ValueError: If template type is invalid
    """
    # Validate template type
    if template_type not in ("basic", "application", "library"):
        raise ValueError(f"Invalid template type: {template_type}")
    
    # Ensure template directory exists
    template_path = template_dir / template_type
    if not template_path.exists():
        raise ValueError(f"Template directory not found: {template_path}")
    
    # Create project directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy template files
    for item in template_path.iterdir():
        if item.is_file():
            shutil.copy2(item, output_dir / item.name)
        elif item.is_dir():
            shutil.copytree(item, output_dir / item.name)
    
    # Create or update manifest
    manifest = Manifest(
        name=name,
        version="0.1.0",
        authors=authors or [],
        edition=edition
    )
    
    # Add template-specific dependencies
    if template_type == "basic":
        manifest.dependencies["elfin-core"] = Dependency("elfin-core", "^1.0.0")
    elif template_type == "application":
        manifest.dependencies["elfin-core"] = Dependency("elfin-core", "^1.0.0")
        manifest.dependencies["elfin-ui"] = Dependency("elfin-ui", "^1.0.0")
    elif template_type == "library":
        manifest.dependencies["elfin-core"] = Dependency("elfin-core", "^1.0.0")
    
    # Save manifest
    manifest.save(output_dir / "elfpkg.toml")
    
    # Update template placeholders in files
    for root, _, files in os.walk(output_dir):
        for filename in files:
            if filename.endswith((".py", ".md", ".txt", ".toml")):
                file_path = Path(root) / filename
                
                # Read file
                with open(file_path, "r") as f:
                    content = f.read()
                
                # Replace placeholders
                content = content.replace("{{name}}", name)
                content = content.replace("{{edition}}", edition)
                
                # Write back
                with open(file_path, "w") as f:
                    f.write(content)
    
    logger.info(f"Created {template_type} project: {name} in {output_dir}")


def download_core_packages(output_dir: Path) -> List[Dict[str, Any]]:
    """
    Download core packages needed for the registry.
    
    This downloads the essential packages like elfin-core and elfin-ui
    from the official registry or GitHub.
    
    Args:
        output_dir: Directory to save downloaded packages
        
    Returns:
        List of package metadata dictionaries
        
    Raises:
        ConnectionError: If download fails
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    packages = []
    
    # Define core packages to download
    core_packages = [
        {
            "name": "elfin-core",
            "version": "1.0.0",
            "url": "https://github.com/elfin-language/elfin-core/archive/v1.0.0.tar.gz"
        },
        {
            "name": "elfin-ui",
            "version": "1.0.0",
            "url": "https://github.com/elfin-language/elfin-ui/archive/v1.0.0.tar.gz"
        }
    ]
    
    for pkg in core_packages:
        try:
            # Download package
            logger.info(f"Downloading {pkg['name']}@{pkg['version']}...")
            
            archive_path = output_dir / f"{pkg['name']}-{pkg['version']}.tar.gz"
            
            response = requests.get(pkg["url"], stream=True)
            response.raise_for_status()
            
            with open(archive_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Add to packages list
            packages.append({
                "name": pkg["name"],
                "version": pkg["version"],
                "archive": archive_path
            })
            
            logger.info(f"Downloaded {pkg['name']}@{pkg['version']}")
            
        except requests.RequestException as e:
            logger.warning(f"Failed to download {pkg['name']}@{pkg['version']}: {e}")
    
    return packages
