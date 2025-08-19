"""
ELFIN Packaging - Package management system for ELFIN projects.

This module provides tools for managing ELFIN packages, dependencies,
and versioning in a manner similar to Cargo for Rust.
"""

from .manifest import Manifest, ManifestError
from .lockfile import Lockfile, LockfileError
from .resolver import DependencyResolver, ResolutionError
from .registry_client import RegistryClient

__all__ = [
    'Manifest',
    'ManifestError',
    'Lockfile',
    'LockfileError',
    'DependencyResolver',
    'ResolutionError',
    'RegistryClient',
]
