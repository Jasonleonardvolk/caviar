"""
Dependency resolver for ELFIN packages.

This module provides tools for resolving package dependencies, similar
to how Cargo resolves dependencies in Rust.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Tuple
import networkx as nx

try:
    import semver
except ImportError:
    raise ImportError("semver package is required. Install with: pip install semver")

from .manifest import Manifest, Dependency
from .lockfile import Lockfile, ResolvedDependency, PackageId


logger = logging.getLogger(__name__)


class ResolutionError(Exception):
    """Exception raised for dependency resolution errors."""
    pass


@dataclass
class PackageInfo:
    """Information about an available package."""
    name: str
    versions: List[str]
    yanked_versions: Set[str] = field(default_factory=set)
    
    def get_latest_matching(self, version_req: str) -> Optional[str]:
        """
        Get latest version matching a requirement.
        
        Args:
            version_req: Version requirement string
            
        Returns:
            Version string or None if no match
        """
        matches = []
        for ver in self.versions:
            if ver in self.yanked_versions:
                continue
                
            try:
                if semver.VersionInfo.parse(ver).match(version_req):
                    matches.append(ver)
            except ValueError:
                logger.warning(f"Invalid version: {ver}")
        
        if not matches:
            return None
        
        # Sort by semver and return latest
        return sorted(matches, key=lambda v: semver.VersionInfo.parse(v), reverse=True)[0]


class DependencyResolver:
    """
    Resolves package dependencies to compatible versions.
    
    This implements Cargo-like resolution with SemVer ranges, allowing
    multiple versions of a package only when required by version constraints.
    """
    
    def __init__(self, registry_client=None):
        """
        Initialize resolver.
        
        Args:
            registry_client: Client for fetching packages from the registry
        """
        self.registry_client = registry_client
        self.package_cache: Dict[str, PackageInfo] = {}
    
    def add_package_info(self, package_info: PackageInfo) -> None:
        """
        Add package information to the cache.
        
        Args:
            package_info: Information about an available package
        """
        self.package_cache[package_info.name] = package_info
    
    def get_package_info(self, name: str) -> Optional[PackageInfo]:
        """
        Get information about a package.
        
        Args:
            name: Package name
            
        Returns:
            PackageInfo or None if not found
        """
        # Check cache first
        if name in self.package_cache:
            return self.package_cache[name]
        
        # Fetch from registry if available
        if self.registry_client:
            try:
                package_info = self.registry_client.get_package_info(name)
                if package_info:
                    self.package_cache[name] = package_info
                    return package_info
            except Exception as e:
                logger.error(f"Failed to fetch package info for {name}: {e}")
        
        return None
    
    def resolve(
        self,
        manifest: Manifest,
        existing_lockfile: Optional[Lockfile] = None
    ) -> Dict[str, ResolvedDependency]:
        """
        Resolve dependencies from a manifest.
        
        Args:
            manifest: Project manifest
            existing_lockfile: Existing lockfile to reuse resolved deps when possible
            
        Returns:
            Mapping of package IDs to resolved dependencies
            
        Raises:
            ResolutionError: If dependencies can't be resolved
        """
        # Build dependency graph
        graph = nx.DiGraph()
        
        # Add root node
        root_id = f"{manifest.name}:{manifest.version}"
        graph.add_node(root_id, name=manifest.name, version=manifest.version, is_root=True)
        
        # Process direct dependencies
        for dep_name, dep in manifest.dependencies.items():
            try:
                self._add_dependency(graph, root_id, dep, existing_lockfile)
            except ResolutionError as e:
                raise ResolutionError(f"Failed to resolve dependency {dep_name}: {e}")
        
        # Extract resolved dependencies
        resolved_deps = {}
        for node in graph.nodes:
            if node == root_id:  # Skip root node
                continue
                
            node_data = graph.nodes[node]
            name = node_data["name"]
            version = node_data["version"]
            
            # Get dependencies of this package
            dependencies = []
            for _, child, _ in graph.edges(node, data=True):
                dependencies.append(child)
            
            # Create resolved dependency
            resolved_deps[node] = ResolvedDependency(
                name=name,
                version=version,
                dependencies=dependencies,
                features=node_data.get("features", [])
            )
        
        return resolved_deps
    
    def _add_dependency(
        self,
        graph: nx.DiGraph,
        parent_id: str,
        dependency: Dependency,
        existing_lockfile: Optional[Lockfile] = None
    ) -> None:
        """
        Add a dependency to the graph.
        
        Args:
            graph: Dependency graph
            parent_id: ID of parent package
            dependency: Dependency to add
            existing_lockfile: Existing lockfile to reuse versions
            
        Raises:
            ResolutionError: If dependency can't be resolved
        """
        # Get package info
        package_info = self.get_package_info(dependency.name)
        if not package_info:
            raise ResolutionError(f"Package {dependency.name} not found")
        
        # Check if we can reuse version from lockfile
        version = None
        if existing_lockfile:
            resolved_dep = existing_lockfile.get_dependency(dependency.name)
            if resolved_dep and dependency.matches(resolved_dep.version):
                version = resolved_dep.version
        
        # Otherwise find latest matching version
        if not version:
            version = package_info.get_latest_matching(dependency.version_req)
            if not version:
                raise ResolutionError(
                    f"No version of {dependency.name} matches requirement {dependency.version_req}"
                )
        
        # Create node ID
        node_id = f"{dependency.name}:{version}"
        
        # Check for version conflicts
        for existing_node in graph.nodes:
            if existing_node.startswith(f"{dependency.name}:") and existing_node != node_id:
                # Allow different major versions to coexist
                existing_version = existing_node.split(":")[1]
                
                # Parse versions
                v1 = semver.VersionInfo.parse(version)
                v2 = semver.VersionInfo.parse(existing_version)
                
                # If same major version but different minor, it's a conflict
                if v1.major >= 1 and v2.major >= 1 and v1.major == v2.major and v1.minor != v2.minor:
                    raise ResolutionError(
                        f"Conflicting versions of {dependency.name}: {version} vs {existing_version}"
                    )
        
        # Add node if it doesn't exist
        if node_id not in graph.nodes:
            graph.add_node(
                node_id,
                name=dependency.name,
                version=version,
                features=dependency.features
            )
            
            # Add dependencies of this package
            # In a real implementation, we would fetch the manifest for this version
            # and recursively process its dependencies
            if self.registry_client:
                try:
                    deps = self.registry_client.get_dependencies(dependency.name, version)
                    for dep in deps:
                        self._add_dependency(graph, node_id, dep, existing_lockfile)
                except Exception as e:
                    logger.warning(f"Failed to fetch dependencies for {dependency.name}@{version}: {e}")
        
        # Add edge from parent to this node
        graph.add_edge(parent_id, node_id)


def resolve_dependencies(
    manifest: Manifest,
    registry_client=None,
    existing_lockfile: Optional[Path] = None
) -> Dict[str, ResolvedDependency]:
    """
    Resolve dependencies for a manifest.
    
    Args:
        manifest: Project manifest
        registry_client: Client for fetching packages from the registry
        existing_lockfile: Path to existing lockfile
        
    Returns:
        Mapping of package IDs to resolved dependencies
        
    Raises:
        ResolutionError: If dependencies can't be resolved
    """
    resolver = DependencyResolver(registry_client)
    
    # Load existing lockfile if available
    lock = None
    if existing_lockfile and existing_lockfile.exists():
        try:
            lock = Lockfile.load(existing_lockfile)
        except Exception as e:
            logger.warning(f"Failed to load existing lockfile: {e}")
    
    # Resolve dependencies
    try:
        return resolver.resolve(manifest, lock)
    except ResolutionError as e:
        raise ResolutionError(f"Dependency resolution failed: {e}")
