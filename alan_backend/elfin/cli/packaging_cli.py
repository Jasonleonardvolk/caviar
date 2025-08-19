"""
CLI commands for the ELFIN package manager.

This module provides command-line interface commands for managing ELFIN
packages, similar to Cargo commands in Rust.
"""

import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import click

from alan_backend.elfin.packaging.manifest import Manifest, ManifestError
from alan_backend.elfin.packaging.lockfile import Lockfile, generate_lockfile
from alan_backend.elfin.packaging.resolver import resolve_dependencies, ResolutionError
from alan_backend.elfin.packaging.registry_client import RegistryClient, RegistryError
from alan_backend.elfin.packaging.setup import (
    initialize_registry,
    seed_registry_with_core_packages,
    download_core_packages,
    create_template_project
)


logger = logging.getLogger(__name__)


def get_registry_client() -> RegistryClient:
    """
    Get a registry client.
    
    Returns:
        Registry client
    """
    # Get registry URL from environment or use default
    registry_url = os.environ.get('ELFIN_REGISTRY_URL', 'https://registry.elfin.dev')
    
    # Get API token from environment (if available)
    api_token = os.environ.get('ELFIN_REGISTRY_TOKEN')
    
    return RegistryClient(registry_url, api_token)


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """ELFIN package manager CLI."""
    pass


@cli.command()
@click.argument('name')
@click.option('--template', '-t', default='basic',
              help='Template to use (basic, application, library)')
@click.option('--edition', default='elfin-1.0',
              help='ELFIN edition to use')
def new(name: str, template: str, edition: str):
    """Create a new ELFIN package."""
    click.echo(f"Creating new package: {name}")
    
    # Create directory
    pkg_dir = Path(name)
    if pkg_dir.exists():
        click.echo(f"Error: {pkg_dir} already exists", err=True)
        sys.exit(1)
    
    pkg_dir.mkdir()
    click.echo(f"Created directory: {pkg_dir}")
    
    # Create manifest
    manifest = Manifest(
        name=name,
        version="0.1.0",
        authors=[],
        edition=edition
    )
    
    # Add template-specific dependencies
    if template == 'basic':
        manifest.dependencies['elfin-core'] = 'Dependency("elfin-core", "^1.0.0")'
    elif template == 'application':
        manifest.dependencies['elfin-core'] = 'Dependency("elfin-core", "^1.0.0")'
        manifest.dependencies['elfin-ui'] = 'Dependency("elfin-ui", "^1.0.0")'
    elif template == 'library':
        manifest.dependencies['elfin-core'] = 'Dependency("elfin-core", "^1.0.0")'
    
    # Save manifest
    try:
        manifest.save(pkg_dir / 'elfpkg.toml')
        click.echo(f"Created manifest: {pkg_dir}/elfpkg.toml")
    except ManifestError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    
    # Create template-specific files
    if template == 'basic':
        # Create basic template files
        with open(pkg_dir / 'README.md', 'w') as f:
            f.write(f"# {name}\n\nA basic ELFIN package.\n")
        with open(pkg_dir / 'src/main.py', 'w') as f:
            f.write('"""Main entry point for the package."""\n\n')
            f.write('def main():\n')
            f.write('    """Run the package."""\n')
            f.write('    print("Hello from ELFIN!")\n\n')
            f.write('if __name__ == "__main__":\n')
            f.write('    main()\n')
        
    elif template == 'application':
        # Create application template files
        with open(pkg_dir / 'README.md', 'w') as f:
            f.write(f"# {name}\n\nAn ELFIN application.\n")
        with open(pkg_dir / 'src/main.py', 'w') as f:
            f.write('"""Main entry point for the application."""\n\n')
            f.write('from elfin_core import get_core\n')
            f.write('from elfin_ui import get_ui\n\n')
            f.write('def main():\n')
            f.write('    """Run the application."""\n')
            f.write('    core = get_core()\n')
            f.write('    ui = get_ui()\n')
            f.write('    print("Hello from ELFIN Application!")\n\n')
            f.write('if __name__ == "__main__":\n')
            f.write('    main()\n')
        
    elif template == 'library':
        # Create library template files
        with open(pkg_dir / 'README.md', 'w') as f:
            f.write(f"# {name}\n\nAn ELFIN library.\n")
        with open(pkg_dir / 'src/lib.py', 'w') as f:
            f.write(f'"""{name} library."""\n\n')
            f.write('from elfin_core import get_core\n\n')
            f.write('def get_lib():\n')
            f.write('    """Get the library."""\n')
            f.write('    core = get_core()\n')
            f.write('    return "ELFIN Library"\n')
    
    click.echo(f"Created {template} template files")
    
    click.echo(f"\nSuccessfully created package: {name}")
    click.echo(f"\nTo get started, run:\n    cd {name}\n    elf build")


@cli.command()
@click.option('--release', is_flag=True, help='Build in release mode')
def build(release: bool):
    """Build the package."""
    click.echo("Building package...")
    
    # Check for manifest
    manifest_path = Path('elfpkg.toml')
    if not manifest_path.exists():
        click.echo("Error: No elfpkg.toml found in current directory", err=True)
        sys.exit(1)
    
    # Load manifest
    try:
        manifest = Manifest.load(manifest_path)
        click.echo(f"Loaded manifest for package: {manifest.name} v{manifest.version}")
    except ManifestError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    
    # Resolve dependencies
    try:
        click.echo("Resolving dependencies...")
        registry_client = get_registry_client()
        deps = resolve_dependencies(manifest, registry_client, Path('elf.lock'))
        click.echo(f"Resolved {len(deps)} dependencies")
        
        # Generate lockfile if it doesn't exist
        lockfile_path = Path('elf.lock')
        if not lockfile_path.exists():
            click.echo("Generating lockfile...")
            generate_lockfile(manifest, deps, lockfile_path)
            click.echo(f"Created lockfile: {lockfile_path}")
    except ResolutionError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    
    # Build package
    click.echo("Building package...")
    
    # Check for src directory
    src_dir = Path('src')
    if not src_dir.exists() or not src_dir.is_dir():
        click.echo("Error: No src directory found", err=True)
        sys.exit(1)
    
    # Create build directory
    build_dir = Path('target/release' if release else 'target/debug')
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy source files to build directory
    for file in src_dir.glob('**/*'):
        if file.is_file():
            rel_path = file.relative_to(src_dir)
            dest_path = build_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest_path)
    
    click.echo(f"Built package: {manifest.name} v{manifest.version}")
    click.echo(f"Output directory: {build_dir}")


@cli.command()
@click.option('--dry-run', is_flag=True, help='Validate package without publishing')
def publish(dry_run: bool):
    """Publish the package to the registry."""
    click.echo("Publishing package...")
    
    # Check for manifest
    manifest_path = Path('elfpkg.toml')
    if not manifest_path.exists():
        click.echo("Error: No elfpkg.toml found in current directory", err=True)
        sys.exit(1)
    
    # Load manifest
    try:
        manifest = Manifest.load(manifest_path)
        click.echo(f"Loaded manifest for package: {manifest.name} v{manifest.version}")
    except ManifestError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    
    if dry_run:
        click.echo("Dry run, not publishing package")
        sys.exit(0)
    
    # Publish package
    try:
        registry_client = get_registry_client()
        click.echo(f"Publishing package to registry: {manifest.name} v{manifest.version}")
        registry_client.publish_package(Path('.'))
        click.echo(f"Successfully published: {manifest.name} v{manifest.version}")
    except RegistryError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('package', required=False)
@click.option('--version', '-v', help='Version to install')
@click.option('--no-deps', is_flag=True, help='Do not install dependencies')
def install(package: Optional[str], version: Optional[str], no_deps: bool):
    """Install a package from the registry."""
    if package:
        # Install specific package
        click.echo(f"Installing package: {package}")
        
        # Use version if specified, otherwise use latest
        version_str = f"@{version}" if version else ""
        click.echo(f"Installing: {package}{version_str}")
        
        # Get registry client
        registry_client = get_registry_client()
        
        try:
            # Get package info
            package_info = registry_client.get_package_info(package)
            if not package_info:
                click.echo(f"Error: Package not found: {package}", err=True)
                sys.exit(1)
            
            # Get version to install
            if version:
                if version not in package_info.versions:
                    click.echo(f"Error: Version not found: {version}", err=True)
                    sys.exit(1)
                install_version = version
            else:
                if not package_info.versions:
                    click.echo(f"Error: No versions available for package: {package}", err=True)
                    sys.exit(1)
                install_version = package_info.versions[0]
            
            click.echo(f"Selected version: {install_version}")
            
            # Create packages directory if it doesn't exist
            packages_dir = Path('.elfin/packages')
            packages_dir.mkdir(parents=True, exist_ok=True)
            
            # Download package
            click.echo(f"Downloading: {package}@{install_version}")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                archive_path = registry_client.download_package(package, install_version, temp_path)
                
                # Extract package
                package_dir = packages_dir / f"{package}-{install_version}"
                click.echo(f"Extracting to: {package_dir}")
                registry_client.extract_package(archive_path, package_dir)
            
            click.echo(f"Successfully installed: {package}@{install_version}")
            
            # Install dependencies if needed
            if not no_deps:
                try:
                    # Load package manifest
                    manifest = Manifest.load(package_dir / 'elfpkg.toml')
                    
                    if manifest.dependencies:
                        click.echo(f"Installing dependencies for {package}@{install_version}...")
                        
                        for dep_name, dep in manifest.dependencies.items():
                            if not dep.optional:  # Skip optional dependencies
                                click.echo(f"Installing dependency: {dep_name}")
                                
                                # Recursively install dependency
                                os.system(f"elf install {dep_name}")
                except Exception as e:
                    click.echo(f"Warning: Failed to install dependencies: {e}", err=True)
        
        except RegistryError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    else:
        # Install all dependencies from manifest
        click.echo("Installing dependencies from manifest...")
        
        # Check for manifest
        manifest_path = Path('elfpkg.toml')
        if not manifest_path.exists():
            click.echo("Error: No elfpkg.toml found in current directory", err=True)
            sys.exit(1)
        
        # Load manifest
        try:
            manifest = Manifest.load(manifest_path)
            click.echo(f"Loaded manifest for package: {manifest.name} v{manifest.version}")
        except ManifestError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        
        # Check for lockfile
        lockfile_path = Path('elf.lock')
        lockfile = None
        
        if lockfile_path.exists():
            try:
                lockfile = Lockfile.load(lockfile_path)
                click.echo("Using existing lockfile")
            except Exception:
                click.echo("Warning: Could not load lockfile, will regenerate")
        
        # Resolve dependencies
        try:
            click.echo("Resolving dependencies...")
            registry_client = get_registry_client()
            deps = resolve_dependencies(manifest, registry_client, lockfile_path if lockfile else None)
            
            # Generate lockfile if needed
            if not lockfile:
                click.echo("Generating lockfile...")
                generate_lockfile(manifest, deps, lockfile_path)
                click.echo(f"Created lockfile: {lockfile_path}")
            
            # Install dependencies
            click.echo(f"Installing {len(deps)} dependencies...")
            
            # Create packages directory if it doesn't exist
            packages_dir = Path('.elfin/packages')
            packages_dir.mkdir(parents=True, exist_ok=True)
            
            for pkg_id, resolved_dep in deps.items():
                dep_name = resolved_dep.name
                dep_version = resolved_dep.version
                
                # Check if already installed
                package_dir = packages_dir / f"{dep_name}-{dep_version}"
                if package_dir.exists():
                    click.echo(f"Package already installed: {dep_name}@{dep_version}")
                    continue
                
                click.echo(f"Installing: {dep_name}@{dep_version}")
                
                try:
                    # Download package
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        archive_path = registry_client.download_package(dep_name, dep_version, temp_path)
                        
                        # Extract package
                        registry_client.extract_package(archive_path, package_dir)
                    
                    click.echo(f"Successfully installed: {dep_name}@{dep_version}")
                    
                except RegistryError as e:
                    click.echo(f"Warning: Failed to install {dep_name}@{dep_version}: {e}", err=True)
            
            click.echo("Dependency installation complete")
            
        except ResolutionError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        except RegistryError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


@cli.command()
def fmt():
    """Format ELFIN source files."""
    click.echo("Formatting ELFIN source files...")
    
    # Check for manifest
    manifest_path = Path('elfpkg.toml')
    if not manifest_path.exists():
        click.echo("Error: No elfpkg.toml found in current directory", err=True)
        sys.exit(1)
    
    # Find .elfin files
    src_dir = Path('src')
    if not src_dir.exists() or not src_dir.is_dir():
        click.echo("Error: No src directory found", err=True)
        sys.exit(1)
    
    elfin_files = list(src_dir.glob('**/*.elfin'))
    click.echo(f"Found {len(elfin_files)} .elfin files")
    
    # Format each file
    for file in elfin_files:
        click.echo(f"Formatting: {file}")
        
        # Read file
        with open(file, 'r') as f:
            content = f.read()
        
        # Format content (placeholder for actual formatter)
        formatted_content = content  # TODO: Implement actual formatting
        
        # Write formatted content back
        with open(file, 'w') as f:
            f.write(formatted_content)
    
    click.echo("Formatting complete")


@cli.command()
def clippy():
    """Run the ELFIN linter."""
    click.echo("Running ELFIN linter...")
    
    # Check for manifest
    manifest_path = Path('elfpkg.toml')
    if not manifest_path.exists():
        click.echo("Error: No elfpkg.toml found in current directory", err=True)
        sys.exit(1)
    
    # Find .elfin files
    src_dir = Path('src')
    if not src_dir.exists() or not src_dir.is_dir():
        click.echo("Error: No src directory found", err=True)
        sys.exit(1)
    
    elfin_files = list(src_dir.glob('**/*.elfin'))
    click.echo(f"Found {len(elfin_files)} .elfin files")
    
    # Lint each file
    warnings = 0
    errors = 0
    
    for file in elfin_files:
        click.echo(f"Linting: {file}")
        
        # Read file
        with open(file, 'r') as f:
            content = f.read()
        
        # Lint content (placeholder for actual linter)
        # TODO: Implement actual linting
        file_warnings = 0
        file_errors = 0
        
        warnings += file_warnings
        errors += file_errors
    
    # Print summary
    click.echo(f"Linting complete: {warnings} warnings, {errors} errors")
    
    if errors > 0:
        sys.exit(1)


@cli.command()
@click.option('--dir', 'registry_dir', default='./registry',
              help='Directory to initialize the registry in')
@click.option('--download-core', is_flag=True,
              help='Download core packages and seed the registry')
def setup_registry(registry_dir: str, download_core: bool):
    """Initialize a local package registry."""
    registry_path = Path(registry_dir)
    
    click.echo(f"Initializing ELFIN registry in: {registry_path}")
    
    try:
        # Create registry directory structure
        initialize_registry(registry_path)
        click.echo(f"Registry structure created successfully")
        
        if download_core:
            click.echo("Downloading core packages...")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Download core packages
                packages = download_core_packages(temp_path)
                
                if packages:
                    # Seed registry with core packages
                    click.echo(f"Seeding registry with {len(packages)} core packages...")
                    seed_registry_with_core_packages(registry_path, packages)
                    click.echo("Registry seeded successfully")
                else:
                    click.echo("Warning: No core packages were downloaded", err=True)
        
        click.echo("\nRegistry setup complete!")
        click.echo("\nTo use this registry, set the following environment variable:")
        click.echo(f"  ELFIN_REGISTRY_URL=file://{registry_path.absolute()}")
        
    except Exception as e:
        click.echo(f"Error: Failed to initialize registry: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('left')
@click.argument('right')
def semver_check(left: str, right: str):
    """Check if two versions are semver compatible."""
    try:
        import semver
        
        # Parse versions
        v1 = semver.VersionInfo.parse(left)
        v2 = semver.VersionInfo.parse(right)
        
        # Check compatibility based on semver rules
        if v1.major >= 1 and v2.major >= 1:
            # For versions >= 1.0.0, major version must match
            compatible = v1.major == v2.major
        else:
            # For versions < 1.0.0, minor must match
            compatible = v1.major == v2.major and v1.minor == v2.minor
        
        # Print result
        if compatible:
            click.echo(f"Compatible: {left} and {right}")
        else:
            click.echo(f"Incompatible: {left} and {right}")
            sys.exit(1)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
