# ELFIN Package Ecosystem

> "Cargo-grade workflows for ELFIN developers"

## Overview

The ELFIN Package Ecosystem provides a robust solution for package management, dependency resolution, and distribution of ELFIN components. Inspired by Rust's Cargo system, it offers a streamlined workflow for creating, building, and publishing packages.

## Key Features

- **Manifest-based Configuration**: Single source of truth (`elfpkg.toml`) for dependencies, solver options, and metadata
- **Git-backed Registry**: 100% OSS approach using a Git index similar to crates.io, with an S3-style blob store
- **Intuitive CLI**: Simple verb-based commands for common operations (`elf new`, `elf build`, `elf publish`, etc.)
- **SemVer Compliance**: Version conflict resolution following Semantic Versioning principles
- **Reproducible Builds**: Lockfile mechanism ensures deterministic dependency resolution

## Getting Started

### Installation

The package manager can be run directly from the source code:

```bash
# Windows
run_elfpkg.bat

# Linux/macOS
./run_elfpkg.sh
```

### Setting Up a Local Registry

Before using the package manager, you should set up a local registry:

```bash
run_elfpkg.bat setup-registry --download-core
```

This command:
1. Creates the required directory structure for the registry
2. Downloads core packages (elfin-core, elfin-ui)
3. Seeds the registry with these packages

### Creating a New Package

```bash
run_elfpkg.bat new my_package
cd my_package
```

This creates a new package with the basic structure:
- `elfpkg.toml`: Package manifest
- `src/`: Source code directory
- `README.md`: Package documentation

### Building a Package

```bash
run_elfpkg.bat build
```

For release builds:

```bash
run_elfpkg.bat build --release
```

### Installing Dependencies

To install all dependencies listed in the manifest:

```bash
run_elfpkg.bat install
```

To install a specific package:

```bash
run_elfpkg.bat install elfin-core
```

With a specific version:

```bash
run_elfpkg.bat install elfin-core --version 1.0.0
```

### Publishing a Package

```bash
run_elfpkg.bat publish
```

Add `--dry-run` to validate without publishing:

```bash
run_elfpkg.bat publish --dry-run
```

## Package Manifest (elfpkg.toml)

The manifest defines your package and its dependencies.

```toml
[package]
name        = "quadrotor_controller"
version     = "0.1.0"
authors     = ["alice@example.com"]
edition     = "elfin-1.0"

[dependencies]
elfin-core  = ">=1.0.0,<2.0.0"
cvxpy       = ">=1.4.0"

[solver]
mosek.msk_license_file = "${HOME}/mosek.lic"
```

### Version Specification

- Exact version: `"1.0.0"`
- Greater than or equal: `">=1.0.0"`
- Compatible range: `">=1.0.0,<2.0.0"` (same as `"^1.0.0"`)
- Any version: `"*"`

## Lockfile (elf.lock)

The lockfile ensures reproducible builds by pinning exact dependency versions:

```toml
version = "1"

[[package]]
name = "elfin-core"
version = "1.0.0"
checksum = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

[[package]]
name = "cvxpy"
version = "1.4.0"
checksum = "f84be1bb0f1b4e4f1c6fd5d05b0e53bde9c1619a72c95fe71b3ef8a9c4d7c39a"
dependencies = [
  "numpy>=1.15.0"
]
```

## Registry Structure

The registry follows a structure similar to crates.io:

```
registry/
├── index/                  # Git repository of package metadata
│   ├── config.json         # Registry configuration
│   ├── 1/                  # 1-letter package names
│   ├── 2/                  # 2-letter package names
│   ├── 3/                  # 3-letter package names
│   ├── e/                  # Packages starting with 'e'
│   │   └── l/              # Packages starting with 'el'
│   │       └── f/          # Packages starting with 'elf'
│   │           └── elfin-core  # Package metadata file
│   └── ...
└── blobs/                  # Package archives
    ├── e3/                 # First 2 hex chars of SHA-256
    │   └── b0/             # Next 2 hex chars
    │       └── e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  # Archive
    └── ...
```

## Command Reference

| Command | Description |
|---------|-------------|
| `new <name>` | Create a new package |
| `build` | Build the package |
| `publish` | Publish to registry |
| `install [package]` | Install dependencies |
| `fmt` | Format ELFIN source files |
| `clippy` | Run the linter |
| `setup-registry` | Initialize a registry |
| `semver-check <v1> <v2>` | Check version compatibility |

## Environment Variables

- `ELFIN_REGISTRY_URL`: URL to the package registry (default: `https://registry.elfin.dev`)
- `ELFIN_REGISTRY_TOKEN`: API token for authenticated operations

## Architecture

The package ecosystem consists of several core components:

1. **Manifest Parser**: Validates and processes `elfpkg.toml` files
2. **Dependency Resolver**: Resolves version requirements to concrete versions
3. **Registry Client**: Communicates with the registry for publishing and downloading
4. **Lockfile Generator**: Creates and updates `elf.lock` files
5. **CLI Interface**: Provides user-friendly command-line tools

## Extending the Ecosystem

### Adding Formatters and Linters

The package ecosystem includes placeholder implementations for formatting (`elffmt`) and linting (`elfclippy`). These can be extended to provide:

- Code style enforcement
- Static analysis
- Best practice suggestions
- Detection of common errors

### Custom Templates

When creating new packages, the ecosystem supports different templates:

- `basic`: Simple package with minimal dependencies
- `application`: Full-featured application with UI components
- `library`: Reusable library package

Additional templates can be added by extending the template directory structure.

## Contributing

To contribute to the ELFIN package ecosystem:

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Submit a pull request

## License

Copyright (c) 2025 ELFIN Project Team. All rights reserved.
