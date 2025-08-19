# 📦 TORI Capsule Builder (`capsbuild.py`)

## Overview

The **Capsule Builder** packages TORI applications into immutable, content-addressed capsules for the Dickbox containerless deployment system. Each capsule is a self-contained unit with all code, dependencies, and configuration needed to run a service.

## Features

- 🔒 **Content-addressed builds** - SHA256 hash of all files becomes the capsule ID
- 📦 **Multiple input formats** - Build from directories or existing ZIP files  
- 🐍 **Python venv support** - Automatically relocates virtual environments
- ✅ **Dependency verification** - Ensures all requirements are included
- 🔐 **Optional signing** - Support for minisign signatures
- 📊 **Build metadata** - Embeds build time, host, and file counts

## Installation

```bash
# Install dependencies
pip install pyyaml

# Make executable
chmod +x capsbuild.py
```

## Usage

### Build from Directory

```bash
# Package the No-DB migration into a capsule
./capsbuild.py \
  --from-dir . \
  --manifest capsule.yml \
  --output tori-metacog-v2.1.tar.gz
```

### Build from ZIP

```bash
# Use our existing No-DB distribution ZIP
./capsbuild.py \
  --from-zip tori_nodb_complete_20250105.zip \
  --manifest capsule.yml
```

### With Signing

```bash
# Sign the capsule for integrity verification
./capsbuild.py \
  --from-dir . \
  --manifest capsule.yml \
  --sign ~/.minisign/tori-release.key
```

## Capsule Structure

```
capsule-tori-metacognitive-a3f2b8c9d1e4.tar.gz
├── capsule.yml              # Manifest file
├── .capsule_metadata.json   # Build metadata
├── bin/                     # Executables
├── venv/                    # Python virtual environment
├── config/                  # Configuration files
├── assets/                  # Static assets
├── alan_backend/            # Application code
├── python/core/             # Core modules
└── requirements_nodb.txt    # Python dependencies
```

## Manifest Format

The `capsule.yml` manifest defines the capsule configuration:

```yaml
name: tori-metacognitive
version: "2.1.0"
entrypoint: alan_backend/start_true_metacognition.py

services:
  - tori-metacog

env:
  TORI_STATE_ROOT: "/opt/tori/state"
  MAX_TOKENS_PER_MIN: "200"

dependencies:
  python: "3.10"
  pip_requirements: "requirements_nodb.txt"

resources:
  cpu_quota: "80%"
  memory_max: "16G"
```

## Integration with No-DB Migration

The capsule builder is designed to work seamlessly with our No-DB migration:

1. **Parquet State** - The `TORI_STATE_ROOT` environment variable points to shared Parquet storage
2. **Pinned Dependencies** - Uses `requirements_nodb.txt` with `~=` version constraints
3. **Resource Limits** - Manifest includes systemd slice configurations
4. **Build Metadata** - BUILD_SHA is embedded for correlation with metrics

## Example: Building No-DB Capsule

```bash
# 1. Ensure No-DB migration is complete
python master_nodb_fix_v2.py

# 2. Create capsule manifest (already provided)
cat capsule.yml

# 3. Build the capsule
./capsbuild.py \
  --from-dir . \
  --manifest capsule.yml \
  --output capsule-metacog-$(date +%Y%m%d).tar.gz

# Output:
# ✅ Capsule built successfully: capsule-metacog-20250105.tar.gz
#    BUILD_SHA: a3f2b8c9d1e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0
#    Size: 124.56 MB
```

## Deployment

Once built, capsules are deployed using `capsdeploy`:

```bash
# Install and activate the capsule
capsdeploy install capsule-metacog-20250105.tar.gz
capsdeploy activate a3f2b8c9d1e4

# Verify
systemctl status tori@a3f2b8c9d1e4.service
```

## Best Practices

1. **Version Everything** - Include version in capsule.yml and use semantic versioning
2. **Pin Dependencies** - Always use `~=` in requirements for reproducible builds
3. **Test Locally** - Extract and test capsule before deployment
4. **Keep Manifests** - Store capsule.yml in version control
5. **Sign Releases** - Use minisign for production capsules

## Troubleshooting

### Missing Entrypoint
```
ERROR: Entrypoint not found: alan_backend/start_true_metacognition.py
```
**Solution**: Ensure entrypoint path is relative to source directory

### Python Version Mismatch
```
WARNING: Python entrypoint found but no venv included
```
**Solution**: Include a virtual environment or ensure system Python matches

### Large Capsule Size
**Solution**: Exclude unnecessary files using `.capsuleignore` (similar to .gitignore)

## Next Steps

1. **Manifest Validator** - `caps_validate.py` to check capsule.yml syntax
2. **Migration Tool** - `caps_migrate.py` to convert legacy deployments
3. **Systemd Templates** - `tori@.service` for capsule runtime

---

Ready to build immutable, content-addressed capsules for Dickbox deployment! 🚀
