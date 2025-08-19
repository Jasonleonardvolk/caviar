# Dickbox - Containerless Deployment System

A lightweight, secure capsule-based deployment system for TORI that provides
container-like benefits without Docker/Kubernetes overhead.

## Key Features

- **Immutable Capsules**: Content-addressed deployment artifacts with signature verification
- **Systemd Integration**: Service management and cgroup-based isolation
- **Blue-Green Deployments**: Zero-downtime upgrades with instant rollback
- **GPU Sharing**: NVIDIA MPS integration for concurrent CUDA workloads
- **Resource Control**: CPU, memory, and I/O limits via systemd slices
- **Fast IPC**: Unix socket communication for local services
- **Observability**: Built-in metrics and version tracking
- **Security**: Capsule signature verification with minisign/sigstore

## Architecture

```
/opt/tori/releases/
├── abc123/          # Old version (content hash)
│   ├── capsule.yml  # Manifest with signature info
│   ├── capsule.sig  # Digital signature
│   ├── bin/         # Executables
│   ├── venv/        # Python environment
│   └── config/      # Configuration
└── xyz789/          # New version
    ├── capsule.yml
    ├── capsule.sig
    ├── bin/
    ├── venv/
    └── config/
```

## Capsule Structure

Each capsule is an immutable directory containing:
- Application code and binaries
- Vendored dependencies (Python venv, Rust binaries)
- Configuration templates
- Metadata manifest (capsule.yml)
- Digital signature (capsule.sig)

## Resource Isolation

Uses systemd slices for workload segregation:
- `tori.slice`: Parent slice for all TORI services
  - `tori-server.slice`: Critical, latency-sensitive services
  - `tori-helper.slice`: Background and batch jobs
  - `tori-build.slice`: Compilation and build tasks

## GPU Management

Leverages NVIDIA MPS for safe GPU sharing:
- Memory isolation on Ampere GPUs
- Concurrent CUDA execution
- Per-service GPU quotas
- Soliton MPS keep-alive for warm contexts

## Communication

- **Local**: gRPC over Unix Domain Sockets
- **Distributed**: ZeroMQ pub/sub with automatic key rotation
- **Service Discovery**: Systemd socket activation

## Capsule Signature Verification

Dickbox supports cryptographic verification of capsules using minisign or sigstore:

### Configuration

In `capsule.yml`:
```yaml
name: my-service
version: 1.0.0
signature: capsule.sig  # Signature file name
public_key: RWQf6LRCGA9i53mlYecO4IzT51TGPpvWucNSCh1CBM  # Base64 public key
```

### Build Process

Use the provided build script:
```bash
./scripts/build_capsule.sh my-service 1.0.0 ./src/my-service
```

This generates:
- Signed capsule tarball
- `results.txt` with SHA256, file list, and metadata
- JSON metadata for CI/CD integration

### Verification Flow

1. Extract capsule to temporary location
2. Check for `signature` and `public_key` in manifest
3. Verify signature against capsule contents
4. If verification fails, remove extracted files
5. If successful or no signature required, proceed with deployment

## GPU Management with MPS

### GPU SM Quota Control

Dickbox integrates with NVIDIA MPS for fine-grained GPU sharing:

```yaml
# In service configuration
services:
  - name: tori-ingest
    gpu_config:
      enabled: true
      visible_devices: "0"  # GPU index
      mps_percentage: 50    # Limit to 50% of GPU SMs
```

### Soliton MPS Keep-Alive

To prevent GPU initialization delays, Dickbox includes a keep-alive service:

```bash
# Automatically started for each GPU
systemctl status soliton-mps@GPU-UUID.service
```

This runs a minimal CUDA kernel every 10 seconds to keep contexts warm.

### GPU Resource Allocation

```python
# Dickbox automatically schedules GPU resources
scheduler = GPUScheduler(mps_manager)
allocation = await scheduler.schedule_service(
    "my-service",
    requirements={
        "memory_mb": 4000,
        "compute_percentage": 25  # 25% of GPU compute
    }
)
```

## Energy Budget Management

Dickbox now supports configurable energy budget paths:

```yaml
# In dickbox_config.yml or environment
energy_budget_path: /var/tmp/tori_energy.json
energy_budget_sync_interval: 60  # seconds
```

This allows mounting the energy budget on tmpfs for better performance:

```bash
# Mount tmpfs for energy budget
mount -t tmpfs -o size=10M tmpfs /var/tmp/tori_energy
export DICKBOX_ENERGY_BUDGET_PATH=/var/tmp/tori_energy/budget.json
```

## ZeroMQ Key Rotation

For secure pub/sub messaging, Dickbox includes automatic key rotation:

### Manual Rotation
```bash
python3 /opt/tori/bin/rotate_zmq_keys.py
```

### Automatic Weekly Rotation
```bash
systemctl enable zmq-key-rotate.timer
systemctl start zmq-key-rotate.timer
```

Keys are stored in `/etc/tori/zmq_keys/` with symlinks to current keys.

## Deployment Flow

1. Build capsule with content hash and signature
2. Extract to `/opt/tori/releases/HASH/`
3. Verify signature if present
4. Create/update systemd service
5. Start new version alongside old
6. Health check new version
7. Switch traffic (HAProxy/symlink)
8. Stop old version
9. Optional: Clean old capsules

This provides atomic, reproducible deployments with instant rollback capability.

## Quick Start

1. Install Dickbox:
```bash
cd scripts
sudo ./install_services.sh
```

2. Create signing keys:
```bash
minisign -G -p /etc/tori/keys/minisign.pub -s /etc/tori/keys/minisign.key
```

3. Build your first capsule:
```bash
./scripts/build_capsule.sh my-service 1.0.0 ./src/my-service
```

4. Deploy:
```python
from kha.mcp_metacognitive.agents.dickbox import create_dickbox_agent

agent = create_dickbox_agent()
result = await agent.execute("deploy_service", {
    "service_name": "my-service",
    "source": "artifacts/my-service-1.0.0.tar.gz"
})
```

See [COOKBOOK.md](COOKBOOK.md) for detailed recipes and examples.
