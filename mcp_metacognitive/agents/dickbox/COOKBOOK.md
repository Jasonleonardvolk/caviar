# Dickbox Cookbook

Practical recipes for using Dickbox in production.

## Table of Contents

1. [Building Capsules](#building-capsules)
2. [Signature Verification](#signature-verification)
3. [GPU Management](#gpu-management)
4. [Blue-Green Deployments](#blue-green-deployments)
5. [Resource Control](#resource-control)
6. [Monitoring and Metrics](#monitoring-and-metrics)
7. [Security Best Practices](#security-best-practices)

## Building Capsules

### Basic Capsule Build

```bash
# Build a simple capsule
./scripts/build_capsule.sh my-service 1.0.0 ./src/my-service

# Build with custom manifest
./scripts/build_capsule.sh my-service 1.0.0 ./src/my-service \
  --manifest ./custom-manifest.yml

# Build without signing (development)
./scripts/build_capsule.sh my-service 1.0.0 ./src/my-service --skip-sign
```

### Capsule Manifest Structure

```yaml
# capsule.yml
name: tori-ingest
version: 1.4.3
entrypoint: bin/ingest_bus

# Dependencies
dependencies:
  python: "3.10"
  rust: "1.65"
  cuda: "11.8"  # For GPU services

# Service configuration
services:
  - name: tori-ingest
    slice: tori-server.slice  # Resource isolation
    
    # Resource limits
    resource_limits:
      cpu_quota: 200      # 2 cores max
      cpu_weight: 200     # Higher priority
      memory_max: 4G      # Hard limit
      memory_high: 3G     # Soft limit
      tasks_max: 512      # Thread limit
      io_weight: 100      # I/O priority
    
    # GPU configuration
    gpu_config:
      enabled: true
      visible_devices: "0"    # GPU index or UUID
      mps_percentage: 50      # 50% of GPU SMs
      mode: mps              # mps, exclusive, or default
    
    # Health check
    health_check_url: "http://localhost:8080/health"
    startup_timeout: 60
    
    # Environment
    environment:
      LOG_LEVEL: info
      METRICS_PORT: "9090"

# Build metadata (auto-generated)
build_info:
  timestamp: 2025-01-10T10:00:00Z
  builder: ci@build-server
  git_sha: abc123def456
  
# Signature info (added by build script)
signature: capsule.sig
public_key: RWQf6LRCGA9i53mlYecO4IzT51TGPpvWucNSCh1CBM
```

## Signature Verification

### Generate Signing Keys

```bash
# Generate minisign keypair
minisign -G -p /etc/tori/keys/minisign.pub -s /etc/tori/keys/minisign.key

# Set permissions
chmod 600 /etc/tori/keys/minisign.key
chmod 644 /etc/tori/keys/minisign.pub
```

### Sign Capsule Manually

```bash
# Sign existing capsule
minisign -Sm my-service-1.0.0.tar.gz \
  -s /etc/tori/keys/minisign.key \
  -x my-service-1.0.0.tar.gz.sig
```

### Verify Signature

```bash
# Verify manually
minisign -Vm my-service-1.0.0.tar.gz \
  -p /etc/tori/keys/minisign.pub \
  -x my-service-1.0.0.tar.gz.sig
```

### Programmatic Verification

```python
# Dickbox handles this automatically, but you can verify manually:
from kha.mcp_metacognitive.agents.dickbox import verify_capsule_signature

verified = await verify_capsule_signature(
    capsule_path=Path("my-service-1.0.0.tar.gz"),
    sig_path=Path("my-service-1.0.0.tar.gz.sig"),
    public_key="RWQf6LRCGA9i53mlYecO4IzT51TGPpvWucNSCh1CBM"
)
```

## GPU Management

### Enable MPS for GPU Sharing

```bash
# Start NVIDIA MPS control daemon
nvidia-cuda-mps-control -d

# Or let Dickbox manage it
from kha.mcp_metacognitive.agents.dickbox import create_dickbox_agent

agent = create_dickbox_agent()
await agent.gpu_manager.start_mps()
```

### Configure GPU Allocation

```python
# Schedule GPU for a service
allocation = await agent.gpu_scheduler.schedule_service(
    "my-ai-service",
    requirements={
        "memory_mb": 8000,      # 8GB VRAM
        "compute_percentage": 25, # 25% of SMs
        "mode": "mps",          # Use MPS sharing
        "prefer_gpu": 0         # Prefer GPU 0
    }
)

# Result includes environment variables to set
print(allocation["environment"])
# {'CUDA_VISIBLE_DEVICES': '0', 'CUDA_MPS_PIPE_DIRECTORY': '/tmp/nvidia-mps'}
```

### Start GPU Keep-Alive Services

```bash
# Enable for all GPUs
systemctl enable soliton-mps@*.service

# Or programmatically
await agent.gpu_manager.start_gpu_keepalive()
```

### Monitor GPU Health

```python
health = await agent.gpu_manager.monitor_gpu_health()

if not health["healthy"]:
    print(f"GPU warnings: {health['warnings']}")
    print(f"GPU errors: {health['errors']}")
```

## Blue-Green Deployments

### Deploy New Version

```python
# Deploy new version alongside old
result = await agent.execute("deploy_service", {
    "service_name": "tori-api",
    "source": "https://artifacts.tori.ai/capsules/tori-api-2.0.0.tar.gz"
})

# Both versions running - old still serving traffic
print(f"New capsule: {result['capsule_id']}")
print(f"Previous: {result['previous_capsule_id']}")
```

### Switch Traffic

```bash
# Update HAProxy configuration
cat > /etc/haproxy/tori-api.cfg << EOF
backend tori-api
    server new-version unix@/var/run/tori/tori-api_${NEW_CAPSULE}.sock
EOF

# Reload HAProxy
systemctl reload haproxy
```

### Rollback if Needed

```python
# Quick rollback to previous version
result = await agent.execute("rollback_service", {
    "service_name": "tori-api"
})
```

## Resource Control

### Configure Systemd Slices

```python
# Create custom slice with limits
from kha.mcp_metacognitive.agents.dickbox import SliceConfig

slice = SliceConfig(
    name="tori-ml.slice",
    parent="tori.slice",
    cpu_weight=150,      # Medium priority
    cpu_quota=800,       # Max 8 cores
    memory_max="32G",    # Hard limit
    memory_high="28G",   # Soft limit
    io_weight=75         # Lower I/O priority
)

await agent.slice_manager.create_slice(slice)
```

### Apply Resource Limits Dynamically

```python
# Adjust limits for running slice
await agent.slice_manager.update_resource_limits(
    "tori-helper.slice",
    cpu_quota=200,      # Reduce to 2 cores
    memory_high="1G"    # Reduce memory
)
```

### Monitor Resource Usage

```python
# Get slice statistics
status = await agent.slice_manager.get_slice_status("tori-server.slice")

print(f"CPU usage: {status['cpu_usage_ns'] / 1e9:.2f} seconds")
print(f"Memory: {status['memory_bytes'] / 1e9:.2f} GB")
print(f"Tasks: {status['task_count']}")
```

## Monitoring and Metrics

### Enable Prometheus Metrics

```python
# Start metrics server
from kha.mcp_metacognitive.agents.dickbox import create_metrics_app

app = create_metrics_app(agent)
# Run with uvicorn on port 9091
```

### Key Metrics to Monitor

```promql
# Deployment success rate
rate(dickbox_deployments_total[5m])

# Average deployment time
histogram_quantile(0.95, dickbox_deployment_duration_seconds_bucket)

# GPU utilization
avg(dickbox_gpu_utilization_percent) by (gpu_name)

# Slice resource usage
dickbox_slice_memory_bytes{slice="tori-server.slice"}
```

### Custom Metrics

```python
# Record custom deployment metrics
from kha.mcp_metacognitive.agents.dickbox import metrics_exporter

metrics_exporter.record_deployment(
    service="my-service",
    status="success",
    duration=45.2  # seconds
)
```

## Security Best Practices

### 1. Always Sign Production Capsules

```bash
# CI/CD pipeline should include
export SIGNING_KEY=/secure/path/to/minisign.key
./scripts/build_capsule.sh $SERVICE $VERSION $SRC_DIR
```

### 2. Rotate ZMQ Keys Regularly

```bash
# Enable automatic rotation
systemctl enable zmq-key-rotate.timer
systemctl start zmq-key-rotate.timer

# Check timer status
systemctl list-timers zmq-key-rotate.timer
```

### 3. Use Energy Budget on tmpfs

```bash
# /etc/fstab entry
tmpfs /var/tmp/tori_energy tmpfs size=10M,mode=0755 0 0

# Environment variable
export DICKBOX_ENERGY_BUDGET_PATH=/var/tmp/tori_energy/budget.json
```

### 4. Restrict Capsule Permissions

```bash
# Set appropriate ownership
chown -R tori:tori /opt/tori/releases/
chmod 750 /opt/tori/releases/

# Each capsule directory
chmod 750 /opt/tori/releases/*/
```

### 5. Use AppArmor/SELinux Profiles

```bash
# Example AppArmor profile for capsules
cat > /etc/apparmor.d/tori.capsule << EOF
#include <tunables/global>

/opt/tori/releases/*/bin/* {
  #include <abstractions/base>
  #include <abstractions/python>
  
  /opt/tori/releases/** r,
  /var/log/tori/** w,
  /var/run/tori/** rw,
  
  # Deny everything else
  deny /** w,
}
EOF

apparmor_parser -r /etc/apparmor.d/tori.capsule
```

### 6. Monitor Failed Verifications

```python
# Set up alerts for signature failures
if psi_archive:
    events = psi_archive.get_events_by_type("deployment_failed")
    
    for event in events:
        if "Signature verification failed" in event.get("error", ""):
            alert_security_team(event)
```

## Advanced Recipes

### Multi-GPU Service with Affinity

```yaml
# In capsule.yml for multi-GPU training
services:
  - name: tori-training
    gpu_config:
      enabled: true
      visible_devices: "0,1,2,3"  # 4 GPUs
      mode: exclusive             # No sharing
    environment:
      NCCL_SOCKET_IFNAME: eth0    # For multi-GPU comms
      NCCL_IB_DISABLE: 1
```

### Canary Deployments

```python
# Deploy canary with limited resources
canary_config = {
    "service_name": "tori-api-canary",
    "source": "new-version.tar.gz",
    "resource_limits": {
        "cpu_quota": 50,      # Only 0.5 cores
        "memory_max": "512M"  # Limited memory
    }
}

# Monitor canary metrics before full rollout
```

### Zero-Downtime Database Migrations

```python
# 1. Deploy new version in migration mode
await agent.execute("deploy_service", {
    "service_name": "tori-api",
    "source": "tori-api-2.0.0-migration.tar.gz",
    "environment": {
        "RUN_MIGRATIONS": "true",
        "SERVE_TRAFFIC": "false"
    }
})

# 2. Wait for migrations
await asyncio.sleep(30)

# 3. Deploy serving version
await agent.execute("deploy_service", {
    "service_name": "tori-api",
    "source": "tori-api-2.0.0.tar.gz"
})
```

This cookbook provides practical examples for production use of Dickbox. Always test these recipes in staging before applying to production systems.
