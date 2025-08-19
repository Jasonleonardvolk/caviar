# 🔧 Beyond Metacognition - Bare Metal Deployment Guide

## Overview

Deployment guide for Beyond Metacognition on your bare-metal TORI setup using virtualenv, systemd, and Ansible.

## 🏗️ Architecture

```
/opt/tori/
├── releases/
│   ├── abc123_20250703_143022/   # Git SHA + timestamp
│   │   ├── kha/                   # Beyond components
│   │   ├── systemd/               # Service files
│   │   ├── ansible/               # Playbooks
│   │   └── scripts/               # Rollback & monitoring
│   └── def456_20250702_090000/   # Previous release
├── current -> releases/abc123...  # Symlink to active
├── venv/                          # Python virtualenv
├── conf/                          # Configuration files
├── data/                          # Runtime data
└── logs/                          # Log files
```

## 📋 Pre-Deployment Checklist

### 1. Prepare Release Package
```bash
# Create Beyond deployment package
python beyond_deploy_bare_metal.py

# This generates:
# - Release directory with all Beyond files
# - Systemd unit files
# - Ansible playbook
# - Rollback scripts
# - Monitoring scripts
```

### 2. Update Python Dependencies
```bash
# Add to your requirements.txt or pyproject.toml
numpy>=1.21.0
pyyaml
psutil

# Update virtualenv
cd /opt/tori/venv
./bin/pip install -r requirements-beyond.txt
```

### 3. Configuration Files

Create/update these config files in `/opt/tori/conf/`:

**runtime.yaml:**
```yaml
beyond_metacognition:
  observer_synthesis:
    reflex_budget: 30  # Start conservative
    measurement_cooldown_ms: 100
  creative_feedback:
    novelty_threshold_high: 0.8
    emergency_threshold: 0.06
  safety:
    auto_rollback_threshold: 0.08
```

**braid.yaml:**
```yaml
temporal_braiding:
  buffers:
    micro:
      capacity: 10000
      window_us: 1000
    meso:
      capacity: 1000
      window_us: 60000000
```

### 4. Ansible Inventory

Update your Ansible inventory:
```ini
[tori_nodes]
tori-node-1 ansible_host=192.168.1.10
tori-node-2 ansible_host=192.168.1.11
tori-node-3 ansible_host=192.168.1.12

[tori_nodes:vars]
ansible_user=ansible
tori_base=/opt/tori
```

## 🚀 Deployment Steps

### 1. Deploy with Ansible
```bash
# Run from your Ansible control node
cd /path/to/release/ansible
ansible-playbook -i inventory.ini beyond_deploy.yml --check  # Dry run
ansible-playbook -i inventory.ini beyond_deploy.yml         # Deploy
```

### 2. Manual Deployment (Single Node)
```bash
# Copy files
sudo -u tori rsync -av /path/to/release/ /opt/tori/releases/new_release/

# Apply patches
cd /opt/tori/releases/new_release
sudo -u tori /opt/tori/venv/bin/python kha/apply_beyond_patches.py --verify

# Update symlink
sudo -u tori ln -sfn /opt/tori/releases/new_release /opt/tori/current

# Install systemd units
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Restart services
sudo systemctl restart tori-api
sudo systemctl start tori-braid-aggregator
sudo systemctl start tori-beyond-monitor
```

### 3. Verify Deployment
```bash
# Check services
systemctl status tori-api tori-braid-aggregator tori-beyond-monitor

# Run verification
/opt/tori/venv/bin/python /opt/tori/current/kha/verify_beyond_integration.py

# Check health endpoint
curl http://localhost:8002/api/health

# Check logs
journalctl -u tori-api -f --grep=BEYOND
```

## 📊 Monitoring Setup

### 1. Prometheus Integration

Add to your Prometheus scrape configs:
```yaml
- job_name: 'tori_beyond'
  static_configs:
    - targets: ['localhost:9090']
  metric_relabel_configs:
    - source_labels: [__name__]
      regex: 'beyond_.*'
      action: keep
```

### 2. Loki/Promtail Setup

Deploy Promtail config:
```bash
sudo cp promtail-beyond.yaml /etc/promtail/
sudo systemctl restart promtail
```

### 3. Nginx Configuration

Add to your TORI nginx server block:
```nginx
# WebSocket for Beyond metrics
location /ws/eigensentry {
    proxy_pass http://localhost:8765;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}

# Metrics endpoint (restricted)
location /metrics {
    proxy_pass http://localhost:9090;
    allow 127.0.0.1;
    allow 192.168.1.0/24;
    deny all;
}
```

## 🔄 Rollback Procedures

### Automatic Rollback
The monitor service watches λ_max and triggers rollback if > 0.08 for 3+ minutes:
```bash
# Monitor runs automatically via systemd
systemctl status tori-beyond-monitor

# Check monitor logs
journalctl -u tori-beyond-monitor -f
```

### Manual Rollback
```bash
# Method 1: Using torictl wrapper
./torictl_beyond rollback-beyond 1  # Rollback 1 version

# Method 2: Direct script
bash /opt/tori/current/scripts/beyond_rollback.sh 1

# Method 3: Manual steps
sudo systemctl stop tori-api tori-braid-aggregator
sudo -u tori ln -sfn /opt/tori/releases/previous_release /opt/tori/current
sudo systemctl start tori-api
```

## 🛠️ Operational Commands

### Using torictl_beyond wrapper:
```bash
# Deploy Beyond
./torictl_beyond deploy-beyond

# Verify installation
./torictl_beyond verify-beyond

# Monitor live metrics
./torictl_beyond monitor-beyond

# Rollback if needed
./torictl_beyond rollback-beyond [n]
```

### Using standard torictl integration:
```bash
# Run demos
/opt/tori/venv/bin/python /opt/tori/current/kha/torictl.py demo emergence

# Check status
/opt/tori/venv/bin/python /opt/tori/current/kha/torictl.py status

# Live monitor
/opt/tori/venv/bin/python /opt/tori/current/kha/torictl.py monitor
```

## 📈 Performance Tuning

### After 1 week of stable operation:

1. **Increase reflex budget:**
   ```yaml
   reflex_budget: 60  # From 30
   ```

2. **Lower novelty threshold:**
   ```yaml
   novelty_threshold_high: 0.7  # From 0.8
   ```

3. **Adjust buffer sizes based on usage:**
   ```bash
   # Check buffer fill ratios in metrics
   curl -s http://localhost:9090/metrics | grep braid_buffer_fill_ratio
   ```

## 🚨 Troubleshooting

### Service Won't Start
```bash
# Check for import errors
/opt/tori/venv/bin/python -c "from alan_backend.origin_sentry import OriginSentry"

# Check patches applied
grep -n "origin_sentry" /opt/tori/current/alan_backend/eigensentry_guard.py

# Check virtualenv has dependencies
/opt/tori/venv/bin/pip list | grep numpy
```

### High Memory Usage
```bash
# Check buffer sizes
du -sh /opt/tori/data/braid_buffers/

# Reduce buffer capacity in braid.yaml
# Restart services
```

### Logs Not Appearing
```bash
# Check journald
journalctl -u tori-api --since "10 minutes ago"

# Check rsyslog
grep BEYOND /var/log/tori/beyond.log

# Verify logging config loaded
/opt/tori/venv/bin/python -c "from kha.beyond_logging_config import BeyondLoggingConfig; BeyondLoggingConfig.setup_logging()"
```

## 📚 Additional Resources

- Technical documentation: `BEYOND_METACOGNITION_COMPLETE.md`
- Quick start guide: `BEYOND_QUICKSTART.md`
- Component tests: `test_beyond_integration.py`
- Diagnostics: `beyond_diagnostics.py`

---

**Your bare-metal TORI is ready to evolve beyond metacognition!** 🌌🔧
