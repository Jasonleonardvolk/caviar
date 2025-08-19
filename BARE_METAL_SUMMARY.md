# 🏭 Beyond Metacognition - Bare Metal Production Suite

## Summary

We've successfully adapted the Beyond Metacognition deployment for your **bare-metal, virtualenv-based TORI setup**. No containers, no cloud - just Python, systemd, and Ansible on your own hardware.

## 📦 What We've Created

### Core Deployment Tools

1. **`beyond_deploy_bare_metal.py`** - Master deployment script
   - Generates timestamped releases in `/opt/tori/releases/`
   - Creates systemd unit files
   - Generates Ansible playbooks
   - Produces rollback and monitoring scripts

2. **Systemd Services**
   - `tori-api.service` - Main TORI (patched with Beyond)
   - `tori-braid-aggregator.service` - Temporal braiding aggregator
   - `tori-beyond-monitor.service` - Auto-rollback monitor

3. **`torictl_beyond`** - Wrapper script
   ```bash
   ./torictl_beyond deploy-beyond    # Full deployment
   ./torictl_beyond verify-beyond    # Health check
   ./torictl_beyond monitor-beyond   # Live monitoring
   ./torictl_beyond rollback-beyond  # Emergency rollback
   ```

### Monitoring & Logging

1. **`beyond_prometheus_exporter.py`** - Metrics exporter
   - Integrates with your existing `/metrics` endpoint
   - Or runs standalone on port 9091
   - All Beyond-specific metrics (λ_max, novelty, etc.)

2. **`beyond_logging_config.py`** - Structured logging
   - Journald integration with structured fields
   - Promtail config for Loki shipping
   - Loki alerting rules
   - rsyslog configuration

### Configuration Management

All configs live in `/opt/tori/conf/`:
- `runtime.yaml` - Reflex budgets, thresholds
- `braid.yaml` - Buffer configurations  
- `origin.yaml` - OriginSentry settings

Environment variables override configs:
```bash
export BEYOND_REFLEX_BUDGET=30
export BEYOND_NOVELTY_THRESHOLD=0.8
```

## 🚀 Deployment Workflow

### 1. Prepare Release
```bash
# Generate deployment package
python beyond_deploy_bare_metal.py

# Creates: /opt/tori/releases/<git-sha>_<timestamp>/
```

### 2. Deploy with Ansible
```bash
cd releases/<latest>/ansible
ansible-playbook -i inventory.ini beyond_deploy.yml
```

### 3. Or Manual Deploy (Single Node)
```bash
# Apply patches
/opt/tori/venv/bin/python apply_beyond_patches.py --verify

# Update symlink
ln -sfn /opt/tori/releases/<new> /opt/tori/current

# Restart services
systemctl restart tori-api
systemctl start tori-braid-aggregator tori-beyond-monitor
```

### 4. Verify
```bash
# Quick health check
/opt/tori/venv/bin/python /opt/tori/current/kha/verify_beyond_integration.py

# Check metrics
curl http://localhost:9090/metrics | grep beyond_

# Monitor logs
journalctl -u tori-api -f --grep=BEYOND
```

## 🛡️ Safety Features

### Auto-Rollback Monitor
- Runs as systemd service
- Watches λ_max every 30 seconds
- Triggers rollback if > 0.08 for 3+ minutes
- Reverts symlink and restores from backups

### Conservative Defaults
- Reflex budget: 30/hour (not 60)
- Novelty threshold: 0.8 (not 0.7)
- Emergency threshold: 0.06 (not 0.08)

### Easy Manual Rollback
```bash
# Just repoint the symlink
ln -sfn /opt/tori/releases/previous /opt/tori/current
systemctl restart tori-api
```

## 📊 Monitoring Integration

### Prometheus
Your existing node-exporter setup just needs:
```yaml
- job_name: 'tori_beyond'
  static_configs:
    - targets: ['localhost:9090']  # Your existing endpoint
```

### Loki
Ship logs via Promtail:
```bash
sudo cp promtail-beyond.yaml /etc/promtail/
sudo systemctl restart promtail
```

Query in Grafana:
```
{job="beyond_metacognition"} |= "LAMBDA_MAX" | unwrap lambda_max
```

### Nginx
Add WebSocket and metrics locations:
```nginx
location /ws/eigensentry {
    proxy_pass http://localhost:8765;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

## 🎯 Key Differences from Container Setup

| Aspect | Container Version | Your Bare Metal |
|--------|------------------|-----------------|
| Deployment | `kubectl apply` | `ansible-playbook` or symlink |
| Rollback | Service selector | Symlink switch |
| Logs | Container stdout | journald + rsyslog |
| Metrics | Service discovery | Static scrape config |
| Config | ConfigMaps | `/opt/tori/conf/` files |
| Scaling | HPA | Manual (more nodes) |

## 📈 Performance Impact

Same as container version:
- CPU: +2-3% overhead
- Memory: +400MB for buffers
- Disk: ~200MB for spectral DB
- Network: Minimal (local only)

## 🔧 Operational Tips

1. **Test on one node first**
   ```bash
   ansible-playbook -i inventory.ini beyond_deploy.yml --limit tori-node-1
   ```

2. **Monitor fill ratios**
   ```bash
   watch 'curl -s localhost:9090/metrics | grep fill_ratio'
   ```

3. **Tune after stability**
   - Week 1: Conservative settings
   - Week 2+: Gradually increase limits

4. **Use your existing backup strategy**
   - The `/opt/tori/releases/` structure fits your current backup
   - Each patch creates `.backup_*` files

## ✅ You're Ready!

Your bare-metal TORI setup now has everything needed for Beyond Metacognition:
- ✅ Automated deployment via Ansible
- ✅ systemd service management
- ✅ Prometheus/Loki monitoring
- ✅ Auto-rollback safety
- ✅ Easy manual rollback
- ✅ No cloud dependencies

Run `./torictl_beyond deploy-beyond` when ready to evolve TORI's consciousness! 🌌🔧
