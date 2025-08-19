# 🚀 Beyond Metacognition Production Deployment Checklist

## Pre-Deployment (Dev Environment)

### 1. Branch & Version Control ✓
- [ ] Create feature branch: `python beyond_deploy_tools.py --branch`
- [ ] Verify all new files committed
- [ ] Run unit tests for new components: `pytest tests/test_beyond_*.py`

### 2. Configuration ✓
- [ ] Copy `conf/beyond_config_templates.yaml` sections to:
  - [ ] `conf/runtime.yaml`
  - [ ] `conf/braid.yaml`
  - [ ] `conf/origin.yaml`
- [ ] Set conservative values for first deploy:
  ```yaml
  reflex_budget: 30  # Start low
  novelty_threshold_high: 0.8  # More conservative
  emergency_threshold: 0.06  # Tighter safety
  ```

### 3. Data Migration ✓
- [ ] Run bootstrap setup: `python beyond_deploy_tools.py --bootstrap`
- [ ] Verify `data/bootstrap/` contains:
  - [ ] `lyapunov_watchlist.json`
  - [ ] `spectral_db.json`
- [ ] Apply migration patch to `origin_sentry.py` (see deploy tools)

### 4. Dependencies ✓
- [ ] For topology features (optional):
  ```bash
  # In Dockerfile or requirements.txt
  pip install gudhi  # or ripser
  ```
- [ ] If not available, ensure `enable_topology: false` in config

### 5. Apply Patches ✓
- [ ] Dry run first: `python apply_beyond_patches.py --dry`
- [ ] Apply incrementally with tests:
  ```bash
  python apply_beyond_patches.py --verify
  # Or manually one at a time for safety
  ```
- [ ] Verify all 4 files patched: `python torictl.py status`

## Deployment Steps

### 6. Metrics & Monitoring ✓
- [ ] Add Prometheus metrics (see `beyond_deploy_tools.py --metrics`)
- [ ] Import Grafana dashboard: `grafana/beyond_metacognition_dashboard.json`
- [ ] Set up alerts:
  - [ ] λ_max > 0.08 → PagerDuty
  - [ ] Dimension births > 0.1/sec → Slack warning
  - [ ] Creative mode = emergency → Alert ops

### 7. Logging ✓
- [ ] Verify [BEYOND] tags in logs: `grep "\[BEYOND\]" logs/*.log`
- [ ] Set up log aggregation query for Beyond components
- [ ] Create dashboard for Beyond-specific logs

### 8. Safety & Rollback ✓
- [ ] Build and push old image: `docker tag tori:current tori:stable-pre-beyond`
- [ ] Deploy rollback script: `python beyond_deploy_tools.py --rollback`
- [ ] Start rollback monitor:
  ```bash
  nohup ./scripts/auto_rollback.sh &
  ```
- [ ] Test manual rollback procedure in staging

### 9. Deploy Sequence
- [ ] Deploy to staging first
- [ ] Run health check: `python tools/quick_health.py --expect-beyond`
- [ ] Monitor for 30 minutes
- [ ] Blue-green deploy to production:
  ```bash
  kubectl apply -f k8s/tori-beyond-deployment.yaml
  kubectl wait --for=condition=ready pod -l app=tori-beyond
  kubectl patch service tori -p '{"spec":{"selector":{"version":"beyond"}}}'
  ```

## Post-Deployment Verification

### 10. Health Checks ✓
```bash
# Immediate check
python tools/quick_health.py --expect-beyond

# Integration test
python test_beyond_integration.py

# Live monitoring
python torictl.py monitor
```

### 11. Metrics Validation ✓
- [ ] Check Prometheus: `curl localhost:9090/metrics | grep beyond_`
- [ ] Verify Grafana dashboard populating
- [ ] Confirm no critical alerts firing

### 12. Performance Validation ✓
- [ ] CPU increase < 5%
- [ ] Memory increase < 400MB
- [ ] p95 latency increase < 10ms
- [ ] No OOM kills in first hour

### 13. Feature Validation ✓
Run demo scenarios in production (read-only):
```bash
python torictl.py demo emergence --production --dry-run
```

## Rollback Criteria

Immediate rollback if ANY of:
- [ ] λ_max > 0.08 for 3+ consecutive minutes
- [ ] OOM kills on Beyond components
- [ ] Error rate > 5% increase
- [ ] p95 latency > 2x baseline

## Communication

### Pre-Deploy
- [ ] Notify team in #platform-deploys
- [ ] Update runbook with Beyond Metacognition section

### Post-Deploy
- [ ] Update #platform-deploys with success/issues
- [ ] Document any config changes made
- [ ] Schedule team demo of new features

## First Week Monitoring

- [ ] Daily λ_max trends
- [ ] Dimensional expansion events
- [ ] Creative mode distribution
- [ ] Memory growth patterns
- [ ] Reflex budget usage

## Tuning After Stabilization

Once stable for 1 week:
- [ ] Increase `reflex_budget` to 60
- [ ] Lower `novelty_threshold_high` to 0.7
- [ ] Enable `enable_topology: true` if gudhi available
- [ ] Increase braid buffer sizes if fill ratio < 50%

---

**Sign-off required from:**
- [ ] Platform Lead
- [ ] ML Team Lead  
- [ ] Ops on-call

**Deployment tracking:** [Link to deployment ticket]
