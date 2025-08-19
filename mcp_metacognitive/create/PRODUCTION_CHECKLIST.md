# MCP Server Creator v2 - Production Checklist

Based on the code review v2, here's the updated comprehensive checklist for deploying the MCP server creation system:

## ✅ Pre-Deployment Checklist

### 1. Directory Structure
- [ ] Verify `${IRIS_ROOT}\mcp_metacognitive\` exists
- [ ] Run `python create/setup.py` to ensure all directories
- [ ] Confirm `data/` directory is created
- [ ] Confirm `agents/` directory structure follows `<agents>/<n>/<n>.py`

### 2. Dependencies
- [ ] Install PyPDF2: `pip install PyPDF2`
- [ ] Install httpx: `pip install httpx`
- [ ] Optional: `pip install scikit-learn numpy` (for advanced features)
  - Note: numpy usage is now properly guarded

### 3. Code Improvements Applied ✅ (v2)
- [x] **Supervisor pattern** for continuous loops with exponential backoff restart
- [x] **Critic hub integration** with `evaluate()` calls
- [x] **Metrics tracking** with ServerMetrics dataclass
- [x] **Configurable intervals** with DEFAULT_ANALYSIS_INTERVAL env var
- [x] **Proper dependency guards** for optional libraries
- [x] **TODOs marked** for future improvements (embeddings, TF-IDF)
- [x] **Health monitoring** with success rate tracking
- [x] **Atomic persistence** patterns maintained

### 4. Configuration
- [ ] Set fast development intervals:
  ```bash
  # Windows
  set DEFAULT_ANALYSIS_INTERVAL=300
  set KAIZEN_ANALYSIS_INTERVAL=300
  set MYSERVER_ANALYSIS_INTERVAL=300
  
  # Linux/Mac
  export DEFAULT_ANALYSIS_INTERVAL=300
  export KAIZEN_ANALYSIS_INTERVAL=300
  export MYSERVER_ANALYSIS_INTERVAL=300
  ```
- [ ] Enable/disable features as needed:
  ```bash
  set MYSERVER_ENABLE_WATCHDOG=true
  set MYSERVER_ENABLE_CRITICS=true
  set MYSERVER_WATCHDOG_TIMEOUT=60
  ```

### 5. Testing
- [ ] Run setup: `python create/setup.py`
- [ ] Run test creation: `python create/test_server_creation.py`
- [ ] Create a test server: `python create/mk_server.py create test "Test server"`
- [ ] Verify with pytest: `pytest -k test`

### 6. Registry Integration
- [ ] Verify agent_registry.py supports the new servers
- [ ] Apply watchdog enhancements if not already present:
  ```python
  from create.watchdog_enhancements import patch_agent_registry
  patch_agent_registry()
  ```
- [ ] Ensure registry wraps tasks in supervisor (see watchdog_enhancements.py)

### 7. Critic Hub Setup
- [ ] Verify `kha/meta_genome/critics/critic_hub.py` exists
- [ ] Confirm `evaluate()` function is available
- [ ] Check critic registration:
  - `{server}_performance` critic (threshold: 0.7)
  - `{server}_health` critic (threshold: 0.7)

### 8. Monitoring Setup
- [ ] Confirm PSI archive logging is working
- [ ] Check WebSocket metrics emission:
  - `{server}_analysis_completed` events
  - `{server}_success_rate` metrics
  - `{server}_loop_restart` events (if crashes occur)
- [ ] Verify critic reports are being evaluated

### 9. Production Deployment
- [ ] Start TORI and watch for:
  ```
  [MCP] Initialized {server} server – analysis every 300 s
  {server} supervisor started
  ```
- [ ] Monitor initial executions for errors
- [ ] Check metrics after first interval (5 minutes in dev)
- [ ] Verify PDF loading if applicable
- [ ] Confirm supervisor restarts on crashes

## 🚀 Quick Commands Reference

```bash
# Setup everything
python create/setup.py

# Create servers
python create/mk_server.py create intent "Intent tracker"
python create/mk_server.py create empathy "Empathy module" paper1.pdf paper2.pdf

# Manage PDFs
python create/mk_server.py add-pdf empathy new_paper.pdf
python create/pdf_manager.py batch-add empathy ./papers/
python create/pdf_manager.py stats

# Windows interactive
create\server_creator.bat
```

## 📊 Metrics to Monitor (v2)

### 1. Performance Metrics
- `{server}_success_rate` (should be >= 0.7)
- `{server}_performance_score` (should be >= 0.7)
- `{server}_error_rate` (should be < 0.3)
- `{server}_timeout_rate` (should be < 0.1)

### 2. Health Indicators
- Supervisor restart events
- Consecutive crash counts
- Backoff durations
- Time since last success

### 3. Critic Evaluations
- Monitor critic hub for health/performance scores
- Check consensus panel integration
- Verify metrics flow to dashboards

## 🔧 Troubleshooting (v2)

### Continuous Loop Crashes
1. Check supervisor logs for restart attempts
2. Look for exponential backoff messages
3. Review crash count and backoff duration
4. Adjust `restart_backoff_base` if needed

### Critic Hub Not Available
1. System gracefully degrades (logs debug message)
2. To enable: ensure `meta_genome/critics/` is set up
3. Import paths must be correct

### Slow Learning in Development
1. Set `DEFAULT_ANALYSIS_INTERVAL=300` (5 minutes)
2. Override per-server: `MYSERVER_ANALYSIS_INTERVAL=60`
3. Don't go below 60s to avoid overload

### Optional Dependencies
1. numpy/sklearn imports are now guarded
2. Features degrade gracefully if not installed
3. Install for full functionality: `pip install scikit-learn numpy`

## 📝 Code Review v2 Compliance

All issues from the v2 review have been addressed:

- ✅ **High**: Supervisor pattern prevents loop crashes
- ✅ **Med**: Critic feedback integrated with evaluate()
- ✅ **Med**: Configurable analysis intervals (env vars)
- ✅ **Low**: Guarded numpy imports
- ✅ **Low**: TODOs added for similarity improvements

## 🎯 Final Deployment Steps

1. Deploy to production environment
2. Monitor supervisor behavior for 24 hours
3. Check critic consensus integration
4. Verify metrics flow to all dashboards
5. Adjust intervals based on performance
6. Scale PDF processing as needed

## 📈 Expected Behavior

- Servers start with supervisor protection
- Crashes trigger exponential backoff restarts
- Metrics flow to critic hub every interval
- Success rates above 70% keep critics happy
- WebSocket emits regular health updates
- System self-heals through supervisor pattern
