# TORI Enhanced Debugging Framework

## Overview
This enhanced debugging framework provides systematic automation and intelligent recovery mechanisms for the TORI system. It addresses the weaknesses in the previous debugging approach by implementing:

- **Root cause analysis** instead of just symptom treatment
- **Automated diagnostics** with prioritized issue detection
- **Self-healing mechanisms** for common problems
- **Continuous health monitoring** with alerting
- **Progressive stabilization** through a three-tier approach

## Quick Start

### 1. Run Complete Stabilization (Recommended)
```bash
cd ${IRIS_ROOT}
python debugging_enhanced/stabilize_tori.py --auto-approve
```

This will:
- Run comprehensive diagnostics
- Apply all automated fixes
- Validate the fixes
- Generate a detailed report

### 2. Monitor System Health
```bash
python debugging_enhanced/monitor_health.py
```

This provides real-time monitoring of all TORI components with alerts.

## Framework Components

### 1. Enhanced Diagnostic System (`enhanced_diagnostic_system.py`)
Performs comprehensive system analysis with automatic issue prioritization.

**Features:**
- Dependency validation
- Port availability checking
- Configuration validation
- File permission verification
- Component isolation testing

**Usage:**
```bash
python debugging_enhanced/enhanced_diagnostic_system.py
```

**Output:**
- Health score (0-100)
- Issues categorized by severity
- Automated fix recommendations
- JSON report with detailed findings

### 2. Automated Fixes (`automated_fixes.py`)
Implements automatic resolution for common issues.

**Fixes Available:**
- Missing Python dependencies installation
- Vite proxy configuration for API/WebSocket
- Missing API endpoints (Soliton routes)
- WebSocket endpoint creation
- WebGPU shader barrier corrections
- TailwindCSS custom utility definitions
- Directory structure creation

**Usage:**
```bash
python debugging_enhanced/automated_fixes.py
```

### 3. System Stabilizer (`stabilize_tori.py`)
Main orchestrator that coordinates diagnostics, fixes, and validation.

**Options:**
```bash
# Full automatic mode
python debugging_enhanced/stabilize_tori.py --auto-approve

# Diagnostics only
python debugging_enhanced/stabilize_tori.py --skip-fixes

# Skip validation
python debugging_enhanced/stabilize_tori.py --skip-validation
```

### 4. Health Monitor (`monitor_health.py`)
Real-time monitoring with component health tracking.

**Features:**
- HTTP and WebSocket endpoint monitoring
- Response time tracking
- Alert generation (after 3 consecutive failures)
- Historical data collection
- Health score calculation

**Usage:**
```bash
# Continuous monitoring (5-second intervals)
python debugging_enhanced/monitor_health.py

# Custom interval
python debugging_enhanced/monitor_health.py --interval 10

# Generate report
python debugging_enhanced/monitor_health.py --report
```

## Enhanced Three-Tier Plan

### Tier 1: Immediate Stabilization (2-4 hours)
**Goal:** Achieve 100% component communication with zero critical errors

**Automated Actions:**
1. Install missing dependencies (torch, deepdiff, sympy, PyPDF2)
2. Fix port conflicts with intelligent fallback
3. Configure Vite proxy for API/WebSocket
4. Add missing API endpoints
5. Fix WebGPU shader compilation errors
6. Validate all components

**Success Metric:** Health score ≥ 80/100

### Tier 2: Architectural Enhancement (1-2 days)
**Goal:** <100ms response time, support for 100+ concurrent users

**Enhancements:**
1. Celery task queue with Redis broker
2. Lazy loading framework
3. Resource pool management
4. Performance monitoring integration

**Success Metric:** Performance score ≥ 90/100

### Tier 3: Continuous Improvement (Ongoing)
**Goal:** 99.9% uptime with automatic issue resolution

**Features:**
1. Self-healing framework
2. Predictive maintenance
3. Automated testing pipeline
4. Intelligent logging and analysis

**Success Metric:** Resilience score ≥ 95/100

## Issue Resolution Guide

### Critical Issues

#### Missing Dependencies
```bash
# Automatic fix
python debugging_enhanced/automated_fixes.py --install-deps

# Manual fix
pip install torch torchvision torchaudio deepdiff sympy PyPDF2
```

#### Port Conflicts
```bash
# Windows: Find and kill process
netstat -ano | findstr :8765
taskkill /PID <PID> /F

# Or use smart port manager in enhanced launcher
```

#### Missing API Endpoints
The automated fix will patch your API files to add:
- `/api/soliton/init`
- `/api/soliton/stats/{user}`
- `/api/soliton/embed`
- `/api/avatar/updates` (WebSocket)

### High Priority Issues

#### Vite Proxy Configuration
Updates `vite.config.js` to properly proxy API calls and WebSocket connections.

#### WebGPU Shader Errors
Refactors shader code to move `workgroupBarrier()` calls outside loops.

## Monitoring and Alerts

### Health Metrics
- **Component Status**: healthy, degraded, failed, unknown
- **Response Time**: Measured for each endpoint
- **Availability**: Percentage uptime per component
- **Health Score**: Overall system health (0-100)

### Alert Thresholds
- 3 consecutive failures trigger an alert
- Critical components weighted higher in health score
- Automatic alert generation for monitoring dashboards

## Best Practices

1. **Always run diagnostics first**
   ```bash
   python debugging_enhanced/enhanced_diagnostic_system.py
   ```

2. **Review automated fixes before applying**
   - Backups are created with `.backup` extension
   - Check the fix summary before proceeding

3. **Monitor after stabilization**
   ```bash
   python debugging_enhanced/monitor_health.py
   ```

4. **Use health snapshots for trend analysis**
   - Located in `debugging_enhanced/health_snapshots/`
   - Daily rotation with JSONL format

5. **Implement gradual rollout**
   - Test fixes in development first
   - Use blue-green deployment for production

## Troubleshooting

### Stabilization fails
1. Check the logs in `debugging_enhanced/logs/`
2. Run diagnostics with verbose mode
3. Apply fixes individually instead of all at once

### Component won't start
1. Check port availability
2. Verify dependencies installed correctly
3. Check configuration files for syntax errors

### Health score remains low
1. Review critical issues in diagnostic report
2. Check for external dependencies (Redis, etc.)
3. Ensure all services have proper permissions

## Advanced Usage

### Custom Component Configuration
Edit component definitions in the monitoring scripts:

```python
self.components = {
    "custom_service": {
        "url": "http://localhost:9000/health",
        "timeout": 5,
        "critical": True
    }
}
```

### Extending Automated Fixes
Add new fix methods to `TORIAutoFixer` class:

```python
def fix_custom_issue(self):
    """Fix description"""
    # Implementation
    self.fixes_applied.append("Fixed custom issue")
    return True
```

## Integration with CI/CD

### Pre-deployment Check
```bash
python debugging_enhanced/enhanced_diagnostic_system.py
if [ $? -ne 0 ]; then
    echo "System unhealthy, aborting deployment"
    exit 1
fi
```

### Post-deployment Validation
```bash
python debugging_enhanced/monitor_health.py --report
# Parse JSON report for automated decisions
```

## Future Enhancements

1. **Machine Learning Integration**
   - Predictive failure detection
   - Automatic fix generation from patterns

2. **Distributed Monitoring**
   - Multi-node health aggregation
   - Centralized alerting system

3. **Performance Profiling**
   - Automatic bottleneck detection
   - Resource optimization suggestions

## Support

For issues or questions:
1. Check the diagnostic reports in `debugging_enhanced/`
2. Review health snapshots for historical data
3. Enable debug logging in the scripts
4. Consult the Enhanced Three-Tier Plan for systematic approach

Remember: The goal is to transform TORI from a complex, failure-prone system into a self-managing, resilient platform that improves automatically over time.
