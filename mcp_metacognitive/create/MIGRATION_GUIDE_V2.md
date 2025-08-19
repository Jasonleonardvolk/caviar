# MCP Server Creator v1 → v2 Migration Guide

This guide helps you upgrade existing servers created with v1 to incorporate v2 improvements based on the code review.

## 🔄 What's New in v2

### Critical Improvements
1. **Supervisor Pattern**: Continuous loops now auto-restart on crashes
2. **Critic Hub Integration**: Proper `evaluate()` calls for consensus
3. **Metrics Tracking**: ServerMetrics dataclass for health monitoring
4. **Configurable Intervals**: Environment-based configuration
5. **Dependency Guards**: Optional libraries properly protected

## 📋 Migration Steps

### 1. Backup Existing Servers

```bash
# Windows
xcopy /E /I agents agents_backup_v1

# Linux/Mac
cp -r agents agents_backup_v1
```

### 2. Update Existing Server Code

For each server in `agents/*/`, apply these changes:

#### A. Add Metrics Tracking

Add after imports:
```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ServerMetrics:
    """Track server performance metrics"""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_timeouts: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 1.0
        return self.successful_executions / self.total_executions
```

In `__init__`:
```python
self.metrics = ServerMetrics()
self._supervisor_task = None
```

#### B. Replace execute() Method

Replace the entire `execute()` method with the v2 version that includes:
- Metrics tracking
- Proper error counting
- Critic hub submission

#### C. Add Supervisor Pattern

Replace `start()` and `_continuous_loop()` with:

```python
async def start(self):
    """Start the continuous processing loop with supervisor"""
    if self.running:
        logger.warning(f"{self.name} is already running")
        return
    
    self.running = True
    self._supervisor_task = asyncio.create_task(self._supervisor_loop())
    logger.info(f"{self.name} supervisor started")

async def _supervisor_loop(self):
    """Supervisor loop that restarts the main loop on crashes"""
    consecutive_crashes = 0
    
    while self.running:
        try:
            logger.info(f"Starting {self.name} continuous loop")
            self._task = asyncio.create_task(self._continuous_loop())
            await self._task
            consecutive_crashes = 0
            
        except asyncio.CancelledError:
            logger.info(f"{self.name} loop cancelled")
            break
            
        except Exception as e:
            consecutive_crashes += 1
            logger.error(f"{self.name} loop crashed (attempt {consecutive_crashes}): {e}")
            
            backoff = min(
                self.config["restart_backoff_base"] * (2 ** consecutive_crashes),
                3600
            )
            
            logger.info(f"Restarting {self.name} in {backoff} seconds...")
            await asyncio.sleep(backoff)
```

#### D. Add Critic Integration

Add this method:
```python
async def _submit_to_critics(self):
    """Submit metrics to critic hub for evaluation"""
    try:
        from kha.meta_genome.critics.critic_hub import evaluate
        
        critic_report = {
            f"{self.name}_success_rate": self.metrics.success_rate,
            f"{self.name}_performance_score": self.metrics.success_rate,
            f"{self.name}_error_rate": self.error_count / max(1, self.metrics.total_executions),
            f"{self.name}_timeout_rate": self.metrics.total_timeouts / max(1, self.metrics.total_executions),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        evaluate(critic_report)
        
    except ImportError:
        logger.debug("Critic hub not available for evaluation")
    except Exception as e:
        logger.error(f"Failed to submit to critics: {e}")
```

Call it in `execute()` after success and in `on_tick()`.

#### E. Update Configuration

Update `_get_config_with_env()`:
```python
config.update({
    "analysis_interval": int(os.getenv(f"{self.name.upper()}_ANALYSIS_INTERVAL", 
                                      os.getenv("DEFAULT_ANALYSIS_INTERVAL", "300"))),
    "enable_critic_hooks": os.getenv(f"{self.name.upper()}_ENABLE_CRITICS", "true").lower() == "true",
    "restart_backoff_base": 60,
})
```

### 3. Update Critic Hooks

Ensure critic registration includes both performance and health:

```python
@critic(f"{server_name}_health")
def server_health(report: dict):
    """Monitor health based on success rate"""
    success_rate = report.get(f"{server_name}_success_rate", 1.0)
    return success_rate, success_rate >= 0.7
```

### 4. Environment Variables

Create a `.env` file or set:
```bash
# Fast development cycles
DEFAULT_ANALYSIS_INTERVAL=300

# Per-server overrides
KAIZEN_ANALYSIS_INTERVAL=300
EMPATHY_ANALYSIS_INTERVAL=600
INTENT_ANALYSIS_INTERVAL=300
```

### 5. Test Migration

```python
# Test script
import asyncio
from agents.myserver.myserver import MyserverServer

async def test():
    server = MyserverServer()
    
    # Test execution
    result = await server.execute({"test": True})
    print(f"Execute result: {result}")
    
    # Test supervisor
    await server.start()
    await asyncio.sleep(10)
    
    # Check metrics
    print(f"Success rate: {server.metrics.success_rate}")
    print(f"Total executions: {server.metrics.total_executions}")
    
    await server.shutdown()

asyncio.run(test())
```

## 🔍 Verification Checklist

- [ ] Supervisor starts without errors
- [ ] Crashes trigger restart with backoff
- [ ] Metrics track properly
- [ ] Critic reports submitted
- [ ] Environment variables respected
- [ ] Logs show proper supervisor messages

## 🚨 Common Issues

### Import Errors
- Ensure `from dataclasses import dataclass`
- Update relative imports if needed

### Critic Hub Missing
- System degrades gracefully
- Check `meta_genome/critics/` setup

### Old Config Format
- Add new config keys with defaults
- Maintain backward compatibility

## 📊 Before/After Comparison

### v1 Behavior
- Continuous loop crashes = server dies
- No automatic recovery
- Limited health visibility
- Hard-coded intervals

### v2 Behavior
- Supervisor auto-restarts crashed loops
- Exponential backoff prevents thrashing
- Full metrics tracking
- Configurable via environment
- Critic consensus integration

## 🎯 Quick Migration Script

For bulk updates, use this helper:

```python
#!/usr/bin/env python
"""Quick migration helper for v1 → v2"""

import os
import re
from pathlib import Path

def migrate_server(server_path: Path):
    """Apply v2 patterns to a server file"""
    content = server_path.read_text()
    
    # Add imports if missing
    if "from dataclasses import dataclass" not in content:
        content = content.replace(
            "from datetime import datetime",
            "from datetime import datetime\nfrom dataclasses import dataclass"
        )
    
    # Add supervisor pattern marker
    if "_supervisor_task" not in content:
        print(f"[NEEDS MANUAL UPDATE] {server_path}")
        print("  - Add supervisor pattern")
        print("  - Add metrics tracking")
        print("  - Update critic integration")
    
    return content

# Run on all servers
agents_dir = Path("agents")
for server_dir in agents_dir.iterdir():
    if server_dir.is_dir():
        server_file = server_dir / f"{server_dir.name}.py"
        if server_file.exists():
            migrate_server(server_file)
```

## ✅ Migration Complete

After migration, your servers will have:
- Crash resilience through supervisor pattern
- Automatic recovery with exponential backoff
- Full metrics and health tracking
- Critic consensus participation
- Environment-based configuration

Monitor the first 24 hours to ensure stability!
