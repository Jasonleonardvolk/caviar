#!/usr/bin/env python3
"""
Fix for nightly_growth_engine.py - Implement proper scheduler
Replace sleep loop with APScheduler or asyncio scheduling
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Dict, Any, Optional, List, Callable
import schedule
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class NightlyGrowthEngine:
    """
    Nightly growth engine with proper scheduling instead of sleep loops
    Uses asyncio for scheduling and concurrent execution
    """
    
    def __init__(self, memory_system, hot_swap_system):
        self.memory_system = memory_system
        self.hot_swap = hot_swap_system
        
        # Configuration
        self.enabled = True
        self.start_hour = 3  # 3 AM by default
        self.max_duration_hours = 2
        self.cpu_limit_percent = 50
        
        # Scheduler state
        self.scheduler = None
        self.is_running = False
        self.last_run_time = None
        self.next_run_time = None
        self.current_task = None
        
        # Task registry
        self.tasks = self._init_default_tasks()
        
        # Execution history
        self.execution_history = []
        
        # Resource management
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    def _init_default_tasks(self) -> List[Dict[str, Any]]:
        """Initialize default nightly tasks"""
        return [
            {
                "name": "switch_to_consolidating",
                "priority": 1,
                "function": self._task_switch_to_consolidating,
                "timeout": 60,  # seconds
                "enabled": True
            },
            {
                "name": "crystallize_memories",
                "priority": 2,
                "function": self._task_crystallize_memories,
                "timeout": 600,  # 10 minutes
                "enabled": True
            },
            {
                "name": "optimize_topology",
                "priority": 3,
                "function": self._task_optimize_topology,
                "timeout": 300,  # 5 minutes
                "enabled": True
            },
            {
                "name": "prune_weak_couplings",
                "priority": 4,
                "function": self._task_prune_couplings,
                "timeout": 300,
                "enabled": True
            },
            {
                "name": "return_to_stable",
                "priority": 5,
                "function": self._task_return_to_stable,
                "timeout": 60,
                "enabled": True
            }
        ]
    
    async def start(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Nightly growth engine already running")
            return
        
        self.is_running = True
        self._calculate_next_run()
        
        # Start the scheduler task
        asyncio.create_task(self._scheduler_loop())
        logger.info(f"Nightly growth engine started. Next run: {self.next_run_time}")
    
    async def stop(self):
        """Stop the scheduler"""
        self.is_running = False
        if self.current_task:
            self.current_task.cancel()
        self.executor.shutdown(wait=False)
        logger.info("Nightly growth engine stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop using asyncio"""
        while self.is_running:
            try:
                now = datetime.now()
                
                # Check if it's time to run
                if self.next_run_time and now >= self.next_run_time:
                    await self._run_nightly_tasks()
                    self._calculate_next_run()
                
                # Sleep until next check (1 minute)
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(300)  # 5 minute backoff on error
    
    def _calculate_next_run(self):
        """Calculate the next run time"""
        now = datetime.now()
        next_run = now.replace(hour=self.start_hour, minute=0, second=0, microsecond=0)
        
        # If the time has passed today, schedule for tomorrow
        if next_run <= now:
            next_run += timedelta(days=1)
        
        self.next_run_time = next_run
        logger.debug(f"Next nightly run scheduled for {self.next_run_time}")
    
    async def _run_nightly_tasks(self):
        """Run all enabled nightly tasks"""
        start_time = datetime.now()
        logger.info("=== Starting Nightly Growth Tasks ===")
        
        # Sort tasks by priority
        enabled_tasks = [t for t in self.tasks if t["enabled"]]
        enabled_tasks.sort(key=lambda x: x["priority"])
        
        results = {}
        
        for task in enabled_tasks:
            # Check if we've exceeded time limit
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > self.max_duration_hours * 3600:
                logger.warning(f"Time limit exceeded, skipping remaining tasks")
                break
            
            # Run task with timeout
            try:
                self.current_task = asyncio.create_task(
                    self._run_task_with_timeout(task)
                )
                result = await self.current_task
                results[task["name"]] = result
                
            except asyncio.TimeoutError:
                logger.error(f"Task {task['name']} timed out")
                results[task["name"]] = {"status": "timeout"}
            except Exception as e:
                logger.error(f"Task {task['name']} failed: {e}")
                results[task["name"]] = {"status": "error", "error": str(e)}
            finally:
                self.current_task = None
        
        # Record execution
        end_time = datetime.now()
        execution_record = {
            "start_time": start_time,
            "end_time": end_time,
            "duration": (end_time - start_time).total_seconds(),
            "results": results
        }
        
        self.execution_history.append(execution_record)
        if len(self.execution_history) > 30:  # Keep last 30 runs
            self.execution_history = self.execution_history[-30:]
        
        self.last_run_time = start_time
        
        logger.info(f"=== Nightly Growth Complete in {execution_record['duration']:.1f}s ===")
    
    async def _run_task_with_timeout(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single task with timeout"""
        logger.info(f"Running task: {task['name']}")
        
        try:
            # Run the task function with timeout
            result = await asyncio.wait_for(
                task["function"](),
                timeout=task["timeout"]
            )
            
            logger.info(f"Task {task['name']} completed successfully")
            return {"status": "success", "result": result}
            
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            logger.error(f"Task {task['name']} error: {e}")
            raise
    
    # Task implementations
    async def _task_switch_to_consolidating(self) -> Dict[str, Any]:
        """Switch to small-world topology for consolidation"""
        if hasattr(self.memory_system, 'lattice'):
            self.hot_swap.switch_topology("small_world")
            return {"topology": "small_world", "switched": True}
        return {"switched": False}
    
    async def _task_crystallize_memories(self) -> Dict[str, Any]:
        """Run memory crystallization"""
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self.memory_system.nightly_crystallization
        )
        return result
    
    async def _task_optimize_topology(self) -> Dict[str, Any]:
        """Analyze and optimize topology based on comfort metrics"""
        # Analyze comfort metrics
        if hasattr(self.memory_system, 'memory_entries'):
            total_stress = 0
            count = 0
            
            for memory in self.memory_system.memory_entries.values():
                if hasattr(memory, 'comfort_metrics'):
                    total_stress += memory.comfort_metrics.stress
                    count += 1
            
            avg_stress = total_stress / max(1, count)
            
            # Decide on topology adjustment
            if avg_stress > 0.7:
                # High stress - switch to more relaxed topology
                self.hot_swap.switch_topology("kagome")
                return {"action": "stress_reduction", "new_topology": "kagome"}
            elif avg_stress < 0.3:
                # Low stress - can handle more complex topology
                self.hot_swap.switch_topology("hexagonal")
                return {"action": "efficiency_boost", "new_topology": "hexagonal"}
        
        return {"action": "none", "reason": "metrics_optimal"}
    
    async def _task_prune_couplings(self) -> Dict[str, Any]:
        """Prune weak couplings to save memory"""
        pruned = 0
        
        if hasattr(self.memory_system, 'lattice'):
            lattice = self.memory_system.lattice
            if hasattr(lattice, 'coupling_matrix'):
                # Find weak couplings
                weak_threshold = 0.01
                keys_to_remove = []
                
                for key, value in lattice.coupling_matrix.items():
                    if abs(value) < weak_threshold:
                        keys_to_remove.append(key)
                
                # Remove weak couplings
                for key in keys_to_remove:
                    del lattice.coupling_matrix[key]
                    pruned += 1
        
        return {"pruned": pruned}
    
    async def _task_return_to_stable(self) -> Dict[str, Any]:
        """Return to stable kagome topology"""
        self.hot_swap.switch_topology("kagome")
        return {"topology": "kagome", "switched": True}
    
    def force_run_now(self):
        """Force an immediate run of nightly tasks"""
        logger.info("Forcing immediate nightly task run")
        asyncio.create_task(self._run_nightly_tasks())
    
    def add_task(self, name: str, function: Callable, priority: int = 10, 
                 timeout: int = 300, enabled: bool = True):
        """Add a custom task to the nightly routine"""
        task = {
            "name": name,
            "priority": priority,
            "function": function,
            "timeout": timeout,
            "enabled": enabled
        }
        
        self.tasks.append(task)
        self.tasks.sort(key=lambda x: x["priority"])
        
        logger.info(f"Added nightly task: {name} (priority {priority})")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the growth engine"""
        return {
            "enabled": self.enabled,
            "is_running": self.is_running,
            "next_run": self.next_run_time.isoformat() if self.next_run_time else None,
            "last_run": self.last_run_time.isoformat() if self.last_run_time else None,
            "current_task": self.current_task is not None,
            "task_count": len([t for t in self.tasks if t["enabled"]]),
            "recent_executions": [
                {
                    "start": ex["start_time"].isoformat(),
                    "duration": ex["duration"],
                    "task_results": list(ex["results"].keys())
                }
                for ex in self.execution_history[-5:]  # Last 5 runs
            ]
        }
    
    def configure(self, start_hour: Optional[int] = None,
                  max_duration_hours: Optional[float] = None,
                  cpu_limit_percent: Optional[int] = None):
        """Update configuration"""
        if start_hour is not None:
            self.start_hour = start_hour
            self._calculate_next_run()
        
        if max_duration_hours is not None:
            self.max_duration_hours = max_duration_hours
        
        if cpu_limit_percent is not None:
            self.cpu_limit_percent = cpu_limit_percent
        
        logger.info(f"Updated nightly growth configuration")
