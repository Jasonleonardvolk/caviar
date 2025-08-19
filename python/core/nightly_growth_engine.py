# python/core/nightly_growth_engine.py

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Optional, Dict, Any, Callable
import json
from pathlib import Path

from .soliton_memory_integration import EnhancedSolitonMemory, VaultStatus
from .memory_crystallization import MemoryCrystallizer
from .hot_swap_laplacian import HotSwappableLaplacian
from .topology_policy import get_topology_policy

logger = logging.getLogger(__name__)

class NightlyGrowthEngine:
    """
    Orchestrates nightly self-improvement cycles.
    Runs memory consolidation, topology optimization, and growth tasks.
    """
    
    def __init__(self,
                 memory_system: EnhancedSolitonMemory,
                 hot_swap: HotSwappableLaplacian,
                 config_path: Optional[Path] = None):
        
        self.memory = memory_system
        self.hot_swap = hot_swap
        self.crystallizer = MemoryCrystallizer(memory_system, hot_swap)
        self.policy = get_topology_policy(hot_swap)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Scheduling
        self.nightly_hour = self.config.get("nightly_hour", 3)  # 3 AM default
        self.enabled = self.config.get("enabled", True)
        self.last_run = None
        
        # Growth tasks registry
        self.growth_tasks: Dict[str, Callable] = {
            "crystallization": self._run_crystallization,
            "soliton_voting": self._run_soliton_voting,
            "topology_optimization": self._run_topology_optimization,
            "comfort_analysis": self._run_comfort_analysis,
        }
        
        # Task history
        self.history = []
        self.max_history = 30  # Keep 30 days
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or defaults"""
        default_config = {
            "enabled": True,
            "nightly_hour": 3,
            "tasks": ["crystallization", "soliton_voting", "topology_optimization"],
            "crystallization": {
                "enabled": True,
                "hot_threshold": 0.7,
                "cold_threshold": 0.1
            },
            "topology": {
                "auto_morph": True,
                "blend_rate": 0.05
            },
            "logging": {
                "verbose": False,
                "save_reports": True
            }
        }
        
        if config_path and config_path.exists():
            with open(config_path) as f:
                user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
                
        return default_config
        
    async def start(self):
        """Start the growth engine background task"""
        if not self.enabled:
            logger.info("Nightly growth engine is disabled")
            return
            
        logger.info("Starting nightly growth engine")
        asyncio.create_task(self._growth_loop())
        
    async def _growth_loop(self):
        """Main loop checking for nightly execution"""
        while self.enabled:
            try:
                now = datetime.now()
                
                # Check if it's time for nightly run
                if self._should_run(now):
                    await self.run_nightly_cycle()
                    self.last_run = now
                    
                # Sleep until next check (1 hour)
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Growth loop error: {e}")
                await asyncio.sleep(300)  # 5 min on error
                
    def _should_run(self, now: datetime) -> bool:
        """Check if nightly cycle should run"""
        # Check hour
        if now.hour != self.nightly_hour:
            return False
            
        # Check if already run today
        if self.last_run:
            if self.last_run.date() == now.date():
                return False
                
        return True
        
    async def run_nightly_cycle(self) -> Dict[str, Any]:
        """Execute full nightly growth cycle"""
        logger.info("=== NIGHTLY GROWTH CYCLE STARTING ===")
        start_time = datetime.now()
        
        report = {
            "start_time": start_time.isoformat(),
            "tasks_run": [],
            "results": {},
            "errors": []
        }
        
        try:
            # Run each configured task
            for task_name in self.config.get("tasks", []):
                if task_name in self.growth_tasks:
                    logger.info(f"Running task: {task_name}")
                    
                    try:
                        task_result = await self.growth_tasks[task_name]()
                        report["results"][task_name] = task_result
                        report["tasks_run"].append(task_name)
                        
                    except Exception as e:
                        logger.error(f"Task {task_name} failed: {e}")
                        report["errors"].append({
                            "task": task_name,
                            "error": str(e)
                        })
                        
            # Save report
            report["end_time"] = datetime.now().isoformat()
            report["duration"] = (datetime.now() - start_time).total_seconds()
            
            self._save_report(report)
            
        except Exception as e:
            logger.error(f"Nightly cycle failed: {e}")
            report["fatal_error"] = str(e)
            
        finally:
            logger.info("=== NIGHTLY GROWTH CYCLE COMPLETE ===")
            
        return report
        
    async def _run_crystallization(self) -> Dict[str, Any]:
        """Run memory crystallization task"""
        return await self.crystallizer.crystallize()
        
    async def _run_soliton_voting(self) -> Dict[str, Any]:
        """Run soliton voting for memory consensus"""
        logger.info("Running soliton voting")
        
        result = {
            "concepts_evaluated": 0,
            "memories_suppressed": 0,
            "conflicts_resolved": 0
        }
        
        # Group memories by concept
        concept_memories = {}
        for entry in self.memory.memory_entries.values():
            if entry.concept_ids:
                concept = entry.concept_ids[0]
                if concept not in concept_memories:
                    concept_memories[concept] = []
                concept_memories[concept].append(entry)
                
        # Evaluate each concept
        for concept, memories in concept_memories.items():
            result["concepts_evaluated"] += 1
            
            # Separate bright and dark
            bright = [m for m in memories if m.polarity == "bright"]
            dark = [m for m in memories if m.polarity == "dark"]
            
            if dark and bright:
                # Voting: sum amplitudes
                bright_strength = sum(m.amplitude for m in bright)
                dark_strength = sum(m.amplitude for m in dark)
                
                if dark_strength >= bright_strength * 0.8:
                    # Suppress bright memories
                    for m in bright:
                        m.vault_status = VaultStatus.QUARANTINE
                        result["memories_suppressed"] += 1
                        
                    result["conflicts_resolved"] += 1
                    
        return result
        
    async def _run_topology_optimization(self) -> Dict[str, Any]:
        """Run topology optimization based on comfort metrics"""
        logger.info("Running topology optimization")
        
        result = {
            "current_topology": self.hot_swap.current_topology,
            "switches": 0,
            "adjustments": 0
        }
        
        # Get comfort analysis
        comfort_report = await self._analyze_comfort()
        
        # Check if topology switch needed
        if comfort_report["average_stress"] > 0.7:
            # High stress - switch to more relaxed topology
            new_topology = await self.policy.evaluate_topology()
            if new_topology and new_topology != self.hot_swap.current_topology:
                success = await self.policy.execute_switch(new_topology)
                if success:
                    result["switches"] += 1
                    result["new_topology"] = new_topology
                    
        # Apply local adjustments
        adjustments = self._apply_comfort_adjustments(comfort_report)
        result["adjustments"] = len(adjustments)
        
        return result
        
    async def _run_comfort_analysis(self) -> Dict[str, Any]:
        """Analyze soliton comfort and lattice stress"""
        return await self._analyze_comfort()
        
    async def _analyze_comfort(self) -> Dict[str, Any]:
        """Analyze comfort metrics for all solitons"""
        total_stress = 0.0
        total_flux = 0.0
        stressed_count = 0
        
        for entry in self.memory.memory_entries.values():
            # Simple comfort calculation
            stress = 1.0 - entry.stability
            flux = 0.0  # Would calculate from coupling in real implementation
            
            total_stress += stress
            total_flux += flux
            
            if stress > 0.7:
                stressed_count += 1
                
        n = len(self.memory.memory_entries)
        
        return {
            "total_memories": n,
            "average_stress": total_stress / max(1, n),
            "average_flux": total_flux / max(1, n),
            "stressed_count": stressed_count,
            "stress_ratio": stressed_count / max(1, n)
        }
        
    def _apply_comfort_adjustments(self, comfort_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply lattice adjustments based on comfort analysis"""
        adjustments = []
        
        # Simplified - in reality would modify coupling matrix
        if comfort_report["average_stress"] > 0.5:
            # Reduce some couplings
            adjustments.append({
                "type": "reduce_coupling",
                "factor": 0.9,
                "reason": "high_average_stress"
            })
            
        return adjustments
        
    def _save_report(self, report: Dict[str, Any]):
        """Save growth cycle report"""
        self.history.append(report)
        
        # Trim history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            
        # Save to file if configured
        if self.config.get("logging", {}).get("save_reports", True):
            report_path = Path("logs/growth_reports")
            report_path.mkdir(parents=True, exist_ok=True)
            
            filename = f"growth_{report['start_time'].replace(':', '-')}.json"
            with open(report_path / filename, 'w') as f:
                json.dump(report, f, indent=2)
                
    def get_status(self) -> Dict[str, Any]:
        """Get current status of growth engine"""
        return {
            "enabled": self.enabled,
            "nightly_hour": self.nightly_hour,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self._next_run_time().isoformat(),
            "configured_tasks": self.config.get("tasks", []),
            "history_length": len(self.history)
        }
        
    def _next_run_time(self) -> datetime:
        """Calculate next scheduled run time"""
        now = datetime.now()
        next_run = now.replace(hour=self.nightly_hour, minute=0, second=0, microsecond=0)
        
        if next_run <= now:
            # Already passed today, schedule for tomorrow
            next_run += timedelta(days=1)
            
        return next_run
