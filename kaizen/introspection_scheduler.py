#!/usr/bin/env python3
"""
Kaizen Introspection Scheduler
Implements continuous self-audit loops for TORI system improvement
"""

import asyncio
import time
import json
import psutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

# Introspection intervals
MICRO_PULSE_INTERVAL = 30.0      # 30 seconds
MESO_PULSE_INTERVAL = 900.0      # 15 minutes
MACRO_PULSE_INTERVAL = 43200.0   # 12 hours

# Log paths
LOGS_DIR = Path("logs")
INTROSPECTION_LOG = LOGS_DIR / "introspection_meso.jl"
KAIZEN_PLAN_FILE = Path("kaizen_plan.json")

@dataclass
class SystemMetrics:
    """Metrics collected during introspection"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    lambda_max: float
    error_count: int
    active_tasks: int
    energy_efficiency: float
    phase_coherence: float
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

class IntrospectionScheduler:
    """
    Continuous introspection loop with multi-scale pulses
    Monitors system health and identifies performance gaps
    """
    
    def __init__(self, tori_system=None):
        self.tori_system = tori_system
        self.running = False
        
        # Metric buffers
        self.micro_buffer = deque(maxlen=100)   # Last 100 micro pulses
        self.meso_buffer = deque(maxlen=100)    # Last 100 meso aggregates
        self.macro_history = []                 # Full macro history
        
        # Pulse counters
        self.micro_count = 0
        self.meso_count = 0
        self.macro_count = 0
        
        # Performance baselines
        self.baseline_metrics = None
        self.anomaly_threshold = 2.0  # Standard deviations
        
        # Callbacks for external systems
        self.metric_callbacks: List[Callable] = []
        
        # Ensure logs directory exists
        LOGS_DIR.mkdir(exist_ok=True)
        
    def register_metric_callback(self, callback: Callable):
        """Register callback for metric updates"""
        self.metric_callbacks.append(callback)
        
    async def start(self):
        """Start introspection loops"""
        if self.running:
            logger.warning("Introspection scheduler already running")
            return
            
        self.running = True
        logger.info("Starting introspection scheduler")
        
        # Start all pulse loops
        tasks = [
            asyncio.create_task(self._micro_pulse_loop()),
            asyncio.create_task(self._meso_pulse_loop()),
            asyncio.create_task(self._macro_pulse_loop())
        ]
        
        # Wait for any to complete (or error)
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Introspection error: {e}")
            self.running = False
            
    async def stop(self):
        """Stop introspection loops"""
        self.running = False
        logger.info("Stopping introspection scheduler")
        
    async def _micro_pulse_loop(self):
        """Micro pulse: immediate metrics every ~30s"""
        while self.running:
            try:
                metrics = await self._collect_immediate_metrics()
                self.micro_buffer.append(metrics)
                self.micro_count += 1
                
                # Notify callbacks
                for callback in self.metric_callbacks:
                    try:
                        await callback('micro', metrics)
                    except Exception as e:
                        logger.error(f"Metric callback error: {e}")
                        
                # Log if anomaly detected
                if self._is_anomaly(metrics):
                    logger.warning(f"Anomaly detected: {metrics}")
                    
            except Exception as e:
                logger.error(f"Micro pulse error: {e}")
                
            await asyncio.sleep(MICRO_PULSE_INTERVAL)
            
    async def _meso_pulse_loop(self):
        """Meso pulse: aggregate metrics every ~15min"""
        while self.running:
            await asyncio.sleep(MESO_PULSE_INTERVAL)
            
            try:
                if len(self.micro_buffer) > 0:
                    # Aggregate recent micro pulses
                    aggregated = self._aggregate_metrics(list(self.micro_buffer))
                    self.meso_buffer.append(aggregated)
                    self.meso_count += 1
                    
                    # Write to log file
                    await self._write_meso_log(aggregated)
                    
                    # Update baseline if needed
                    if self.baseline_metrics is None and len(self.meso_buffer) > 10:
                        self._update_baseline()
                        
                    # Notify callbacks
                    for callback in self.metric_callbacks:
                        try:
                            await callback('meso', aggregated)
                        except Exception as e:
                            logger.error(f"Meso callback error: {e}")
                            
            except Exception as e:
                logger.error(f"Meso pulse error: {e}")
                
    async def _macro_pulse_loop(self):
        """Macro pulse: deep analysis every ~12h"""
        while self.running:
            await asyncio.sleep(MACRO_PULSE_INTERVAL)
            
            try:
                # Trigger Kaizen analysis
                await self._trigger_kaizen_analysis()
                self.macro_count += 1
                
                # Notify callbacks
                for callback in self.metric_callbacks:
                    try:
                        await callback('macro', {'analysis_triggered': True})
                    except Exception as e:
                        logger.error(f"Macro callback error: {e}")
                        
            except Exception as e:
                logger.error(f"Macro pulse error: {e}")
                
    async def _collect_immediate_metrics(self) -> SystemMetrics:
        """Collect immediate system metrics"""
        # Basic system stats
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # TORI-specific metrics
        lambda_max = 0.0
        error_count = 0
        active_tasks = 0
        energy_efficiency = 1.0
        phase_coherence = 1.0
        
        if self.tori_system:
            try:
                # Get eigenvalue stability
                if hasattr(self.tori_system, 'eigen_sentry'):
                    status = self.tori_system.eigen_sentry.get_status()
                    lambda_max = status.get('current_max_eigenvalue', 0.0)
                    
                # Get CCL metrics
                if hasattr(self.tori_system, 'ccl'):
                    ccl_status = self.tori_system.ccl.get_status()
                    active_tasks = ccl_status.get('active_tasks', 0)
                    energy_efficiency = ccl_status.get('efficiency_ratio', 1.0)
                    
                # Get error count from logs
                # This is simplified - in production would track actual errors
                error_count = 0
                
            except Exception as e:
                logger.error(f"Failed to collect TORI metrics: {e}")
                
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=memory.used / 1024 / 1024,
            lambda_max=lambda_max,
            error_count=error_count,
            active_tasks=active_tasks,
            energy_efficiency=energy_efficiency,
            phase_coherence=phase_coherence,
            custom_metrics={}
        )
        
    def _aggregate_metrics(self, metrics_list: List[SystemMetrics]) -> Dict[str, Any]:
        """Aggregate a list of metrics"""
        if not metrics_list:
            return {}
            
        # Extract numeric fields
        cpu_values = [m.cpu_percent for m in metrics_list]
        memory_values = [m.memory_percent for m in metrics_list]
        lambda_values = [m.lambda_max for m in metrics_list]
        error_counts = [m.error_count for m in metrics_list]
        efficiency_values = [m.energy_efficiency for m in metrics_list]
        
        return {
            'timestamp': time.time(),
            'window_start': metrics_list[0].timestamp,
            'window_end': metrics_list[-1].timestamp,
            'sample_count': len(metrics_list),
            'cpu': {
                'mean': np.mean(cpu_values),
                'std': np.std(cpu_values),
                'max': np.max(cpu_values)
            },
            'memory': {
                'mean': np.mean(memory_values),
                'std': np.std(memory_values),
                'max': np.max(memory_values)
            },
            'lambda_max': {
                'mean': np.mean(lambda_values),
                'std': np.std(lambda_values),
                'max': np.max(lambda_values)
            },
            'errors': {
                'total': sum(error_counts),
                'rate': sum(error_counts) / len(error_counts)
            },
            'efficiency': {
                'mean': np.mean(efficiency_values),
                'min': np.min(efficiency_values)
            }
        }
        
    async def _write_meso_log(self, aggregated: Dict[str, Any]):
        """Write aggregated metrics to meso log"""
        try:
            # Convert to JSON line
            log_entry = json.dumps(aggregated) + '\n'
            
            # Append to log file
            with open(INTROSPECTION_LOG, 'a') as f:
                f.write(log_entry)
                
        except Exception as e:
            logger.error(f"Failed to write meso log: {e}")
            
    def _update_baseline(self):
        """Update performance baseline from recent history"""
        if len(self.meso_buffer) < 10:
            return
            
        recent = list(self.meso_buffer)[-20:]  # Last 20 meso pulses
        
        self.baseline_metrics = {
            'cpu_mean': np.mean([m['cpu']['mean'] for m in recent]),
            'cpu_std': np.std([m['cpu']['mean'] for m in recent]),
            'memory_mean': np.mean([m['memory']['mean'] for m in recent]),
            'memory_std': np.std([m['memory']['mean'] for m in recent]),
            'lambda_mean': np.mean([m['lambda_max']['mean'] for m in recent]),
            'lambda_std': np.std([m['lambda_max']['mean'] for m in recent]),
            'error_rate_mean': np.mean([m['errors']['rate'] for m in recent]),
            'error_rate_std': np.std([m['errors']['rate'] for m in recent])
        }
        
        logger.info("Updated performance baseline")
        
    def _is_anomaly(self, metrics: SystemMetrics) -> bool:
        """Check if metrics indicate an anomaly"""
        if self.baseline_metrics is None:
            return False
            
        # Check each metric against baseline
        anomalies = []
        
        # CPU anomaly
        cpu_z = abs(metrics.cpu_percent - self.baseline_metrics['cpu_mean']) / (self.baseline_metrics['cpu_std'] + 1e-6)
        if cpu_z > self.anomaly_threshold:
            anomalies.append(f"cpu_z={cpu_z:.2f}")
            
        # Memory anomaly
        mem_z = abs(metrics.memory_percent - self.baseline_metrics['memory_mean']) / (self.baseline_metrics['memory_std'] + 1e-6)
        if mem_z > self.anomaly_threshold:
            anomalies.append(f"memory_z={mem_z:.2f}")
            
        # Eigenvalue anomaly
        lambda_z = abs(metrics.lambda_max - self.baseline_metrics['lambda_mean']) / (self.baseline_metrics['lambda_std'] + 1e-6)
        if lambda_z > self.anomaly_threshold:
            anomalies.append(f"lambda_z={lambda_z:.2f}")
            
        # Error rate anomaly
        if metrics.error_count > self.baseline_metrics['error_rate_mean'] + self.anomaly_threshold * self.baseline_metrics['error_rate_std']:
            anomalies.append(f"errors={metrics.error_count}")
            
        if anomalies:
            logger.warning(f"Anomalies detected: {', '.join(anomalies)}")
            return True
            
        return False
        
    async def _trigger_kaizen_analysis(self):
        """Trigger Kaizen gap analysis and learning"""
        logger.info("Triggering Kaizen analysis")
        
        try:
            # Import gap finder
            from kaizen.miner import find_gaps
            from kaizen.ingest_papers import plan_ingestion
            
            # Find gaps in recent performance
            gaps = find_gaps(INTROSPECTION_LOG)
            
            if gaps:
                logger.info(f"Found {len(gaps)} performance gaps")
                
                # Plan research ingestion
                plan = plan_ingestion(gaps)
                
                # Save plan
                with open(KAIZEN_PLAN_FILE, 'w') as f:
                    json.dump(plan, f, indent=2)
                    
                logger.info(f"Saved Kaizen plan to {KAIZEN_PLAN_FILE}")
                
        except ImportError:
            logger.warning("Kaizen modules not available, skipping analysis")
        except Exception as e:
            logger.error(f"Kaizen analysis failed: {e}")
            
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        return {
            'running': self.running,
            'pulse_counts': {
                'micro': self.micro_count,
                'meso': self.meso_count,
                'macro': self.macro_count
            },
            'buffer_sizes': {
                'micro': len(self.micro_buffer),
                'meso': len(self.meso_buffer),
                'macro': len(self.macro_history)
            },
            'baseline_established': self.baseline_metrics is not None,
            'callbacks_registered': len(self.metric_callbacks)
        }
        
    def get_recent_metrics(self, scale: str = 'micro', count: int = 10) -> List[Any]:
        """Get recent metrics at specified scale"""
        if scale == 'micro':
            return list(self.micro_buffer)[-count:]
        elif scale == 'meso':
            return list(self.meso_buffer)[-count:]
        elif scale == 'macro':
            return self.macro_history[-count:]
        else:
            raise ValueError(f"Unknown scale: {scale}")

# Test function
async def test_introspection():
    """Test the introspection scheduler"""
    print("🔍 Testing Introspection Scheduler")
    print("=" * 50)
    
    # Create scheduler
    scheduler = IntrospectionScheduler()
    
    # Add test callback
    async def test_callback(scale, data):
        print(f"📊 {scale.upper()} pulse: {type(data)}")
        if scale == 'micro' and isinstance(data, SystemMetrics):
            print(f"  CPU: {data.cpu_percent:.1f}%, Memory: {data.memory_percent:.1f}%")
            
    scheduler.register_metric_callback(test_callback)
    
    # Start scheduler
    task = asyncio.create_task(scheduler.start())
    
    # Run for a short time
    print("Running for 2 minutes...")
    await asyncio.sleep(120)
    
    # Stop and get status
    await scheduler.stop()
    status = scheduler.get_status()
    
    print("\n📈 Final Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
        
    # Cancel task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(test_introspection())
