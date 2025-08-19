#!/usr/bin/env python3
"""
Prometheus Metrics Integration for TORI/KHA
Provides comprehensive metrics without containers or databases
Uses file-based metric storage with MCP server compatibility
"""

import time
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, deque
import threading
import asyncio
from enum import Enum
from dataclasses import dataclass, asdict

# Try to import prometheus_client
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest,
        write_to_textfile, REGISTRY
    )
    from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not available - using file-based metrics")

logger = logging.getLogger(__name__)

# Metric types for file-based fallback
class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: float
    metric_type: MetricType

class TORIMetricsCollector:
    """
    Comprehensive metrics collector for TORI/KHA system
    Supports both Prometheus and file-based metrics
    """
    
    def __init__(self, storage_path: Path = Path("data/metrics")):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # File-based metrics storage
        self.metrics_file = self.storage_path / "tori_metrics.jsonl"
        self.summary_file = self.storage_path / "metrics_summary.json"
        
        # In-memory buffers
        self.metric_buffers = defaultdict(lambda: deque(maxlen=10000))
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        
        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
        else:
            logger.info("Using file-based metrics (Prometheus not available)")
        
        # Background writer thread
        self.writer_thread = threading.Thread(target=self._metric_writer_loop, daemon=True)
        self.writer_thread.start()
        
        # Register default metrics
        self._register_default_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metric objects"""
        # Chaos metrics
        self.chaos_events = Counter(
            'tori_chaos_events_total',
            'Total chaos events by type',
            ['chaos_type', 'module']
        )
        
        self.chaos_energy_consumed = Counter(
            'tori_chaos_energy_consumed_total',
            'Total energy consumed in chaos operations',
            ['module']
        )
        
        self.chaos_efficiency = Histogram(
            'tori_chaos_efficiency_ratio',
            'Chaos efficiency ratio distribution',
            buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0)
        )
        
        # Eigenvalue metrics
        self.eigenvalue_max = Gauge(
            'tori_eigenvalue_max',
            'Maximum eigenvalue (stability indicator)'
        )
        
        self.eigenvalue_computation_time = Histogram(
            'tori_eigenvalue_computation_seconds',
            'Eigenvalue computation time',
            ['backend']  # cpu, gpu_pytorch, gpu_cupy
        )
        
        # Cognitive processing metrics
        self.query_processing_time = Histogram(
            'tori_query_processing_seconds',
            'Query processing time distribution',
            ['query_type', 'chaos_enabled']
        )
        
        self.cognitive_state_stability = Gauge(
            'tori_cognitive_state_stability',
            'Current cognitive state stability score'
        )
        
        # Memory metrics
        self.memory_entries = Gauge(
            'tori_memory_entries',
            'Number of memory entries by type',
            ['memory_type']
        )
        
        self.memory_operations = Counter(
            'tori_memory_operations_total',
            'Memory operations by type',
            ['operation', 'memory_type']
        )
        
        # Safety metrics
        self.safety_level = Gauge(
            'tori_safety_level',
            'Current safety level (0=emergency, 1=critical, 2=degraded, 3=nominal, 4=optimal)'
        )
        
        self.safety_interventions = Counter(
            'tori_safety_interventions_total',
            'Total safety interventions by type',
            ['intervention_type']
        )
        
        # System info
        self.system_info = Info(
            'tori_system',
            'TORI system information'
        )
        
        self.system_info.info({
            'version': '2.0.0',
            'chaos_enabled': 'true',
            'storage_type': 'file_based'
        })
    
    def _register_default_metrics(self):
        """Register default metric collectors"""
        # System resource metrics
        self.register_gauge('system_cpu_percent', 'CPU usage percentage')
        self.register_gauge('system_memory_mb', 'Memory usage in MB')
        self.register_gauge('system_disk_free_gb', 'Free disk space in GB')
        
        # Processing metrics
        self.register_counter('queries_processed', 'Total queries processed')
        self.register_counter('errors_total', 'Total errors by type', ['error_type'])
        
        # Uptime metric
        self.start_time = time.time()
        self.register_gauge('uptime_seconds', 'System uptime in seconds')
    
    def register_counter(self, name: str, description: str, labels: List[str] = None):
        """Register a counter metric"""
        if PROMETHEUS_AVAILABLE:
            # Dynamic Prometheus counter creation would require custom collector
            pass
        
        # Always track in file-based system
        self.counters[name] = 0.0
        logger.debug(f"Registered counter: {name}")
    
    def register_gauge(self, name: str, description: str, labels: List[str] = None):
        """Register a gauge metric"""
        self.gauges[name] = 0.0
        logger.debug(f"Registered gauge: {name}")
    
    def register_histogram(self, name: str, description: str, buckets: List[float] = None):
        """Register a histogram metric"""
        self.histograms[name] = []
        logger.debug(f"Registered histogram: {name}")
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        if PROMETHEUS_AVAILABLE:
            # Use appropriate Prometheus counter
            if name == 'chaos_events':
                self.chaos_events.labels(**labels).inc(value)
            elif name == 'memory_operations':
                self.memory_operations.labels(**labels).inc(value)
            elif name == 'safety_interventions':
                self.safety_interventions.labels(**labels).inc(value)
        
        # File-based tracking
        key = self._make_key(name, labels)
        self.counters[key] += value
        
        # Record metric point
        self._record_metric(MetricPoint(
            name=name,
            value=value,
            labels=labels or {},
            timestamp=time.time(),
            metric_type=MetricType.COUNTER
        ))
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric"""
        if PROMETHEUS_AVAILABLE:
            # Use appropriate Prometheus gauge
            if name == 'eigenvalue_max':
                self.eigenvalue_max.set(value)
            elif name == 'cognitive_state_stability':
                self.cognitive_state_stability.set(value)
            elif name == 'safety_level':
                self.safety_level.set(value)
            elif name == 'memory_entries' and labels:
                self.memory_entries.labels(**labels).set(value)
        
        # File-based tracking
        key = self._make_key(name, labels)
        self.gauges[key] = value
        
        # Update uptime automatically
        if name == 'uptime_seconds':
            value = time.time() - self.start_time
            self.gauges['uptime_seconds'] = value
        
        # Record metric point
        self._record_metric(MetricPoint(
            name=name,
            value=value,
            labels=labels or {},
            timestamp=time.time(),
            metric_type=MetricType.GAUGE
        ))
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe a histogram metric"""
        if PROMETHEUS_AVAILABLE:
            # Use appropriate Prometheus histogram
            if name == 'chaos_efficiency':
                self.chaos_efficiency.observe(value)
            elif name == 'eigenvalue_computation_time' and labels:
                self.eigenvalue_computation_time.labels(**labels).observe(value)
            elif name == 'query_processing_time' and labels:
                self.query_processing_time.labels(**labels).observe(value)
        
        # File-based tracking
        key = self._make_key(name, labels)
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)
        
        # Keep bounded size
        if len(self.histograms[key]) > 10000:
            self.histograms[key] = self.histograms[key][-5000:]
        
        # Record metric point
        self._record_metric(MetricPoint(
            name=name,
            value=value,
            labels=labels or {},
            timestamp=time.time(),
            metric_type=MetricType.HISTOGRAM
        ))
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create unique key for metric with labels"""
        if not labels:
            return name
        
        label_str = ','.join(f'{k}={v}' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _record_metric(self, metric: MetricPoint):
        """Record metric point to buffer"""
        self.metric_buffers[metric.name].append(metric)
    
    def _metric_writer_loop(self):
        """Background thread to write metrics to disk"""
        while True:
            try:
                time.sleep(10)  # Write every 10 seconds
                self._write_metrics_to_disk()
                self._write_summary()
            except Exception as e:
                logger.error(f"Metric writer error: {e}")
    
    def _write_metrics_to_disk(self):
        """Write buffered metrics to JSONL file"""
        if not any(self.metric_buffers.values()):
            return
        
        metrics_to_write = []
        
        # Collect metrics from buffers
        for name, buffer in self.metric_buffers.items():
            while buffer:
                metric = buffer.popleft()
                metrics_to_write.append({
                    'name': metric.name,
                    'value': metric.value,
                    'labels': metric.labels,
                    'timestamp': metric.timestamp,
                    'type': metric.metric_type.value
                })
        
        # Append to JSONL file
        if metrics_to_write:
            with open(self.metrics_file, 'a') as f:
                for metric in metrics_to_write:
                    f.write(json.dumps(metric) + '\n')
    
    def _write_summary(self):
        """Write current metric summary"""
        summary = {
            'timestamp': time.time(),
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histogram_stats': {}
        }
        
        # Calculate histogram statistics
        for key, values in self.histograms.items():
            if values:
                summary['histogram_stats'][key] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'p50': self._percentile(values, 0.5),
                    'p95': self._percentile(values, 0.95),
                    'p99': self._percentile(values, 0.99)
                }
        
        # Write atomic
        temp_file = self.summary_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(summary, f, indent=2)
        temp_file.replace(self.summary_file)
    
    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        if PROMETHEUS_AVAILABLE:
            # Use prometheus_client's format
            return generate_latest(REGISTRY).decode('utf-8')
        else:
            # Generate Prometheus-compatible format manually
            lines = []
            
            # Counters
            for key, value in self.counters.items():
                lines.append(f"tori_{key} {value}")
            
            # Gauges
            for key, value in self.gauges.items():
                lines.append(f"tori_{key} {value}")
            
            # Histograms (simplified)
            for key, values in self.histograms.items():
                if values:
                    stats = {
                        'count': len(values),
                        'sum': sum(values)
                    }
                    lines.append(f"tori_{key}_count {stats['count']}")
                    lines.append(f"tori_{key}_sum {stats['sum']}")
            
            return '\n'.join(lines)
    
    def write_prometheus_file(self, filepath: Optional[Path] = None):
        """Write metrics to Prometheus text file format"""
        if filepath is None:
            filepath = self.storage_path / 'tori_metrics.prom'
        
        if PROMETHEUS_AVAILABLE:
            write_to_textfile(str(filepath), REGISTRY)
        else:
            # Write our custom format
            with open(filepath, 'w') as f:
                f.write(self.export_prometheus())
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        if self.summary_file.exists():
            with open(self.summary_file, 'r') as f:
                return json.load(f)
        
        return {
            'timestamp': time.time(),
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histogram_stats': {}
        }
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.metric_buffers.clear()
        
        # Re-register defaults
        self._register_default_metrics()
        
        logger.info("Metrics reset")

class TORIMetricsDecorator:
    """
    Decorator for automatic metric collection
    """
    
    def __init__(self, collector: TORIMetricsCollector):
        self.collector = collector
    
    def count_calls(self, name: str, labels: Dict[str, str] = None):
        """Decorator to count function calls"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                self.collector.increment_counter(f"{name}_calls", labels=labels)
                try:
                    result = func(*args, **kwargs)
                    self.collector.increment_counter(f"{name}_success", labels=labels)
                    return result
                except Exception as e:
                    self.collector.increment_counter(f"{name}_errors", labels=labels)
                    raise
            return wrapper
        return decorator
    
    def measure_time(self, name: str, labels: Dict[str, str] = None):
        """Decorator to measure execution time"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.collector.observe_histogram(f"{name}_duration", duration, labels)
            
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.collector.observe_histogram(f"{name}_duration", duration, labels)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return wrapper
        return decorator

# Global metrics instance
_metrics_collector = None

def get_metrics_collector() -> TORIMetricsCollector:
    """Get or create global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = TORIMetricsCollector()
    return _metrics_collector

# Convenience decorators
metrics = TORIMetricsDecorator(get_metrics_collector())

# Integration helper for TORI system
class TORIMetricsIntegration:
    """
    Helper class to integrate metrics with TORI production system
    """
    
    def __init__(self, tori_system):
        self.tori = tori_system
        self.collector = get_metrics_collector()
        
        # Start background monitoring
        self.monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def _monitor_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                await asyncio.sleep(5)  # Update every 5 seconds
                
                # Collect system metrics
                status = self.tori.get_status()
                
                # Update gauges
                self.collector.set_gauge('queries_processed', status['statistics']['queries_processed'])
                self.collector.set_gauge('chaos_events', status['statistics']['chaos_events'])
                
                # Safety level
                safety_map = {'emergency': 0, 'critical': 1, 'degraded': 2, 'nominal': 3, 'optimal': 4}
                safety_level = safety_map.get(status['safety']['current_safety_level'].lower(), 2)
                self.collector.set_gauge('safety_level', safety_level)
                
                # Eigenvalue metrics
                eigen_status = status.get('eigensentry', {})
                if 'max_eigenvalue' in eigen_status:
                    self.collector.set_gauge('eigenvalue_max', eigen_status['max_eigenvalue'])
                
                # Memory metrics
                memory_status = self.tori.metacognitive_system.memory_vault.get_status()
                for mem_type, count in memory_status.get('entries_by_type', {}).items():
                    self.collector.set_gauge('memory_entries', count, {'memory_type': mem_type})
                
                # CCL metrics
                ccl_status = status.get('ccl', {})
                if 'efficiency_ratio' in ccl_status:
                    self.collector.observe_histogram('chaos_efficiency', ccl_status['efficiency_ratio'])
                
            except Exception as e:
                logger.error(f"Metrics monitor error: {e}")
    
    def stop(self):
        """Stop monitoring"""
        self.monitor_task.cancel()

# Example Prometheus endpoint server (no containers)
def start_metrics_server(port: int = 9090):
    """
    Start simple HTTP server for Prometheus metrics
    No containers needed - just pure Python
    """
    from http.server import HTTPServer, BaseHTTPRequestHandler
    
    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/metrics':
                collector = get_metrics_collector()
                metrics_text = collector.export_prometheus()
                
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; version=0.0.4')
                self.end_headers()
                self.wfile.write(metrics_text.encode())
            else:
                self.send_response(404)
                self.end_headers()
        
        def log_message(self, format, *args):
            # Suppress default logging
            pass
    
    server = HTTPServer(('', port), MetricsHandler)
    logger.info(f"Metrics server started on port {port}")
    
    # Run in thread
    import threading
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    
    return server

# Example usage
if __name__ == "__main__":
    # Initialize metrics
    collector = get_metrics_collector()
    
    # Start metrics server
    server = start_metrics_server(9090)
    
    # Simulate some metrics
    for i in range(10):
        collector.increment_counter('test_counter', labels={'test': 'true'})
        collector.set_gauge('test_gauge', i * 10)
        collector.observe_histogram('test_histogram', i * 0.1)
        time.sleep(1)
    
    # Export metrics
    print("Prometheus format:")
    print(collector.export_prometheus())
    
    print("\nSummary:")
    print(json.dumps(collector.get_metrics_summary(), indent=2))
    
    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
