#!/usr/bin/env python3
"""
beyond_prometheus_exporter.py - Custom Prometheus metrics for Beyond Metacognition
Runs as part of TORI's /metrics endpoint or as standalone exporter
"""

from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest
from prometheus_client.core import CollectorRegistry
import time
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent.parent))

# Create metrics
registry = CollectorRegistry()

# Counters
origin_dim_expansions_total = Counter(
    'origin_dim_expansions_total',
    'Total dimensional expansions detected by OriginSentry',
    registry=registry
)

origin_gap_births_total = Counter(
    'origin_gap_births_total', 
    'Total spectral gap births detected',
    registry=registry
)

braid_retrocoherence_events_total = Counter(
    'braid_retrocoherence_events_total',
    'Total retro-coherent labeling events',
    registry=registry
)

creative_injections_total = Counter(
    'creative_injections_total',
    'Total entropy injections by creative feedback',
    registry=registry
)

# Gauges
beyond_lambda_max = Gauge(
    'beyond_lambda_max',
    'Current maximum eigenvalue',
    registry=registry
)

beyond_novelty_score = Gauge(
    'beyond_novelty_score',
    'Current novelty score from OriginSentry',
    registry=registry
)

creative_mode = Gauge(
    'creative_mode',
    'Current creative feedback mode (0=stable, 1=exploring, 0.5=consolidating, -1=emergency)',
    registry=registry
)

braid_buffer_fill_ratio = Gauge(
    'braid_buffer_fill_ratio',
    'Fill ratio of temporal braid buffers',
    ['scale'],  # Label for micro/meso/macro
    registry=registry
)

observer_reflex_remaining = Gauge(
    'observer_reflex_remaining',
    'Remaining reflex budget for self-measurement',
    registry=registry
)

# Histograms
origin_classify_duration = Histogram(
    'origin_classify_duration_seconds',
    'Time to classify spectral state',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    registry=registry
)

braid_aggregation_duration = Histogram(
    'braid_aggregation_duration_seconds',
    'Time to aggregate temporal buffers',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    registry=registry
)

# Info
beyond_info = Info(
    'beyond_metacognition',
    'Beyond Metacognition deployment info',
    registry=registry
)

class BeyondMetricsCollector:
    """Collects metrics from Beyond Metacognition components"""
    
    def __init__(self):
        self.components_available = False
        self._try_import_components()
        
    def _try_import_components(self):
        """Try to import Beyond components"""
        try:
            from alan_backend.origin_sentry import OriginSentry
            from python.core.braid_buffers import get_braiding_engine, TimeScale
            from python.core.observer_synthesis import get_observer_synthesis
            from python.core.creative_feedback import get_creative_feedback
            
            self.origin = OriginSentry()
            self.braiding = get_braiding_engine()
            self.observer = get_observer_synthesis()
            self.creative = get_creative_feedback()
            self.TimeScale = TimeScale
            
            self.components_available = True
            
            # Set info
            beyond_info.info({
                'version': '2.0.0',
                'deployment_date': str(Path(__file__).stat().st_mtime),
                'components': 'origin,braiding,observer,creative'
            })
            
        except ImportError as e:
            print(f"Warning: Beyond components not available: {e}")
            beyond_info.info({
                'version': 'not_deployed',
                'error': str(e)
            })
    
    def collect_metrics(self):
        """Collect current metrics from all components"""
        if not self.components_available:
            return
        
        try:
            # Origin Sentry metrics
            origin_metrics = self.origin.metrics
            beyond_lambda_max.set(origin_metrics.get('lambda_max', 0.0))
            beyond_novelty_score.set(origin_metrics.get('novelty_score', 0.0))
            
            # Set counters to current totals (in production, track deltas)
            origin_dim_expansions_total._value.set(
                origin_metrics.get('dimension_expansions', 0)
            )
            origin_gap_births_total._value.set(
                origin_metrics.get('gap_births', 0)
            )
            
            # Creative feedback metrics
            creative_metrics = self.creative.get_creative_metrics()
            mode_map = {
                'stable': 0, 
                'exploring': 1, 
                'consolidating': 0.5, 
                'emergency': -1
            }
            creative_mode.set(
                mode_map.get(creative_metrics['current_mode'], 0)
            )
            creative_injections_total._value.set(
                creative_metrics.get('total_injections', 0)
            )
            
            # Temporal braiding metrics
            for scale in self.TimeScale:
                buffer = self.braiding.buffers[scale]
                fill_ratio = len(buffer.buffer) / buffer.capacity
                braid_buffer_fill_ratio.labels(scale=scale.value).set(fill_ratio)
            
            # Observer synthesis metrics
            observer_reflex_remaining.set(
                self.observer._get_reflex_budget_remaining()
            )
            
        except Exception as e:
            print(f"Error collecting metrics: {e}")
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        self.collect_metrics()
        return generate_latest(registry)

# Integration with TORI's existing metrics endpoint
def integrate_with_tori_metrics():
    """
    Add this to your existing metrics endpoint handler:
    
    from kha.beyond_prometheus_exporter import BeyondMetricsCollector
    
    collector = BeyondMetricsCollector()
    beyond_metrics = collector.get_metrics()
    
    # Append to your existing metrics
    all_metrics = existing_metrics + b'\n' + beyond_metrics
    """
    pass

# Standalone exporter
if __name__ == "__main__":
    from http.server import HTTPServer, BaseHTTPRequestHandler
    
    collector = BeyondMetricsCollector()
    
    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/metrics':
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; version=0.0.4')
                self.end_headers()
                self.wfile.write(collector.get_metrics())
            else:
                self.send_response(404)
                self.end_headers()
    
    print("Starting Beyond Metacognition metrics exporter on :9091")
    server = HTTPServer(('', 9091), MetricsHandler)
    server.serve_forever()
