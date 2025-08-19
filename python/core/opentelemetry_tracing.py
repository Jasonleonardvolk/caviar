#!/usr/bin/env python3
"""
OpenTelemetry Tracing Integration for TORI/KHA
Distributed tracing without containers - file-based trace storage
Compatible with MCP servers and no database requirement
"""

import time
import json
import logging
import asyncio
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import traceback

# Try to import OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider, Span
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor, ConsoleSpanExporter, SpanExporter, SpanExportResult
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logging.warning("OpenTelemetry not available - using file-based tracing")

logger = logging.getLogger(__name__)

# Span status for file-based fallback
class SpanStatus(Enum):
    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"

@dataclass
class TraceSpan:
    """Represents a single trace span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    start_time: float
    end_time: Optional[float]
    attributes: Dict[str, Any]
    events: List[Dict[str, Any]]
    status: SpanStatus
    status_description: Optional[str]
    
    def duration_ms(self) -> Optional[float]:
        """Calculate duration in milliseconds"""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

class FileSpanExporter(SpanExporter if OTEL_AVAILABLE else object):
    """
    Custom span exporter that writes to files instead of network
    Compatible with OpenTelemetry SDK
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.traces_dir = storage_path / 'traces'
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        
        # Current trace file (rotate hourly)
        self.current_hour = datetime.now(timezone.utc).hour
        self.trace_file = self._get_trace_file()
        
        # Buffer for batch writing
        self.buffer = []
        self.buffer_lock = threading.Lock()
        
        # Start writer thread
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
    
    def _get_trace_file(self) -> Path:
        """Get current trace file path"""
        now = datetime.now(timezone.utc)
        filename = f"traces_{now.strftime('%Y%m%d_%H')}.jsonl"
        return self.traces_dir / filename
    
    def export(self, spans) -> SpanExportResult:
        """Export spans to file storage"""
        try:
            with self.buffer_lock:
                for span in spans:
                    # Convert OTEL span to our format
                    trace_data = {
                        'trace_id': format(span.context.trace_id, '032x'),
                        'span_id': format(span.context.span_id, '016x'),
                        'parent_span_id': format(span.parent.span_id, '016x') if span.parent else None,
                        'name': span.name,
                        'start_time': span.start_time / 1e9,  # Convert to seconds
                        'end_time': span.end_time / 1e9 if span.end_time else None,
                        'attributes': dict(span.attributes) if span.attributes else {},
                        'events': [
                            {
                                'name': event.name,
                                'timestamp': event.timestamp / 1e9,
                                'attributes': dict(event.attributes) if event.attributes else {}
                            }
                            for event in span.events
                        ],
                        'status': span.status.status_code.name,
                        'status_description': span.status.description
                    }
                    self.buffer.append(trace_data)
            
            return SpanExportResult.SUCCESS
            
        except Exception as e:
            logger.error(f"Failed to export spans: {e}")
            return SpanExportResult.FAILURE
    
    def _writer_loop(self):
        """Background thread to write spans to disk"""
        while True:
            try:
                time.sleep(1)  # Write every second
                self._write_buffer()
            except Exception as e:
                logger.error(f"Trace writer error: {e}")
    
    def _write_buffer(self):
        """Write buffered spans to file"""
        # Check for hourly rotation
        current_hour = datetime.now(timezone.utc).hour
        if current_hour != self.current_hour:
            self.current_hour = current_hour
            self.trace_file = self._get_trace_file()
        
        # Write spans
        with self.buffer_lock:
            if not self.buffer:
                return
            
            spans_to_write = self.buffer.copy()
            self.buffer.clear()
        
        with open(self.trace_file, 'a') as f:
            for span in spans_to_write:
                f.write(json.dumps(span) + '\n')
    
    def shutdown(self):
        """Shutdown exporter"""
        self._write_buffer()

class TORITracer:
    """
    Main tracer for TORI system with file-based fallback
    """
    
    def __init__(self, service_name: str = "tori", storage_path: Path = Path("data/traces")):
        self.service_name = service_name
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        if OTEL_AVAILABLE:
            # Initialize OpenTelemetry
            resource = Resource.create({
                "service.name": service_name,
                "service.version": "2.0.0",
                "deployment.environment": "production"
            })
            
            provider = TracerProvider(resource=resource)
            
            # Add file exporter
            file_exporter = FileSpanExporter(storage_path)
            provider.add_span_processor(BatchSpanProcessor(file_exporter))
            
            # Set global tracer provider
            trace.set_tracer_provider(provider)
            
            # Get tracer
            self.tracer = trace.get_tracer(__name__, "1.0.0")
            
            # Propagator for distributed tracing
            self.propagator = TraceContextTextMapPropagator()
            
        else:
            # File-based fallback
            self.tracer = None
            self.active_spans = {}
            self.completed_spans = deque(maxlen=10000)
            
        logger.info(f"TORI Tracer initialized (OpenTelemetry: {OTEL_AVAILABLE})")
    
    @contextmanager
    def span(self, name: str, attributes: Dict[str, Any] = None, kind=None):
        """Create a trace span"""
        if OTEL_AVAILABLE and self.tracer:
            # Use OpenTelemetry
            with self.tracer.start_as_current_span(
                name,
                attributes=attributes,
                kind=kind or trace.SpanKind.INTERNAL
            ) as span:
                yield span
        else:
            # File-based fallback
            span = self._create_fallback_span(name, attributes)
            self.active_spans[span.span_id] = span
            
            try:
                yield span
                span.status = SpanStatus.OK
            except Exception as e:
                span.status = SpanStatus.ERROR
                span.status_description = str(e)
                raise
            finally:
                span.end_time = time.time()
                self.completed_spans.append(span)
                del self.active_spans[span.span_id]
                self._write_fallback_span(span)
    
    def _create_fallback_span(self, name: str, attributes: Dict[str, Any] = None) -> TraceSpan:
        """Create fallback span when OpenTelemetry not available"""
        # Generate IDs
        trace_id = uuid.uuid4().hex
        span_id = uuid.uuid4().hex[:16]
        
        # Check for parent span
        parent_span_id = None
        for span in reversed(list(self.active_spans.values())):
            parent_span_id = span.span_id
            trace_id = span.trace_id  # Use parent's trace ID
            break
        
        return TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            start_time=time.time(),
            end_time=None,
            attributes=attributes or {},
            events=[],
            status=SpanStatus.OK,
            status_description=None
        )
    
    def _write_fallback_span(self, span: TraceSpan):
        """Write fallback span to file"""
        trace_file = self.storage_path / f"traces_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        
        with open(trace_file, 'a') as f:
            f.write(json.dumps(asdict(span)) + '\n')
    
    def add_event(self, name: str, attributes: Dict[str, Any] = None):
        """Add event to current span"""
        if OTEL_AVAILABLE:
            span = trace.get_current_span()
            if span and span.is_recording():
                span.add_event(name, attributes=attributes)
        else:
            # Add to active fallback span
            for span in reversed(list(self.active_spans.values())):
                span.events.append({
                    'name': name,
                    'timestamp': time.time(),
                    'attributes': attributes or {}
                })
                break
    
    def set_attribute(self, key: str, value: Any):
        """Set attribute on current span"""
        if OTEL_AVAILABLE:
            span = trace.get_current_span()
            if span and span.is_recording():
                span.set_attribute(key, value)
        else:
            # Set on active fallback span
            for span in reversed(list(self.active_spans.values())):
                span.attributes[key] = value
                break
    
    def set_status(self, status: Union[Status, SpanStatus], description: str = None):
        """Set status on current span"""
        if OTEL_AVAILABLE:
            span = trace.get_current_span()
            if span and span.is_recording():
                if isinstance(status, Status):
                    span.set_status(status)
                else:
                    # Convert to OTEL status
                    otel_status = Status(
                        StatusCode.ERROR if status == SpanStatus.ERROR else StatusCode.OK,
                        description
                    )
                    span.set_status(otel_status)
        else:
            # Set on active fallback span
            for span in reversed(list(self.active_spans.values())):
                span.status = status
                span.status_description = description
                break
    
    def get_trace_context(self) -> Dict[str, str]:
        """Get current trace context for propagation"""
        if OTEL_AVAILABLE:
            carrier = {}
            self.propagator.inject(carrier)
            return carrier
        else:
            # Fallback context
            for span in reversed(list(self.active_spans.values())):
                return {
                    'traceparent': f'00-{span.trace_id}-{span.span_id}-01'
                }
            return {}
    
    def set_trace_context(self, carrier: Dict[str, str]):
        """Set trace context from propagation"""
        if OTEL_AVAILABLE:
            ctx = self.propagator.extract(carrier)
            # Would need to set context, implementation depends on usage
        else:
            # Parse traceparent header
            traceparent = carrier.get('traceparent', '')
            if traceparent:
                parts = traceparent.split('-')
                if len(parts) >= 4:
                    # Store for next span creation
                    self._parent_trace_id = parts[1]
                    self._parent_span_id = parts[2]

class TracingDecorators:
    """Decorators for automatic tracing"""
    
    def __init__(self, tracer: TORITracer):
        self.tracer = tracer
    
    def trace(self, name: str = None, attributes: Dict[str, Any] = None):
        """Decorator to trace function execution"""
        def decorator(func):
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            def wrapper(*args, **kwargs):
                with self.tracer.span(span_name, attributes) as span:
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        self.tracer.set_status(
                            SpanStatus.ERROR,
                            f"{type(e).__name__}: {str(e)}"
                        )
                        self.tracer.add_event("exception", {
                            "exception.type": type(e).__name__,
                            "exception.message": str(e),
                            "exception.stacktrace": traceback.format_exc()
                        })
                        raise
            
            async def async_wrapper(*args, **kwargs):
                with self.tracer.span(span_name, attributes) as span:
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except Exception as e:
                        self.tracer.set_status(
                            SpanStatus.ERROR,
                            f"{type(e).__name__}: {str(e)}"
                        )
                        self.tracer.add_event("exception", {
                            "exception.type": type(e).__name__,
                            "exception.message": str(e),
                            "exception.stacktrace": traceback.format_exc()
                        })
                        raise
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return wrapper
        return decorator

# Trace analysis tools
class TraceAnalyzer:
    """Analyze traces stored in files"""
    
    def __init__(self, storage_path: Path = Path("data/traces")):
        self.storage_path = storage_path
        self.traces_dir = storage_path / 'traces'
    
    def load_traces(self, start_time: float = None, end_time: float = None) -> List[TraceSpan]:
        """Load traces from files"""
        traces = []
        
        # Find relevant trace files
        trace_files = sorted(self.traces_dir.glob("traces_*.jsonl"))
        
        for trace_file in trace_files:
            with open(trace_file, 'r') as f:
                for line in f:
                    try:
                        span_data = json.loads(line.strip())
                        span = TraceSpan(**span_data)
                        
                        # Filter by time if specified
                        if start_time and span.start_time < start_time:
                            continue
                        if end_time and span.start_time > end_time:
                            continue
                        
                        traces.append(span)
                    except Exception as e:
                        logger.warning(f"Failed to parse span: {e}")
        
        return traces
    
    def analyze_latencies(self, traces: List[TraceSpan]) -> Dict[str, Dict[str, float]]:
        """Analyze latencies by operation"""
        latencies = defaultdict(list)
        
        for span in traces:
            if span.duration_ms() is not None:
                latencies[span.name].append(span.duration_ms())
        
        # Calculate statistics
        stats = {}
        for name, values in latencies.items():
            if values:
                stats[name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'p50': self._percentile(values, 0.5),
                    'p95': self._percentile(values, 0.95),
                    'p99': self._percentile(values, 0.99)
                }
        
        return stats
    
    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile"""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def find_slow_operations(self, traces: List[TraceSpan], threshold_ms: float = 1000) -> List[TraceSpan]:
        """Find operations that exceed latency threshold"""
        slow_ops = []
        
        for span in traces:
            duration = span.duration_ms()
            if duration and duration > threshold_ms:
                slow_ops.append(span)
        
        return sorted(slow_ops, key=lambda s: s.duration_ms(), reverse=True)
    
    def build_trace_tree(self, trace_id: str, spans: List[TraceSpan]) -> Dict[str, Any]:
        """Build tree structure for a trace"""
        # Filter spans for this trace
        trace_spans = [s for s in spans if s.trace_id == trace_id]
        
        # Build parent-child relationships
        children = defaultdict(list)
        roots = []
        
        for span in trace_spans:
            if span.parent_span_id:
                children[span.parent_span_id].append(span)
            else:
                roots.append(span)
        
        def build_node(span: TraceSpan) -> Dict[str, Any]:
            node = {
                'span_id': span.span_id,
                'name': span.name,
                'duration_ms': span.duration_ms(),
                'attributes': span.attributes,
                'status': span.status.value if isinstance(span.status, SpanStatus) else span.status,
                'children': [build_node(child) for child in children.get(span.span_id, [])]
            }
            return node
        
        return {
            'trace_id': trace_id,
            'roots': [build_node(root) for root in roots],
            'total_spans': len(trace_spans)
        }
    
    def generate_report(self, start_time: float = None, end_time: float = None) -> Dict[str, Any]:
        """Generate comprehensive trace analysis report"""
        traces = self.load_traces(start_time, end_time)
        
        if not traces:
            return {'error': 'No traces found'}
        
        # Group by trace ID
        traces_by_id = defaultdict(list)
        for span in traces:
            traces_by_id[span.trace_id].append(span)
        
        # Analyze
        report = {
            'total_traces': len(traces_by_id),
            'total_spans': len(traces),
            'time_range': {
                'start': min(s.start_time for s in traces),
                'end': max(s.end_time or s.start_time for s in traces)
            },
            'latency_stats': self.analyze_latencies(traces),
            'slow_operations': [
                {
                    'name': span.name,
                    'duration_ms': span.duration_ms(),
                    'trace_id': span.trace_id,
                    'attributes': span.attributes
                }
                for span in self.find_slow_operations(traces)[:10]  # Top 10
            ],
            'error_rate': sum(1 for s in traces if s.status == SpanStatus.ERROR) / len(traces) * 100
        }
        
        return report

# Integration with TORI system
class TORITracingIntegration:
    """Integrate tracing with TORI production system"""
    
    def __init__(self, tori_system, tracer: TORITracer = None):
        self.tori = tori_system
        self.tracer = tracer or TORITracer("tori-production")
        self.decorators = TracingDecorators(self.tracer)
        
        # Instrument key methods
        self._instrument_methods()
    
    def _instrument_methods(self):
        """Add tracing to TORI methods"""
        # Instrument query processing
        original_process = self.tori.process_query
        
        @self.decorators.trace("tori.process_query")
        async def traced_process_query(query: str, context: Optional[Dict[str, Any]] = None):
            self.tracer.set_attribute("query.text", query[:100])  # First 100 chars
            self.tracer.set_attribute("query.length", len(query))
            
            if context:
                self.tracer.set_attribute("context.chaos_enabled", context.get('enable_chaos', False))
            
            result = await original_process(query, context)
            
            # Add result attributes
            self.tracer.set_attribute("result.response_length", len(result.get('response', '')))
            self.tracer.set_attribute("result.chaos_used", result['metadata'].get('chaos_enabled', False))
            self.tracer.set_attribute("result.safety_level", result['metadata'].get('safety_level', 'unknown'))
            
            return result
        
        self.tori.process_query = traced_process_query

# Trace visualization server (no containers)
def start_trace_viewer(port: int = 9091, storage_path: Path = Path("data/traces")):
    """
    Start simple HTTP server for trace visualization
    No containers needed - serves a simple HTML interface
    """
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse
    
    analyzer = TraceAnalyzer(storage_path)
    
    class TraceHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                # Serve HTML interface
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                
                html = """
<!DOCTYPE html>
<html>
<head>
    <title>TORI Trace Viewer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .trace { border: 1px solid #ccc; margin: 10px 0; padding: 10px; }
        .span { margin-left: 20px; padding: 5px; border-left: 2px solid #ddd; }
        .error { background-color: #fee; }
        .slow { background-color: #ffd; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>TORI Trace Viewer</h1>
    <div id="content">Loading...</div>
    <script>
        fetch('/api/report')
            .then(r => r.json())
            .then(data => {
                let html = '<h2>Trace Report</h2>';
                html += `<p>Total traces: ${data.total_traces}, Total spans: ${data.total_spans}</p>`;
                html += `<p>Error rate: ${data.error_rate.toFixed(2)}%</p>`;
                
                html += '<h3>Slow Operations</h3><table><tr><th>Operation</th><th>Duration (ms)</th><th>Trace ID</th></tr>';
                data.slow_operations.forEach(op => {
                    html += `<tr class="slow"><td>${op.name}</td><td>${op.duration_ms.toFixed(2)}</td><td>${op.trace_id}</td></tr>`;
                });
                html += '</table>';
                
                html += '<h3>Latency Statistics</h3><table><tr><th>Operation</th><th>Count</th><th>Mean (ms)</th><th>P95 (ms)</th><th>P99 (ms)</th></tr>';
                Object.entries(data.latency_stats).forEach(([name, stats]) => {
                    html += `<tr><td>${name}</td><td>${stats.count}</td><td>${stats.mean.toFixed(2)}</td><td>${stats.p95.toFixed(2)}</td><td>${stats.p99.toFixed(2)}</td></tr>`;
                });
                html += '</table>';
                
                document.getElementById('content').innerHTML = html;
            });
    </script>
</body>
</html>
                """
                self.wfile.write(html.encode())
                
            elif self.path == '/api/report':
                # Serve JSON report
                report = analyzer.generate_report()
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(report).encode())
                
            else:
                self.send_response(404)
                self.end_headers()
        
        def log_message(self, format, *args):
            # Suppress default logging
            pass
    
    server = HTTPServer(('', port), TraceHandler)
    logger.info(f"Trace viewer started on port {port}")
    
    # Run in thread
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    
    return server

# Global tracer instance
_global_tracer = None

def get_tracer() -> TORITracer:
    """Get or create global tracer"""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = TORITracer()
    return _global_tracer

# Convenience decorators
tracer = get_tracer()
trace = TracingDecorators(tracer).trace

# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Initialize tracer
    tracer = TORITracer("example-service")
    decorators = TracingDecorators(tracer)
    
    # Example traced function
    @decorators.trace("process_data")
    async def process_data(data: str) -> str:
        tracer.set_attribute("data.length", len(data))
        
        # Simulate processing
        await asyncio.sleep(0.1)
        
        # Add event
        tracer.add_event("processing_complete", {"items_processed": 42})
        
        return data.upper()
    
    # Run example
    async def main():
        # Start trace viewer
        viewer = start_trace_viewer(9091)
        
        # Generate some traces
        for i in range(5):
            result = await process_data(f"hello world {i}")
            print(f"Operation {i} result: {result}")
            await asyncio.sleep(0.5)
        
        print(f"\nView traces at http://localhost:9091")
        
        # Keep running
        await asyncio.sleep(3600)
    
    asyncio.run(main())
