"""
Core Metrics Module
Provides system-wide metrics and monitoring helpers
"""

import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json

# Try to import Prometheus client if available
try:
    from prometheus_client import Gauge, Counter, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes if not available
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, value): pass
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, amount=1): pass
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, value): pass
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, value): pass

logger = logging.getLogger(__name__)

# Prometheus metrics
mesh_compact_needed = Gauge(
    'mesh_compact_needed',
    'Whether mesh compaction is needed (1=yes, 0=no)',
    ['scope', 'scope_id']
)

mesh_size_bytes = Gauge(
    'mesh_size_bytes',
    'Size of mesh data in bytes',
    ['scope', 'scope_id']
)

wal_size_bytes = Gauge(
    'wal_size_bytes',
    'Size of WAL in bytes',
    ['scope', 'scope_id']
)

compaction_duration_seconds = Histogram(
    'compaction_duration_seconds',
    'Time taken for compaction',
    ['scope']
)

concepts_total = Gauge(
    'concepts_total',
    'Total number of concepts',
    ['scope', 'scope_id']
)

relations_total = Gauge(
    'relations_total',
    'Total number of relations',
    ['scope', 'scope_id']
)

@dataclass
class CompactionMetrics:
    """Metrics for mesh compaction needs"""
    scope: str
    scope_id: str
    mesh_size_mb: float
    wal_size_mb: float
    last_modified_hours: float
    last_compact_hours: Optional[float]
    needs_compaction: bool
    reason: str


class MetricsCollector:
    """Collects and analyzes system metrics"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path('data')
        self.snapshot_dir = self.data_dir / 'snapshots'
        
        # Compaction thresholds
        self.max_wal_size_mb = 50
        self.max_mesh_size_mb = 100
        self.min_compact_interval_hours = 12
        self.stale_data_hours = 24
    
    def needs_compact(self, scope: str, scope_id: str) -> CompactionMetrics:
        """
        Check if a mesh needs compaction
        
        Args:
            scope: "user" or "group" 
            scope_id: ID of the mesh
            
        Returns:
            CompactionMetrics with analysis
        """
        metrics = CompactionMetrics(
            scope=scope,
            scope_id=scope_id,
            mesh_size_mb=0,
            wal_size_mb=0,
            last_modified_hours=0,
            last_compact_hours=None,
            needs_compaction=False,
            reason="No compaction needed"
        )
        
        try:
            # Check mesh directory
            mesh_dir = self.data_dir / 'concept_mesh' / scope / scope_id
            if not mesh_dir.exists():
                metrics.reason = "Mesh directory not found"
                return metrics
            
            # Calculate mesh size
            mesh_size = 0
            for file in mesh_dir.rglob('*.json'):
                mesh_size += file.stat().st_size
            metrics.mesh_size_mb = mesh_size / (1024 * 1024)
            
            # Check WAL size
            wal_dir = self.data_dir / 'wal' / scope
            wal_file = wal_dir / f"{scope_id}.wal"
            checkpoint_file = wal_dir / f"{scope_id}.checkpoint"
            
            wal_size = 0
            if wal_file.exists():
                wal_size += wal_file.stat().st_size
            if checkpoint_file.exists():
                wal_size += checkpoint_file.stat().st_size
            
            # Check for compressed checkpoint
            checkpoint_gz = wal_dir / f"{scope_id}.checkpoint.gz"
            if checkpoint_gz.exists():
                wal_size += checkpoint_gz.stat().st_size
                
            metrics.wal_size_mb = wal_size / (1024 * 1024)
            
            # Check last modification time
            if mesh_dir.exists():
                file_times = [f.stat().st_mtime for f in mesh_dir.rglob('*') if f.is_file()]
                if file_times:
                    oldest_time = min(file_times)
                    metrics.last_modified_hours = (time.time() - oldest_time) / 3600
            
            # Check last compaction time
            last_compact_file = self.snapshot_dir / scope / scope_id / '.last_compact'
            if last_compact_file.exists():
                last_compact_time = last_compact_file.stat().st_mtime
                metrics.last_compact_hours = (time.time() - last_compact_time) / 3600
            
            # Determine if compaction needed
            reasons = []
            
            if metrics.wal_size_mb > self.max_wal_size_mb:
                reasons.append(f"WAL too large ({metrics.wal_size_mb:.1f}MB > {self.max_wal_size_mb}MB)")
                
            if metrics.mesh_size_mb > self.max_mesh_size_mb:
                reasons.append(f"Mesh too large ({metrics.mesh_size_mb:.1f}MB > {self.max_mesh_size_mb}MB)")
                
            if metrics.last_modified_hours > self.stale_data_hours:
                reasons.append(f"Data stale ({metrics.last_modified_hours:.1f}h old)")
                
            if metrics.last_compact_hours is None:
                reasons.append("Never compacted")
            elif metrics.last_compact_hours > self.min_compact_interval_hours:
                reasons.append(f"Compaction overdue ({metrics.last_compact_hours:.1f}h ago)")
            
            if reasons:
                metrics.needs_compaction = True
                metrics.reason = "; ".join(reasons)
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                mesh_compact_needed.labels(scope=scope, scope_id=scope_id).set(
                    1 if metrics.needs_compaction else 0
                )
                mesh_size_bytes.labels(scope=scope, scope_id=scope_id).set(
                    metrics.mesh_size_mb * 1024 * 1024
                )
                wal_size_bytes.labels(scope=scope, scope_id=scope_id).set(
                    metrics.wal_size_mb * 1024 * 1024
                )
            
        except Exception as e:
            logger.error(f"Error checking compaction need for {scope}:{scope_id}: {e}")
            metrics.needs_compaction = True
            metrics.reason = f"Error during check: {str(e)}"
        
        return metrics
    
    def check_all_meshes(self) -> List[CompactionMetrics]:
        """Check all meshes and return list of metrics"""
        all_metrics = []
        
        # Check user meshes
        user_dir = self.data_dir / 'concept_mesh' / 'user'
        if user_dir.exists():
            for user_dir in user_dir.iterdir():
                if user_dir.is_dir():
                    metrics = self.needs_compact('user', user_dir.name)
                    all_metrics.append(metrics)
        
        # Check group meshes
        group_dir = self.data_dir / 'concept_mesh' / 'group'
        if group_dir.exists():
            for group_dir in group_dir.iterdir():
                if group_dir.is_dir():
                    metrics = self.needs_compact('group', group_dir.name)
                    all_metrics.append(metrics)
        
        return all_metrics
    
    def get_compaction_report(self) -> Dict[str, Any]:
        """Generate a compaction needs report"""
        all_metrics = self.check_all_meshes()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_meshes': len(all_metrics),
            'needs_compaction': sum(1 for m in all_metrics if m.needs_compaction),
            'total_mesh_size_mb': sum(m.mesh_size_mb for m in all_metrics),
            'total_wal_size_mb': sum(m.wal_size_mb for m in all_metrics),
            'details': []
        }
        
        # Add details for meshes needing compaction
        for metrics in all_metrics:
            if metrics.needs_compaction:
                report['details'].append({
                    'scope': metrics.scope,
                    'scope_id': metrics.scope_id,
                    'mesh_size_mb': round(metrics.mesh_size_mb, 2),
                    'wal_size_mb': round(metrics.wal_size_mb, 2),
                    'reason': metrics.reason
                })
        
        return report
    
    def save_compaction_report(self, report_path: Path = None) -> Path:
        """Save compaction report to file"""
        if report_path is None:
            report_path = Path('logs') / f"compaction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = self.get_compaction_report()
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved compaction report to {report_path}")
        return report_path


# Helper function for scripts
def needs_compact(scope: str, scope_id: str, data_dir: Path = None) -> bool:
    """
    Simple helper to check if a mesh needs compaction
    
    Args:
        scope: "user" or "group"
        scope_id: ID of the mesh
        data_dir: Optional data directory path
        
    Returns:
        True if compaction is needed
    """
    collector = MetricsCollector(data_dir)
    metrics = collector.needs_compact(scope, scope_id)
    return metrics.needs_compaction


# CLI interface
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Check mesh compaction needs')
    parser.add_argument('--scope', help='Check specific scope:id (e.g. user:alice)')
    parser.add_argument('--report', action='store_true', help='Generate full report')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    collector = MetricsCollector()
    
    if args.scope:
        # Check specific mesh
        scope, scope_id = args.scope.split(':')
        metrics = collector.needs_compact(scope, scope_id)
        
        if args.json:
            print(json.dumps({
                'scope': metrics.scope,
                'scope_id': metrics.scope_id,
                'needs_compaction': metrics.needs_compaction,
                'reason': metrics.reason,
                'mesh_size_mb': metrics.mesh_size_mb,
                'wal_size_mb': metrics.wal_size_mb
            }, indent=2))
        else:
            print(f"Mesh: {scope}:{scope_id}")
            print(f"Needs compaction: {'YES' if metrics.needs_compaction else 'NO'}")
            print(f"Reason: {metrics.reason}")
            print(f"Mesh size: {metrics.mesh_size_mb:.2f} MB")
            print(f"WAL size: {metrics.wal_size_mb:.2f} MB")
    
    elif args.report:
        # Generate full report
        report = collector.get_compaction_report()
        
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"Compaction Report - {report['timestamp']}")
            print(f"Total meshes: {report['total_meshes']}")
            print(f"Need compaction: {report['needs_compaction']}")
            print(f"Total mesh size: {report['total_mesh_size_mb']:.2f} MB")
            print(f"Total WAL size: {report['total_wal_size_mb']:.2f} MB")
            
            if report['details']:
                print("\nMeshes needing compaction:")
                for detail in report['details']:
                    print(f"  - {detail['scope']}:{detail['scope_id']}: {detail['reason']}")
            
            # Save report
            report_path = collector.save_compaction_report()
            print(f"\nReport saved to: {report_path}")
    
    else:
        # Quick summary
        all_metrics = collector.check_all_meshes()
        need_compact = sum(1 for m in all_metrics if m.needs_compaction)
        
        print(f"Checked {len(all_metrics)} meshes")
        print(f"{need_compact} need compaction")
        
        if need_compact > 0:
            print("\nRun with --report for details")
