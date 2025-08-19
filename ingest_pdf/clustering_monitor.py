"""
TORI Clustering Production Monitor
Real-time monitoring and alerting for clustering performance and quality.
"""

import json
import time
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clustering_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TORIClusteringMonitor')

@dataclass
class ClusteringMetrics:
    """Comprehensive clustering metrics for monitoring."""
    timestamp: float
    method: str
    n_concepts: int
    n_clusters: int
    avg_cohesion: float
    silhouette_score: float
    runtime_seconds: float
    memory_usage_mb: float
    convergence_efficiency: float
    singleton_ratio: float
    quality_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AlertThresholds:
    """Alert thresholds for clustering quality monitoring."""
    min_cohesion: float = 0.25
    min_silhouette: float = 0.2
    max_runtime_seconds: float = 10.0
    max_memory_mb: float = 500.0
    min_convergence_efficiency: float = 0.5
    max_singleton_ratio: float = 0.4
    min_quality_score: float = 0.6

class ClusteringMonitor:
    """Production monitoring for TORI clustering system."""
    
    def __init__(self, 
                 alert_thresholds: Optional[AlertThresholds] = None,
                 history_file: str = "clustering_history.json",
                 max_history_days: int = 30):
        
        self.alert_thresholds = alert_thresholds or AlertThresholds()
        self.history_file = history_file
        self.max_history_days = max_history_days
        self.metrics_history: List[ClusteringMetrics] = []
        self.alert_callbacks: Dict[str, List[Callable]] = {
            'quality_degradation': [],
            'performance_issues': [],
            'system_health': [],
            'optimization_opportunities': []
        }
        
        # Load existing history
        self._load_history()
        
        logger.info("Clustering monitor initialized")
    
    def register_alert_callback(self, alert_type: str, callback: Callable):
        """Register a callback for specific alert types."""
        if alert_type in self.alert_callbacks:
            self.alert_callbacks[alert_type].append(callback)
        else:
            self.alert_callbacks[alert_type] = [callback]
    
    def record_clustering_result(self, result: Dict[str, Any], 
                               embeddings: np.ndarray,
                               start_time: float) -> ClusteringMetrics:
        """Record clustering results and perform quality analysis."""
        
        # Calculate comprehensive metrics
        runtime = time.time() - start_time
        memory_usage = self._estimate_memory_usage(embeddings, result)
        singleton_ratio = self._calculate_singleton_ratio(result)
        quality_score = self._calculate_quality_score(result)
        convergence_efficiency = self._calculate_convergence_efficiency(result)
        
        metrics = ClusteringMetrics(
            timestamp=time.time(),
            method=result.get('method', 'unknown'),
            n_concepts=len(embeddings),
            n_clusters=result.get('n_clusters', 0),
            avg_cohesion=result.get('avg_cohesion', 0.0),
            silhouette_score=result.get('silhouette_score', 0.0),
            runtime_seconds=runtime,
            memory_usage_mb=memory_usage,
            convergence_efficiency=convergence_efficiency,
            singleton_ratio=singleton_ratio,
            quality_score=quality_score
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        self._save_history()
        self._cleanup_old_history()
        
        # Check for alerts
        self._check_alerts(metrics)
        
        logger.info(f"Recorded clustering metrics: {metrics.method} - "
                   f"Quality: {quality_score:.3f}, Runtime: {runtime:.3f}s")
        
        return metrics
    
    def _estimate_memory_usage(self, embeddings: np.ndarray, result: Dict[str, Any]) -> float:
        """Estimate memory usage in MB."""
        embedding_size = embeddings.nbytes / (1024 * 1024)
        result_size = len(str(result)) / (1024 * 1024)  # Rough estimate
        return embedding_size + result_size
    
    def _calculate_singleton_ratio(self, result: Dict[str, Any]) -> float:
        """Calculate ratio of singleton clusters."""
        if 'clusters' not in result:
            return 0.0
        
        clusters = result['clusters']
        if not clusters:
            return 0.0
        
        singleton_count = sum(1 for cluster in clusters.values() if len(cluster) == 1)
        return singleton_count / len(clusters)
    
    def _calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """Calculate composite quality score (0-1)."""
        cohesion = result.get('avg_cohesion', 0.0)
        silhouette = result.get('silhouette_score', 0.0)
        
        # Normalize silhouette score to 0-1 range
        normalized_silhouette = (silhouette + 1) / 2
        
        # Weighted composite score
        quality_score = (cohesion * 0.6) + (normalized_silhouette * 0.4)
        return min(1.0, max(0.0, quality_score))
    
    def _calculate_convergence_efficiency(self, result: Dict[str, Any]) -> float:
        """Calculate convergence efficiency for oscillator method."""
        if result.get('method') != 'oscillator':
            return 0.5  # Default for non-oscillator methods
        
        convergence_step = result.get('convergence_step', 0)
        total_steps = result.get('total_steps', 1)
        
        return 1.0 - (convergence_step / total_steps)
    
    def _check_alerts(self, metrics: ClusteringMetrics):
        """Check metrics against alert thresholds and trigger callbacks."""
        alerts = []
        
        # Quality degradation alerts
        if metrics.avg_cohesion < self.alert_thresholds.min_cohesion:
            alerts.append(('quality_degradation', f'Low cohesion: {metrics.avg_cohesion:.3f}'))
        
        if metrics.silhouette_score < self.alert_thresholds.min_silhouette:
            alerts.append(('quality_degradation', f'Low silhouette score: {metrics.silhouette_score:.3f}'))
        
        if metrics.quality_score < self.alert_thresholds.min_quality_score:
            alerts.append(('quality_degradation', f'Low quality score: {metrics.quality_score:.3f}'))
        
        # Performance issues
        if metrics.runtime_seconds > self.alert_thresholds.max_runtime_seconds:
            alerts.append(('performance_issues', f'Slow runtime: {metrics.runtime_seconds:.3f}s'))
        
        if metrics.memory_usage_mb > self.alert_thresholds.max_memory_mb:
            alerts.append(('performance_issues', f'High memory usage: {metrics.memory_usage_mb:.1f}MB'))
        
        # System health
        if metrics.singleton_ratio > self.alert_thresholds.max_singleton_ratio:
            alerts.append(('system_health', f'Too many singletons: {metrics.singleton_ratio:.3f}'))
        
        if metrics.convergence_efficiency < self.alert_thresholds.min_convergence_efficiency:
            alerts.append(('system_health', f'Poor convergence: {metrics.convergence_efficiency:.3f}'))
        
        # Trigger callbacks
        for alert_type, message in alerts:
            logger.warning(f"ALERT [{alert_type}]: {message}")
            for callback in self.alert_callbacks.get(alert_type, []):
                try:
                    callback(metrics, message)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance trends over the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {"error": "No recent metrics available"}
        
        # Calculate trends
        quality_scores = [m.quality_score for m in recent_metrics]
        runtimes = [m.runtime_seconds for m in recent_metrics]
        cohesion_scores = [m.avg_cohesion for m in recent_metrics]
        
        trends = {
            "period_hours": hours,
            "total_runs": len(recent_metrics),
            "avg_quality": np.mean(quality_scores),
            "quality_trend": "improving" if quality_scores[-1] > quality_scores[0] else "degrading",
            "avg_runtime": np.mean(runtimes),
            "runtime_trend": "improving" if runtimes[-1] < runtimes[0] else "degrading",
            "avg_cohesion": np.mean(cohesion_scores),
            "method_distribution": {},
            "alerts_triggered": sum(1 for m in recent_metrics if m.quality_score < self.alert_thresholds.min_quality_score)
        }
        
        # Method distribution
        method_counts = {}
        for m in recent_metrics:
            method_counts[m.method] = method_counts.get(m.method, 0) + 1
        trends["method_distribution"] = method_counts
        
        return trends
    
    def generate_health_report(self) -> str:
        """Generate comprehensive health report."""
        if not self.metrics_history:
            return "No clustering metrics available for health report."
        
        recent_trends = self.get_performance_trends(24)
        latest_metrics = self.metrics_history[-1]
        
        report = f"""
🏥 TORI CLUSTERING SYSTEM HEALTH REPORT
=======================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 CURRENT STATUS
Latest Run: {datetime.fromtimestamp(latest_metrics.timestamp).strftime('%Y-%m-%d %H:%M:%S')}
Method: {latest_metrics.method}
Quality Score: {latest_metrics.quality_score:.3f}/1.0 {'✅' if latest_metrics.quality_score >= 0.7 else '⚠️' if latest_metrics.quality_score >= 0.5 else '❌'}
Cohesion: {latest_metrics.avg_cohesion:.3f}
Runtime: {latest_metrics.runtime_seconds:.3f}s
Memory: {latest_metrics.memory_usage_mb:.1f}MB

📈 24-HOUR TRENDS
Total Runs: {recent_trends['total_runs']}
Average Quality: {recent_trends['avg_quality']:.3f}
Quality Trend: {recent_trends['quality_trend']} {'📈' if recent_trends['quality_trend'] == 'improving' else '📉'}
Average Runtime: {recent_trends['avg_runtime']:.3f}s
Runtime Trend: {recent_trends['runtime_trend']} {'📈' if recent_trends['runtime_trend'] == 'improving' else '📉'}
Alerts Triggered: {recent_trends['alerts_triggered']}

🔧 METHOD USAGE
"""
        
        for method, count in recent_trends['method_distribution'].items():
            percentage = (count / recent_trends['total_runs']) * 100
            report += f"{method}: {count} runs ({percentage:.1f}%)\n"
        
        # Recommendations
        report += "\n💡 RECOMMENDATIONS\n"
        if latest_metrics.quality_score < 0.6:
            report += "• Quality below threshold - consider parameter tuning\n"
        if latest_metrics.runtime_seconds > 5.0:
            report += "• Consider batch processing for better performance\n"
        if latest_metrics.singleton_ratio > 0.3:
            report += "• High singleton ratio - adjust cohesion threshold\n"
        if recent_trends['alerts_triggered'] > 0:
            report += f"• {recent_trends['alerts_triggered']} alerts in last 24h - investigate issues\n"
        
        if latest_metrics.quality_score >= 0.7 and latest_metrics.runtime_seconds <= 2.0:
            report += "• System performing optimally ✅\n"
        
        return report
    
    def _load_history(self):
        """Load metrics history from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.metrics_history = [ClusteringMetrics(**item) for item in data]
                logger.info(f"Loaded {len(self.metrics_history)} historical metrics")
            except Exception as e:
                logger.error(f"Failed to load history: {e}")
                self.metrics_history = []
    
    def _save_history(self):
        """Save metrics history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump([m.to_dict() for m in self.metrics_history], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def _cleanup_old_history(self):
        """Remove metrics older than max_history_days."""
        cutoff_time = time.time() - (self.max_history_days * 24 * 3600)
        self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]

# Example usage and default alert callbacks
def default_quality_alert(metrics: ClusteringMetrics, message: str):
    """Default quality degradation alert."""
    print(f"🚨 QUALITY ALERT: {message}")
    print(f"   Method: {metrics.method}, Concepts: {metrics.n_concepts}")

def default_performance_alert(metrics: ClusteringMetrics, message: str):
    """Default performance alert."""
    print(f"⚠️  PERFORMANCE ALERT: {message}")
    print(f"   Consider optimizing clustering parameters or using batch processing")

def default_health_alert(metrics: ClusteringMetrics, message: str):
    """Default system health alert."""
    print(f"🔧 SYSTEM HEALTH ALERT: {message}")
    print(f"   Review clustering configuration and data quality")

# Factory function for easy setup
def create_production_monitor(config_file: Optional[str] = None) -> ClusteringMonitor:
    """Create a production-ready clustering monitor with default settings."""
    
    # Load config if provided
    thresholds = AlertThresholds()
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                thresholds = AlertThresholds(**config.get('alert_thresholds', {}))
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    
    # Create monitor
    monitor = ClusteringMonitor(alert_thresholds=thresholds)
    
    # Register default callbacks
    monitor.register_alert_callback('quality_degradation', default_quality_alert)
    monitor.register_alert_callback('performance_issues', default_performance_alert)
    monitor.register_alert_callback('system_health', default_health_alert)
    
    return monitor

if __name__ == "__main__":
    # Demo the monitoring system
    print("🔍 TORI Clustering Monitor Demo")
    print("===============================")
    
    monitor = create_production_monitor()
    
    # Simulate some clustering results
    import numpy as np
    from clustering import run_oscillator_clustering_with_metrics
    
    for i in range(3):
        print(f"\n📊 Simulating clustering run {i+1}...")
        embeddings = np.random.randn(50 + i*25, 100)
        start_time = time.time()
        
        try:
            result = run_oscillator_clustering_with_metrics(embeddings, enable_logging=False)
            metrics = monitor.record_clustering_result(result, embeddings, start_time)
            print(f"   Quality: {metrics.quality_score:.3f}, Runtime: {metrics.runtime_seconds:.3f}s")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Generate health report
    print("\n" + monitor.generate_health_report())
