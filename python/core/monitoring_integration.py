#!/usr/bin/env python3
"""
TORI/KHA Monitoring Integration Implementation
Complete monitoring setup with Prometheus, OpenTelemetry, and GraphQL
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass, field

# The monitoring components were already created in the previous files:
# - prometheus_metrics.py
# - opentelemetry_tracing.py  
# - graphql_api.py

# Now let's create the main monitoring integration

from python.core.prometheus_metrics import PrometheusMetricsExporter
from python.core.opentelemetry_tracing import OpenTelemetryTracer
from python.core.graphql_api import GraphQLAPI
from python.core.tori_production import TORIProductionSystem

class TORIMonitoringIntegration:
    """Complete monitoring integration for TORI/KHA"""
    
    def __init__(self, tori_system: TORIProductionSystem):
        self.tori = tori_system
        self.config = self.tori.config
        
        # Initialize monitoring components
        self.metrics_exporter = None
        self.tracer = None
        self.graphql_api = None
        
        # Monitoring state
        self.monitoring_enabled = False
        self.start_time = time.time()
        
    async def initialize(self):
        """Initialize all monitoring components"""
        print("🔍 Initializing TORI Monitoring Integration...")
        
        # 1. Prometheus Metrics
        if self.config.enable_prometheus_metrics:
            self.metrics_exporter = PrometheusMetricsExporter(
                port=self.config.prometheus_port or 9090
            )
            await self.metrics_exporter.start()
            
            # Register TORI metrics collectors
            self._register_metrics_collectors()
            
        # 2. OpenTelemetry Tracing
        if self.config.enable_tracing:
            self.tracer = OpenTelemetryTracer(
                service_name="tori-kha",
                endpoint=self.config.otlp_endpoint or "http://localhost:4317"
            )
            
            # Instrument TORI components
            self._instrument_components()
            
        # 3. GraphQL API
        if self.config.enable_graphql_api:
            self.graphql_api = GraphQLAPI(
                tori_system=self.tori,
                port=self.config.graphql_port or 8080
            )
            await self.graphql_api.start()
            
        self.monitoring_enabled = True
        print("✅ Monitoring integration initialized")
        
    def _register_metrics_collectors(self):
        """Register custom metrics collectors"""
        
        # Chaos metrics collector
        async def collect_chaos_metrics():
            if hasattr(self.tori, 'ccl') and self.tori.ccl:
                status = self.tori.ccl.get_status()
                
                # Update gauges
                self.metrics_exporter.chaos_tasks_active.set(
                    status.get('active_tasks', 0)
                )
                self.metrics_exporter.chaos_tasks_completed.set(
                    len(status.get('completed_tasks', []))
                )
                
                # Efficiency metrics
                for task in status.get('completed_tasks', [])[-10:]:
                    if task.efficiency_gain:
                        self.metrics_exporter.chaos_efficiency_gain.set(
                            task.efficiency_gain
                        )
                        
        # Eigenvalue metrics collector  
        async def collect_eigenvalue_metrics():
            if hasattr(self.tori, 'eigen_sentry') and self.tori.eigen_sentry:
                metrics = self.tori.eigen_sentry.get_instability_metrics()
                
                self.metrics_exporter.eigenvalue_max.set(
                    metrics.get('max_eigenvalue', 0)
                )
                self.metrics_exporter.eigenvalue_spectral_radius.set(
                    metrics.get('spectral_radius', 0)
                )
                
        # Memory metrics collector
        async def collect_memory_metrics():
            if hasattr(self.tori, 'memory_vault') and self.tori.memory_vault:
                stats = self.tori.memory_vault.get_statistics()
                
                for memory_type, count in stats.get('by_type', {}).items():
                    self.metrics_exporter.memory_entries.labels(
                        type=memory_type
                    ).set(count)
                    
        # Schedule periodic collection
        async def metrics_collection_loop():
            while self.monitoring_enabled:
                try:
                    await collect_chaos_metrics()
                    await collect_eigenvalue_metrics()
                    await collect_memory_metrics()
                except Exception as e:
                    print(f"Metrics collection error: {e}")
                    
                await asyncio.sleep(5)  # Collect every 5 seconds
                
        asyncio.create_task(metrics_collection_loop())
        
    def _instrument_components(self):
        """Add tracing instrumentation to components"""
        
        # Instrument cognitive engine
        if hasattr(self.tori, 'cognitive_engine'):
            original_process = self.tori.cognitive_engine.process
            
            async def traced_process(*args, **kwargs):
                with self.tracer.tracer.start_as_current_span("cognitive_process"):
                    return await original_process(*args, **kwargs)
                    
            self.tori.cognitive_engine.process = traced_process
            
        # Instrument chaos control layer
        if hasattr(self.tori, 'ccl'):
            original_submit = self.tori.ccl.submit_task
            
            async def traced_submit(*args, **kwargs):
                with self.tracer.tracer.start_as_current_span("chaos_submit"):
                    return await original_submit(*args, **kwargs)
                    
            self.tori.ccl.submit_task = traced_submit
            
    async def export_monitoring_config(self, output_dir: Path):
        """Export monitoring configuration files"""
        output_dir.mkdir(exist_ok=True)
        
        # 1. Prometheus configuration
        prometheus_config = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'tori-kha'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    
  - job_name: 'tori-chaos'
    static_configs:
      - targets: ['localhost:9091']
    metrics_path: '/chaos/metrics'
    
  - job_name: 'tori-quantum'
    static_configs:
      - targets: ['localhost:9092']
    metrics_path: '/quantum/metrics'

rule_files:
  - 'alerts/tori_alerts.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']
"""
        (output_dir / "prometheus.yml").write_text(prometheus_config)
        
        # 2. Alert rules
        alert_rules = """
groups:
  - name: tori_chaos_alerts
    interval: 30s
    rules:
      - alert: ChaosCriticalInstability
        expr: tori_eigenvalue_max > 0.95
        for: 2m
        labels:
          severity: critical
          component: chaos
        annotations:
          summary: "Chaos system approaching critical instability"
          description: "Max eigenvalue {{ $value }} exceeds safety threshold"
          
      - alert: ChaosEfficiencyLow
        expr: avg_over_time(tori_chaos_efficiency_gain[5m]) < 2
        for: 10m
        labels:
          severity: warning
          component: chaos
        annotations:
          summary: "Chaos efficiency below minimum threshold"
          description: "Average efficiency {{ $value }}x is below target 2x"
          
  - name: tori_memory_alerts
    interval: 30s
    rules:
      - alert: MemoryVaultFull
        expr: tori_memory_vault_size_bytes > 10737418240  # 10GB
        for: 5m
        labels:
          severity: warning
          component: memory
        annotations:
          summary: "Memory vault approaching capacity"
          description: "Size {{ $value | humanize }} exceeds threshold"
          
      - alert: MemoryFragmentation
        expr: rate(tori_memory_vault_fragmentation[5m]) > 0.5
        for: 10m
        labels:
          severity: warning
          component: memory
        annotations:
          summary: "High memory fragmentation detected"
          
  - name: tori_system_alerts
    interval: 30s
    rules:
      - alert: SystemOverload
        expr: rate(tori_requests_total[1m]) > 1000
        for: 5m
        labels:
          severity: warning
          component: system
        annotations:
          summary: "System experiencing high load"
          
      - alert: HighErrorRate
        expr: rate(tori_requests_total{status="error"}[5m]) / rate(tori_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
          component: system
        annotations:
          summary: "Error rate exceeds 5%"
"""
        alerts_dir = output_dir / "alerts"
        alerts_dir.mkdir(exist_ok=True)
        (alerts_dir / "tori_alerts.yml").write_text(alert_rules)
        
        # 3. Grafana dashboard
        grafana_dashboard = {
            "dashboard": {
                "id": None,
                "uid": "tori-kha-main",
                "title": "TORI/KHA System Dashboard",
                "timezone": "browser",
                "schemaVersion": 30,
                "version": 1,
                "refresh": "5s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "panels": [
                    {
                        "id": 1,
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "type": "graph",
                        "title": "Chaos Efficiency Gain",
                        "targets": [
                            {
                                "expr": "tori_chaos_efficiency_gain",
                                "legendFormat": "Efficiency {{mode}}"
                            }
                        ],
                        "yaxes": [
                            {"format": "short", "label": "Efficiency (x)"},
                            {"format": "short"}
                        ]
                    },
                    {
                        "id": 2,
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                        "type": "graph",
                        "title": "Eigenvalue Stability",
                        "targets": [
                            {
                                "expr": "tori_eigenvalue_max",
                                "legendFormat": "Max Eigenvalue"
                            },
                            {
                                "expr": "tori_eigenvalue_spectral_radius",
                                "legendFormat": "Spectral Radius"
                            }
                        ],
                        "alert": {
                            "conditions": [
                                {
                                    "evaluator": {
                                        "params": [0.95],
                                        "type": "gt"
                                    },
                                    "query": {
                                        "params": ["A", "5m", "now"]
                                    },
                                    "reducer": {
                                        "params": [],
                                        "type": "avg"
                                    },
                                    "type": "query"
                                }
                            ],
                            "executionErrorState": "alerting",
                            "for": "5m",
                            "frequency": "1m",
                            "handler": 1,
                            "name": "Eigenvalue Critical",
                            "noDataState": "no_data",
                            "notifications": []
                        }
                    },
                    {
                        "id": 3,
                        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 8},
                        "type": "stat",
                        "title": "Active Chaos Tasks",
                        "targets": [
                            {
                                "expr": "tori_chaos_tasks_active",
                                "legendFormat": "Active"
                            }
                        ],
                        "options": {
                            "colorMode": "value",
                            "graphMode": "area",
                            "justifyMode": "auto"
                        }
                    },
                    {
                        "id": 4,
                        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 8},
                        "type": "piechart",
                        "title": "Memory Distribution",
                        "targets": [
                            {
                                "expr": "tori_memory_entries",
                                "legendFormat": "{{type}}"
                            }
                        ]
                    },
                    {
                        "id": 5,
                        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 8},
                        "type": "gauge",
                        "title": "System Safety Level",
                        "targets": [
                            {
                                "expr": "1 - tori_eigenvalue_max",
                                "legendFormat": "Safety"
                            }
                        ],
                        "options": {
                            "showThresholdLabels": True,
                            "showThresholdMarkers": True
                        },
                        "fieldConfig": {
                            "defaults": {
                                "thresholds": {
                                    "mode": "absolute",
                                    "steps": [
                                        {"color": "red", "value": 0},
                                        {"color": "yellow", "value": 0.3},
                                        {"color": "green", "value": 0.7}
                                    ]
                                },
                                "unit": "percentunit"
                            }
                        }
                    },
                    {
                        "id": 6,
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16},
                        "type": "table",
                        "title": "Recent Chaos Tasks",
                        "targets": [
                            {
                                "expr": "tori_chaos_task_duration_seconds",
                                "format": "table",
                                "instant": True
                            }
                        ]
                    }
                ]
            }
        }
        
        dashboards_dir = output_dir / "grafana" / "dashboards"
        dashboards_dir.mkdir(parents=True, exist_ok=True)
        with open(dashboards_dir / "tori_main.json", 'w') as f:
            json.dump(grafana_dashboard, f, indent=2)
            
        # 4. Docker compose for monitoring stack
        docker_compose = """
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: tori-prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./alerts:/etc/prometheus/alerts:ro
      - prometheus-data:/prometheus
    networks:
      - tori-monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: tori-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./grafana/datasources:/etc/grafana/provisioning/datasources:ro
    networks:
      - tori-monitoring

  alertmanager:
    image: prom/alertmanager:latest
    container_name: tori-alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - alertmanager-data:/alertmanager
    networks:
      - tori-monitoring

  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: tori-jaeger
    ports:
      - "16686:16686"  # UI
      - "14268:14268"  # HTTP collector
      - "4317:4317"    # OTLP gRPC
      - "4318:4318"    # OTLP HTTP
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    networks:
      - tori-monitoring

volumes:
  prometheus-data:
  grafana-data:
  alertmanager-data:

networks:
  tori-monitoring:
    driver: bridge
"""
        (output_dir / "docker-compose.monitoring.yml").write_text(docker_compose)
        
        # 5. Grafana datasource
        datasources_dir = output_dir / "grafana" / "datasources"
        datasources_dir.mkdir(parents=True, exist_ok=True)
        
        datasource_config = """
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    
  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
    editable: true
"""
        (datasources_dir / "prometheus.yml").write_text(datasource_config)
        
        # 6. Alertmanager configuration
        alertmanager_config = """
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'component']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'tori-team'
  
  routes:
    - match:
        severity: critical
      receiver: 'tori-critical'
      continue: true

receivers:
  - name: 'tori-team'
    webhook_configs:
      - url: 'http://localhost:8000/api/alerts/webhook'
        send_resolved: true
        
  - name: 'tori-critical'
    webhook_configs:
      - url: 'http://localhost:8000/api/alerts/critical'
        send_resolved: true
    email_configs:
      - to: 'ops@example.com'
        from: 'tori-alerts@example.com'
        smarthost: 'smtp.example.com:587'
        auth_username: 'alerts@example.com'
        auth_password: 'password'
"""
        (output_dir / "alertmanager.yml").write_text(alertmanager_config)
        
        print(f"✅ Monitoring configuration exported to {output_dir}")
        
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "enabled": self.monitoring_enabled,
            "uptime": time.time() - self.start_time,
            "components": {
                "prometheus": self.metrics_exporter is not None,
                "opentelemetry": self.tracer is not None,
                "graphql": self.graphql_api is not None
            },
            "metrics": {
                "exported": self.metrics_exporter.get_metric_count() if self.metrics_exporter else 0,
                "traces": self.tracer.get_trace_count() if self.tracer else 0
            }
        }
        
    async def shutdown(self):
        """Shutdown monitoring components"""
        print("🛑 Shutting down monitoring integration...")
        
        self.monitoring_enabled = False
        
        if self.metrics_exporter:
            await self.metrics_exporter.stop()
            
        if self.graphql_api:
            await self.graphql_api.stop()
            
        print("✅ Monitoring integration shutdown complete")


# Monitoring setup script
async def setup_monitoring():
    """Complete monitoring setup for TORI/KHA"""
    print("🚀 Setting up TORI/KHA Monitoring")
    print("=" * 60)
    
    # Initialize TORI system
    from python.core.tori_production import TORIProductionConfig
    
    config = TORIProductionConfig(
        enable_prometheus_metrics=True,
        enable_tracing=True,
        enable_graphql_api=True,
        prometheus_port=9090,
        graphql_port=8080,
        otlp_endpoint="http://localhost:4317"
    )
    
    tori = TORIProductionSystem(config)
    await tori.start()
    
    # Initialize monitoring
    monitoring = TORIMonitoringIntegration(tori)
    await monitoring.initialize()
    
    # Export configuration
    config_dir = Path("monitoring_config")
    await monitoring.export_monitoring_config(config_dir)
    
    print("\n📊 Monitoring URLs:")
    print(f"  - Prometheus: http://localhost:9090")
    print(f"  - Grafana: http://localhost:3000")
    print(f"  - Jaeger: http://localhost:16686")
    print(f"  - GraphQL: http://localhost:8080/graphql")
    
    print("\n🚀 To start monitoring stack:")
    print(f"  cd {config_dir}")
    print("  docker-compose -f docker-compose.monitoring.yml up -d")
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(60)
            status = monitoring.get_monitoring_status()
            print(f"\n📈 Monitoring Status: {status}")
    except KeyboardInterrupt:
        print("\n⏹️  Stopping monitoring...")
        await monitoring.shutdown()
        await tori.stop()


if __name__ == "__main__":
    asyncio.run(setup_monitoring())
