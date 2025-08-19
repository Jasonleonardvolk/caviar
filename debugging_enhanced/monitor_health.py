#!/usr/bin/env python3
"""
TORI Health Monitor
Real-time monitoring of TORI system health
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Container for health metrics"""
    component: str
    status: ComponentStatus
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    last_check: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        return {
            "component": self.component,
            "status": self.status.value,
            "response_time": self.response_time,
            "error_message": self.error_message,
            "last_check": self.last_check.isoformat()
        }


class TORIHealthMonitor:
    """Real-time health monitoring for TORI components"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.components = {
            "api": {
                "url": "http://localhost:8002/api/health",
                "timeout": 5,
                "critical": True
            },
            "mcp": {
                "url": "http://localhost:8100/api/system/status",
                "timeout": 5,
                "critical": True
            },
            "frontend": {
                "url": "http://localhost:5173",
                "timeout": 10,
                "critical": True
            },
            "audio_bridge": {
                "url": "ws://localhost:8765",
                "timeout": 5,
                "critical": False,
                "type": "websocket"
            },
            "hologram_bridge": {
                "url": "ws://localhost:8766",
                "timeout": 5,
                "critical": False,
                "type": "websocket"
            }
        }
        
        self.metrics: Dict[str, HealthMetric] = {}
        self.history: List[Dict] = []
        self.alert_threshold = 3  # Consecutive failures before alert
        self.failure_counts: Dict[str, int] = {}
        
    async def check_http_endpoint(self, name: str, config: Dict) -> HealthMetric:
        """Check HTTP endpoint health"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    config["url"],
                    timeout=aiohttp.ClientTimeout(total=config["timeout"])
                ) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        return HealthMetric(
                            component=name,
                            status=ComponentStatus.HEALTHY,
                            response_time=response_time
                        )
                    else:
                        return HealthMetric(
                            component=name,
                            status=ComponentStatus.DEGRADED,
                            response_time=response_time,
                            error_message=f"HTTP {response.status}"
                        )
                        
        except asyncio.TimeoutError:
            return HealthMetric(
                component=name,
                status=ComponentStatus.FAILED,
                error_message="Timeout"
            )
        except Exception as e:
            return HealthMetric(
                component=name,
                status=ComponentStatus.FAILED,
                error_message=str(e)
            )
    
    async def check_websocket_endpoint(self, name: str, config: Dict) -> HealthMetric:
        """Check WebSocket endpoint health"""
        import websockets
        
        start_time = time.time()
        
        try:
            async with websockets.connect(
                config["url"],
                timeout=config["timeout"]
            ) as websocket:
                # Send ping
                await websocket.ping()
                response_time = time.time() - start_time
                
                return HealthMetric(
                    component=name,
                    status=ComponentStatus.HEALTHY,
                    response_time=response_time
                )
                
        except asyncio.TimeoutError:
            return HealthMetric(
                component=name,
                status=ComponentStatus.FAILED,
                error_message="Timeout"
            )
        except Exception as e:
            return HealthMetric(
                component=name,
                status=ComponentStatus.FAILED,
                error_message=str(e)
            )
    
    async def check_component(self, name: str, config: Dict) -> HealthMetric:
        """Check a single component's health"""
        if config.get("type") == "websocket":
            return await self.check_websocket_endpoint(name, config)
        else:
            return await self.check_http_endpoint(name, config)
    
    async def check_all_components(self) -> Dict[str, HealthMetric]:
        """Check all components in parallel"""
        tasks = []
        
        for name, config in self.components.items():
            task = self.check_component(name, config)
            tasks.append((name, task))
        
        results = {}
        for name, task in tasks:
            try:
                metric = await task
                results[name] = metric
                self.metrics[name] = metric
                
                # Track failures for alerting
                if metric.status == ComponentStatus.FAILED:
                    self.failure_counts[name] = self.failure_counts.get(name, 0) + 1
                else:
                    self.failure_counts[name] = 0
                    
            except Exception as e:
                logger.error(f"Failed to check {name}: {e}")
                results[name] = HealthMetric(
                    component=name,
                    status=ComponentStatus.UNKNOWN,
                    error_message=str(e)
                )
        
        return results
    
    def calculate_health_score(self) -> int:
        """Calculate overall system health score (0-100)"""
        if not self.metrics:
            return 0
            
        total_weight = 0
        weighted_score = 0
        
        for name, metric in self.metrics.items():
            config = self.components[name]
            weight = 10 if config.get("critical") else 5
            
            if metric.status == ComponentStatus.HEALTHY:
                score = 100
            elif metric.status == ComponentStatus.DEGRADED:
                score = 50
            else:
                score = 0
                
            weighted_score += score * weight
            total_weight += weight
            
        return int(weighted_score / total_weight) if total_weight > 0 else 0
    
    def get_alerts(self) -> List[str]:
        """Get current alerts based on failure counts"""
        alerts = []
        
        for name, count in self.failure_counts.items():
            if count >= self.alert_threshold:
                config = self.components[name]
                severity = "CRITICAL" if config.get("critical") else "WARNING"
                alerts.append(f"{severity}: {name} has failed {count} times")
                
        return alerts
    
    def print_status(self):
        """Print current status to console"""
        health_score = self.calculate_health_score()
        
        # Clear screen (works on Windows)
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Print header
        print("="*60)
        print(f"TORI HEALTH MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Overall health
        score_color = ""
        if health_score >= 80:
            score_color = "\033[92m"  # Green
        elif health_score >= 60:
            score_color = "\033[93m"  # Yellow
        else:
            score_color = "\033[91m"  # Red
            
        print(f"\nOverall Health: {score_color}{health_score}/100\033[0m")
        
        # Component status
        print("\nComponent Status:")
        print("-"*60)
        print(f"{'Component':<20} {'Status':<12} {'Response Time':<15} {'Error'}")
        print("-"*60)
        
        for name, metric in sorted(self.metrics.items()):
            status_color = {
                ComponentStatus.HEALTHY: "\033[92m",
                ComponentStatus.DEGRADED: "\033[93m",
                ComponentStatus.FAILED: "\033[91m",
                ComponentStatus.UNKNOWN: "\033[90m"
            }.get(metric.status, "")
            
            response_str = f"{metric.response_time:.3f}s" if metric.response_time else "N/A"
            error_str = metric.error_message or ""
            
            print(f"{name:<20} {status_color}{metric.status.value:<12}\033[0m {response_str:<15} {error_str}")
        
        # Alerts
        alerts = self.get_alerts()
        if alerts:
            print(f"\n\033[91mALERTS:\033[0m")
            for alert in alerts:
                print(f"  • {alert}")
        
        # Statistics
        print("\nStatistics:")
        healthy = sum(1 for m in self.metrics.values() if m.status == ComponentStatus.HEALTHY)
        total = len(self.metrics)
        print(f"  Healthy Components: {healthy}/{total}")
        
        avg_response = [m.response_time for m in self.metrics.values() if m.response_time]
        if avg_response:
            print(f"  Average Response Time: {sum(avg_response)/len(avg_response):.3f}s")
    
    async def monitor_loop(self, interval: int = 5):
        """Main monitoring loop"""
        print("Starting TORI Health Monitor...")
        print(f"Checking every {interval} seconds. Press Ctrl+C to stop.\n")
        
        try:
            while True:
                # Check all components
                await self.check_all_components()
                
                # Update history
                snapshot = {
                    "timestamp": datetime.now().isoformat(),
                    "health_score": self.calculate_health_score(),
                    "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
                    "alerts": self.get_alerts()
                }
                self.history.append(snapshot)
                
                # Keep history limited
                if len(self.history) > 1000:
                    self.history = self.history[-1000:]
                
                # Display status
                self.print_status()
                
                # Save snapshot to file
                self.save_snapshot(snapshot)
                
                # Wait for next check
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
            
    def save_snapshot(self, snapshot: Dict):
        """Save snapshot to file for historical analysis"""
        snapshot_dir = Path("debugging_enhanced/health_snapshots")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Daily rotation
        filename = snapshot_dir / f"health_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(filename, 'a') as f:
            f.write(json.dumps(snapshot) + '\n')
    
    async def generate_report(self) -> Dict:
        """Generate health report"""
        if not self.history:
            return {"error": "No data collected yet"}
            
        # Calculate statistics
        health_scores = [h["health_score"] for h in self.history]
        
        report = {
            "report_time": datetime.now().isoformat(),
            "monitoring_duration": len(self.history) * 5,  # seconds
            "current_health": self.calculate_health_score(),
            "statistics": {
                "average_health": sum(health_scores) / len(health_scores),
                "min_health": min(health_scores),
                "max_health": max(health_scores),
                "total_alerts": sum(len(h["alerts"]) for h in self.history)
            },
            "component_availability": {},
            "recent_alerts": self.history[-1]["alerts"] if self.history else []
        }
        
        # Component availability
        for component in self.components:
            healthy_count = sum(
                1 for h in self.history
                if h["metrics"].get(component, {}).get("status") == "healthy"
            )
            availability = (healthy_count / len(self.history) * 100) if self.history else 0
            report["component_availability"][component] = f"{availability:.1f}%"
        
        return report


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TORI Health Monitor")
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Check interval in seconds (default: 5)"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate report and exit"
    )
    
    args = parser.parse_args()
    
    monitor = TORIHealthMonitor()
    
    if args.report:
        # Generate report mode
        print("Generating health report...")
        
        # Run a few checks to gather data
        for _ in range(5):
            await monitor.check_all_components()
            await asyncio.sleep(1)
        
        report = await monitor.generate_report()
        
        report_path = Path(f"debugging_enhanced/health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"Report saved to: {report_path}")
        print(json.dumps(report, indent=2))
    else:
        # Continuous monitoring mode
        await monitor.monitor_loop(interval=args.interval)


if __name__ == "__main__":
    asyncio.run(main())
