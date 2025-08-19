#!/usr/bin/env python3
"""
TORI Health Monitor
Real-time monitoring of all system components
"""

import asyncio
import aiohttp
import redis
import json
from datetime import datetime
from typing import Dict, List, Optional
import os

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    CLEAR = '\033[2J\033[H'  # Clear screen and move cursor to top

class HealthMonitor:
    def __init__(self):
        self.checks = {
            "Redis": {
                "type": "redis",
                "url": "redis://localhost:6379",
                "critical": True
            },
            "API Server": {
                "type": "http",
                "url": "http://localhost:8002/api/health",
                "critical": True
            },
            "Frontend": {
                "type": "http",
                "url": "http://localhost:5173",
                "critical": True
            },
            "API Docs": {
                "type": "http",
                "url": "http://localhost:8002/docs",
                "critical": False
            },
            "MCP Server": {
                "type": "http",
                "url": "http://localhost:8100/api/system/status",
                "critical": False
            },
            "Audio Bridge": {
                "type": "tcp",
                "host": "localhost",
                "port": 8765,
                "critical": False
            },
            "Concept Bridge": {
                "type": "tcp",
                "host": "localhost",
                "port": 8766,
                "critical": False
            }
        }
        
        self.status_history = {}
        self.check_count = 0
        self.start_time = datetime.now()
        
    async def check_http(self, url: str, timeout: int = 5) -> Dict:
        """Check HTTP endpoint health"""
        start_time = datetime.now()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=timeout) as response:
                    response_time = (datetime.now() - start_time).total_seconds() * 1000
                    
                    return {
                        "healthy": response.status == 200,
                        "status_code": response.status,
                        "response_time_ms": round(response_time, 1),
                        "error": None
                    }
        except asyncio.TimeoutError:
            return {
                "healthy": False,
                "status_code": None,
                "response_time_ms": timeout * 1000,
                "error": "Timeout"
            }
        except Exception as e:
            return {
                "healthy": False,
                "status_code": None,
                "response_time_ms": None,
                "error": str(e)[:50]
            }
    
    async def check_redis(self, url: str) -> Dict:
        """Check Redis health"""
        start_time = datetime.now()
        
        try:
            r = redis.from_url(url, socket_connect_timeout=2)
            result = r.ping()
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Try to get some stats
            info = r.info()
            memory_mb = info.get('used_memory', 0) / 1024 / 1024
            
            return {
                "healthy": result,
                "response_time_ms": round(response_time, 1),
                "memory_mb": round(memory_mb, 1),
                "error": None
            }
        except Exception as e:
            return {
                "healthy": False,
                "response_time_ms": None,
                "memory_mb": None,
                "error": str(e)[:50]
            }
    
    async def check_tcp(self, host: str, port: int, timeout: int = 5) -> Dict:
        """Check TCP port availability"""
        import socket
        
        start_time = datetime.now()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        try:
            result = sock.connect_ex((host, port))
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "healthy": result == 0,
                "response_time_ms": round(response_time, 1),
                "error": None if result == 0 else f"Port {port} not accessible"
            }
        except Exception as e:
            return {
                "healthy": False,
                "response_time_ms": None,
                "error": str(e)[:50]
            }
        finally:
            sock.close()
    
    async def check_component(self, name: str, config: Dict) -> Dict:
        """Check a single component"""
        check_type = config["type"]
        
        if check_type == "http":
            result = await self.check_http(config["url"])
        elif check_type == "redis":
            result = await self.check_redis(config["url"])
        elif check_type == "tcp":
            result = await self.check_tcp(config["host"], config["port"])
        else:
            result = {"healthy": False, "error": f"Unknown check type: {check_type}"}
        
        result["name"] = name
        result["critical"] = config.get("critical", False)
        result["timestamp"] = datetime.now().isoformat()
        
        # Update history
        if name not in self.status_history:
            self.status_history[name] = []
        
        self.status_history[name].append(result["healthy"])
        # Keep only last 10 results
        self.status_history[name] = self.status_history[name][-10:]
        
        return result
    
    async def check_all(self) -> List[Dict]:
        """Check all components"""
        tasks = []
        for name, config in self.checks.items():
            tasks.append(self.check_component(name, config))
        
        results = await asyncio.gather(*tasks)
        self.check_count += 1
        return results
    
    def calculate_uptime(self, name: str) -> float:
        """Calculate uptime percentage for a component"""
        if name not in self.status_history or not self.status_history[name]:
            return 0.0
        
        healthy_count = sum(1 for status in self.status_history[name] if status)
        total_count = len(self.status_history[name])
        
        return (healthy_count / total_count) * 100
    
    def print_dashboard(self, results: List[Dict]):
        """Print formatted dashboard"""
        # Clear screen (optional - comment out if you prefer scrolling)
        # print(Colors.CLEAR, end='')
        
        # Header
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'TORI SYSTEM HEALTH MONITOR':^80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
        
        # System info
        runtime = datetime.now() - self.start_time
        runtime_str = str(runtime).split('.')[0]  # Remove microseconds
        
        print(f"\n{Colors.BOLD}System Information:{Colors.RESET}")
        print(f"  Monitoring Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Runtime: {runtime_str}")
        print(f"  Checks Performed: {self.check_count}")
        print(f"  Last Check: {datetime.now().strftime('%H:%M:%S')}")
        
        # Component status
        print(f"\n{Colors.BOLD}Component Status:{Colors.RESET}")
        print(f"{'─'*80}")
        print(f"{'Component':<20} {'Status':<10} {'Response':<12} {'Uptime':<10} {'Details':<30}")
        print(f"{'─'*80}")
        
        critical_healthy = 0
        critical_total = 0
        
        for result in results:
            name = result["name"]
            healthy = result["healthy"]
            critical = result["critical"]
            response_time = result.get("response_time_ms", "N/A")
            error = result.get("error", "")
            uptime = self.calculate_uptime(name)
            
            if critical:
                critical_total += 1
                if healthy:
                    critical_healthy += 1
            
            # Status icon and color
            if healthy:
                status = f"{Colors.GREEN}● UP{Colors.RESET}"
            else:
                status = f"{Colors.RED}● DOWN{Colors.RESET}"
            
            # Response time formatting
            if response_time != "N/A" and response_time is not None:
                if response_time < 100:
                    resp_color = Colors.GREEN
                elif response_time < 500:
                    resp_color = Colors.YELLOW
                else:
                    resp_color = Colors.RED
                response_str = f"{resp_color}{response_time}ms{Colors.RESET}"
            else:
                response_str = "N/A"
            
            # Uptime color
            if uptime >= 99:
                uptime_color = Colors.GREEN
            elif uptime >= 90:
                uptime_color = Colors.YELLOW
            else:
                uptime_color = Colors.RED
            uptime_str = f"{uptime_color}{uptime:.1f}%{Colors.RESET}"
            
            # Details
            if healthy:
                if name == "Redis" and "memory_mb" in result:
                    details = f"Memory: {result['memory_mb']}MB"
                elif "status_code" in result and result["status_code"]:
                    details = f"HTTP {result['status_code']}"
                else:
                    details = "Healthy"
            else:
                details = error or "Connection failed"
            
            # Critical marker
            critical_marker = "*" if critical else " "
            
            print(f"{name:<19}{critical_marker} {status:<10} {response_str:<12} {uptime_str:<10} {details:<30}")
        
        print(f"{'─'*80}")
        print(f"{Colors.BOLD}* = Critical component{Colors.RESET}")
        
        # Overall health
        print(f"\n{Colors.BOLD}Overall System Health:{Colors.RESET}")
        
        all_healthy = all(r["healthy"] for r in results)
        critical_ok = critical_healthy == critical_total
        
        if all_healthy:
            print(f"  {Colors.GREEN}● ALL SYSTEMS OPERATIONAL{Colors.RESET}")
        elif critical_ok:
            print(f"  {Colors.YELLOW}● DEGRADED - Non-critical components down{Colors.RESET}")
        else:
            print(f"  {Colors.RED}● CRITICAL - Essential components down{Colors.RESET}")
        
        print(f"  Critical Components: {critical_healthy}/{critical_total} healthy")
        print(f"  Total Components: {sum(1 for r in results if r['healthy'])}/{len(results)} healthy")
        
        # Alerts
        down_components = [r["name"] for r in results if not r["healthy"]]
        if down_components:
            print(f"\n{Colors.BOLD}{Colors.RED}⚠️  ALERTS:{Colors.RESET}")
            for comp in down_components:
                print(f"  - {comp} is DOWN")
        
        print(f"\n{Colors.BLUE}Refreshing every 10 seconds... Press Ctrl+C to stop{Colors.RESET}")
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        print(f"{Colors.BOLD}{Colors.GREEN}Starting TORI Health Monitor...{Colors.RESET}")
        
        while True:
            try:
                results = await self.check_all()
                self.print_dashboard(results)
                
                # Save to file for other tools
                self.save_status(results)
                
                await asyncio.sleep(10)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"{Colors.RED}Monitor error: {e}{Colors.RESET}")
                await asyncio.sleep(10)
    
    def save_status(self, results: List[Dict]):
        """Save current status to file"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "check_count": self.check_count,
            "runtime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "components": {r["name"]: r for r in results},
            "overall_health": all(r["healthy"] for r in results),
            "critical_health": all(r["healthy"] for r in results if r["critical"])
        }
        
        try:
            with open("health_status.json", "w") as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            print(f"Failed to save status: {e}")

async def main():
    """Main entry point"""
    monitor = HealthMonitor()
    
    try:
        await monitor.monitor_loop()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Health monitor stopped by user{Colors.RESET}")
        
        # Print final summary
        print(f"\n{Colors.BOLD}Session Summary:{Colors.RESET}")
        print(f"  Total checks: {monitor.check_count}")
        print(f"  Runtime: {datetime.now() - monitor.start_time}")
        
        # Calculate overall uptime
        for name in monitor.status_history:
            uptime = monitor.calculate_uptime(name)
            print(f"  {name} uptime: {uptime:.1f}%")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
