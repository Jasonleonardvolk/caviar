#!/usr/bin/env python3
"""
TORI System Monitor - Continuous Health Checking
Monitors all TORI services and automatically attempts recovery
"""

import os
import sys
import time
import json
import socket
import subprocess
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import psutil

class TORISystemMonitor:
    """Continuous monitoring and auto-recovery for TORI system"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.config = {
            'ports': {
                'api': 8002,
                'frontend': 5173,
                'audio_bridge': 8765,
                'concept_mesh': 8766,
                'mcp': 8100
            },
            'check_interval': 30,  # seconds
            'recovery_attempts': 3,
            'recovery_delay': 5,   # seconds between attempts
        }
        
        # Setup logging
        self.setup_logging()
        
        # Service status tracking
        self.service_status = {
            'api': {'healthy': False, 'last_check': None, 'consecutive_failures': 0},
            'frontend': {'healthy': False, 'last_check': None, 'consecutive_failures': 0},
            'audio_bridge': {'healthy': False, 'last_check': None, 'consecutive_failures': 0},
            'concept_mesh': {'healthy': False, 'last_check': None, 'consecutive_failures': 0},
            'mcp': {'healthy': False, 'last_check': None, 'consecutive_failures': 0}
        }
        
        # Recovery in progress flag
        self.recovery_in_progress = False
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.script_dir / 'logs' / 'monitor'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f'monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def check_port(self, port: int, service: str) -> bool:
        """Check if a port is listening"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                result = s.connect_ex(('127.0.0.1', port))
                return result == 0
        except Exception as e:
            self.logger.error(f"Error checking port {port} for {service}: {e}")
            return False
    
    def check_api_health(self) -> Tuple[bool, Optional[Dict]]:
        """Check API server health endpoint"""
        try:
            response = requests.get(
                f'http://localhost:{self.config["ports"]["api"]}/api/health',
                timeout=5
            )
            if response.status_code == 200:
                return True, response.json()
            return False, None
        except Exception as e:
            self.logger.debug(f"API health check failed: {e}")
            return False, None
    
    def check_frontend_health(self) -> bool:
        """Check if frontend is responding"""
        try:
            response = requests.get(
                f'http://localhost:{self.config["ports"]["frontend"]}/',
                timeout=5
            )
            return response.status_code in [200, 304]
        except Exception:
            return False
    
    def check_bridge_health(self, port: int, service: str) -> bool:
        """Check if bridge WebSocket is accessible"""
        # For now, just check if port is listening
        # Could be enhanced with actual WebSocket ping
        return self.check_port(port, service)
    
    def check_all_services(self) -> Dict[str, bool]:
        """Check health of all services"""
        results = {}
        
        # Check API
        api_healthy, api_data = self.check_api_health()
        results['api'] = api_healthy
        if api_healthy and api_data:
            self.logger.debug(f"API health data: {api_data}")
        
        # Check Frontend
        results['frontend'] = self.check_frontend_health()
        
        # Check Audio Bridge
        results['audio_bridge'] = self.check_bridge_health(
            self.config['ports']['audio_bridge'], 
            'audio_bridge'
        )
        
        # Check Concept Mesh Bridge
        results['concept_mesh'] = self.check_bridge_health(
            self.config['ports']['concept_mesh'],
            'concept_mesh'
        )
        
        # Check MCP
        results['mcp'] = self.check_port(self.config['ports']['mcp'], 'mcp')
        
        return results
    
    def update_service_status(self, service: str, healthy: bool):
        """Update service status tracking"""
        status = self.service_status[service]
        status['last_check'] = datetime.now()
        
        if healthy:
            if not status['healthy']:
                self.logger.info(f"✅ {service} is now HEALTHY")
            status['healthy'] = True
            status['consecutive_failures'] = 0
        else:
            if status['healthy']:
                self.logger.warning(f"⚠️ {service} is now UNHEALTHY")
            status['healthy'] = False
            status['consecutive_failures'] += 1
    
    def get_process_info(self, port: int) -> Optional[Dict]:
        """Get information about process using a port"""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    try:
                        proc = psutil.Process(conn.pid)
                        return {
                            'pid': conn.pid,
                            'name': proc.name(),
                            'cmdline': ' '.join(proc.cmdline()),
                            'create_time': datetime.fromtimestamp(proc.create_time())
                        }
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        except Exception as e:
            self.logger.error(f"Error getting process info for port {port}: {e}")
        return None
    
    def attempt_service_recovery(self, service: str):
        """Attempt to recover a failed service"""
        if self.recovery_in_progress:
            self.logger.info(f"Recovery already in progress, skipping {service}")
            return
        
        self.recovery_in_progress = True
        self.logger.warning(f"🔧 Attempting recovery for {service}")
        
        try:
            if service in ['audio_bridge', 'concept_mesh']:
                self.recover_bridge_service(service)
            elif service == 'api':
                self.logger.error("API server down - manual restart required")
                self.send_alert("API server is down and requires manual restart")
            elif service == 'frontend':
                self.logger.warning("Frontend down - may be rebuilding")
            elif service == 'mcp':
                self.logger.warning("MCP server down - checking if it can be restarted")
                
        except Exception as e:
            self.logger.error(f"Recovery failed for {service}: {e}")
        finally:
            self.recovery_in_progress = False
    
    def recover_bridge_service(self, service: str):
        """Recover a bridge service"""
        port = self.config['ports'][service]
        script_name = 'audio_hologram_bridge.py' if service == 'audio_bridge' else 'concept_mesh_hologram_bridge.py'
        script_path = self.script_dir / script_name
        
        if not script_path.exists():
            self.logger.error(f"Bridge script not found: {script_path}")
            return
        
        # Kill any existing process on the port
        self.logger.info(f"Killing any process on port {port}")
        self.kill_process_on_port(port)
        
        # Wait for port to be freed
        time.sleep(2)
        
        # Start the bridge
        self.logger.info(f"Starting {service} on port {port}")
        
        try:
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            cmd = [
                sys.executable, 
                str(script_path),
                '--host', '127.0.0.1',
                '--port', str(port)
            ]
            
            process = subprocess.Popen(
                cmd,
                cwd=str(self.script_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a moment for startup
            time.sleep(3)
            
            # Check if it started successfully
            if process.poll() is None and self.check_port(port, service):
                self.logger.info(f"✅ {service} recovered successfully (PID: {process.pid})")
            else:
                self.logger.error(f"Failed to recover {service}")
                
        except Exception as e:
            self.logger.error(f"Error starting {service}: {e}")
    
    def kill_process_on_port(self, port: int):
        """Kill process using a specific port"""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port:
                    try:
                        proc = psutil.Process(conn.pid)
                        self.logger.info(f"Terminating {proc.name()} (PID: {conn.pid})")
                        proc.terminate()
                        proc.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        self.logger.warning(f"Force killing PID {conn.pid}")
                        proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        except Exception as e:
            self.logger.error(f"Error killing process on port {port}: {e}")
    
    def send_alert(self, message: str):
        """Send alert for critical issues"""
        self.logger.critical(f"🚨 ALERT: {message}")
        
        # Could be extended to send email, SMS, or system notification
        # For now, just log prominently
        
        alert_file = self.script_dir / 'logs' / 'ALERTS.txt'
        with open(alert_file, 'a') as f:
            f.write(f"[{datetime.now()}] {message}\n")
    
    def generate_status_report(self) -> str:
        """Generate a status report"""
        report = ["TORI System Status Report", "=" * 40]
        
        for service, status in self.service_status.items():
            health = "✅ HEALTHY" if status['healthy'] else "❌ UNHEALTHY"
            failures = status['consecutive_failures']
            last_check = status['last_check'].strftime('%H:%M:%S') if status['last_check'] else 'Never'
            
            report.append(f"{service.upper()}: {health}")
            report.append(f"  Last Check: {last_check}")
            if failures > 0:
                report.append(f"  Consecutive Failures: {failures}")
            
            # Get process info if service is up
            if status['healthy']:
                port = self.config['ports'][service]
                proc_info = self.get_process_info(port)
                if proc_info:
                    report.append(f"  PID: {proc_info['pid']} ({proc_info['name']})")
        
        report.append("=" * 40)
        return '\n'.join(report)
    
    def monitor_loop(self):
        """Main monitoring loop"""
        self.logger.info("🔍 Starting TORI system monitoring")
        self.logger.info(f"Check interval: {self.config['check_interval']} seconds")
        
        while True:
            try:
                # Check all services
                results = self.check_all_services()
                
                # Update status and check for recovery needs
                for service, healthy in results.items():
                    self.update_service_status(service, healthy)
                    
                    # Check if recovery is needed
                    status = self.service_status[service]
                    if not healthy and status['consecutive_failures'] >= 3:
                        if status['consecutive_failures'] == 3:  # First time hitting threshold
                            self.attempt_service_recovery(service)
                
                # Generate and log status report every 5 checks
                if hasattr(self, 'check_count'):
                    self.check_count += 1
                else:
                    self.check_count = 1
                
                if self.check_count % 5 == 0:
                    report = self.generate_status_report()
                    self.logger.info(f"\n{report}")
                
                # Sleep until next check
                time.sleep(self.config['check_interval'])
                
            except KeyboardInterrupt:
                self.logger.info("Monitor shutdown requested")
                break
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                time.sleep(self.config['check_interval'])
    
    def run(self):
        """Run the monitor"""
        try:
            self.monitor_loop()
        except Exception as e:
            self.logger.critical(f"Fatal error in monitor: {e}")
            raise
        finally:
            self.logger.info("Monitor stopped")


def main():
    """Main entry point"""
    print("=" * 60)
    print("TORI SYSTEM MONITOR v1.0")
    print("Continuous health checking and auto-recovery")
    print("=" * 60)
    print()
    
    monitor = TORISystemMonitor()
    
    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\nMonitor stopped by user")
        return 0
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
