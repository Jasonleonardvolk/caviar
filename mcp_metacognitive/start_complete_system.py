#!/usr/bin/env python3
"""
Start Complete TORI System with Phase 2 Components
=================================================

This script starts all TORI components:
- MCP Metacognitive Server
- Daniel (Cognitive Engine)
- Kaizen (Continuous Improvement)
- Celery Workers (if available)
- Redis (if needed)
"""

import os
import sys
import subprocess
import time
import asyncio
import signal
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TORISystemLauncher:
    """Launcher for complete TORI system"""
    
    def __init__(self):
        self.processes = {}
        self.base_dir = Path(__file__).parent
        
    def check_redis(self) -> bool:
        """Check if Redis is running"""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379)
            r.ping()
            logger.info("✅ Redis is running")
            return True
        except:
            logger.warning("⚠️  Redis not running")
            return False
    
    def start_redis(self):
        """Attempt to start Redis"""
        if self.check_redis():
            return True
        
        logger.info("Starting Redis...")
        
        # Try common Redis commands
        redis_commands = [
            'redis-server',
            'redis-server.exe',
            '/usr/local/bin/redis-server',
            'C:\\Program Files\\Redis\\redis-server.exe'
        ]
        
        for cmd in redis_commands:
            try:
                self.processes['redis'] = subprocess.Popen(
                    [cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                time.sleep(2)  # Give Redis time to start
                
                if self.check_redis():
                    logger.info("✅ Redis started successfully")
                    return True
            except:
                continue
        
        logger.warning("⚠️  Could not start Redis - Celery features will be limited")
        return False
    
    def start_celery_worker(self):
        """Start Celery worker"""
        logger.info("Starting Celery worker...")
        
        try:
            worker_cmd = [
                sys.executable, '-m', 'celery',
                '-A', 'mcp_metacognitive.tasks.celery_tasks',
                'worker',
                '--loglevel=info',
                '--concurrency=4',
                '--queues=tori.default,tori.cognitive,tori.analysis,tori.tools,tori.learning'
            ]
            
            self.processes['celery_worker'] = subprocess.Popen(
                worker_cmd,
                cwd=str(self.base_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info("✅ Celery worker started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Celery worker: {e}")
            return False
    
    def start_celery_beat(self):
        """Start Celery beat scheduler"""
        logger.info("Starting Celery beat...")
        
        try:
            beat_cmd = [
                sys.executable, '-m', 'celery',
                '-A', 'mcp_metacognitive.tasks.celery_tasks',
                'beat',
                '--loglevel=info'
            ]
            
            self.processes['celery_beat'] = subprocess.Popen(
                beat_cmd,
                cwd=str(self.base_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info("✅ Celery beat started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Celery beat: {e}")
            return False
    
    def start_flower(self, port: int = 5555):
        """Start Flower monitoring (optional)"""
        logger.info(f"Starting Flower on port {port}...")
        
        try:
            flower_cmd = [
                sys.executable, '-m', 'celery',
                '-A', 'mcp_metacognitive.tasks.celery_tasks',
                'flower',
                f'--port={port}'
            ]
            
            self.processes['flower'] = subprocess.Popen(
                flower_cmd,
                cwd=str(self.base_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info(f"✅ Flower started on http://localhost:{port}")
            return True
            
        except Exception as e:
            logger.warning(f"Could not start Flower: {e}")
            return False
    
    def start_mcp_server(self):
        """Start the main MCP server"""
        logger.info("Starting TORI MCP server...")
        
        try:
            server_cmd = [sys.executable, 'server.py']
            
            env = os.environ.copy()
            env.update({
                'TRANSPORT_TYPE': 'sse',
                'SERVER_PORT': '8100',
                'SERVER_HOST': '0.0.0.0',
                'PYTHONIOENCODING': 'utf-8',
                'KAIZEN_AUTO_START': 'true',
                'DANIEL_MODEL_BACKEND': 'mock'  # Use mock for testing
            })
            
            self.processes['mcp_server'] = subprocess.Popen(
                server_cmd,
                cwd=str(self.base_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
            
            logger.info("✅ TORI MCP server started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor all processes"""
        while True:
            try:
                time.sleep(5)
                
                # Check each process
                for name, process in self.processes.items():
                    if process and process.poll() is not None:
                        logger.warning(f"⚠️  {name} process died (exit code: {process.returncode})")
                        
                        # Try to restart critical processes
                        if name == 'mcp_server':
                            logger.info(f"Restarting {name}...")
                            self.start_mcp_server()
                            
            except KeyboardInterrupt:
                break
    
    def shutdown(self):
        """Shutdown all processes"""
        logger.info("Shutting down TORI system...")
        
        for name, process in self.processes.items():
            if process and process.poll() is None:
                logger.info(f"Terminating {name}...")
                process.terminate()
                
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        logger.info("TORI system shutdown complete")
    
    def run(self):
        """Run the complete system"""
        print("🚀 Starting TORI Complete System")
        print("=" * 50)
        
        # Check/start Redis
        redis_available = self.start_redis()
        
        # Start Celery components if Redis is available
        if redis_available:
            self.start_celery_worker()
            self.start_celery_beat()
            self.start_flower()
        else:
            print("⚠️  Running without Celery (no async tasks)")
        
        # Start main server
        self.start_mcp_server()
        
        # Give everything time to start
        time.sleep(3)
        
        print("\n✅ TORI System Started!")
        print("=" * 50)
        print("🌐 MCP Server: http://localhost:8100")
        print("📊 System Status: http://localhost:8100/api/system/status")
        print("🧠 Query Endpoint: http://localhost:8100/api/query")
        print("💡 Insights: http://localhost:8100/api/insights")
        
        if redis_available:
            print("🌸 Flower Monitor: http://localhost:5555")
            print("✅ Background tasks: ENABLED")
        else:
            print("⚠️  Background tasks: DISABLED (Redis not available)")
        
        print("\n💡 Press Ctrl+C to shutdown")
        print("=" * 50)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, lambda s, f: self.shutdown())
        signal.signal(signal.SIGTERM, lambda s, f: self.shutdown())
        
        # Monitor processes
        try:
            self.monitor_processes()
        except KeyboardInterrupt:
            self.shutdown()

def main():
    """Main entry point"""
    launcher = TORISystemLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
