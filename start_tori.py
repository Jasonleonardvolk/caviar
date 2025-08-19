#!/usr/bin/env python3
"""
TORI/KHA Startup Script - Production Deployment
Starts all Python services and Node.js frontend with proper coordination
"""

import subprocess
import sys
import time
import signal
import json
import os
from pathlib import Path
from typing import List, Optional
import threading
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TORILauncher:
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.shutdown_requested = False
        
        # Project paths
        self.project_root = Path(__file__).parent
        self.python_root = self.project_root / "python"
        
        # Service configurations
        self.services = {
            'cognitive_engine': {
                'script': self.python_root / 'core' / 'CognitiveEngine.py',
                'port': 8001,
                'required': True
            },
            'memory_vault': {
                'script': self.python_root / 'core' / 'memory_vault.py', 
                'port': 8002,
                'required': True
            },
            'eigenvalue_monitor': {
                'script': self.python_root / 'stability' / 'eigenvalue_monitor.py',
                'port': 8003,
                'required': True
            },
            'lyapunov_analyzer': {
                'script': self.python_root / 'stability' / 'lyapunov_analyzer.py',
                'port': 8004,
                'required': False
            },
            'koopman_operator': {
                'script': self.python_root / 'stability' / 'koopman_operator.py',
                'port': 8005,
                'required': False
            }
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_requested = True
        self.shutdown()
        sys.exit(0)
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        logger.info("Checking dependencies...")
        
        # Check Python packages
        required_packages = [
            'numpy', 'scipy', 'asyncio', 'sqlite3', 
            'json', 'pathlib', 'dataclasses'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✓ {package}")
            except ImportError:
                logger.error(f"✗ {package} - not found")
                return False
        
        # Check Node.js availability
        try:
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info(f"✓ Node.js {result.stdout.strip()}")
            else:
                logger.error("✗ Node.js - not available")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.error("✗ Node.js - not found")
            return False
        
        # Check npm availability (Windows uses npm.cmd)
        npm_cmd = 'npm.cmd' if sys.platform == 'win32' else 'npm'
        try:
            result = subprocess.run([npm_cmd, '--version'], 
                                  capture_output=True, text=True, timeout=5, shell=True)
            if result.returncode == 0:
                logger.info(f"✓ npm {result.stdout.strip()}")
            else:
                logger.error("✗ npm - not available")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.error("✗ npm - not found")
            return False
        
        logger.info("All dependencies check passed ✓")
        return True
    
    def setup_environment(self):
        """Setup environment variables and directories"""
        logger.info("Setting up environment...")
        
        # Create data directories
        data_dirs = [
            'data/cognitive',
            'data/memory_vault',
            'data/eigenvalue_monitor',
            'data/lyapunov',
            'data/koopman'
        ]
        
        for dir_path in data_dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {full_path}")
        
        # Set environment variables
        os.environ['PYTHONPATH'] = str(self.python_root)
        os.environ['TORI_PROJECT_ROOT'] = str(self.project_root)
        os.environ['TORI_MODE'] = 'production'
        
        logger.info("Environment setup complete ✓")
    
    def start_python_service(self, service_name: str, config: dict) -> Optional[subprocess.Popen]:
        """Start a Python service"""
        script_path = config['script']
        
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return None
        
        logger.info(f"Starting {service_name}...")
        
        try:
            # Start as a test module first to validate
            python_root_str = str(self.python_root).replace('\\', '/')
            cmd = [
                sys.executable, '-c',
                f"""
import sys
sys.path.insert(0, '{python_root_str}')

# Test import
try:
    if '{service_name}' == 'cognitive_engine':
        from python.core.CognitiveEngine import CognitiveEngine
        engine = CognitiveEngine({{'storage_path': 'data/cognitive'}})
        print("CognitiveEngine initialized successfully")
    elif '{service_name}' == 'memory_vault':
        from python.core.memory_vault import UnifiedMemoryVault
        vault = UnifiedMemoryVault({{'storage_path': 'data/memory_vault'}})
        print("MemoryVault initialized successfully")
    elif '{service_name}' == 'eigenvalue_monitor':
        from python.stability.eigenvalue_monitor import EigenvalueMonitor
        monitor = EigenvalueMonitor({{'storage_path': 'data/eigenvalue_monitor'}})
        print("EigenvalueMonitor initialized successfully")
    elif '{service_name}' == 'lyapunov_analyzer':
        from python.stability.lyapunov_analyzer import LyapunovAnalyzer
        analyzer = LyapunovAnalyzer({{'storage_path': 'data/lyapunov'}})
        print("LyapunovAnalyzer initialized successfully")
    elif '{service_name}' == 'koopman_operator':
        from python.stability.koopman_operator import KoopmanOperator
        operator = KoopmanOperator({{'storage_path': 'data/koopman'}})
        print("KoopmanOperator initialized successfully")
    
    print(f"{service_name} service ready")
    
    # Keep service running (in production, this would be a proper server)
    import time
    while True:
        time.sleep(1)
        
except Exception as e:
    print(f"Error starting {service_name}: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.project_root
            )
            
            # Give it a moment to start
            time.sleep(2)
            
            # Check if it's still running
            if process.poll() is None:
                logger.info(f"✓ {service_name} started successfully (PID: {process.pid})")
                self.processes.append(process)
                return process
            else:
                stdout, stderr = process.communicate()
                logger.error(f"✗ {service_name} failed to start:")
                logger.error(f"stdout: {stdout}")
                logger.error(f"stderr: {stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error starting {service_name}: {e}")
            return None
    
    def start_frontend(self) -> Optional[subprocess.Popen]:
        """Start the SvelteKit frontend"""
        logger.info("Starting frontend...")
        
        try:
            # First install dependencies if needed
            package_json = self.project_root / "package.json"
            node_modules = self.project_root / "node_modules"
            
            if package_json.exists() and not node_modules.exists():
                logger.info("Installing npm dependencies...")
                npm_cmd = 'npm.cmd' if sys.platform == 'win32' else 'npm'
                install_process = subprocess.run(
                    [npm_cmd, 'install'],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    shell=True
                )
                
                if install_process.returncode != 0:
                    logger.error(f"npm install failed: {install_process.stderr}")
                    return None
                
                logger.info("npm dependencies installed ✓")
            
            # Start the dev server
            npm_cmd = 'npm.cmd' if sys.platform == 'win32' else 'npm'
            process = subprocess.Popen(
                [npm_cmd, 'run', 'dev', '--', '--port', '5173'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.project_root,
                shell=True
            )
            
            # Wait a bit and check if it started
            time.sleep(5)
            
            if process.poll() is None:
                logger.info(f"✓ Frontend started successfully (PID: {process.pid})")
                logger.info("Frontend available at: http://localhost:5173")
                self.processes.append(process)
                return process
            else:
                stdout, stderr = process.communicate()
                logger.error("✗ Frontend failed to start:")
                logger.error(f"stdout: {stdout}")
                logger.error(f"stderr: {stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error starting frontend: {e}")
            return None
    
    def monitor_services(self):
        """Monitor running services"""
        logger.info("Starting service monitoring...")
        
        while not self.shutdown_requested:
            try:
                # Check each process
                for i, process in enumerate(self.processes):
                    if process.poll() is not None:
                        logger.warning(f"Process {process.pid} has terminated")
                        # In production, you might want to restart it here
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5)
    
    def start_all(self):
        """Start all services"""
        logger.info("🚀 Starting TORI/KHA system...")
        
        # Check dependencies
        if not self.check_dependencies():
            logger.error("Dependency check failed. Aborting startup.")
            return False
        
        # Setup environment
        self.setup_environment()
        
        # Start Python services
        started_services = 0
        required_services = 0
        
        for service_name, config in self.services.items():
            if config['required']:
                required_services += 1
            
            process = self.start_python_service(service_name, config)
            if process:
                started_services += 1
            elif config['required']:
                logger.error(f"Required service {service_name} failed to start")
                self.shutdown()
                return False
        
        logger.info(f"Started {started_services}/{len(self.services)} Python services")
        
        # Start frontend
        frontend_process = self.start_frontend()
        if not frontend_process:
            logger.error("Frontend failed to start")
            self.shutdown()
            return False
        
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=self.monitor_services, daemon=True)
        monitor_thread.start()
        
        logger.info("✅ TORI/KHA system started successfully!")
        logger.info("=" * 60)
        logger.info("🌐 Frontend: http://localhost:5173")
        logger.info("🧠 Python services: Running")
        logger.info("📊 Monitoring: Active")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to shutdown")
        
        return True
    
    def shutdown(self):
        """Shutdown all services"""
        logger.info("🛑 Shutting down TORI/KHA system...")
        
        self.shutdown_requested = True
        
        # Terminate all processes
        for process in self.processes:
            try:
                logger.info(f"Terminating process {process.pid}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                    logger.info(f"Process {process.pid} terminated gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing process {process.pid}...")
                    process.kill()
                    process.wait()
                    
            except Exception as e:
                logger.error(f"Error terminating process {process.pid}: {e}")
        
        self.processes.clear()
        logger.info("✅ TORI/KHA system shutdown complete")
    
    def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        try:
            while not self.shutdown_requested:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.shutdown()

def main():
    """Main entry point"""
    launcher = TORILauncher()
    
    if launcher.start_all():
        launcher.wait_for_shutdown()
    else:
        logger.error("Failed to start TORI/KHA system")
        sys.exit(1)

if __name__ == "__main__":
    main()
