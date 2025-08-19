#!/usr/bin/env python3
"""
🚀 ENHANCED UNIFIED TORI LAUNCHER - GRACEFUL SHUTDOWN EDITION v3.0
Now with proper shutdown handling, no more DLL errors!
"""

import socket
import json
import os
import sys
import time
import subprocess
import requests
import asyncio
import atexit
import logging
import threading
import traceback
import signal
import psutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import uvicorn

# Add utils to path for graceful shutdown
sys.path.insert(0, str(Path(__file__).parent))
from utils.graceful_shutdown import GracefulShutdownHandler, AsyncioGracefulShutdown, delayed_keyboard_interrupt

# Suppress startup warnings
import warnings
warnings.filterwarnings("ignore", message=".*already exists.*")
warnings.filterwarnings("ignore", message=".*shadows an attribute.*")
warnings.filterwarnings("ignore", message=".*0 concepts.*")

# Enhanced error handling and encoding
import locale
import codecs

# Set UTF-8 encoding globally
if sys.platform.startswith('win'):
    # Windows UTF-8 setup
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Global shutdown handler
shutdown_handler = GracefulShutdownHandler()


class EnhancedUnifiedToriLauncher:
    """Enhanced TORI launcher with graceful shutdown support"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processes = {}
        self.api_port = None
        self.frontend_port = None
        self.mcp_port = None
        self.running = True
        
        # Setup shutdown handling
        self._setup_shutdown_handling()
        
    def _setup_shutdown_handling(self):
        """Setup graceful shutdown handlers"""
        # Register signal handlers
        shutdown_handler.setup_signal_handlers()
        
        # Add cleanup callback
        shutdown_handler.add_cleanup_callback(self._cleanup)
        
        # Register atexit handler as backup
        atexit.register(self._cleanup)
        
    def _cleanup(self):
        """Cleanup function called during shutdown"""
        self.logger.info("🧹 Starting TORI cleanup sequence...")
        self.running = False
        
        # Save any critical state
        try:
            self._save_state()
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
        
        # Stop monitoring threads
        self.logger.info("✅ TORI cleanup completed")
        
    def _save_state(self):
        """Save critical state before shutdown"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "api_port": self.api_port,
            "frontend_port": self.frontend_port,
            "mcp_port": self.mcp_port,
            "processes": {name: info.pid for name, info in shutdown_handler.processes.items()}
        }
        
        state_file = Path("tori_state.json")
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        self.logger.info(f"💾 Saved state to {state_file}")
        
    def launch(self) -> int:
        """Main launch sequence with protected initialization"""
        try:
            # Protected initialization phase
            with delayed_keyboard_interrupt():
                self.logger.info("🚀 TORI initialization (protected from interruption)...")
                
                # Initialize core components
                self._initialize_components()
                
                # Start API server
                api_port = self._start_api_server()
                if not api_port:
                    return 1
                    
                # Start MCP Metacognitive server
                if not self._start_mcp_server():
                    self.logger.error("Failed to start MCP server")
                    # Continue anyway, not critical
                    
                # Start frontend (optional)
                self._start_frontend()
                
                self.logger.info("✅ TORI initialization complete")
            
            # Main running phase
            self.logger.info("🎯 TORI system ready! Press Ctrl+C to shutdown gracefully.")
            
            # Keep running until shutdown
            while self.running:
                try:
                    time.sleep(1)
                    # Optionally check system health
                    if not self._check_health():
                        self.logger.warning("System health check failed")
                except KeyboardInterrupt:
                    self.logger.info("Received shutdown signal")
                    break
                    
            return 0
            
        except Exception as e:
            self.logger.error(f"Critical error: {e}")
            self.logger.error(traceback.format_exc())
            return 1
        finally:
            # Ensure cleanup runs
            if self.running:
                self._cleanup()
                
    def _initialize_components(self):
        """Initialize core TORI components"""
        self.logger.info("Initializing TORI components...")
        
        # Import and initialize Python components
        try:
            # These imports may take time, protect them
            from python.core import (
                CognitiveEngine, UnifiedMemoryVault, 
                ConceptMesh, CognitiveInterface
            )
            
            self.cognitive_engine = CognitiveEngine()
            self.memory_vault = UnifiedMemoryVault()
            self.concept_mesh = ConceptMesh()
            self.cognitive_interface = CognitiveInterface()
            
            self.logger.info("✅ Core components initialized")
        except ImportError as e:
            self.logger.warning(f"Some components unavailable: {e}")
            
    def _start_api_server(self) -> Optional[int]:
        """Start the API server with proper process management"""
        self.logger.info("🌐 Starting API server...")
        
        # Find available port
        self.api_port = self._find_available_port(8002)
        
        # Prepare environment
        env = os.environ.copy()
        env['TORI_API_PORT'] = str(self.api_port)
        env['PYTHONUNBUFFERED'] = '1'
        
        # Start API process
        cmd = [sys.executable, "-m", "prajna_atomic"]
        
        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Register with shutdown handler
            shutdown_handler.register_process(
                "api_server", 
                proc.pid, 
                is_critical=True,
                shutdown_timeout=10.0
            )
            
            # Wait for startup
            if self._wait_for_port(self.api_port, timeout=30):
                self.logger.info(f"✅ API server started on port {self.api_port}")
                return self.api_port
            else:
                self.logger.error("API server failed to start")
                proc.terminate()
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}")
            return None
            
    def _start_mcp_server(self) -> bool:
        """Start MCP server with fixed parameters"""
        self.logger.info("🧠 Starting MCP Metacognitive server...")
        
        # Find available port
        self.mcp_port = self._find_available_port(8100)
        
        # Use the fixed server script
        server_script = Path(__file__).parent / "mcp_metacognitive" / "server_fixed.py"
        if not server_script.exists():
            # Fall back to original if fixed version doesn't exist
            server_script = Path(__file__).parent / "mcp_metacognitive" / "server.py"
            
        env = os.environ.copy()
        env['SERVER_PORT'] = str(self.mcp_port)
        env['SERVER_HOST'] = '0.0.0.0'
        env['PYTHONUNBUFFERED'] = '1'
        
        cmd = [sys.executable, str(server_script)]
        
        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Register with shutdown handler
            shutdown_handler.register_process(
                "mcp_server",
                proc.pid,
                is_critical=False,
                shutdown_timeout=5.0
            )
            
            # Monitor output in thread
            threading.Thread(
                target=self._monitor_process_output,
                args=(proc, "MCP"),
                daemon=True
            ).start()
            
            # Give it time to start
            time.sleep(2)
            
            if proc.poll() is None:
                self.logger.info(f"✅ MCP server started on port {self.mcp_port}")
                return True
            else:
                self.logger.error("MCP server exited immediately")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start MCP server: {e}")
            return False
            
    def _start_frontend(self) -> bool:
        """Start the SvelteKit frontend"""
        self.logger.info("🎨 Starting frontend...")
        
        # Check if frontend exists
        frontend_dir = Path(__file__).parent / "tori_ui_svelte"
        if not frontend_dir.exists():
            self.logger.warning("Frontend directory not found, skipping")
            return False
            
        # Find available port
        self.frontend_port = self._find_available_port(5173)
        
        env = os.environ.copy()
        env['PORT'] = str(self.frontend_port)
        env['HOST'] = '0.0.0.0'
        
        # Use npm to start frontend
        cmd = ["npm", "run", "dev", "--", "--port", str(self.frontend_port), "--host", "0.0.0.0"]
        
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(frontend_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Register with shutdown handler
            shutdown_handler.register_process(
                "frontend",
                proc.pid,
                is_critical=False,
                shutdown_timeout=3.0
            )
            
            # Monitor output
            threading.Thread(
                target=self._monitor_process_output,
                args=(proc, "Frontend"),
                daemon=True
            ).start()
            
            # Wait for startup
            if self._wait_for_port(self.frontend_port, timeout=30):
                self.logger.info(f"✅ Frontend started on port {self.frontend_port}")
                
                # Open browser
                import webbrowser
                webbrowser.open(f"http://localhost:{self.frontend_port}")
                
                return True
            else:
                self.logger.error("Frontend failed to start")
                proc.terminate()
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start frontend: {e}")
            return False
            
    def _monitor_process_output(self, proc, name):
        """Monitor process output in a thread"""
        try:
            for line in proc.stdout:
                if line.strip():
                    self.logger.info(f"[{name}] {line.strip()}")
                    
            # Also check stderr
            for line in proc.stderr:
                if line.strip():
                    self.logger.warning(f"[{name}] {line.strip()}")
        except Exception as e:
            self.logger.error(f"Error monitoring {name}: {e}")
            
    def _find_available_port(self, start_port: int) -> int:
        """Find an available port starting from start_port"""
        for port in range(start_port, start_port + 100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except:
                continue
        raise RuntimeError(f"No available ports found starting from {start_port}")
        
    def _wait_for_port(self, port: int, timeout: int = 30) -> bool:
        """Wait for a port to become active"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(('localhost', port))
                    if result == 0:
                        return True
            except:
                pass
            time.sleep(0.5)
        return False
        
    def _check_health(self) -> bool:
        """Check system health"""
        try:
            # Check if API is responsive
            if self.api_port:
                response = requests.get(
                    f"http://localhost:{self.api_port}/api/health",
                    timeout=2
                )
                return response.status_code == 200
        except:
            pass
        return True  # Don't fail health check if API isn't ready


def main():
    """Main entry point with proper logging setup"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return 1
        
    # Launch TORI
    launcher = EnhancedUnifiedToriLauncher()
    return launcher.launch()


if __name__ == "__main__":
    sys.exit(main())
