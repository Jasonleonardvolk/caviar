#!/usr/bin/env python3
"""
Enhanced launcher with graceful shutdown support
"""

import os
import sys
import signal
import asyncio
import subprocess
import threading
import time
import logging
from pathlib import Path

# Import our graceful shutdown manager
sys.path.insert(0, str(Path(__file__).parent))
from core.graceful_shutdown import shutdown_manager, register_shutdown_handler, install_shutdown_handlers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tori.launcher')

class TORILauncher:
    def __init__(self):
        self.processes = {}
        self.threads = {}
        self.services = {}
        self.running = True
        
    def start_api_server(self):
        """Start the API server"""
        logger.info("Starting API server...")
        # Import and start Prajna API
        try:
            from prajna.api.prajna_api import app as prajna_app
            import uvicorn
            
            # Create shutdown handler for API
            async def api_shutdown():
                logger.info("Shutting down API server...")
                # Uvicorn will handle the shutdown
                
            register_shutdown_handler(api_shutdown, "API Server")
            
            # Run in thread
            def run_api():
                uvicorn.run(
                    prajna_app,
                    host="0.0.0.0",
                    port=8002,
                    log_level="info"
                )
            
            api_thread = threading.Thread(target=run_api, daemon=True)
            api_thread.start()
            self.threads['api'] = api_thread
            logger.info("✅ API server started on port 8002")
            
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            
    def start_lattice_runner(self):
        """Start the lattice evolution runner"""
        logger.info("Starting lattice evolution runner...")
        try:
            from python.core.lattice_evolution_runner import run_forever, request_shutdown
            
            # Register shutdown handler
            register_shutdown_handler(request_shutdown, "Lattice Evolution")
            
            # Run in asyncio thread
            def run_lattice():
                asyncio.run(run_forever())
                
            lattice_thread = threading.Thread(target=run_lattice, daemon=True)
            lattice_thread.start()
            self.threads['lattice'] = lattice_thread
            logger.info("✅ Lattice evolution runner started")
            
        except Exception as e:
            logger.error(f"Failed to start lattice runner: {e}")
            
    def start_mcp_server(self):
        """Start the MCP metacognitive server"""
        logger.info("Starting MCP server...")
        try:
            cmd = [
                sys.executable,
                "-m", "mcp_metacognitive.server"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            self.processes['mcp'] = process
            
            # Register shutdown handler
            def mcp_shutdown():
                logger.info("Shutting down MCP server...")
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        
            register_shutdown_handler(mcp_shutdown, "MCP Server")
            logger.info("✅ MCP server started")
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            
    def start_frontend(self):
        """Start the frontend server"""
        logger.info("Starting frontend...")
        try:
            frontend_dir = Path(__file__).parent / "tori_ui_svelte"
            
            cmd = ["npm", "run", "dev", "--", "--host", "--port", "5173"]
            
            process = subprocess.Popen(
                cmd,
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            self.processes['frontend'] = process
            
            # Register shutdown handler
            def frontend_shutdown():
                logger.info("Shutting down frontend...")
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        
            register_shutdown_handler(frontend_shutdown, "Frontend")
            logger.info("✅ Frontend started on port 5173")
            
        except Exception as e:
            logger.error(f"Failed to start frontend: {e}")
            
    def monitor_services(self):
        """Monitor running services"""
        while self.running:
            # Check if any critical service has died
            for name, process in self.processes.items():
                if process.poll() is not None:
                    logger.warning(f"Service {name} has stopped unexpectedly!")
                    
            time.sleep(5)  # Check every 5 seconds
            
    def shutdown(self):
        """Shutdown all services"""
        logger.info("Launcher shutdown initiated...")
        self.running = False

def main():
    """Main launcher function"""
    logger.info("🚀 TORI Enhanced Launcher starting...")
    
    # Create launcher instance
    launcher = TORILauncher()
    
    # Register launcher shutdown
    register_shutdown_handler(launcher.shutdown, "TORI Launcher")
    
    # Install signal handlers
    install_shutdown_handlers()
    
    # Start all services
    launcher.start_api_server()
    time.sleep(2)  # Give API time to start
    
    launcher.start_lattice_runner()
    launcher.start_mcp_server()
    launcher.start_frontend()
    
    logger.info("✅ All TORI services started!")
    logger.info("Press Ctrl+C to shutdown gracefully...")
    
    try:
        # Monitor services
        launcher.monitor_services()
    except KeyboardInterrupt:
        # This should not happen as signal handler intercepts it
        pass
    
    logger.info("TORI Launcher exiting...")

if __name__ == "__main__":
    main()
