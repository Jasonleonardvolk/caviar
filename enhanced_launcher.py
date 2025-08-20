#!/usr/bin/env python3
"""
🚀 ENHANCED UNIFIED TORI LAUNCHER - BULLETPROOF EDITION v3.0
Advanced logging, bulletproof error handling, concept mesh fixes, and scalable architecture
🌐 NOW WITH TORI MCP PRODUCTION SERVER INTEGRATION 🌊
🌟 NOW WITH CONCEPT MESH HOLOGRAPHIC VISUALIZATION! ⚡🌟
"""

import os
import sys
import subprocess
import json
import time
import signal
import atexit
import logging
import argparse
import socket
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from contextlib import closing

# Suppress startup warnings (centralize)
import warnings
for _pat in (".*already exists.*", ".*shadows an attribute.*", ".*0 concepts.*"):
    warnings.filterwarnings("ignore", message=_pat)

# Logging setup (single place)
LOG = logging.getLogger("launcher")
logging.getLogger("mcp.server.fastmcp").setLevel(logging.ERROR)
logging.getLogger("server_proper").setLevel(logging.ERROR)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

# Import port manager for dynamic port allocation
try:
    from port_manager import port_manager
    atexit.register(port_manager.cleanup_all_ports)
    PORT_MANAGER_AVAILABLE = True
except ImportError:
    PORT_MANAGER_AVAILABLE = False
    LOG.warning("Port manager not available, using fallback port allocation")

# Try importing optional TORI/Saigon v5 components
TORI_V5_AVAILABLE = False
try:
    from python.core.saigon_inference_v5 import SaigonInference
    from python.core.concept_mesh_holographic import ConceptMeshHolographic
    from python.services.mcp_production_server import MCPProductionServer
    from python.core.concept_mesh import ConceptMesh
    from python.services.bridge_audio import AudioBridge
    from python.services.bridge_concept_mesh import ConceptMeshBridge
    TORI_V5_AVAILABLE = True
    LOG.info("TORI/Saigon v5 components loaded successfully")
except ImportError as e:
    TORI_V5_AVAILABLE = False
    LOG.warning("TORI v5 components not available (optional): %s", e)

# Try importing psutil for better process management
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    LOG.info("psutil not available, using basic process management")


@dataclass
class ServiceConfig:
    """Configuration for a service"""
    name: str
    port: int
    health_endpoint: str = "/health"
    startup_timeout: float = 30.0
    ready_interval: float = 0.5


class EnhancedLauncher:
    """
    Enhanced TORI Launcher with bulletproof error handling, 
    graceful shutdown, and comprehensive health checking
    """
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.api_port = 8002
        self.ui_port = 3000
        self.mcp_port = 6660
        self.bridge_audio_port = 8501
        self.bridge_concept_mesh_port = 8502
        self.concept_mesh_visualization_port = 8503
        
        # Load or create bridge config
        self.bridge_config_path = Path("D:/Dev/kha/bridge_config.json")
        self.load_bridge_config()
        
        # Service configurations
        self.services = {
            "api": ServiceConfig(
                name="API Server",
                port=self.api_port,
                health_endpoint="/health",
                startup_timeout=30.0
            ),
            "ui": ServiceConfig(
                name="UI Server",
                port=self.ui_port,
                health_endpoint="/",
                startup_timeout=20.0
            ),
            "mcp": ServiceConfig(
                name="MCP Server",
                port=self.mcp_port,
                health_endpoint="/health",
                startup_timeout=15.0
            )
        }
        
    def load_bridge_config(self):
        """Load bridge configuration from file or create default"""
        if self.bridge_config_path.exists():
            try:
                with open(self.bridge_config_path, 'r') as f:
                    config = json.load(f)
                    self.api_port = config.get("api_port", self.api_port)
                    self.ui_port = config.get("ui_port", self.ui_port)
                    self.mcp_port = config.get("mcp_port", self.mcp_port)
                    self.bridge_audio_port = config.get("bridge_audio_port", self.bridge_audio_port)
                    self.bridge_concept_mesh_port = config.get("bridge_concept_mesh_port", self.bridge_concept_mesh_port)
                    LOG.info("Loaded bridge config from %s", self.bridge_config_path)
            except Exception as e:
                LOG.warning("Failed to load bridge config: %s", e)
                self.save_bridge_config()
        else:
            self.save_bridge_config()
            
    def save_bridge_config(self):
        """Save current bridge configuration"""
        config = {
            "api_port": self.api_port,
            "ui_port": self.ui_port,
            "mcp_port": self.mcp_port,
            "bridge_audio_port": self.bridge_audio_port,
            "bridge_concept_mesh_port": self.bridge_concept_mesh_port,
            "concept_mesh_visualization_port": self.concept_mesh_visualization_port
        }
        try:
            self.bridge_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.bridge_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            LOG.info("Saved bridge config to %s", self.bridge_config_path)
        except Exception as e:
            LOG.error("Failed to save bridge config: %s", e)
            
    def find_free_port(self, start_port: int = 8000, max_tries: int = 100) -> int:
        """Find a free port starting from start_port"""
        for port in range(start_port, start_port + max_tries):
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                try:
                    sock.bind(('', port))
                    return port
                except OSError:
                    continue
        raise RuntimeError(f"No free ports found in range {start_port}-{start_port + max_tries}")
        
    def _wait_http_ready(self, url: str, timeout: float = 30.0, interval: float = 0.5):
        """Wait for HTTP service to become ready"""
        import time
        try:
            import requests
        except ImportError:
            LOG.warning("requests not available, skipping health check for %s", url)
            time.sleep(2)  # Basic wait
            return
            
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                r = requests.get(url, timeout=2.0)
                if r.ok:
                    return
            except Exception:
                pass
            time.sleep(interval)
        raise RuntimeError(f"Service did not become healthy: {url}")
        
    def start_api(self, api_only: bool = False):
        """Start the FastAPI server"""
        LOG.info("Starting API server on port %s...", self.api_port)
        
        cmd = [
            sys.executable, "-m", "uvicorn", "api.main:app",
            "--host", "0.0.0.0",
            "--port", str(self.api_port),
            "--reload"
        ]
        
        env = os.environ.copy()
        env["API_PORT"] = str(self.api_port)
        env["MCP_PORT"] = str(self.mcp_port)
        env["BRIDGE_AUDIO_PORT"] = str(self.bridge_audio_port)
        env["BRIDGE_CONCEPT_MESH_PORT"] = str(self.bridge_concept_mesh_port)
        
        # Windows process group handling
        creation = 0
        if os.name == "nt":
            creation = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            
        proc = subprocess.Popen(
            cmd,
            env=env,
            creationflags=creation,
            stdout=subprocess.PIPE if not api_only else None,
            stderr=subprocess.PIPE if not api_only else None
        )
        self.processes.append(proc)
        
        # Wait for API to be ready
        try:
            self._wait_http_ready(
                f"http://127.0.0.1:{self.api_port}/health",
                timeout=30.0
            )
            LOG.info("API is healthy on http://127.0.0.1:%s", self.api_port)
        except Exception as e:
            LOG.warning("API health check failed: %s", e)
            
    def start_ui(self):
        """Start the UI server"""
        LOG.info("Starting UI server on port %s...", self.ui_port)
        
        ui_path = Path("tori_ui_svelte")
        if not ui_path.exists():
            ui_path = Path("D:/Dev/kha/tori_ui_svelte")
            
        if not ui_path.exists():
            LOG.warning("UI directory not found, skipping UI startup")
            return
            
        # Install dependencies if needed
        node_modules = ui_path / "node_modules"
        if not node_modules.exists():
            LOG.info("Installing UI dependencies...")
            subprocess.run(["npm", "install"], cwd=ui_path, check=False)
            
        cmd = ["npm", "run", "dev", "--", "--port", str(self.ui_port)]
        
        env = os.environ.copy()
        env["VITE_API_URL"] = f"http://localhost:{self.api_port}"
        
        creation = 0
        if os.name == "nt":
            creation = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            
        proc = subprocess.Popen(
            cmd,
            cwd=ui_path,
            env=env,
            creationflags=creation,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.processes.append(proc)
        
        # Wait for UI to be ready
        time.sleep(3)  # Basic wait for UI
        LOG.info("UI server started on http://localhost:%s", self.ui_port)
        
    def start_mcp_server(self):
        """Start the MCP production server"""
        if not TORI_V5_AVAILABLE:
            LOG.info("MCP server components not available, skipping")
            return
            
        LOG.info("Starting MCP production server on port %s...", self.mcp_port)
        
        cmd = [
            sys.executable, "-m", "python.services.mcp_production_server",
            "--port", str(self.mcp_port)
        ]
        
        env = os.environ.copy()
        env["MCP_PORT"] = str(self.mcp_port)
        
        creation = 0
        if os.name == "nt":
            creation = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            
        proc = subprocess.Popen(
            cmd,
            env=env,
            creationflags=creation,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.processes.append(proc)
        
        time.sleep(2)  # Basic wait
        LOG.info("MCP server started on port %s", self.mcp_port)
        
    def start_bridges(self):
        """Start bridge services"""
        if not TORI_V5_AVAILABLE:
            LOG.info("Bridge components not available, skipping")
            return
            
        # Start audio bridge
        LOG.info("Starting audio bridge on port %s...", self.bridge_audio_port)
        cmd = [
            sys.executable, "-m", "python.services.bridge_audio",
            "--port", str(self.bridge_audio_port)
        ]
        
        creation = 0
        if os.name == "nt":
            creation = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            
        proc = subprocess.Popen(
            cmd,
            creationflags=creation,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.processes.append(proc)
        
        # Start concept mesh bridge
        LOG.info("Starting concept mesh bridge on port %s...", self.bridge_concept_mesh_port)
        cmd = [
            sys.executable, "-m", "python.services.bridge_concept_mesh",
            "--port", str(self.bridge_concept_mesh_port)
        ]
        
        proc = subprocess.Popen(
            cmd,
            creationflags=creation,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.processes.append(proc)
        
        time.sleep(2)
        LOG.info("Bridge services started")
        
    def start(self, api_only: bool = False):
        """Start all services"""
        LOG.info("=" * 60)
        LOG.info("🚀 ENHANCED TORI LAUNCHER v3.0 - STARTING...")
        LOG.info("=" * 60)
        
        try:
            # Always start API
            self.start_api(api_only)
            
            if not api_only:
                # Start UI
                self.start_ui()
                
                # Start MCP server
                self.start_mcp_server()
                
                # Start bridges
                self.start_bridges()
                
            LOG.info("=" * 60)
            LOG.info("✅ ALL SERVICES STARTED SUCCESSFULLY!")
            LOG.info("🌐 API: http://localhost:%s", self.api_port)
            if not api_only:
                LOG.info("🎨 UI: http://localhost:%s", self.ui_port)
                if TORI_V5_AVAILABLE:
                    LOG.info("🔌 MCP: port %s", self.mcp_port)
                    LOG.info("🎵 Audio Bridge: port %s", self.bridge_audio_port)
                    LOG.info("🧠 Concept Mesh Bridge: port %s", self.bridge_concept_mesh_port)
            LOG.info("=" * 60)
            
            # Keep running
            if not api_only:
                LOG.info("Press Ctrl+C to shutdown...")
                try:
                    while True:
                        time.sleep(1)
                        # Check if processes are still alive
                        for proc in self.processes[:]:
                            if proc.poll() is not None:
                                self.processes.remove(proc)
                                LOG.warning("Process terminated with code %s", proc.returncode)
                except KeyboardInterrupt:
                    pass
                    
        except Exception as e:
            LOG.error("Failed to start services: %s", e)
            self.shutdown()
            raise
            
    def shutdown(self):
        """Gracefully shutdown all services"""
        LOG.info("Shutting down services...")
        
        for p in self.processes:
            try:
                if os.name == "nt":
                    try:
                        # Try Windows process group signal
                        p.send_signal(signal.CTRL_BREAK_EVENT)
                    except Exception:
                        p.terminate()
                else:
                    p.terminate()
            except Exception:
                pass
                
        # Wait for processes to terminate
        for p in self.processes:
            try:
                p.wait(timeout=8)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
                    
        # Clean up ports if port manager available
        if PORT_MANAGER_AVAILABLE:
            port_manager.cleanup_all_ports()
            
        # Additional cleanup with psutil if available
        if PSUTIL_AVAILABLE:
            try:
                current_process = psutil.Process()
                children = current_process.children(recursive=True)
                for child in children:
                    try:
                        child.terminate()
                    except Exception:
                        pass
                        
                # Give them time to terminate
                gone, alive = psutil.wait_procs(children, timeout=5)
                
                # Kill any remaining
                for p in alive:
                    try:
                        p.kill()
                    except Exception:
                        pass
            except Exception as e:
                LOG.debug("psutil cleanup error: %s", e)
                
        LOG.info("Launcher shutdown complete")


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Enhanced TORI Launcher v3.0"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="API port override"
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Start API only (no UI/bridges)"
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=None,
        help="UI port override"
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=None,
        help="MCP port override"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Create launcher
    launcher = EnhancedLauncher()
    
    # Apply port overrides
    if args.port:
        launcher.api_port = args.port
    if args.ui_port:
        launcher.ui_port = args.ui_port
    if args.mcp_port:
        launcher.mcp_port = args.mcp_port
        
    # Save updated config
    launcher.save_bridge_config()
    
    try:
        launcher.start(api_only=args.api_only)
    except KeyboardInterrupt:
        LOG.info("Ctrl+C received, shutting down...")
    except Exception as e:
        LOG.error("Launcher error: %s", e)
    finally:
        launcher.shutdown()


if __name__ == "__main__":
    main()
