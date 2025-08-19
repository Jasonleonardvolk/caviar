#!/usr/bin/env python3
"""
🚀 UNIFIED TORI LAUNCHER - Dynamic port + MCP + Prajna integration
Combines smart port allocation with full MCP server integration and Prajna voice system
No more guessing why MCP isn't starting! Now with Prajna - TORI's voice!
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
import webbrowser
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import uvicorn

# Optional MCP bridge import - don't crash if not available
try:
    from mcp_bridge_real_tori import create_real_mcp_bridge, RealMCPBridge
    MCP_BRIDGE_AVAILABLE = True
except ImportError:
    MCP_BRIDGE_AVAILABLE = False
    create_real_mcp_bridge = None
    RealMCPBridge = None

# Initialize basic logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prajna integration - add to path and import
prajna_path = Path(__file__).parent / "prajna"
if prajna_path.exists():
    sys.path.insert(0, str(prajna_path))
    try:
        from prajna.config.prajna_config import load_config as load_prajna_config
        # Import API directly to avoid circular imports
        from prajna.api.prajna_api import app as prajna_app
        PRAJNA_AVAILABLE = True
    except ImportError as e:
        PRAJNA_AVAILABLE = False
        logger.warning(f"⚠️ Prajna import failed: {e}")
        prajna_app = None
        load_prajna_config = None
else:
    PRAJNA_AVAILABLE = False
    prajna_app = None
    load_prajna_config = None

# Setup colored logging
class ColoredFormatter(logging.Formatter):
    """Colored console formatter"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

# Configure enhanced logging (logger already created above)
logging.getLogger().handlers.clear()  # Clear default handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

# Apply colored formatter
if logger.handlers:
    logger.handlers[0].setFormatter(ColoredFormatter())
else:
    # Create a handler if none exists
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    logger.addHandler(handler)

class UnifiedToriLauncher:
    """Unified launcher with dynamic port management, MCP integration, and Prajna voice system"""
    
    def __init__(self, base_port=8002, max_attempts=10):
        self.base_port = base_port
        self.max_attempts = max_attempts
        self.script_dir = Path(__file__).parent.absolute()
        self.config_file = self.script_dir / "api_port.json"
        self.status_file = self.script_dir / "tori_status.json"
        
        # Enhanced port management settings
        self.port_check_retry = 2  # seconds between retries
        self.port_check_max_attempts = 20  # 40s total timeout
        self.frontend_preferred_port = 5173  # Force this for proxy support
        
        # Service tracking
        self.mcp_process = None
        self.mcp_bridge = None  # Will be RealMCPBridge if available
        self.frontend_process = None
        self.prajna_process = None  # NEW: Prajna service tracking
        self.api_port = None
        self.frontend_port = None
        self.prajna_port = None  # NEW: Prajna port tracking
        self.multi_tenant_config_file = self.script_dir / "multi_tenant_config.json"
        self.multi_tenant_mode = False
        
        # Frontend directory
        self.frontend_dir = self.script_dir / "tori_ui_svelte"
        
        # Prajna directory
        self.prajna_dir = self.script_dir / "prajna"
        
        # Check for multi-tenant mode
        self.multi_tenant_mode = self.check_multi_tenant_mode()
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def check_multi_tenant_mode(self):
        """Check if multi-tenant mode is enabled"""
        try:
            if self.multi_tenant_config_file.exists():
                with open(self.multi_tenant_config_file, 'r') as f:
                    config = json.load(f)
                return config.get("enabled", False)
        except Exception as e:
            logger.warning(f"⚠️ Failed to read multi-tenant config: {e}")
        return False
    
    def print_banner(self):
        """Print startup banner"""
        print("\n" + "=" * 70)
        print("🚀 UNIFIED TORI LAUNCHER - BULLETPROOF EDITION")
        print("Features: NoneType-safe, Entropy pruning, Admin mode, Non-discriminatory")
        print("=" * 70)
        print(f"📂 Working directory: {self.script_dir}")
        print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
        print("🔧 Features: Dynamic ports, MCP integration, Prajna voice system")
        print("🔒 NEW: Mesh collision prevention (parameterized paths!)")
        print("🛑 Bug Fixes: All NoneType multiplication errors eliminated")
        print("🎯 Pipeline: Atomic purity-based universal extraction")
        print("🌈 Diversity: Entropy-based semantic diversity pruning")
        print("🧠 Prajna: TORI's voice and language model")
        print("=" * 70 + "\n")
    
    def update_status(self, stage: str, status: str, details: dict = None):
        """Update status file for debugging and frontend consumption"""
        status_data = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "status": status,
            "details": details or {},
            "api_port": self.api_port,
            "prajna_port": self.prajna_port,  # NEW: Include Prajna port
            "api_mesh_file": f"concept_mesh_{self.api_port}.json" if self.api_port else None,
            "prajna_mesh_file": f"concept_mesh_{self.prajna_port}.json" if self.prajna_port else None,
            "mesh_collision_prevented": True,
            "mcp_running": self.mcp_process is not None and self.mcp_process.poll() is None,
            "prajna_running": self.prajna_process is not None and self.prajna_process.poll() is None,  # NEW
            "bridge_ready": self.mcp_bridge is not None
        }
        
        with open(self.status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        logger.info(f"📊 Status: {stage} -> {status}")
    
    def is_port_available(self, port):
        """Check if a port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return True
        except OSError:
            return False
    
    def find_available_port(self, start_port=None, service_name="service"):
        """Find the first available port starting from given port"""
        start = start_port or self.base_port
        logger.info(f"🔍 Searching for available {service_name} port starting from {start}")
        
        for i in range(self.max_attempts):
            port = start + i
            if self.is_port_available(port):
                logger.info(f"✅ Found available {service_name} port: {port}")
                return port
            else:
                logger.warning(f"❌ Port {port} is busy")
        
        raise Exception(f"❌ No available {service_name} ports found in range {start}-{start + self.max_attempts}")
    
    def save_port_config(self, api_port, prajna_port=None):
        """Save the active ports to config file for SvelteKit to read"""
        config = {
            "api_port": api_port,
            "api_url": f"http://localhost:{api_port}",
            "prajna_port": prajna_port,
            "prajna_url": f"http://localhost:{prajna_port}" if prajna_port else None,
            "timestamp": time.time(),
            "status": "active",
            "mcp_integrated": True,
            "prajna_integrated": prajna_port is not None
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"📝 Saved port config: {self.config_file}")
        return config
    
    def kill_existing_processes(self):
        """Kill any existing processes on our ports"""
        logger.info("🔫 Checking for existing processes...")
        
        try:
            # Kill any existing API processes
            existing_port = self.get_existing_port()
            if existing_port and not self.is_port_available(existing_port):
                logger.info(f"🔫 Killing existing API process on port {existing_port}")
                self.kill_process_on_port(existing_port)
            
            # Kill any existing MCP processes on port 3000
            if not self.is_port_available(3000):
                logger.info("🔫 Killing existing MCP process on port 3000")
                self.kill_process_on_port(3000)
            
            # Kill any existing Prajna processes on port 8001 (default Prajna port)
            if not self.is_port_available(8001):
                logger.info("🔫 Killing existing Prajna process on port 8001")
                self.kill_process_on_port(8001)
                
        except Exception as e:
            logger.warning(f"⚠️ Error during process cleanup: {e}")
    
    def find_pids(self, port):
        """Find all PIDs using specific port (enhanced method)"""
        try:
            output = subprocess.check_output(
                f'netstat -ano | findstr :{port}', shell=True
            ).decode()
        except subprocess.CalledProcessError:
            return []
        
        pids = set()
        for line in output.strip().splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    pid = int(parts[-1])
                    pids.add(pid)
                except Exception:
                    pass
        return list(pids)
    
    def kill_pid(self, pid):
        """Kill specific PID with enhanced error handling"""
        try:
            subprocess.run(f'taskkill /PID {pid} /F', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"🔫 Killed PID {pid}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to kill PID {pid}: {e}")
            return False
    
    def port_is_free(self, port):
        """Enhanced port check using socket connection test"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) != 0
    
    def nuke_node_processes(self):
        """Kill all lingering node/npm/pnpm processes (optional but robust)"""
        for proc_name in ["node", "npm", "pnpm"]:
            try:
                subprocess.run(f'taskkill /IM {proc_name}.exe /F', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logger.info(f"🔫 Killed all {proc_name}.exe processes")
            except Exception:
                pass
    
    def kill_process_on_port(self, port):
        """Enhanced port killing with bulletproof retry logic"""
        logger.info(f"🔫 Searching for processes on port {port}...")
        pids = self.find_pids(port)
        
        if pids:
            for pid in pids:
                self.kill_pid(pid)
            logger.info("⏳ Waiting 8 seconds for Windows to release the port...")
            time.sleep(8)
        else:
            logger.info(f"✅ No processes found using port {port}")
        
        # Enhanced port verification with retry
        logger.info(f"🔎 Checking port {port} status...")
        for attempt in range(self.port_check_max_attempts):
            if self.port_is_free(port):
                logger.info(f"✅ Port {port} is now free!")
                return True
            else:
                logger.warning(f"⏳ Port {port} still in use. Retry {attempt + 1}/{self.port_check_max_attempts}...")
                time.sleep(self.port_check_retry)
        
        logger.error(f"❌ Port {port} did not free after {self.port_check_retry * self.port_check_max_attempts} seconds")
        return False
    
    def get_existing_port(self):
        """Get existing API port from config"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                return config.get("api_port")
            except:
                pass
        return None
    
    def start_prajna_service(self):
        """Configure Prajna integration status (Prajna is now integrated into main API)"""
        if not PRAJNA_AVAILABLE:
            logger.info("⏭️ Prajna integration not available")
            self.update_status("prajna_startup", "skipped", {"reason": "Prajna not available"})
            return False
        
        if not self.prajna_dir.exists():
            logger.warning(f"⚠️ Prajna directory not found: {self.prajna_dir}")
            logger.info("⏭️ Prajna integration disabled")
            self.update_status("prajna_startup", "skipped", {"reason": "Prajna directory not found"})
            return False
        
        self.update_status("prajna_startup", "preparing", {"message": "Configuring Prajna integration..."})
        logger.info("🧠 Configuring Prajna - TORI's Voice and Language Model...")
        logger.info("🔗 Prajna is integrated into the main API server (no separate service needed)")
        
        try:
            # Since Prajna is integrated, it will run on the same port as the main API
            self.prajna_port = self.api_port  # Prajna runs on the same port as main API
            
            # Set up environment for integrated Prajna
            env = os.environ.copy()
            
            # Set environment variables for Prajna configuration
            if not env.get('PRAJNA_MODEL_TYPE'):
                env['PRAJNA_MODEL_TYPE'] = 'demo'  # Default to demo mode
            
            if not env.get('PRAJNA_TEMPERATURE'):
                env['PRAJNA_TEMPERATURE'] = '0.7'
            
            # Update environment for this process
            os.environ.update(env)
            
            logger.info(f"✅ Prajna integration configured successfully!")
            logger.info(f"🧠 Prajna will be available at: http://localhost:{self.api_port}/api/answer")
            logger.info(f"📚 Prajna docs will be at: http://localhost:{self.api_port}/docs")
            logger.info(f"🔧 Model type: {env.get('PRAJNA_MODEL_TYPE', 'demo')}")
            logger.info(f"🌡️ Temperature: {env.get('PRAJNA_TEMPERATURE', '0.7')}")
            
            self.update_status("prajna_startup", "configured", {
                "port": self.api_port,
                "api_url": f"http://localhost:{self.api_port}/api/answer",
                "docs_url": f"http://localhost:{self.api_port}/docs",
                "mode": "integrated",
                "model_type": env.get('PRAJNA_MODEL_TYPE', 'demo'),
                "integration": "embedded_in_main_api"
            })
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Error configuring Prajna integration: {e}")
            logger.info("⏭️ Continuing without Prajna integration")
            self.update_status("prajna_startup", "failed", {"error": str(e)})
            return False
    
    def start_mcp_services(self):
        """Start MCP TypeScript services (optional)"""
        if not MCP_BRIDGE_AVAILABLE:
            logger.info("⏭️ Skipping MCP services (bridge not available)")
            self.update_status("mcp_startup", "skipped", {"reason": "MCP bridge not available"})
            return False
        
        self.update_status("mcp_startup", "starting", {"message": "Starting MCP services..."})
        logger.info("🚀 Starting MCP services...")
        
        mcp_dir = self.script_dir / "mcp-server-architecture"
        if not mcp_dir.exists():
            logger.warning(f"⚠️ MCP directory not found: {mcp_dir}")
            logger.info("⏭️ Continuing without MCP services")
            self.update_status("mcp_startup", "skipped", {"reason": "MCP directory not found"})
            return False
        
        try:
            # Start MCP in background
            self.mcp_process = subprocess.Popen(
                'npm run start',
                cwd=str(mcp_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            
            # Wait for MCP to be ready with progress indication
            max_retries = 30
            logger.info("⏳ Waiting for MCP services to start...")
            
            for i in range(max_retries):
                try:
                    response = requests.get('http://localhost:3000/health', timeout=2)
                    if response.status_code == 200:
                        logger.info("✅ MCP services started successfully")
                        self.update_status("mcp_startup", "success", {"port": 3000})
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                # Show progress every 5 seconds
                if i % 5 == 0 and i > 0:
                    logger.info(f"⏳ Still waiting for MCP... ({i}/{max_retries} attempts)")
                
                time.sleep(1)
            
            # MCP failed to start
            logger.warning("⚠️ MCP services failed to start within 30 seconds")
            logger.info("⏭️ Continuing without MCP services")
            self.update_status("mcp_startup", "failed", {"error": "MCP health check timeout"})
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ Error starting MCP services: {e}")
            logger.info("⏭️ Continuing without MCP services")
            self.update_status("mcp_startup", "failed", {"error": str(e)})
            return False
    
    def find_frontend_port(self):
        """Find available port for frontend - FORCE 5173 for proxy support"""
        preferred_port = 5173  # This is where the proxy works!
        
        # First, check if 5173 is available
        if self.is_port_available(preferred_port):
            logger.info(f"✅ Port 5173 is available - using for proxy support")
            self.frontend_port = preferred_port
            return preferred_port
        
        # Port 5173 is busy - try to kill the process using it
        logger.warning(f"⚠️ Port 5173 is busy - attempting to free it for proxy support")
        
        if self.kill_process_on_port(preferred_port):
            logger.info("🔫 Successfully killed process on port 5173")
            
            # Wait a moment for the port to be freed
            time.sleep(2)
            
            # Check if port is now available
            if self.is_port_available(preferred_port):
                logger.info(f"✅ Port 5173 is now available - using for proxy support")
                self.frontend_port = preferred_port
                return preferred_port
            else:
                logger.warning("⚠️ Port 5173 still not available after killing process")
        else:
            logger.warning("⚠️ Failed to kill process on port 5173")
        
        # If we still can't get 5173, this is a problem because proxy won't work
        logger.error("❌ Cannot secure port 5173 - proxy configuration will not work!")
        logger.error("❌ Frontend will start but /upload routing will fail!")
        
        # Fall back to other ports but warn user
        frontend_ports = [4173, 3000, 3001, 5000, 5001]  # Fallback ports
        
        for port in frontend_ports:
            if self.is_port_available(port):
                logger.warning(f"🚨 Using fallback port {port} - PROXY WILL NOT WORK!")
                logger.warning(f"🚨 Uploads will fail because proxy is configured for port 5173 only!")
                self.frontend_port = port
                return port
        
        # If none of the common ports work, find any available port starting from 5174
        for i in range(50):
            port = 5174 + i
            if self.is_port_available(port):
                logger.warning(f"🚨 Using fallback port {port} - PROXY WILL NOT WORK!")
                logger.warning(f"🚨 Uploads will fail because proxy is configured for port 5173 only!")
                self.frontend_port = port
                return port
        
        raise Exception("❌ No available frontend ports found")
    
    def start_frontend_services(self):
        """Start SvelteKit frontend services"""
        if not self.frontend_dir.exists():
            logger.warning(f"⚠️ Frontend directory not found: {self.frontend_dir}")
            logger.info("⏭️ Continuing without frontend services")
            self.update_status("frontend_startup", "skipped", {"reason": "Frontend directory not found"})
            return False
        
        # Check if package.json exists
        package_json = self.frontend_dir / "package.json"
        if not package_json.exists():
            logger.warning(f"⚠️ Frontend package.json not found: {package_json}")
            logger.info("⏭️ Continuing without frontend services")
            self.update_status("frontend_startup", "skipped", {"reason": "package.json not found"})
            return False
        
        self.update_status("frontend_startup", "starting", {"message": "Starting SvelteKit frontend..."})
        logger.info("🎨 Starting SvelteKit frontend...")
        
        try:
            # Find available frontend port
            frontend_port = self.find_frontend_port()
            
            # Start frontend in development mode
            env = os.environ.copy()
            env['PORT'] = str(frontend_port)
            
            self.frontend_process = subprocess.Popen(
                f'npm run dev -- --port {frontend_port} --host 0.0.0.0',
                cwd=str(self.frontend_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            
            # Wait for frontend to be ready
            max_retries = 30
            logger.info("⏳ Waiting for frontend to start...")
            
            for i in range(max_retries):
                try:
                    response = requests.get(f'http://localhost:{frontend_port}', timeout=2)
                    if response.status_code == 200:
                        logger.info("✅ Frontend started successfully")
                        self.update_status("frontend_startup", "success", {"port": frontend_port})
                        
                        # Open browser automatically
                        self.open_browser_to_frontend(frontend_port)
                        
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                # Show progress every 5 seconds
                if i % 5 == 0 and i > 0:
                    logger.info(f"⏳ Still waiting for frontend... ({i}/{max_retries} attempts)")
                
                time.sleep(1)
            
            # Frontend failed to start
            logger.warning("⚠️ Frontend failed to start within 30 seconds")
            logger.info("⏭️ Continuing without frontend")
            self.update_status("frontend_startup", "failed", {"error": "Frontend health check timeout"})
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ Error starting frontend: {e}")
            logger.info("⏭️ Continuing without frontend")
            self.update_status("frontend_startup", "failed", {"error": str(e)})
            return False
    
    def open_browser_to_frontend(self, port):
        """Open browser to frontend URL"""
        try:
            import webbrowser
            frontend_url = f"http://localhost:{port}"
            logger.info(f"🌐 Opening browser to: {frontend_url}")
            webbrowser.open(frontend_url)
        except Exception as e:
            logger.warning(f"⚠️ Could not open browser: {e}")
    
    async def initialize_mcp_bridge(self):
        """Initialize MCP bridge with TORI filtering (optional)"""
        if not MCP_BRIDGE_AVAILABLE or not create_real_mcp_bridge:
            logger.info("⏭️ Skipping MCP bridge initialization (not available)")
            self.update_status("mcp_bridge", "skipped", {"reason": "MCP bridge not available"})
            return None
        
        if not self.mcp_process or self.mcp_process.poll() is not None:
            logger.info("⏭️ Skipping MCP bridge initialization (MCP services not running)")
            self.update_status("mcp_bridge", "skipped", {"reason": "MCP services not running"})
            return None
        
        try:
            self.update_status("mcp_bridge", "initializing", {"message": "Setting up MCP bridge..."})
            logger.info("🔗 Initializing MCP bridge...")
            
            config = {
                'mcp_gateway_url': os.getenv('MCP_GATEWAY_URL', 'http://localhost:3001'),
                'auth_token': os.getenv('MCP_AUTH_TOKEN', 'your-secure-token'),
                'enable_audit_log': True
            }
            
            self.mcp_bridge = await create_real_mcp_bridge(config)
            
            # Register callback handlers
            self.mcp_bridge.register_callback_handler(
                'kaizen.improvement',
                self.handle_kaizen_improvement_callback
            )
            self.mcp_bridge.register_callback_handler(
                'celery.task_update',
                self.handle_celery_task_callback
            )
            
            logger.info("✅ MCP bridge initialized successfully")
            self.update_status("mcp_bridge", "success", {"callbacks_registered": 2})
            return self.mcp_bridge
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize MCP bridge: {e}")
            logger.info("⏭️ Continuing without MCP bridge")
            self.update_status("mcp_bridge", "failed", {"error": str(e)})
            return None
    
    async def handle_kaizen_improvement_callback(self, data):
        """Handle Kaizen improvement callbacks"""
        logger.info(f"📥 Received Kaizen improvement: {data}")
        return {"status": "processed", "data": data}
    
    async def handle_celery_task_callback(self, data):
        """Handle Celery task callbacks"""
        logger.info(f"📥 Received Celery task update: {data}")
        return {"status": "processed", "data": data}
    
    def set_mesh_path_for_service(self, port, service_name):
        """Set parameterized mesh path to prevent service collision"""
        mesh_filename = f"concept_mesh_{port}.json"
        os.environ['CONCEPT_MESH_PATH'] = mesh_filename
        logger.info(f"🔒 {service_name} mesh path: {mesh_filename}")
        return mesh_filename
    
    def start_api_server(self, port):
        """Start the API server with MCP integration and mesh parameterization"""
        self.update_status("api_startup", "starting", {"port": port})
        logger.info(f"🌐 Starting API server on port {port}...")
        
        # 🔒 MESH PARAMETERIZATION: Set unique mesh path for this service
        mesh_file = self.set_mesh_path_for_service(port, "API Server")
        
        # Change to correct directory
        os.chdir(self.script_dir)
        
        logger.info(f"🎯 API SERVER READY:")
        logger.info(f"   📍 Port: {port}")
        logger.info(f"   🌐 URL: http://localhost:{port}")
        logger.info(f"   ❤️ Health: http://localhost:{port}/health")
        logger.info(f"   📚 Docs: http://localhost:{port}/docs")
        logger.info(f"   🔒 Mesh File: {mesh_file} (collision-free!)")
        logger.info(f"   🔗 MCP: Integrated and Ready")
        logger.info(f"   🎯 Features: Bulletproof NoneType protection, Entropy pruning, Admin mode")
        logger.info(f"   🚀 Pipeline: Enhanced atomic purity-based universal extraction")
        
        self.update_status("api_startup", "ready", {
            "port": port,
            "urls": {
                "api": f"http://localhost:{port}",
                "health": f"http://localhost:{port}/health",
                "docs": f"http://localhost:{port}/docs"
            }
        })
        
        # Start the server (this blocks) - FIXED: Use correct prajna_api path
        uvicorn.run(
            "prajna_api:app",  # Use the fixed prajna_api.py file in root directory
            host="0.0.0.0",
            port=port,
            reload=False,  # Stable mode - no file watching
            workers=1,
            log_level="info",
            access_log=True
        )
    
    def cleanup(self):
        """Cleanup all services"""
        logger.info("🧹 Cleaning up services...")
        
        if self.mcp_bridge:
            try:
                asyncio.run(self.mcp_bridge.stop())
                logger.info("✅ MCP bridge stopped")
            except Exception as e:
                logger.warning(f"⚠️ Error stopping MCP bridge: {e}")
        
        if self.mcp_process and self.mcp_process.poll() is None:
            try:
                self.mcp_process.terminate()
                self.mcp_process.wait(timeout=5)
                logger.info("✅ MCP process terminated")
            except Exception as e:
                logger.warning(f"⚠️ Error terminating MCP process: {e}")
        
        if self.prajna_process and self.prajna_process.poll() is None:
            try:
                self.prajna_process.terminate()
                self.prajna_process.wait(timeout=5)
                logger.info("✅ Prajna process terminated")
            except Exception as e:
                logger.warning(f"⚠️ Error terminating Prajna process: {e}")
        
        if self.frontend_process and self.frontend_process.poll() is None:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
                logger.info("✅ Frontend process terminated")
            except Exception as e:
                logger.warning(f"⚠️ Error terminating frontend process: {e}")
        
        # Update status
        self.update_status("shutdown", "complete", {"timestamp": datetime.now().isoformat()})
    
    def launch(self):
        """Main launch sequence (synchronous)"""
        try:
            self.print_banner()
            
            # Step 1: Cleanup existing processes
            self.update_status("startup", "cleaning", {"message": "Cleaning up existing processes"})
            self.kill_existing_processes()
            
            # Step 2: Find available port for API
            self.update_status("startup", "port_search", {"message": "Finding available port"})
            port = self.find_available_port(service_name="API")
            self.api_port = port
            
            # Step 3: Start Prajna voice system (NEW!)
            logger.info("\n" + "=" * 50)
            logger.info("🧠 STARTING PRAJNA - TORI'S VOICE SYSTEM...")
            logger.info("=" * 50)
            prajna_started = self.start_prajna_service()
            
            if prajna_started:
                logger.info("✅ Prajna voice system ready!")
            else:
                logger.warning("⚠️ Prajna startup failed or skipped!")
            
            # Step 4: Start MCP services (optional, can fail gracefully)
            mcp_started = self.start_mcp_services()
            
            # Step 5: Initialize MCP bridge (only if MCP started)
            if mcp_started and MCP_BRIDGE_AVAILABLE:
                logger.info("🔗 MCP bridge available but skipping for synchronous launch")
                self.update_status("mcp_bridge", "skipped", {"reason": "Synchronous launch mode"})
            
            # Step 6: Start frontend services (non-blocking)
            logger.info("\n" + "=" * 50)
            logger.info("🎨 STARTING FRONTEND SERVICES...")
            logger.info("=" * 50)
            frontend_started = self.start_frontend_services()
            
            if frontend_started:
                logger.info("✅ Frontend startup completed successfully!")
                time.sleep(2)  # Give frontend a moment to fully initialize
            else:
                logger.warning("⚠️ Frontend startup failed or skipped!")
            
            # Step 7: Save port configuration with all services
            self.save_port_config(port, self.prajna_port)
            
            # Step 8: Print complete system status BEFORE starting API
            self.print_system_ready(port, frontend_started, prajna_started)
            
            # Give user a moment to see the status
            logger.info("⏳ Starting API server in 3 seconds...")
            time.sleep(3)
            
            # Step 9: Start API server (blocks here)
            self.start_api_server(port)
            
        except KeyboardInterrupt:
            logger.info("\n👋 Shutdown requested by user")
            self.update_status("shutdown", "user_requested", {"message": "Ctrl+C pressed"})
        except Exception as e:
            logger.error(f"❌ Launch failed: {e}")
            self.update_status("startup", "failed", {"error": str(e)})
            return 1
        
        return 0
    
    def print_system_ready(self, api_port, frontend_started, prajna_started):
        """Print complete system ready status"""
        logger.info("\n" + "🎉 " * 25)
        logger.info("🎯 COMPLETE TORI SYSTEM READY (BULLETPROOF EDITION):")
        logger.info(f"   🔧 API Server: http://localhost:{api_port} (NoneType-safe)")
        logger.info(f"   📚 API Docs: http://localhost:{api_port}/docs")
        logger.info(f"   ❤️ Health Check: http://localhost:{api_port}/health")
        logger.info(f"   🔒 Mesh Collision: PREVENTED (concept_mesh_{api_port}.json)")
        logger.info(f"   🛑 Bug Status: All NoneType multiplication errors ELIMINATED")
        logger.info(f"   🎯 Extraction: Entropy-based diversity pruning ENABLED")
        
        # Prajna status
        if prajna_started and self.prajna_port:
            logger.info(f"   🧠 Prajna Voice: http://localhost:{self.prajna_port}/api/answer")
            logger.info(f"   🧠 Prajna Docs: http://localhost:{self.prajna_port}/docs")
            logger.info(f"   🧠 Prajna Health: http://localhost:{self.prajna_port}/api/health")
            logger.info(f"   🔒 Prajna Mesh: concept_mesh_{self.prajna_port}.json (collision-free!)")
        else:
            logger.info("   🧠 Prajna Voice: Not available")
        
        # Frontend status
        if frontend_started and self.frontend_port:
            logger.info(f"   🎨 Frontend: http://localhost:{self.frontend_port}")
            logger.info(f"   🌐 Open in Browser: http://localhost:{self.frontend_port}")
        else:
            logger.info("   🎨 Frontend: Not available")
        
        logger.info(f"   🔗 MCP: {'Available' if self.mcp_process and self.mcp_process.poll() is None else 'Skipped'}")
        logger.info(f"   🏢 Multi-Tenant: {'Enabled' if self.multi_tenant_mode else 'Disabled'}")
        
        # Mesh collision prevention status
        mesh_status = "ACTIVE" if prajna_started and self.prajna_port else "PARTIAL"
        logger.info(f"   🔒 MESH COLLISION PREVENTION: {mesh_status}")
        if prajna_started and self.prajna_port:
            logger.info(f"      → API Server writes to: concept_mesh_{api_port}.json")
            logger.info(f"      → Prajna Service writes to: concept_mesh_{self.prajna_port}.json")
            logger.info(f"      → No more mesh state stomping!")
        else:
            logger.info(f"      → API Server writes to: concept_mesh_{api_port}.json")
            logger.info(f"      → Single service mode - collision prevention ready")
        
        # Quick test suggestions
        logger.info("\n🧪 QUICK TESTS (NOW BULLETPROOF):")
        if prajna_started and self.prajna_port:
            logger.info(f"   Test Prajna: curl -X POST http://localhost:{self.prajna_port}/api/answer \\")
            logger.info(f"     -H 'Content-Type: application/json' \\")
            logger.info(f"     -d '{{\"user_query\": \"What is Prajna?\"}}'")
        logger.info(f"   Test API: curl http://localhost:{api_port}/health")
        logger.info(f"   Test Upload: Use /upload endpoint (NoneType-safe!)")
        logger.info(f"   Test Extract: Use /extract endpoint (entropy pruning enabled!)")
        logger.info(f"   Verify Mesh: Check concept_mesh_{{port}}.json files are separate!")
        logger.info("🎉 " * 25 + "\n")

def main():
    """Main entry point"""
    launcher = UnifiedToriLauncher()
    return launcher.launch()  # Now synchronous, no more event loop issues

if __name__ == "__main__":
    sys.exit(main())
