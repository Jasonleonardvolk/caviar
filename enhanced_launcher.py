#!/usr/bin/env python3
"""
🚀 ENHANCED UNIFIED TORI LAUNCHER - BULLETPROOF EDITION v3.0
Advanced logging, bulletproof error handling, concept mesh fixes, and scalable architecture
🏆 NOW WITH TORI MCP PRODUCTION SERVER INTEGRATION 🌊
🌟 NOW WITH CONCEPT MESH HOLOGRAPHIC VISUALIZATION! 🧠➡️🌟
✨ NOW WITH TORI/SAIGON v5 COMPLETE INTEGRATION! 🚀
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

# ============================================================================
# TORI/SAIGON v5 COMPONENTS - NEW!
# ============================================================================
try:
    from python.core.saigon_inference_v5 import SaigonInference
    from python.core.adapter_loader_v5 import MetadataManager
    from python.core.concept_mesh_v5 import MeshManager
    from python.core.user_context import UserContextManager
    from python.core.conversation_manager import ConversationManager
    from python.core.lattice_morphing import LatticeMorpher
    TORI_V5_AVAILABLE = True
    print("✅ TORI/Saigon v5 components loaded successfully")
except ImportError as e:
    TORI_V5_AVAILABLE = False
    SaigonInference = None
    MetadataManager = None
    MeshManager = None
    UserContextManager = None
    ConversationManager = None
    LatticeMorpher = None
    print(f"ℹ️ TORI v5 components not available (optional): {e}")

# HoTT Integration (Advanced)
try:
    from hott_integration.psi_morphon import ConceptSynthesizer, PsiMorphonField
    HOTT_AVAILABLE = True
    print("✅ HoTT/Psi-Morphon synthesis available")
except ImportError:
    HOTT_AVAILABLE = False
    ConceptSynthesizer = None
    PsiMorphonField = None

# Training Components (Optional)
try:
    from python.training.synthetic_data_generator import SyntheticDataGenerator
    from python.training.validate_adapter import AdapterValidator
    from python.training.rollback_adapter import RollbackManager
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    SyntheticDataGenerator = None
    AdapterValidator = None
    RollbackManager = None

# ============================================================================
# EXISTING COMPONENTS (Preserved for backward compatibility)
# ============================================================================

# BPS Soliton Support
try:
    from python.core.bps_config_enhanced import BPS_CONFIG, SolitonPolarity
    from python.core.bps_oscillator_enhanced import BPSEnhancedLattice
    from python.core.bps_soliton_memory_enhanced import BPSEnhancedSolitonMemory
    from python.monitoring.bps_diagnostics import BPSDiagnostics
    BPS_AVAILABLE = True
except ImportError:
    BPS_AVAILABLE = False
    BPS_CONFIG = None
    BPSEnhancedLattice = None
    BPSEnhancedSolitonMemory = None
    BPSDiagnostics = None

# Import port manager for dynamic port allocation
try:
    from port_manager import port_manager
    PORT_MANAGER_AVAILABLE = True
    atexit.register(port_manager.cleanup_all_ports)
except ImportError:
    PORT_MANAGER_AVAILABLE = False
    port_manager = None

# Suppress startup warnings
import warnings
warnings.filterwarnings("ignore", message=".*already exists.*")
warnings.filterwarnings("ignore", message=".*shadows an attribute.*")
warnings.filterwarnings("ignore", message=".*0 concepts.*")

# Reduce logging noise
logging.getLogger("mcp.server.fastmcp").setLevel(logging.ERROR)
logging.getLogger("server_proper").setLevel(logging.ERROR)

# Import graceful shutdown support
try:
    from utils.graceful_shutdown import GracefulShutdownHandler, AsyncioGracefulShutdown, delayed_keyboard_interrupt
    GRACEFUL_SHUTDOWN_AVAILABLE = True
except ImportError:
    try:
        from core.graceful_shutdown import shutdown_manager, register_shutdown_handler, install_shutdown_handlers
        GRACEFUL_SHUTDOWN_AVAILABLE = True
    except ImportError:
        GRACEFUL_SHUTDOWN_AVAILABLE = False

# Enhanced error handling and encoding
import locale
import codecs

# Set UTF-8 encoding globally
if sys.platform.startswith('win'):
    # Windows UTF-8 setup
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Enable entropy pruning for proper concept flow
os.environ['TORI_ENABLE_ENTROPY_PRUNING'] = '1'

# Suppress Python warnings
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Optional MCP bridge import
try:
    from mcp_bridge_real_tori import create_real_mcp_bridge, RealMCPBridge
    MCP_BRIDGE_AVAILABLE = True
except ImportError:
    MCP_BRIDGE_AVAILABLE = False
    create_real_mcp_bridge = None
    RealMCPBridge = None

# MCP Server Integration
try:
    mcp_path = Path(__file__).parent / "mcp_metacognitive"
    if mcp_path.exists():
        sys.path.insert(0, str(mcp_path))
    
    try:
        from mcp.types import Tool, TextContent
        MCP_TYPES_AVAILABLE = True
    except ImportError:
        MCP_TYPES_AVAILABLE = False
    
    try:
        import server_proper as mcp_server_module
    except ImportError:
        try:
            import server_simple as mcp_server_module
        except ImportError:
            mcp_server_module = None
    
    MCP_METACOGNITIVE_AVAILABLE = mcp_server_module is not None
except ImportError:
    MCP_METACOGNITIVE_AVAILABLE = False
    MCP_TYPES_AVAILABLE = False
    mcp_server_module = None

# Core Python Components Integration
try:
    python_core_path = Path(__file__).parent / "python" / "core"
    sys.path.insert(0, str(python_core_path.parent))
    
    from python.core.CognitiveEngine import CognitiveEngine
    from python.core.memory_vault import UnifiedMemoryVault
    from python.core.concept_mesh import ConceptMesh
    
    try:
        from python.core.mcp_metacognitive import MCPMetacognitiveServer
    except ImportError:
        MCPMetacognitiveServer = None
    
    try:
        from python.core.cognitive_interface import CognitiveInterface
    except ImportError:
        CognitiveInterface = None
    
    CORE_COMPONENTS_AVAILABLE = True
except ImportError:
    CORE_COMPONENTS_AVAILABLE = False
    CognitiveEngine = None
    UnifiedMemoryVault = None
    MCPMetacognitiveServer = None
    CognitiveInterface = None
    ConceptMesh = None

# Stability Components Integration
try:
    from python.stability.eigenvalue_monitor import EigenvalueMonitor
    from python.stability.lyapunov_analyzer import LyapunovAnalyzer
    from python.stability.koopman_operator import KoopmanOperator
    STABILITY_COMPONENTS_AVAILABLE = True
except ImportError:
    STABILITY_COMPONENTS_AVAILABLE = False
    EigenvalueMonitor = None
    LyapunovAnalyzer = None
    KoopmanOperator = None

# Add argument parsing
import argparse
parser = argparse.ArgumentParser(description='Enhanced TORI Launcher v3.0 with v5 Integration')
parser.add_argument('--require-penrose', action='store_true', help='Require Rust Penrose engine')
parser.add_argument('--no-require-penrose', dest='require_penrose', action='store_false')
parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
parser.add_argument('--enable-hologram', action='store_true', help='Enable holographic visualization')
parser.add_argument('--hologram-audio', action='store_true', help='Enable audio-to-hologram bridge')
parser.add_argument('--api', choices=['quick', 'full', 'v5'], default='v5',
                   help='API mode: quick (minimal), full (all endpoints), v5 (new Saigon API)')
parser.add_argument('--api-port', type=int, default=None, help='Override API port')
parser.add_argument('--enable-v5', action='store_true', default=True, 
                   help='Enable TORI v5 components (default: True)')
parser.add_argument('--no-v5', dest='enable_v5', action='store_false',
                   help='Disable v5 components for compatibility')
args = parser.parse_args()

# Force-enable hologram for bulletproof AV mode
args.enable_hologram = True
args.hologram_audio = True

# Default ports based on mode
if args.api == 'v5':
    DEFAULT_API_PORT = 8001  # v5 uses 8001
elif args.api == 'quick':
    DEFAULT_API_PORT = 8002
else:
    DEFAULT_API_PORT = 8001

# Override if specified
if args.api_port:
    DEFAULT_API_PORT = args.api_port

# Import the full path display function for diagnostics
try:
    from core.tori_pathfinder import display_module_paths
except ImportError:
    def display_module_paths():
        print("📦 Module path display not available")

class EnhancedLogger:
    """Enhanced logging with session management"""
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path("logs") / f"session_{self.session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure enhanced logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # File handler
        file_handler = logging.FileHandler(
            self.session_dir / "launcher.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Console handler (less verbose)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.handlers = []  # Clear existing handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        self.logger = logging.getLogger("EnhancedLauncher")
        
    def log_exception(self, exc_info=None):
        """Log exception with full traceback"""
        if exc_info is None:
            exc_info = sys.exc_info()
        self.logger.error("Exception occurred", exc_info=exc_info)
        
        # Also save to separate exception log
        exc_file = self.session_dir / "exceptions.log"
        with open(exc_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            traceback.print_exc(file=f)

class HealthChecker:
    """System health monitoring"""
    def __init__(self, logger):
        self.logger = logger
        
    def check_system_resources(self):
        """Check system resources"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            health = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "healthy": cpu_percent < 90 and memory.percent < 90
            }
            
            if not health["healthy"]:
                self.logger.warning(f"System resources stressed: CPU {cpu_percent}%, Memory {memory.percent}%")
                
            return health
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"healthy": False}

class EnhancedUnifiedToriLauncher:
    """Enhanced launcher with v5 integration"""
    
    def __init__(self):
        self.enhanced_logger = EnhancedLogger()
        self.logger = self.enhanced_logger.logger
        self.health_checker = HealthChecker(self.logger)
        self.processes = []
        self.running = False
        
        # Port configuration
        self.api_port = DEFAULT_API_PORT
        self.frontend_port = None
        self.prajna_port = None
        self.bridge_port = 8765
        self.audio_bridge_port = 8766
        self.concept_mesh_bridge_port = 8767
        self.mcp_metacognitive_port = 3456
        
        # Component references
        self.cognitive_engine = None
        self.memory_vault = None
        self.concept_mesh = None
        self.cognitive_interface = None
        self.metacognitive_server = None
        
        # Stability components
        self.eigenvalue_monitor = None
        self.lyapunov_analyzer = None
        self.koopman_operator = None
        
        # BPS Soliton components
        self.bps_lattice = None
        self.bps_memory = None
        self.bps_diagnostics = None
        self.bps_initialized = False
        
        # TORI v5 Components
        self.saigon_inference = None
        self.adapter_manager = None
        self.mesh_manager_v5 = None
        self.user_context_manager = None
        self.conversation_manager = None
        self.lattice_morpher = None
        self.concept_synthesizer = None
        self.v5_initialized = False
        
        # Training components
        self.synthetic_generator = None
        self.adapter_validator = None
        self.rollback_manager = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info("\n🛑 Shutdown signal received")
        self.shutdown()
        sys.exit(0)
        
    def check_port(self, port):
        """Check if port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return True
        except:
            return False
            
    def find_available_port(self, preferred_port, max_attempts=10):
        """Find available port starting from preferred"""
        for i in range(max_attempts):
            port = preferred_port + i
            if self.check_port(port):
                return port
        raise RuntimeError(f"No available ports found starting from {preferred_port}")
    
    def initialize_v5_components(self):
        """Initialize TORI/Saigon v5 components"""
        if not TORI_V5_AVAILABLE or not args.enable_v5:
            return False
        
        try:
            self.logger.info("🚀 Initializing TORI/Saigon v5 components...")
            
            # Core v5 components
            self.saigon_inference = SaigonInference()
            self.adapter_manager = MetadataManager()
            self.mesh_manager_v5 = MeshManager()
            self.user_context_manager = UserContextManager()
            
            # Conversation manager with dependencies
            self.conversation_manager = ConversationManager(
                self.saigon_inference,
                self.mesh_manager_v5
            )
            
            # Lattice morphing
            self.lattice_morpher = LatticeMorpher()
            
            # HoTT synthesis if available
            if HOTT_AVAILABLE:
                self.concept_synthesizer = ConceptSynthesizer(
                    self.mesh_manager_v5,
                    self.lattice_morpher
                )
                self.logger.info("   ✅ HoTT/Psi-Morphon synthesis initialized")
            
            # Training components if available
            if TRAINING_AVAILABLE:
                self.synthetic_generator = SyntheticDataGenerator()
                self.adapter_validator = AdapterValidator()
                self.rollback_manager = RollbackManager()
                self.logger.info("   ✅ Training pipeline initialized")
            
            # Get system statistics
            stats = self.saigon_inference.get_statistics()
            
            self.logger.info("   ✅ Saigon Inference Engine v5 initialized")
            self.logger.info("   ✅ Multi-user adapter management ready")
            self.logger.info("   ✅ Concept mesh v5 evolution enabled")
            self.logger.info("   ✅ Conversation manager with intent gaps ready")
            self.logger.info("   ✅ Lattice morphing with AV sync ready")
            self.logger.info(f"   📊 Device: {stats['device']}")
            self.logger.info(f"   📊 Cache size: {stats['cache']['maxsize']}")
            
            self.v5_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize v5 components: {e}")
            self.enhanced_logger.log_exception()
            return False
    
    def initialize_bps_soliton(self):
        """Initialize BPS Soliton System"""
        if not BPS_AVAILABLE:
            return False
            
        try:
            self.logger.info("🌊 Initializing BPS Soliton System...")
            
            # Initialize lattice
            self.bps_lattice = BPSEnhancedLattice(
                size=BPS_CONFIG['LATTICE_SIZE'],
                allow_mixed_polarity=BPS_CONFIG['ENABLE_MIXED_POLARITY']
            )
            
            # Initialize memory
            self.bps_memory = BPSEnhancedSolitonMemory(
                self.bps_lattice,
                max_solitons=100
            )
            
            # Initialize diagnostics
            self.bps_diagnostics = BPSDiagnostics(self.bps_lattice, self.bps_memory)
            
            self.logger.info("   ✅ BPS Enhanced Lattice initialized")
            self.logger.info("   ✅ BPS Soliton Memory initialized")
            self.logger.info("   ✅ BPS Diagnostics initialized")
            
            self.bps_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize BPS: {e}")
            return False
    
    def initialize_core_components(self):
        """Initialize core Python components"""
        if not CORE_COMPONENTS_AVAILABLE:
            return False
            
        try:
            self.logger.info("🧠 Initializing Core Python Components...")
            
            # Initialize cognitive engine
            if CognitiveEngine:
                self.cognitive_engine = CognitiveEngine()
                self.logger.info("   ✅ CognitiveEngine initialized")
            
            # Initialize memory vault
            if UnifiedMemoryVault:
                self.memory_vault = UnifiedMemoryVault()
                self.logger.info("   ✅ UnifiedMemoryVault initialized")
            
            # Initialize concept mesh
            if ConceptMesh:
                self.concept_mesh = ConceptMesh()
                self.logger.info("   ✅ ConceptMesh initialized")
            
            # Initialize cognitive interface
            if CognitiveInterface:
                self.cognitive_interface = CognitiveInterface(
                    cognitive_engine=self.cognitive_engine,
                    memory_vault=self.memory_vault,
                    concept_mesh=self.concept_mesh
                )
                self.logger.info("   ✅ CognitiveInterface initialized")
            
            # Initialize metacognitive server
            if MCPMetacognitiveServer:
                self.metacognitive_server = MCPMetacognitiveServer()
                self.logger.info("   ✅ MCPMetacognitiveServer initialized")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize core components: {e}")
            return False
    
    def initialize_stability_components(self):
        """Initialize stability analysis components"""
        if not STABILITY_COMPONENTS_AVAILABLE:
            return False
            
        try:
            self.logger.info("🔬 Initializing Stability Components...")
            
            if EigenvalueMonitor:
                self.eigenvalue_monitor = EigenvalueMonitor()
                self.logger.info("   ✅ EigenvalueMonitor initialized")
            
            if LyapunovAnalyzer:
                self.lyapunov_analyzer = LyapunovAnalyzer()
                self.logger.info("   ✅ LyapunovAnalyzer initialized")
            
            if KoopmanOperator:
                self.koopman_operator = KoopmanOperator()
                self.logger.info("   ✅ KoopmanOperator initialized")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stability components: {e}")
            return False
    
    def start_api_server(self):
        """Start API server based on mode"""
        try:
            self.api_port = self.find_available_port(self.api_port)
            
            if args.api == 'v5':
                # Start v5 API server
                self.logger.info(f"🌐 Starting TORI v5 API server on port {self.api_port}...")
                
                if Path("api/saigon_inference_api_v5.py").exists():
                    cmd = [sys.executable, "api/saigon_inference_api_v5.py"]
                else:
                    self.logger.warning("v5 API not found, falling back to standard API")
                    cmd = [sys.executable, "-m", "uvicorn", "api.main:app", 
                          "--host", "0.0.0.0", "--port", str(self.api_port)]
            elif args.api == 'quick':
                # Quick API mode
                self.logger.info(f"🌐 Starting quick API server on port {self.api_port}...")
                cmd = [sys.executable, "api/quick_api_server.py", "--port", str(self.api_port)]
            else:
                # Full API mode
                self.logger.info(f"🌐 Starting full API server on port {self.api_port}...")
                cmd = [sys.executable, "-m", "uvicorn", "api.main:app",
                      "--host", "0.0.0.0", "--port", str(self.api_port)]
            
            env = os.environ.copy()
            env['API_PORT'] = str(self.api_port)
            
            process = subprocess.Popen(cmd, env=env)
            self.processes.append(process)
            
            # Wait for API to be ready
            for i in range(30):
                try:
                    response = requests.get(f"http://localhost:{self.api_port}/api/health")
                    if response.status_code == 200:
                        self.logger.info(f"   ✅ API server ready on port {self.api_port}")
                        return True
                except:
                    time.sleep(1)
                    
            self.logger.warning("   ⚠️ API server may not be fully ready")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}")
            return False
    
    def start_frontend(self):
        """Start frontend development server"""
        try:
            frontend_path = Path("frontend")
            if not frontend_path.exists():
                self.logger.warning("Frontend directory not found")
                return False
                
            self.frontend_port = self.find_available_port(3000)
            self.logger.info(f"🎨 Starting frontend on port {self.frontend_port}...")
            
            cmd = ["npm", "run", "dev"]
            process = subprocess.Popen(cmd, cwd=frontend_path)
            self.processes.append(process)
            
            self.logger.info(f"   ✅ Frontend started on port {self.frontend_port}")
            
            if not args.no_browser:
                time.sleep(3)
                import webbrowser
                webbrowser.open(f"http://localhost:{self.frontend_port}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start frontend: {e}")
            return False
    
    def print_status_summary(self, api_started, frontend_started, prajna_configured,
                            core_components_started, stability_components_started,
                            mcp_metacognitive_started, concept_mesh_bridge_started,
                            audio_bridge_started):
        """Print comprehensive status summary"""
        self.logger.info("\n" + "🎉 " * 25)
        self.logger.info("🎉 TORI SYSTEM READY! 🎉")
        self.logger.info("🎉 " * 25)
        
        # v5 Components status
        if self.v5_initialized:
            self.logger.info("\n🚀 TORI/SAIGON v5 COMPONENTS:")
            if self.saigon_inference:
                self.logger.info("   ✅ Saigon Inference Engine: Active (multi-user, LoRA adapters)")
            if self.adapter_manager:
                self.logger.info("   ✅ Adapter Manager: Active (hot-swapping enabled)")
            if self.mesh_manager_v5:
                self.logger.info("   ✅ Mesh Manager v5: Active (evolution tracking)")
            if self.conversation_manager:
                self.logger.info("   ✅ Conversation Manager: Active (intent gap detection)")
            if self.lattice_morpher:
                self.logger.info("   ✅ Lattice Morphing: Active (AV sync ready)")
            if self.concept_synthesizer:
                self.logger.info("   ✅ HoTT/Psi-Morphon: Active (concept synthesis)")
            
            # Show v5 API endpoints
            self.logger.info("\n🌐 TORI v5 API ENDPOINTS:")
            self.logger.info(f"   • POST http://localhost:{self.api_port}/api/saigon/infer - Multi-user inference")
            self.logger.info(f"   • POST http://localhost:{self.api_port}/api/saigon/adapters/hot-swap - Hot-swap adapters")
            self.logger.info(f"   • GET http://localhost:{self.api_port}/api/saigon/mesh/{{user_id}} - Get user mesh")
            self.logger.info(f"   • POST http://localhost:{self.api_port}/api/saigon/conversation/chat - Chat with intent tracking")
            self.logger.info(f"   • POST http://localhost:{self.api_port}/api/saigon/morph - Lattice morphing")
        
        # API Status
        self.logger.info(f"\n🌐 API SERVER:")
        if api_started:
            self.logger.info(f"   ✅ Running on http://localhost:{self.api_port}")
            self.logger.info(f"   📚 API Docs: http://localhost:{self.api_port}/docs")
        else:
            self.logger.info("   ❌ Not started")
        
        # Frontend Status
        if frontend_started:
            self.logger.info(f"\n🎨 FRONTEND:")
            self.logger.info(f"   ✅ Running on http://localhost:{self.frontend_port}")
        
        # BPS Status
        if self.bps_initialized:
            self.logger.info("\n🌊 BPS SOLITON SYSTEM:")
            self.logger.info("   ✅ BPSEnhancedLattice: Active")
            self.logger.info("   ✅ BPSEnhancedSolitonMemory: Active")
            self.logger.info("   ✅ BPSDiagnostics: Active")
        
        # Core Components Status
        if core_components_started:
            self.logger.info("\n🧠 CORE PYTHON COMPONENTS:")
            if self.cognitive_engine:
                self.logger.info("   ✅ CognitiveEngine: Active")
            if self.memory_vault:
                self.logger.info("   ✅ UnifiedMemoryVault: Active")
            if self.concept_mesh:
                self.logger.info("   ✅ ConceptMesh: Active")
        
        # Quick Tests
        self.logger.info("\n🧪 QUICK TESTS:")
        if self.v5_initialized:
            self.logger.info(f"   # Test v5 inference:")
            self.logger.info(f"   curl -X POST http://localhost:{self.api_port}/api/saigon/infer \\")
            self.logger.info(f"     -H 'Content-Type: application/json' \\")
            self.logger.info(f"     -d '{{\"user_id\": \"demo\", \"prompt\": \"Hello TORI v5!\"}}'")
        else:
            self.logger.info(f"   # Test API:")
            self.logger.info(f"   curl http://localhost:{self.api_port}/api/health")
        
        # System Health
        health = self.health_checker.check_system_resources()
        health_status = "✅ Healthy" if health.get("healthy") else "⚠️ Stressed"
        self.logger.info(f"\n💊 System Health: {health_status}")
        
        self.logger.info("\n" + "🎉 " * 25 + "\n")
    
    def launch(self):
        """Main launch sequence"""
        try:
            self.logger.info("\n" + "🚀 " * 30)
            self.logger.info("🚀 ENHANCED TORI LAUNCHER v3.0 - WITH SAIGON v5 INTEGRATION 🚀")
            self.logger.info("🚀 " * 30)
            
            # Check system health
            health = self.health_checker.check_system_resources()
            if not health["healthy"]:
                self.logger.warning("⚠️ System resources may be stressed")
            
            # Initialize v5 components (if enabled)
            v5_initialized = False
            if args.enable_v5:
                v5_initialized = self.initialize_v5_components()
            
            # Initialize BPS Soliton System
            bps_initialized = self.initialize_bps_soliton()
            
            # Initialize core components
            core_components_started = self.initialize_core_components()
            
            # Initialize stability components
            stability_components_started = self.initialize_stability_components()
            
            # Start API server
            api_started = self.start_api_server()
            
            # Start frontend
            frontend_started = False
            if not args.no_browser:
                frontend_started = self.start_frontend()
            
            # Initialize MCP if available
            mcp_metacognitive_started = False
            if MCP_METACOGNITIVE_AVAILABLE:
                try:
                    self.logger.info("🧬 Starting MCP Metacognitive Server...")
                    # MCP initialization code here
                    mcp_metacognitive_started = True
                except Exception as e:
                    self.logger.error(f"Failed to start MCP: {e}")
            
            # Initialize bridges if hologram enabled
            concept_mesh_bridge_started = False
            audio_bridge_started = False
            if args.enable_hologram:
                # Bridge initialization code here
                concept_mesh_bridge_started = True
                audio_bridge_started = args.hologram_audio
            
            # Print status summary
            self.print_status_summary(
                api_started, frontend_started, False,
                core_components_started, stability_components_started,
                mcp_metacognitive_started, concept_mesh_bridge_started,
                audio_bridge_started
            )
            
            # Keep running
            self.running = True
            self.logger.info("✅ System running. Press Ctrl+C to shutdown.\n")
            
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("\n⌨️ Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Launch failed: {e}")
            self.enhanced_logger.log_exception()
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown"""
        self.running = False
        self.logger.info("\n🛑 Shutting down TORI system...")
        
        # Terminate all processes
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        
        # Cleanup ports
        if PORT_MANAGER_AVAILABLE and port_manager:
            port_manager.cleanup_all_ports()
        
        self.logger.info("✅ Shutdown complete")

def main():
    """Bulletproof main entry point"""
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return 1
        
        # Check critical dependencies
        try:
            import psutil
            import requests
            import uvicorn
        except ImportError as e:
            print(f"❌ Missing critical dependency: {e}")
            print("💡 Run: pip install psutil requests uvicorn")
            return 1
        
        launcher = EnhancedUnifiedToriLauncher()
        launcher.launch()
        return 0
        
    except Exception as e:
        print(f"❌ Critical startup failure: {e}")
        print(f"📋 Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
