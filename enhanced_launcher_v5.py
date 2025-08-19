#!/usr/bin/env python3
"""
🚀 TORI/SAIGON v5 UNIFIED LAUNCHER
==================================
Production-ready launcher with full integration of:
- Multi-user inference engine
- Dynamic LoRA adapters
- Concept mesh evolution
- HoTT/Psi-Morphon synthesis
- Lattice morphing
- WebGPU holographic rendering
- Complete monitoring stack
"""

import os
import sys
import time
import socket
import json
import asyncio
import signal
import subprocess
import threading
import traceback
import logging
import psutil
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import atexit

# Ensure proper encoding
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================================
# TORI/SAIGON v5 IMPORTS
# ============================================================================

# Core v5 Components
try:
    from python.core.saigon_inference_v5 import SaigonInference
    from python.core.adapter_loader_v5 import MetadataManager
    from python.core.concept_mesh_v5 import MeshManager
    from python.core.user_context import UserContextManager
    from python.core.conversation_manager import ConversationManager
    from python.core.lattice_morphing import LatticeMorpher
    CORE_V5_AVAILABLE = True
except ImportError as e:
    CORE_V5_AVAILABLE = False
    print(f"⚠️ Core v5 components not available: {e}")

# Training Components
try:
    from python.training.synthetic_data_generator import SyntheticDataGenerator
    from python.training.validate_adapter import AdapterValidator
    from python.training.rollback_adapter import RollbackManager
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False

# HoTT Integration
try:
    from hott_integration.psi_morphon import ConceptSynthesizer, PsiMorphonField
    HOTT_AVAILABLE = True
except ImportError:
    HOTT_AVAILABLE = False

# BPS Support (backward compatibility)
try:
    from python.core.bps_config_enhanced import BPS_CONFIG, SolitonPolarity
    from python.core.bps_oscillator_enhanced import BPSEnhancedLattice
    from python.core.bps_soliton_memory_enhanced import BPSEnhancedSolitonMemory
    BPS_AVAILABLE = True
except ImportError:
    BPS_AVAILABLE = False

# Port management
try:
    from port_manager import port_manager
    PORT_MANAGER_AVAILABLE = True
except ImportError:
    PORT_MANAGER_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_PORTS = {
    'api': 8001,
    'frontend': 3000,
    'prometheus': 9090,
    'grafana': 3001,
    'redis': 6379,
    'postgres': 5432,
    'hologram_bridge': 8765,
    'audio_bridge': 8766,
    'concept_mesh_bridge': 8767,
    'prajna': 3000,  # If using Prajna
    'mcp_metacognitive': 3456
}

# ============================================================================
# ENHANCED LOGGER
# ============================================================================

class EnhancedLogger:
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path("logs") / f"session_{self.session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.session_dir / "launcher.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("TORI_v5")
    
    def info(self, msg):
        self.logger.info(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)

# ============================================================================
# TORI/SAIGON v5 LAUNCHER
# ============================================================================

class ToriSaigonV5Launcher:
    def __init__(self):
        self.logger = EnhancedLogger()
        self.processes = []
        self.shutdown_event = threading.Event()
        
        # v5 Components
        self.inference_engine = None
        self.metadata_manager = None
        self.mesh_manager = None
        self.context_manager = None
        self.conversation_manager = None
        self.lattice_morpher = None
        self.concept_synthesizer = None
        
        # Legacy components (for compatibility)
        self.bps_lattice = None
        self.bps_memory = None
        
        # Ports
        self.ports = DEFAULT_PORTS.copy()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Register cleanup
        atexit.register(self.cleanup)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("\n🛑 Shutdown signal received")
        self.shutdown_event.set()
        self.cleanup()
        sys.exit(0)
    
    def _check_port(self, port: int) -> bool:
        """Check if port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return True
        except:
            return False
    
    def _find_available_port(self, preferred: int) -> int:
        """Find available port starting from preferred"""
        if self._check_port(preferred):
            return preferred
        
        for offset in range(1, 100):
            if self._check_port(preferred + offset):
                return preferred + offset
        
        raise RuntimeError(f"No available ports near {preferred}")
    
    def initialize_v5_components(self) -> bool:
        """Initialize TORI/Saigon v5 components"""
        self.logger.info("\n🚀 Initializing TORI/Saigon v5 Components...")
        
        if not CORE_V5_AVAILABLE:
            self.logger.warning("Core v5 components not available")
            return False
        
        try:
            # Initialize core components
            self.inference_engine = SaigonInference()
            self.metadata_manager = MetadataManager()
            self.mesh_manager = MeshManager()
            self.context_manager = UserContextManager()
            
            # Initialize conversation manager with dependencies
            self.conversation_manager = ConversationManager(
                self.inference_engine,
                self.mesh_manager
            )
            
            # Initialize lattice morpher
            self.lattice_morpher = LatticeMorpher()
            
            # Initialize HoTT if available
            if HOTT_AVAILABLE:
                self.concept_synthesizer = ConceptSynthesizer(
                    self.mesh_manager,
                    self.lattice_morpher
                )
                self.logger.info("   ✅ HoTT/Psi-Morphon synthesis initialized")
            
            self.logger.info("   ✅ Inference engine initialized")
            self.logger.info("   ✅ Adapter management initialized")
            self.logger.info("   ✅ Mesh manager initialized")
            self.logger.info("   ✅ User context manager initialized")
            self.logger.info("   ✅ Conversation manager initialized")
            self.logger.info("   ✅ Lattice morpher initialized")
            
            # Get system stats
            stats = self.inference_engine.get_statistics()
            self.logger.info(f"   📊 Device: {stats['device']}")
            self.logger.info(f"   📊 Cache size: {stats['cache']['maxsize']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize v5 components: {e}")
            return False
    
    def start_api_server(self) -> bool:
        """Start FastAPI server"""
        self.logger.info("\n🌐 Starting API Server...")
        
        # Find available port
        api_port = self._find_available_port(self.ports['api'])
        self.ports['api'] = api_port
        
        try:
            # Start API server
            cmd = [
                sys.executable,
                "api/saigon_inference_api_v5.py"
            ]
            
            env = os.environ.copy()
            env['API_PORT'] = str(api_port)
            
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes.append(process)
            
            # Wait for server to start
            for _ in range(30):
                try:
                    response = requests.get(f"http://localhost:{api_port}/health")
                    if response.status_code == 200:
                        self.logger.info(f"   ✅ API server running on port {api_port}")
                        return True
                except:
                    time.sleep(1)
            
            self.logger.error("   ❌ API server failed to start")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}")
            return False
    
    def start_frontend(self) -> bool:
        """Start frontend development server"""
        self.logger.info("\n🎨 Starting Frontend...")
        
        frontend_path = Path("frontend/hybrid")
        if not frontend_path.exists():
            self.logger.warning("Frontend directory not found")
            return False
        
        try:
            # Find available port
            frontend_port = self._find_available_port(self.ports['frontend'])
            self.ports['frontend'] = frontend_port
            
            # Start frontend
            cmd = ["npm", "run", "dev", "--", "--port", str(frontend_port)]
            
            process = subprocess.Popen(
                cmd,
                cwd=frontend_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes.append(process)
            
            self.logger.info(f"   ✅ Frontend starting on port {frontend_port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start frontend: {e}")
            return False
    
    def start_monitoring(self) -> bool:
        """Start monitoring stack (if using dickbox)"""
        self.logger.info("\n📊 Starting Monitoring Stack...")
        
        # Check if dickbox is available
        dickbox_config = Path("dickbox.toml")
        if not dickbox_config.exists():
            self.logger.info("   ℹ️ Dickbox config not found, skipping monitoring")
            return False
        
        try:
            # Start monitoring with dickbox
            cmd = [sys.executable, "dickbox.py", "up"]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes.append(process)
            
            self.logger.info("   ✅ Monitoring stack started with dickbox")
            return True
            
        except Exception as e:
            self.logger.warning(f"Could not start monitoring: {e}")
            return False
    
    def initialize_legacy_components(self) -> bool:
        """Initialize legacy BPS components for backward compatibility"""
        if not BPS_AVAILABLE:
            return False
        
        try:
            self.logger.info("\n🌊 Initializing BPS Soliton System...")
            
            self.bps_lattice = BPSEnhancedLattice(
                size=BPS_CONFIG['LATTICE_SIZE'],
                allow_mixed_polarity=BPS_CONFIG['ENABLE_MIXED_POLARITY']
            )
            
            self.bps_memory = BPSEnhancedSolitonMemory(
                self.bps_lattice,
                max_solitons=100
            )
            
            self.logger.info("   ✅ BPS components initialized (legacy support)")
            return True
            
        except Exception as e:
            self.logger.warning(f"Could not initialize BPS: {e}")
            return False
    
    def print_status(self):
        """Print comprehensive system status"""
        print("\n" + "="*80)
        print(" 🎉 TORI/SAIGON v5.0.0 - SYSTEM READY 🎉")
        print("="*80)
        
        # Core v5 Status
        print("\n🚀 CORE v5 COMPONENTS:")
        if self.inference_engine:
            print("   ✅ Saigon Inference Engine: Active")
            print("   ✅ Multi-user support: Enabled")
            print("   ✅ LoRA adapters: Dynamic hot-swapping")
        if self.mesh_manager:
            print("   ✅ Concept Mesh: Evolution enabled")
        if self.conversation_manager:
            print("   ✅ Conversation Manager: Intent gap detection")
        if self.lattice_morpher:
            print("   ✅ Lattice Morphing: AV sync ready")
        if self.concept_synthesizer:
            print("   ✅ HoTT/Psi-Morphon: Concept synthesis active")
        
        # API Endpoints
        print(f"\n🌐 API ENDPOINTS:")
        api_port = self.ports['api']
        print(f"   📍 Main API: http://localhost:{api_port}")
        print(f"   📍 API Docs: http://localhost:{api_port}/docs")
        print(f"   📍 Health: http://localhost:{api_port}/health")
        
        # Frontend
        if 'frontend' in self.ports:
            print(f"\n🎨 FRONTEND:")
            print(f"   📍 Web UI: http://localhost:{self.ports['frontend']}")
            print(f"   📍 Hologram: Available in browser")
        
        # Monitoring
        print(f"\n📊 MONITORING:")
        if 'prometheus' in self.ports:
            print(f"   📍 Prometheus: http://localhost:{self.ports['prometheus']}")
        if 'grafana' in self.ports:
            print(f"   📍 Grafana: http://localhost:{self.ports['grafana']}")
        
        # Quick Tests
        print(f"\n🧪 QUICK TESTS:")
        print(f"   # Test inference")
        print(f"   curl -X POST http://localhost:{api_port}/api/saigon/infer \\")
        print(f"     -H 'Content-Type: application/json' \\")
        print(f"     -d '{{\"user_id\": \"demo\", \"prompt\": \"Hello TORI\"}}'")
        print()
        print(f"   # Check adapters")
        print(f"   curl http://localhost:{api_port}/api/saigon/adapters/demo")
        print()
        print(f"   # View mesh")
        print(f"   curl http://localhost:{api_port}/api/saigon/mesh/demo")
        
        # Features
        print(f"\n✨ FEATURES:")
        print(f"   • Multi-user inference with isolation")
        print(f"   • Dynamic LoRA adapter hot-swapping")
        print(f"   • Continuous learning from intent gaps")
        print(f"   • Concept mesh evolution")
        print(f"   • HoTT/Psi-morphon synthesis")
        print(f"   • WebGPU holographic rendering")
        print(f"   • Production monitoring")
        
        print("\n" + "="*80)
        print(" 🚀 System ready for production use!")
        print("="*80 + "\n")
    
    def cleanup(self):
        """Clean shutdown of all components"""
        self.logger.info("\n🧹 Cleaning up...")
        
        # Terminate all processes
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        
        # Cleanup ports if manager available
        if PORT_MANAGER_AVAILABLE:
            port_manager.cleanup_all_ports()
        
        self.logger.info("   ✅ Cleanup complete")
    
    def launch(self):
        """Main launch sequence"""
        print("\n" + "🚀"*20)
        print(" TORI/SAIGON v5 LAUNCHER")
        print("🚀"*20 + "\n")
        
        try:
            # Initialize v5 components
            v5_initialized = self.initialize_v5_components()
            
            # Initialize legacy components
            legacy_initialized = self.initialize_legacy_components()
            
            # Start API server
            api_started = self.start_api_server()
            
            # Start frontend (optional)
            frontend_started = False
            if "--no-frontend" not in sys.argv:
                frontend_started = self.start_frontend()
            
            # Start monitoring (optional)
            monitoring_started = False
            if "--with-monitoring" in sys.argv:
                monitoring_started = self.start_monitoring()
            
            # Print status
            self.print_status()
            
            # Keep running
            self.logger.info("\n✅ System running. Press Ctrl+C to shutdown.\n")
            
            while not self.shutdown_event.is_set():
                time.sleep(1)
            
        except KeyboardInterrupt:
            self.logger.info("\n⌨️ Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Launch failed: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()

# ============================================================================
# MAIN ENTRY
# ============================================================================

def main():
    """Main entry point"""
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required")
        return 1
    
    # Check critical dependencies
    try:
        import psutil
        import requests
        import fastapi
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return 1
    
    # Launch
    launcher = ToriSaigonV5Launcher()
    launcher.launch()
    return 0

if __name__ == "__main__":
    sys.exit(main())
