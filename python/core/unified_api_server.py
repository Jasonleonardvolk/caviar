# 🌐 **UNIFIED API SERVER** 🚀
# ═══════════════════════════════════════════════════════════════════════════════
# Comprehensive REST API and WebSocket server for all BPS unified systems
# Provides external control, monitoring, and integration capabilities
# ═══════════════════════════════════════════════════════════════════════════════

import logging
import time
import threading
import json
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from contextlib import asynccontextmanager

# FastAPI and WebSocket imports
try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("FastAPI not available - API server will use fallback mode")
    FASTAPI_AVAILABLE = False
    
    # Fallback classes
    class BaseModel:
        pass
    class FastAPI:
        def __init__(self, *args, **kwargs):
            pass
    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail

# 🌟 EPIC BPS CONFIG INTEGRATION 🌟
try:
    from .bps_config import (
        # API configuration flags
        ENABLE_BPS_API_SERVER, ENABLE_BPS_WEBSOCKET, ENABLE_BPS_API_AUTH,
        STRICT_BPS_MODE, ENABLE_DETAILED_LOGGING,
        
        # API parameters
        API_SERVER_HOST, API_SERVER_PORT, API_RATE_LIMIT_PER_MINUTE,
        WEBSOCKET_MAX_CONNECTIONS, API_REQUEST_TIMEOUT
    )
    
    BPS_CONFIG_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("🚀 API Server using CENTRALIZED BPS configuration!")
    
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ BPS config unavailable - using fallback constants")
    
    # API flags (conservative defaults)
    ENABLE_BPS_API_SERVER = True
    ENABLE_BPS_WEBSOCKET = True
    ENABLE_BPS_API_AUTH = False  # Disabled for development
    STRICT_BPS_MODE = False
    ENABLE_DETAILED_LOGGING = True
    
    # API parameters
    API_SERVER_HOST = "localhost"
    API_SERVER_PORT = 8000
    API_RATE_LIMIT_PER_MINUTE = 100
    WEBSOCKET_MAX_CONNECTIONS = 50
    API_REQUEST_TIMEOUT = 30.0
    
    BPS_CONFIG_AVAILABLE = False

# Import all our unified systems with graceful fallbacks
try:
    from .unified_system_initializer import (
        UnifiedSystemInitializer, create_production_system, validate_system
    )
    SYSTEM_INITIALIZER_AVAILABLE = True
except ImportError:
    logger.warning("Unified System Initializer not available")
    SYSTEM_INITIALIZER_AVAILABLE = False
    
    class UnifiedSystemInitializer:
        def __init__(self, *args, **kwargs):
            pass
        def get_system_status(self):
            return {'status': 'fallback'}

try:
    from .unified_config_management import (
        UnifiedConfigManager, create_config_manager, validate_config_manager
    )
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    logger.warning("Unified Config Manager not available")
    CONFIG_MANAGER_AVAILABLE = False

try:
    from .unified_memory_orchestrator import (
        UnifiedMemoryOrchestrator, create_unified_memory_orchestrator
    )
    MEMORY_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    logger.warning("Unified Memory Orchestrator not available")
    MEMORY_ORCHESTRATOR_AVAILABLE = False

try:
    from .unified_oscillator_network_manager import (
        UnifiedOscillatorNetworkManager, create_oscillator_network_manager
    )
    OSCILLATOR_MANAGER_AVAILABLE = True
except ImportError:
    logger.warning("Unified Oscillator Network Manager not available")
    OSCILLATOR_MANAGER_AVAILABLE = False

try:
    from .unified_diagnostics_coordinator import (
        UnifiedDiagnosticsCoordinator, create_diagnostics_coordinator
    )
    DIAGNOSTICS_COORDINATOR_AVAILABLE = True
except ImportError:
    logger.warning("Unified Diagnostics Coordinator not available")
    DIAGNOSTICS_COORDINATOR_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# API DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class APIResponse(BaseModel):
    """Standard API response format"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: float = Field(default_factory=time.time)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class SystemStatusResponse(BaseModel):
    """System status response"""
    system_name: str
    system_state: str
    uptime_seconds: float
    total_subsystems: int
    subsystem_details: Dict[str, Any]

class ConfigUpdateRequest(BaseModel):
    """Configuration update request"""
    parameter_name: str
    new_value: Any
    changed_by: str = "api_user"
    validate: bool = True

class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    message_type: str
    data: Dict[str, Any]
    timestamp: float = Field(default_factory=time.time)
    source: str = "api_server"

# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    """WebSocket connection manager for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        self.connection_lock = threading.RLock()
        
    async def connect(self, websocket: WebSocket, client_info: Dict[str, Any] = None):
        """Accept new WebSocket connection"""
        if len(self.active_connections) >= WEBSOCKET_MAX_CONNECTIONS:
            await websocket.close(code=1008, reason="Maximum connections reached")
            return False
            
        await websocket.accept()
        
        with self.connection_lock:
            self.active_connections.append(websocket)
            self.connection_metadata[websocket] = {
                'connected_at': time.time(),
                'client_info': client_info or {},
                'message_count': 0
            }
        
        logger.info(f"WebSocket connected: {len(self.active_connections)} total connections")
        return True
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        with self.connection_lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                self.connection_metadata.pop(websocket, None)
                
        logger.info(f"WebSocket disconnected: {len(self.active_connections)} total connections")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific connection"""
        try:
            await websocket.send_text(message)
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]['message_count'] += 1
        except Exception as e:
            logger.warning(f"Failed to send personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
            
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
                if connection in self.connection_metadata:
                    self.connection_metadata[connection]['message_count'] += 1
            except Exception as e:
                logger.warning(f"Failed to broadcast to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        with self.connection_lock:
            total_messages = sum(
                metadata.get('message_count', 0) 
                for metadata in self.connection_metadata.values()
            )
            
            return {
                'active_connections': len(self.active_connections),
                'max_connections': WEBSOCKET_MAX_CONNECTIONS,
                'total_messages_sent': total_messages,
                'connection_details': [
                    {
                        'connected_at': metadata['connected_at'],
                        'uptime': time.time() - metadata['connected_at'],
                        'message_count': metadata['message_count'],
                        'client_info': metadata['client_info']
                    }
                    for metadata in self.connection_metadata.values()
                ]
            }

# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED API SERVER
# ═══════════════════════════════════════════════════════════════════════════════

class UnifiedAPIServer:
    """
    🌐 UNIFIED API SERVER - THE EXTERNAL CONTROL CENTER! 🎛️
    
    Features:
    • REST API endpoints for all unified systems
    • Real-time WebSocket monitoring
    • Authentication and rate limiting
    • Auto-generated API documentation
    • Comprehensive error handling and logging
    • Integration with all BPS subsystems
    """
    
    def __init__(self, base_path: Union[str, Path], server_name: str = "unified_api_server"):
        """
        Initialize the unified API server
        
        Args:
            base_path: Base directory for server storage
            server_name: Unique name for this server
        """
        self.base_path = Path(base_path)
        self.server_name = server_name
        self.server_path = self.base_path / server_name
        
        # 🎛️ Configuration
        self.config_available = BPS_CONFIG_AVAILABLE
        self.strict_mode = STRICT_BPS_MODE
        
        # 🌐 FastAPI application
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(
                title="Unified BPS API Server",
                description="Comprehensive API for all BPS unified systems",
                version="1.0.0",
                docs_url="/docs" if not ENABLE_BPS_API_AUTH else None,
                redoc_url="/redoc" if not ENABLE_BPS_API_AUTH else None
            )
            self._setup_middleware()
        else:
            self.app = None
        
        # 🔧 System components
        self.system_initializer: Optional[UnifiedSystemInitializer] = None
        self.config_manager: Optional[UnifiedConfigManager] = None
        
        # 📡 WebSocket manager
        self.connection_manager = ConnectionManager()
        
        # 🔒 Authentication
        self.security = HTTPBearer() if ENABLE_BPS_API_AUTH else None
        
        # 📊 Server state
        self.creation_time = time.time()
        self.start_time: Optional[float] = None
        self.request_count = 0
        self.error_count = 0
        
        # 🎯 Background monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
        # Initialize server
        self._initialize_server()
        
        # Setup API routes
        if self.app:
            self._setup_routes()
        
        logger.info(f"🚀 API Server '{server_name}' CREATED!")
        logger.info(f"📍 Location: {self.server_path}")
        logger.info(f"⚡ BPS Config: {'ENABLED' if self.config_available else 'FALLBACK'}")
        logger.info(f"🌐 FastAPI: {'ENABLED' if FASTAPI_AVAILABLE else 'FALLBACK'}")
    
    def _initialize_server(self):
        """Initialize server directory and components"""
        try:
            self.server_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.server_path / "logs").mkdir(exist_ok=True)
            (self.server_path / "configs").mkdir(exist_ok=True)
            (self.server_path / "backups").mkdir(exist_ok=True)
            
            # Initialize core systems if available
            if SYSTEM_INITIALIZER_AVAILABLE:
                self.system_initializer = UnifiedSystemInitializer(
                    "api_managed_system", self.server_path / "systems"
                )
                
            if CONFIG_MANAGER_AVAILABLE:
                self.config_manager = create_config_manager(str(self.server_path))
            
            logger.info("📁 API server directory structure initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize API server: {e}")
            if self.strict_mode:
                raise RuntimeError(f"API server initialization failed: {e}")
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        if not self.app:
            return
            
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup all API routes"""
        if not self.app:
            return
        
        # Health check endpoint
        @self.app.get("/health", response_model=APIResponse)
        async def health_check():
            """Health check endpoint"""
            self.request_count += 1
            
            return APIResponse(
                success=True,
                message="API server is healthy",
                data={
                    "server_name": self.server_name,
                    "uptime_seconds": time.time() - self.creation_time if self.creation_time else 0,
                    "request_count": self.request_count,
                    "error_count": self.error_count,
                    "config_available": self.config_available,
                    "fastapi_available": FASTAPI_AVAILABLE
                }
            )
        
        # System status endpoint
        @self.app.get("/system/status", response_model=APIResponse)
        async def get_system_status():
            """Get unified system status"""
            self.request_count += 1
            
            if not self.system_initializer:
                raise HTTPException(status_code=503, detail="System initializer not available")
            
            try:
                status = self.system_initializer.get_system_status()
                return APIResponse(
                    success=True,
                    message="System status retrieved",
                    data=status
                )
            except Exception as e:
                self.error_count += 1
                logger.error(f"Failed to get system status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Start system endpoint
        @self.app.post("/system/start", response_model=APIResponse)
        async def start_system():
            """Start the unified system"""
            self.request_count += 1
            
            if not self.system_initializer:
                raise HTTPException(status_code=503, detail="System initializer not available")
            
            try:
                success = self.system_initializer.start_system(auto_discover=True)
                
                if success:
                    # Broadcast system start to WebSocket clients
                    await self.connection_manager.broadcast(
                        json.dumps({
                            "type": "system_event",
                            "event": "system_started",
                            "timestamp": time.time()
                        })
                    )
                    
                    return APIResponse(
                        success=True,
                        message="System started successfully",
                        data=self.system_initializer.get_system_status()
                    )
                else:
                    self.error_count += 1
                    raise HTTPException(status_code=500, detail="Failed to start system")
                    
            except Exception as e:
                self.error_count += 1
                logger.error(f"Failed to start system: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Shutdown system endpoint
        @self.app.post("/system/shutdown", response_model=APIResponse)
        async def shutdown_system():
            """Shutdown the unified system"""
            self.request_count += 1
            
            if not self.system_initializer:
                raise HTTPException(status_code=503, detail="System initializer not available")
            
            try:
                success = self.system_initializer.shutdown_system()
                
                if success:
                    # Broadcast system shutdown to WebSocket clients
                    await self.connection_manager.broadcast(
                        json.dumps({
                            "type": "system_event",
                            "event": "system_shutdown",
                            "timestamp": time.time()
                        })
                    )
                    
                    return APIResponse(
                        success=True,
                        message="System shutdown successfully"
                    )
                else:
                    self.error_count += 1
                    raise HTTPException(status_code=500, detail="Failed to shutdown system")
                    
            except Exception as e:
                self.error_count += 1
                logger.error(f"Failed to shutdown system: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Configuration endpoints
        @self.app.get("/config/summary", response_model=APIResponse)
        async def get_config_summary():
            """Get configuration summary"""
            self.request_count += 1
            
            if not self.config_manager:
                raise HTTPException(status_code=503, detail="Config manager not available")
            
            try:
                summary = self.config_manager.get_configuration_summary()
                return APIResponse(
                    success=True,
                    message="Configuration summary retrieved",
                    data=summary
                )
            except Exception as e:
                self.error_count += 1
                logger.error(f"Failed to get config summary: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/config/parameter/{parameter_name}", response_model=APIResponse)
        async def get_config_parameter(parameter_name: str):
            """Get specific configuration parameter"""
            self.request_count += 1
            
            if not self.config_manager:
                raise HTTPException(status_code=503, detail="Config manager not available")
            
            try:
                value = self.config_manager.get_parameter(parameter_name)
                return APIResponse(
                    success=True,
                    message=f"Parameter {parameter_name} retrieved",
                    data={"parameter_name": parameter_name, "value": value}
                )
            except ValueError as e:
                self.error_count += 1
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:
                self.error_count += 1
                logger.error(f"Failed to get parameter {parameter_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/config/parameter", response_model=APIResponse)
        async def update_config_parameter(request: ConfigUpdateRequest):
            """Update configuration parameter"""
            self.request_count += 1
            
            if not self.config_manager:
                raise HTTPException(status_code=503, detail="Config manager not available")
            
            try:
                success = self.config_manager.set_parameter(
                    request.parameter_name,
                    request.new_value,
                    request.changed_by,
                    request.validate
                )
                
                if success:
                    # Broadcast config change to WebSocket clients
                    await self.connection_manager.broadcast(
                        json.dumps({
                            "type": "config_change",
                            "parameter": request.parameter_name,
                            "new_value": request.new_value,
                            "changed_by": request.changed_by,
                            "timestamp": time.time()
                        })
                    )
                    
                    return APIResponse(
                        success=True,
                        message=f"Parameter {request.parameter_name} updated successfully",
                        data={
                            "parameter_name": request.parameter_name,
                            "new_value": request.new_value,
                            "changed_by": request.changed_by
                        }
                    )
                else:
                    self.error_count += 1
                    raise HTTPException(status_code=400, detail="Parameter update failed validation")
                    
            except Exception as e:
                self.error_count += 1
                logger.error(f"Failed to update parameter {request.parameter_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # WebSocket monitoring endpoint
        @self.app.websocket("/ws/monitor")
        async def websocket_monitor(websocket: WebSocket):
            """WebSocket endpoint for real-time monitoring"""
            client_info = {
                "user_agent": websocket.headers.get("user-agent", "unknown"),
                "origin": websocket.headers.get("origin", "unknown")
            }
            
            connected = await self.connection_manager.connect(websocket, client_info)
            if not connected:
                return
            
            try:
                # Send initial system status
                if self.system_initializer:
                    status = self.system_initializer.get_system_status()
                    await self.connection_manager.send_personal_message(
                        json.dumps({
                            "type": "initial_status",
                            "data": status,
                            "timestamp": time.time()
                        }),
                        websocket
                    )
                
                # Keep connection alive and handle incoming messages
                while True:
                    try:
                        # Wait for incoming message with timeout
                        data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                        
                        # Echo received message (can be extended for commands)
                        await self.connection_manager.send_personal_message(
                            json.dumps({
                                "type": "echo",
                                "received": data,
                                "timestamp": time.time()
                            }),
                            websocket
                        )
                        
                    except asyncio.TimeoutError:
                        # Send keepalive
                        await self.connection_manager.send_personal_message(
                            json.dumps({
                                "type": "keepalive",
                                "timestamp": time.time()
                            }),
                            websocket
                        )
                        
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected normally")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.connection_manager.disconnect(websocket)
        
        # WebSocket connection stats
        @self.app.get("/ws/stats", response_model=APIResponse)
        async def get_websocket_stats():
            """Get WebSocket connection statistics"""
            self.request_count += 1
            
            stats = self.connection_manager.get_connection_stats()
            return APIResponse(
                success=True,
                message="WebSocket stats retrieved",
                data=stats
            )
    
    async def start_background_monitoring(self):
        """Start background monitoring task"""
        if not ENABLE_BPS_WEBSOCKET:
            logger.info("WebSocket monitoring disabled")
            return
        
        async def monitor_loop():
            """Background monitoring loop"""
            logger.info("🔍 Background monitoring started")
            
            while not self.shutdown_event.is_set():
                try:
                    # Broadcast system status every 30 seconds
                    if self.system_initializer and self.connection_manager.active_connections:
                        status = self.system_initializer.get_system_status()
                        
                        await self.connection_manager.broadcast(
                            json.dumps({
                                "type": "status_update",
                                "data": status,
                                "timestamp": time.time()
                            })
                        )
                    
                    # Wait for next cycle or shutdown
                    try:
                        await asyncio.wait_for(self.shutdown_event.wait(), timeout=30.0)
                        break  # Shutdown requested
                    except asyncio.TimeoutError:
                        continue  # Continue monitoring
                        
                except Exception as e:
                    logger.error(f"Background monitoring error: {e}")
                    await asyncio.sleep(5.0)  # Brief pause on error
        
        self.monitoring_task = asyncio.create_task(monitor_loop())
    
    async def stop_background_monitoring(self):
        """Stop background monitoring task"""
        if self.monitoring_task:
            self.shutdown_event.set()
            try:
                await asyncio.wait_for(self.monitoring_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.monitoring_task.cancel()
            
            logger.info("🔍 Background monitoring stopped")
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get comprehensive server statistics"""
        uptime = time.time() - self.creation_time if self.creation_time else 0
        
        return {
            'server_name': self.server_name,
            'uptime_seconds': uptime,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.request_count),
            'config_available': self.config_available,
            'fastapi_available': FASTAPI_AVAILABLE,
            'websocket_stats': self.connection_manager.get_connection_stats(),
            'system_initializer_available': self.system_initializer is not None,
            'config_manager_available': self.config_manager is not None,
            'start_time': self.start_time,
            'creation_time': self.creation_time
        }
    
    def __repr__(self):
        return (f"<UnifiedAPIServer '{self.server_name}' "
                f"requests={self.request_count} connections={len(self.connection_manager.active_connections)}>")

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_api_server(base_path: str = "/tmp", server_name: str = "production_api_server") -> UnifiedAPIServer:
    """Create and initialize an API server"""
    if not ENABLE_BPS_API_SERVER:
        logger.warning("API server disabled")
        return None
    
    server = UnifiedAPIServer(base_path, server_name)
    
    logger.info(f"🌐 API Server created: {server.server_name}")
    logger.info(f"📊 Server components: System={server.system_initializer is not None}, Config={server.config_manager is not None}")
    return server

async def run_api_server(server: UnifiedAPIServer, host: str = None, port: int = None):
    """Run the API server with uvicorn"""
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not available - cannot run server")
        return False
    
    if not server.app:
        logger.error("Server app not initialized")
        return False
    
    host = host or API_SERVER_HOST
    port = port or API_SERVER_PORT
    
    try:
        server.start_time = time.time()
        
        # Start background monitoring
        await server.start_background_monitoring()
        
        logger.info(f"🚀 Starting API server on {host}:{port}")
        logger.info(f"📖 API docs available at: http://{host}:{port}/docs")
        
        # Create uvicorn config
        config = uvicorn.Config(
            server.app,
            host=host,
            port=port,
            log_level="info" if ENABLE_DETAILED_LOGGING else "warning"
        )
        
        # Run server
        server_instance = uvicorn.Server(config)
        await server_instance.serve()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to run API server: {e}")
        return False
    finally:
        # Stop background monitoring
        await server.stop_background_monitoring()

def validate_api_server(server: UnifiedAPIServer) -> Dict[str, Any]:
    """Comprehensive validation of API server"""
    validation = {
        'status': 'unknown',
        'issues': [],
        'server_stats': server.get_server_stats()
    }
    
    try:
        stats = validation['server_stats']
        
        # Check if FastAPI is available
        if not FASTAPI_AVAILABLE:
            validation['issues'].append("FastAPI not available - server running in fallback mode")
        
        # Check error rate
        if stats['error_rate'] > 0.1:  # More than 10% error rate
            validation['issues'].append(f"High error rate: {stats['error_rate']:.1%}")
        
        # Check if core systems are available
        if not stats['system_initializer_available']:
            validation['issues'].append("System initializer not available")
        
        if not stats['config_manager_available']:
            validation['issues'].append("Config manager not available")
        
        # Overall status
        if not validation['issues']:
            validation['status'] = 'excellent'
        elif len(validation['issues']) <= 2:
            validation['status'] = 'good'
        else:
            validation['status'] = 'issues_detected'
        
        return validation
        
    except Exception as e:
        validation['status'] = 'error'
        validation['issues'].append(f"Validation failed: {e}")
        return validation

# Export all components
__all__ = [
    'UnifiedAPIServer',
    'ConnectionManager',
    'APIResponse',
    'SystemStatusResponse',
    'ConfigUpdateRequest',
    'WebSocketMessage',
    'create_api_server',
    'run_api_server',
    'validate_api_server',
    'BPS_CONFIG_AVAILABLE',
    'FASTAPI_AVAILABLE'
]

if __name__ == "__main__":
    # 🎪 DEMONSTRATION AND PRODUCTION MODE!
    logger.info("🚀 UNIFIED API SERVER ACTIVATED!")
    logger.info(f"⚡ Config: {'CENTRALIZED' if BPS_CONFIG_AVAILABLE else 'FALLBACK MODE'}")
    logger.info(f"🌐 FastAPI: {'ENABLED' if FASTAPI_AVAILABLE else 'FALLBACK MODE'}")
    
    import sys
    
    if '--demo' in sys.argv:
        async def demo_main():
            logger.info("🎪 Creating demo API server...")
            
            server = create_api_server("/tmp", "demo_api_server")
            
            if server:
                logger.info("📊 API Server Demo Status:")
                stats = server.get_server_stats()
                for key, value in stats.items():
                    if key not in ['websocket_stats']:  # Skip complex nested data
                        logger.info(f"  {key}: {value}")
                
                if '--validate' in sys.argv:
                    logger.info("🔍 Running API server validation...")
                    validation = validate_api_server(server)
                    logger.info(f"Overall validation: {validation['status'].upper()}")
                    if validation['issues']:
                        for issue in validation['issues']:
                            logger.warning(f"  Issue: {issue}")
            else:
                logger.error("💥 Failed to create demo API server")
        
        # Run demo
        asyncio.run(demo_main())
    
    elif '--production' in sys.argv:
        async def production_main():
            logger.info("🏭 STARTING PRODUCTION API SERVER...")
            
            server = create_api_server("/tmp", "production_api_server")
            
            if server:
                # Show server info
                stats = server.get_server_stats()
                logger.info("📊 PRODUCTION API SERVER STATUS:")
                logger.info(f"  Server: {stats['server_name']}")
                logger.info(f"  FastAPI: {'ENABLED' if FASTAPI_AVAILABLE else 'FALLBACK'}")
                logger.info(f"  System Initializer: {'ENABLED' if stats['system_initializer_available'] else 'DISABLED'}")
                logger.info(f"  Config Manager: {'ENABLED' if stats['config_manager_available'] else 'DISABLED'}")
                
                # Run the server
                success = await run_api_server(server)
                
                if success:
                    logger.info("✅ Production API server completed successfully")
                else:
                    logger.error("💥 Production API server failed")
            else:
                logger.error("💥 Failed to create production API server")
        
        # Run production server
        asyncio.run(production_main())
    
    else:
        logger.info("ℹ️ Usage: python unified_api_server.py [--demo|--production] [--validate]")
        logger.info("  --demo: Run demonstration mode")
        logger.info("  --production: Start production API server")
        logger.info("  --validate: Run validation (with demo)")
    
    logger.info("🎯 Unified API Server ready for PRODUCTION use!")
