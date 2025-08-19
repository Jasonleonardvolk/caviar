"""
MCP Server Integration with Dynamic Discovery
===========================================

This module integrates all dynamically discovered servers into the MCP ecosystem.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Import core components
from ..core.agent_registry import agent_registry
from ..core.psi_archive import psi_archive
from ..core.state_manager import state_manager
from ..core.dynamic_discovery import server_discovery

# Import Celery tasks
try:
    from ..tasks.celery_tasks import (
        app as celery_app,
        process_complex_query,
        run_kaizen_analysis,
        monitor_system_health,
        CeleryManager
    )
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    celery_app = None

logger = logging.getLogger(__name__)

class TORIIntegration:
    """Integration layer for all TORI components with dynamic discovery"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.celery_manager = None
        self.is_initialized = False
        self.discovered_servers = {}
        
    async def initialize(self):
        """Initialize all components using dynamic discovery"""
        if self.is_initialized:
            logger.info("TORI integration already initialized")
            return
        
        logger.info("Initializing TORI integration with dynamic discovery...")
        
        # Discover all available servers
        self.discovered_servers = server_discovery.discover_servers()
        logger.info(f"🔍 Discovered {len(self.discovered_servers)} servers")
        
        # Start all enabled servers
        start_results = await server_discovery.start_all_servers()
        
        # Log results
        for server_name, success in start_results.items():
            if success:
                logger.info(f"✅ {server_name}: Started successfully")
            else:
                logger.warning(f"⚠️ {server_name}: Failed to start")
        
        # Initialize Celery if available
        if CELERY_AVAILABLE:
            try:
                self.celery_manager = CeleryManager()
                logger.info("✅ Celery task manager initialized")
                
                # Check if Redis is available
                import redis
                r = redis.Redis(host='localhost', port=6379)
                r.ping()
                logger.info("✅ Redis connection successful")
                
            except Exception as e:
                logger.warning(f"Celery/Redis not available: {e}")
                logger.info("⚠️  Running without async task support")
        else:
            logger.warning("Celery not installed - async tasks disabled")
        
        # Log initialization complete
        psi_archive.log_event("tori_integration_initialized", {
            "servers_discovered": list(self.discovered_servers.keys()),
            "servers_started": [name for name, success in start_results.items() if success],
            "celery_active": CELERY_AVAILABLE,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        self.is_initialized = True
        logger.info("🎉 TORI integration initialized successfully!")
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a query using Daniel (if available)"""
        daniel = agent_registry.get("daniel")
        if not daniel:
            return {
                "status": "error",
                "error": "Daniel cognitive engine not initialized"
            }
        
        # Check if this should be async
        if CELERY_AVAILABLE and context and context.get("async", False):
            # Process asynchronously via Celery
            task = process_complex_query.delay(query, context)
            return {
                "status": "queued",
                "task_id": task.id,
                "message": "Query queued for async processing"
            }
        else:
            # Process synchronously
            return await daniel.execute(query, context)
    
    async def get_insights(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent insights from Kaizen (if available)"""
        kaizen = agent_registry.get("kaizen")
        if not kaizen:
            return {
                "status": "error",
                "error": "Kaizen not initialized"
            }
        
        return kaizen.get_recent_insights(limit)
    
    async def trigger_analysis(self) -> Dict[str, Any]:
        """Manually trigger Kaizen analysis (if available)"""
        kaizen = agent_registry.get("kaizen")
        if not kaizen:
            return {
                "status": "error",
                "error": "Kaizen not initialized"
            }
        
        if CELERY_AVAILABLE:
            # Run async via Celery
            task = run_kaizen_analysis.delay()
            return {
                "status": "queued",
                "task_id": task.id,
                "message": "Analysis queued"
            }
        else:
            # Run synchronously
            return await kaizen.run_analysis_cycle()
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status with all discovered servers"""
        # Get dynamic server status
        server_status = await server_discovery.get_server_status()
        
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "servers": server_status,
            "discovery": {
                "total_discovered": len(self.discovered_servers),
                "enabled": sum(1 for s in server_status.values() if s.get("enabled")),
                "running": sum(1 for s in server_status.values() if s.get("running"))
            },
            "celery": {
                "available": CELERY_AVAILABLE,
                "redis_connected": False
            },
            "cognitive_state": await state_manager.get_current_state()
        }
        
        # Check Celery/Redis
        if CELERY_AVAILABLE:
            try:
                import redis
                r = redis.Redis(host='localhost', port=6379)
                r.ping()
                status["celery"]["redis_connected"] = True
            except:
                pass
        
        return status
    
    async def shutdown(self):
        """Graceful shutdown of all components"""
        logger.info("Shutting down TORI integration...")
        
        # Stop all servers using dynamic discovery
        await server_discovery.stop_all_servers()
        
        # Log shutdown
        psi_archive.log_event("tori_integration_shutdown", {
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info("TORI integration shutdown complete")

# Global integration instance
tori_integration = TORIIntegration()

# FastAPI endpoint additions (if using FastAPI mode)
def register_integration_endpoints(app):
    """Register integration endpoints with FastAPI app"""
    
    @app.post("/api/query")
    async def process_query_endpoint(query: str, context: Optional[Dict[str, Any]] = None):
        """Process a query using Daniel"""
        return await tori_integration.process_query(query, context)
    
    @app.get("/api/insights")
    async def get_insights_endpoint(limit: int = 10):
        """Get recent Kaizen insights"""
        return await tori_integration.get_insights(limit)
    
    @app.post("/api/analyze")
    async def trigger_analysis_endpoint():
        """Trigger Kaizen analysis"""
        return await tori_integration.trigger_analysis()
    
    @app.get("/api/system/status")
    async def get_system_status_endpoint():
        """Get system status"""
        return await tori_integration.get_system_status()
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize integration on startup"""
        await tori_integration.initialize()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown integration"""
        await tori_integration.shutdown()

# MCP tool registration (if using MCP mode)
def register_integration_tools(mcp):
    """Register integration tools with MCP server"""
    
    @mcp.tool()
    async def process_query(query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a query using Daniel cognitive engine"""
        return await tori_integration.process_query(query, context)
    
    @mcp.tool()
    async def get_insights(limit: int = 10) -> Dict[str, Any]:
        """Get recent insights from Kaizen"""
        return await tori_integration.get_insights(limit)
    
    @mcp.tool()
    async def trigger_analysis() -> Dict[str, Any]:
        """Manually trigger Kaizen analysis"""
        return await tori_integration.trigger_analysis()
    
    @mcp.tool()
    async def get_system_status() -> Dict[str, Any]:
        """Get comprehensive system status"""
        return await tori_integration.get_system_status()

# Export
__all__ = [
    'TORIIntegration',
    'tori_integration',
    'register_integration_endpoints',
    'register_integration_tools'
]
