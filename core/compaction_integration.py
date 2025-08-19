"""
TORI Compaction Integration
Hooks for integrating compaction with the main TORI system
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Import compaction components
try:
    from scripts.compact_all_meshes import MeshCompactor
    from core.metrics import MetricsCollector
    COMPACTION_AVAILABLE = True
except ImportError:
    COMPACTION_AVAILABLE = False

logger = logging.getLogger(__name__)


class CompactionIntegration:
    """
    Integration layer for TORI compaction
    Can be called from API routes or background tasks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not COMPACTION_AVAILABLE:
            logger.warning("Compaction components not available")
            self.enabled = False
            return
            
        self.config = config or {}
        self.enabled = self.config.get('compaction_enabled', True)
        
        if self.enabled:
            self.compactor = MeshCompactor(self.config.get('compaction', {}))
            self.collector = MetricsCollector()
            logger.info("Compaction integration initialized")
    
    async def compact_if_needed(self, scope: str, scope_id: str) -> bool:
        """
        Check and compact a specific mesh if needed
        
        Args:
            scope: "user" or "group"
            scope_id: ID of the mesh
            
        Returns:
            True if compaction was performed
        """
        if not self.enabled:
            return False
            
        try:
            # Check if compaction needed
            metrics = self.collector.needs_compact(scope, scope_id)
            
            if metrics.needs_compaction:
                logger.info(f"Auto-compacting {scope}:{scope_id} - {metrics.reason}")
                success = await self.compactor._compact_single_mesh(scope, scope_id)
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Auto-compaction failed for {scope}:{scope_id}: {e}")
            return False
    
    async def run_scheduled_compaction(self) -> Dict[str, Any]:
        """
        Run scheduled compaction for all meshes
        Called by cron/scheduler
        """
        if not self.enabled:
            return {'error': 'Compaction disabled'}
            
        logger.info("Running scheduled compaction")
        return await self.compactor.compact_all_meshes()
    
    async def get_compaction_status(self) -> Dict[str, Any]:
        """Get current compaction status"""
        if not self.enabled:
            return {'error': 'Compaction disabled'}
            
        return self.collector.get_compaction_report()
    
    async def create_backup_snapshot(self, scope: str, scope_id: str, 
                                   tag: Optional[str] = None) -> Optional[Path]:
        """
        Create a backup snapshot for disaster recovery
        
        Args:
            scope: "user" or "group"
            scope_id: ID of the mesh
            tag: Optional tag for the snapshot
            
        Returns:
            Path to snapshot file or None
        """
        if not self.enabled:
            return None
            
        try:
            from python.core.scoped_concept_mesh import ScopedConceptMesh
            
            mesh = ScopedConceptMesh.get_instance(scope, scope_id)
            snapshot_path = await self.compactor._create_snapshot(scope, scope_id, mesh)
            
            # Add tag if provided
            if tag and snapshot_path.exists():
                tagged_path = snapshot_path.parent / f"{snapshot_path.stem}_{tag}{snapshot_path.suffix}"
                snapshot_path.rename(tagged_path)
                snapshot_path = tagged_path
            
            logger.info(f"Created backup snapshot: {snapshot_path}")
            return snapshot_path
            
        except Exception as e:
            logger.error(f"Failed to create backup snapshot: {e}")
            return None
    
    async def restore_from_backup(self, scope: str, scope_id: str,
                                snapshot_path: Optional[Path] = None) -> bool:
        """
        Restore mesh from backup snapshot
        
        Args:
            scope: "user" or "group"
            scope_id: ID of the mesh
            snapshot_path: Specific snapshot or None for latest
            
        Returns:
            True if successful
        """
        if not self.enabled:
            return False
            
        return await self.compactor.restore_from_snapshot(scope, scope_id, snapshot_path)
    
    def register_api_routes(self, app):
        """
        Register compaction API routes
        
        Args:
            app: FastAPI application instance
        """
        if not self.enabled:
            return
            
        from fastapi import APIRouter, HTTPException, BackgroundTasks
        from pydantic import BaseModel
        
        router = APIRouter(prefix="/api/compaction", tags=["compaction"])
        
        class CompactRequest(BaseModel):
            scope: str
            scope_id: str
            force: bool = False
        
        @router.get("/status")
        async def get_status():
            """Get compaction status for all meshes"""
            return await self.get_compaction_status()
        
        @router.post("/compact")
        async def compact_mesh(request: CompactRequest, background_tasks: BackgroundTasks):
            """Compact a specific mesh"""
            if request.force:
                # Run in background
                background_tasks.add_task(
                    self.compactor._compact_single_mesh,
                    request.scope, request.scope_id
                )
                return {"status": "scheduled", "scope": request.scope, "scope_id": request.scope_id}
            else:
                # Check if needed first
                metrics = self.collector.needs_compact(request.scope, request.scope_id)
                if metrics.needs_compaction:
                    background_tasks.add_task(
                        self.compactor._compact_single_mesh,
                        request.scope, request.scope_id
                    )
                    return {"status": "scheduled", "reason": metrics.reason}
                else:
                    return {"status": "not_needed", "reason": "Compaction not required"}
        
        @router.post("/compact-all")
        async def compact_all(background_tasks: BackgroundTasks, force: bool = False):
            """Compact all meshes"""
            background_tasks.add_task(self.compactor.compact_all_meshes, force)
            return {"status": "scheduled", "message": "Full compaction started"}
        
        @router.post("/snapshot/{scope}/{scope_id}")
        async def create_snapshot(scope: str, scope_id: str, tag: Optional[str] = None):
            """Create backup snapshot"""
            path = await self.create_backup_snapshot(scope, scope_id, tag)
            if path:
                return {"status": "success", "path": str(path)}
            else:
                raise HTTPException(status_code=500, detail="Failed to create snapshot")
        
        @router.post("/restore/{scope}/{scope_id}")
        async def restore_snapshot(scope: str, scope_id: str, snapshot_path: Optional[str] = None):
            """Restore from snapshot"""
            path = Path(snapshot_path) if snapshot_path else None
            success = await self.restore_from_backup(scope, scope_id, path)
            
            if success:
                return {"status": "success", "message": f"Restored {scope}:{scope_id}"}
            else:
                raise HTTPException(status_code=500, detail="Failed to restore from snapshot")
        
        # Register routes
        app.include_router(router)
        logger.info("Registered compaction API routes")
    
    def start_background_monitor(self, interval_seconds: int = 3600):
        """
        Start background monitoring task
        Checks for compaction needs periodically
        
        Args:
            interval_seconds: Check interval (default: 1 hour)
        """
        if not self.enabled:
            return
            
        async def monitor_loop():
            while True:
                try:
                    # Check all meshes
                    report = self.collector.get_compaction_report()
                    
                    if report['needs_compaction'] > 0:
                        logger.info(f"Background monitor: {report['needs_compaction']} meshes need compaction")
                        
                        # Auto-compact if configured
                        if self.config.get('auto_compact', False):
                            for detail in report['details']:
                                await self.compact_if_needed(detail['scope'], detail['scope_id'])
                    
                    # Wait for next check
                    await asyncio.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Background monitor error: {e}")
                    await asyncio.sleep(60)  # Brief pause on error
        
        # Start background task
        asyncio.create_task(monitor_loop())
        logger.info(f"Started background compaction monitor (interval: {interval_seconds}s)")


# Singleton instance
_compaction_integration = None

def get_compaction_integration(config: Optional[Dict[str, Any]] = None) -> CompactionIntegration:
    """Get singleton compaction integration"""
    global _compaction_integration
    if _compaction_integration is None:
        _compaction_integration = CompactionIntegration(config)
    return _compaction_integration


# FastAPI integration example
def setup_compaction_routes(app, config: Optional[Dict[str, Any]] = None):
    """
    Setup compaction routes on FastAPI app
    Call this from your main API setup
    """
    integration = get_compaction_integration(config)
    integration.register_api_routes(app)
    
    # Optionally start background monitor
    if config and config.get('compaction', {}).get('background_monitor', False):
        integration.start_background_monitor()
