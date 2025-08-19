"""
TORI Live Re-Ingestion System
Handles document updates with version tracking and system synchronization

Features:
- Document version detection via hash comparison
- Automatic archival of previous versions
- Live updates to all TORI systems
- Assistant notification hooks
- Complete audit trail
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import hashlib
import uuid

# Import TORI components
from production_file_handlers import file_handlers, ParsedPayload
from psi_mesh_verification_layer import verify_concept_extraction_integrity
from ingestRouter import tori_router

logger = logging.getLogger("tori-reingest.live_system")

class LiveReingestionManager:
    """
    Manages live re-ingestion of updated documents with version tracking
    """
    
    def __init__(self):
        self.document_registry = {}  # doc_id -> document metadata
        self.version_history = {}    # doc_id -> list of versions
        self.reingest_hooks = []     # callbacks for re-ingestion events
        
        # Storage paths
        self.base_path = Path(__file__).parent.parent / "concept-mesh-data"
        self.versions_path = self.base_path / "versions"
        self.registry_path = self.base_path / "document_registry.json"
        
        # Create directories
        self.versions_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry
        self._load_document_registry()
        
        logger.info("Live Re-ingestion Manager initialized")
    
    def _load_document_registry(self):
        """Load existing document registry"""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    self.document_registry = data.get('documents', {})
                    self.version_history = data.get('versions', {})
                logger.info(f"Loaded {len(self.document_registry)} documents from registry")
        except Exception as e:
            logger.warning(f"Could not load document registry: {e}")
    
    def _save_document_registry(self):
        """Save document registry"""
        try:
            registry_data = {
                'documents': self.document_registry,
                'versions': self.version_history,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save document registry: {e}")
    
    def register_reingest_hook(self, callback):
        """Register callback for re-ingestion events"""
        self.reingest_hooks.append(callback)
    
    async def check_document_changes(self, doc_id: str, file_content: bytes, 
                                   force: bool = False) -> Dict[str, Any]:
        """
        Check if document has changed and needs re-ingestion
        
        Args:
            doc_id: Document identifier
            file_content: New file content
            force: Force re-ingestion even if no changes
            
        Returns:
            Dictionary with change detection results
        """
        new_hash = hashlib.sha256(file_content).hexdigest()
        
        # Check if document exists
        if doc_id not in self.document_registry:
            return {
                'needs_reingest': True,
                'reason': 'new_document',
                'current_hash': None,
                'new_hash': new_hash
            }
        
        current_doc = self.document_registry[doc_id]
        current_hash = current_doc.get('current_hash')
        
        if force:
            return {
                'needs_reingest': True,
                'reason': 'forced_reingest',
                'current_hash': current_hash,
                'new_hash': new_hash
            }
        
        if current_hash != new_hash:
            return {
                'needs_reingest': True,
                'reason': 'content_changed',
                'current_hash': current_hash,
                'new_hash': new_hash
            }
        
        return {
            'needs_reingest': False,
            'reason': 'no_changes',
            'current_hash': current_hash,
            'new_hash': new_hash
        }
    
    async def reingest_document(self, doc_id: str, file_content: bytes, 
                              file_type: str, filename: str,
                              metadata: Dict[str, Any] = None,
                              force: bool = False) -> Dict[str, Any]:
        """
        Perform live re-ingestion of an updated document
        
        Args:
            doc_id: Document identifier
            file_content: Updated file content
            file_type: Document type
            filename: Filename
            metadata: Additional metadata
            force: Force re-ingestion
            
        Returns:
            Re-ingestion results
        """
        start_time = datetime.now()
        
        logger.info(f"Starting re-ingestion for document: {doc_id}")
        
        # Check if re-ingestion is needed
        change_check = await self.check_document_changes(doc_id, file_content, force)
        
        if not change_check['needs_reingest']:
            logger.info(f"No re-ingestion needed for {doc_id}: {change_check['reason']}")
            return {
                'status': 'skipped',
                'reason': change_check['reason'],
                'doc_id': doc_id
            }
        
        reingest_results = {
            'doc_id': doc_id,
            'version_id': f"{doc_id}::{int(start_time.timestamp())}",
            'started_at': start_time.isoformat(),
            'change_detection': change_check,
            'stages': {},
            'status': 'processing'
        }
        
        try:
            # Stage 1: Archive previous version
            if doc_id in self.document_registry:
                logger.info(f"Archiving previous version of {doc_id}")
                archive_result = await self._archive_previous_version(doc_id)
                reingest_results['stages']['archive_previous'] = archive_result
            
            # Stage 2: Process new document version
            logger.info(f"Processing new version of {doc_id}")
            
            # Use the existing TORI router for complete processing
            processing_result = await tori_router.route_document(
                file_content, file_type, filename, metadata or {}
            )
            
            reingest_results['stages']['process_new_version'] = processing_result
            
            # Stage 3: Update system links
            logger.info(f"Updating system links for {doc_id}")
            link_update_result = await self._update_system_links(
                doc_id, processing_result, change_check
            )
            reingest_results['stages']['update_links'] = link_update_result
            
            # Stage 4: Update document registry
            logger.info(f"Updating document registry for {doc_id}")
            registry_result = await self._update_document_registry(
                doc_id, reingest_results, file_content, metadata or {}
            )
            reingest_results['stages']['update_registry'] = registry_result
            
            # Stage 5: Notify systems and hooks
            logger.info(f"Sending re-ingestion notifications for {doc_id}")
            notification_result = await self._send_reingest_notifications(
                doc_id, reingest_results
            )
            reingest_results['stages']['notifications'] = notification_result
            
            # Final status
            reingest_results['status'] = 'completed'
            reingest_results['completed_at'] = datetime.now().isoformat()
            reingest_results['processing_duration'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Re-ingestion completed for {doc_id} in {reingest_results['processing_duration']:.2f}s")
            
        except Exception as e:
            logger.exception(f"Error during re-ingestion of {doc_id}: {e}")
            reingest_results['status'] = 'failed'
            reingest_results['error'] = str(e)
            reingest_results['failed_at'] = datetime.now().isoformat()
        
        return reingest_results
    
    async def _archive_previous_version(self, doc_id: str) -> Dict[str, Any]:
        """Archive the previous version of a document"""
        try:
            if doc_id not in self.document_registry:
                return {'status': 'skipped', 'reason': 'no_previous_version'}
            
            current_doc = self.document_registry[doc_id]
            
            # Create version archive entry
            version_entry = {
                'doc_id': doc_id,
                'version_hash': current_doc.get('current_hash'),
                'archived_at': datetime.now().isoformat(),
                'routing_id': current_doc.get('routing_id'),
                'system_uuids': current_doc.get('system_uuids', {}),
                'metadata': current_doc.get('metadata', {})
            }
            
            # Add to version history
            if doc_id not in self.version_history:
                self.version_history[doc_id] = []
            
            self.version_history[doc_id].append(version_entry)
            
            # Mark previous ConceptMesh entries as superseded
            await self._mark_concepts_superseded(doc_id, version_entry)
            
            # Archive BraidMemory segments
            await self._archive_memory_segments(doc_id, version_entry)
            
            self._save_document_registry()
            
            return {
                'status': 'completed',
                'version_archived': version_entry['version_hash'][:8],
                'total_versions': len(self.version_history[doc_id])
            }
            
        except Exception as e:
            logger.exception(f"Error archiving previous version of {doc_id}: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _mark_concepts_superseded(self, doc_id: str, version_entry: Dict[str, Any]):
        """Mark previous ConceptMesh entries as superseded"""
        try:
            system_uuids = version_entry.get('system_uuids', {})
            concept_mesh_uuid = system_uuids.get('concept_mesh')
            
            if concept_mesh_uuid:
                # Load existing concept mesh data
                concept_file = self.base_path / "nodes" / f"{concept_mesh_uuid}_concepts.json"
                
                if concept_file.exists():
                    with open(concept_file, 'r') as f:
                        concept_data = json.load(f)
                    
                    # Mark as superseded
                    concept_data['superseded_at'] = datetime.now().isoformat()
                    concept_data['superseded_by'] = 'pending_new_version'
                    concept_data['version_status'] = 'archived'
                    
                    # Save updated data
                    with open(concept_file, 'w') as f:
                        json.dump(concept_data, f, indent=2)
                    
                    logger.info(f"Marked ConceptMesh {concept_mesh_uuid} as superseded")
        
        except Exception as e:
            logger.warning(f"Could not mark concepts as superseded: {e}")
    
    async def _archive_memory_segments(self, doc_id: str, version_entry: Dict[str, Any]):
        """Archive BraidMemory segments for previous version"""
        try:
            system_uuids = version_entry.get('system_uuids', {})
            braid_memory_uuid = system_uuids.get('braid_memory')
            
            if braid_memory_uuid:
                # Load existing memory data
                memory_file = self.base_path / "braid_memory" / f"{braid_memory_uuid}_memory.json"
                
                if memory_file.exists():
                    with open(memory_file, 'r') as f:
                        memory_data = json.load(f)
                    
                    # Move to version history
                    archived_file = self.versions_path / f"{braid_memory_uuid}_memory_archived.json"
                    
                    memory_data['archived_at'] = datetime.now().isoformat()
                    memory_data['version_status'] = 'archived'
                    
                    with open(archived_file, 'w') as f:
                        json.dump(memory_data, f, indent=2)
                    
                    logger.info(f"Archived BraidMemory {braid_memory_uuid}")
        
        except Exception as e:
            logger.warning(f"Could not archive memory segments: {e}")
    
    async def _update_system_links(self, doc_id: str, processing_result: Dict[str, Any], 
                                 change_check: Dict[str, Any]) -> Dict[str, Any]:
        """Update cross-system links for the new version"""
        try:
            new_system_uuids = processing_result.get('system_uuids', {})
            
            # Update ConceptMesh links to point to new version
            await self._update_concept_mesh_links(doc_id, new_system_uuids, change_check)
            
            # Update PsiArc trajectory with re-ingestion info
            await self._update_psiarc_trajectory(doc_id, new_system_uuids, change_check)
            
            return {
                'status': 'completed',
                'updated_systems': list(new_system_uuids.keys()),
                'new_system_uuids': new_system_uuids
            }
            
        except Exception as e:
            logger.exception(f"Error updating system links: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _update_concept_mesh_links(self, doc_id: str, system_uuids: Dict[str, str], 
                                       change_check: Dict[str, Any]):
        """Update ConceptMesh with re-ingestion metadata"""
        try:
            concept_mesh_uuid = system_uuids.get('concept_mesh')
            if not concept_mesh_uuid:
                return
            
            concept_file = self.base_path / "nodes" / f"{concept_mesh_uuid}_concepts.json"
            
            if concept_file.exists():
                with open(concept_file, 'r') as f:
                    concept_data = json.load(f)
                
                # Add re-ingestion metadata
                concept_data['reingest_metadata'] = {
                    'original_doc_id': doc_id,
                    'reingest_timestamp': datetime.now().isoformat(),
                    'change_reason': change_check['reason'],
                    'previous_hash': change_check.get('current_hash'),
                    'new_hash': change_check['new_hash']
                }
                
                with open(concept_file, 'w') as f:
                    json.dump(concept_data, f, indent=2)
        
        except Exception as e:
            logger.warning(f"Could not update ConceptMesh links: {e}")
    
    async def _update_psiarc_trajectory(self, doc_id: str, system_uuids: Dict[str, str], 
                                      change_check: Dict[str, Any]):
        """Update PsiArc with re-ingestion trajectory"""
        try:
            psiarc_uuid = system_uuids.get('psi_arc')
            if not psiarc_uuid:
                return
            
            psiarc_file = Path(__file__).parent.parent / "psiarc_logs" / f"{psiarc_uuid}_trajectory.json"
            
            if psiarc_file.exists():
                with open(psiarc_file, 'r') as f:
                    trajectory_data = json.load(f)
                
                # Add re-ingestion event
                if 'reingest_events' not in trajectory_data:
                    trajectory_data['reingest_events'] = []
                
                trajectory_data['reingest_events'].append({
                    'timestamp': datetime.now().isoformat(),
                    'change_reason': change_check['reason'],
                    'hash_change': {
                        'from': change_check.get('current_hash'),
                        'to': change_check['new_hash']
                    }
                })
                
                with open(psiarc_file, 'w') as f:
                    json.dump(trajectory_data, f, indent=2)
        
        except Exception as e:
            logger.warning(f"Could not update PsiArc trajectory: {e}")
    
    async def _update_document_registry(self, doc_id: str, reingest_results: Dict[str, Any], 
                                      file_content: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Update the document registry with new version info"""
        try:
            new_hash = hashlib.sha256(file_content).hexdigest()
            processing_result = reingest_results['stages']['process_new_version']
            
            # Update registry entry
            self.document_registry[doc_id] = {
                'doc_id': doc_id,
                'current_hash': new_hash,
                'routing_id': processing_result.get('routing_id'),
                'system_uuids': processing_result.get('system_uuids', {}),
                'last_reingested_at': reingest_results['started_at'],
                'version_count': len(self.version_history.get(doc_id, [])) + 1,
                'metadata': metadata
            }
            
            self._save_document_registry()
            
            return {
                'status': 'completed',
                'new_hash': new_hash[:8],
                'version_count': self.document_registry[doc_id]['version_count']
            }
            
        except Exception as e:
            logger.exception(f"Error updating document registry: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _send_reingest_notifications(self, doc_id: str, reingest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Send notifications about re-ingestion completion"""
        try:
            notifications_sent = 0
            
            # Call registered hooks
            for hook in self.reingest_hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook(doc_id, reingest_results)
                    else:
                        hook(doc_id, reingest_results)
                    notifications_sent += 1
                except Exception as hook_error:
                    logger.warning(f"Re-ingestion hook failed: {hook_error}")
            
            # Log to LoopRecord
            loop_record_data = {
                'event_type': 'document_reingest',
                'doc_id': doc_id,
                'version_id': reingest_results['version_id'],
                'timestamp': datetime.now().isoformat(),
                'change_reason': reingest_results['change_detection']['reason'],
                'processing_duration': reingest_results.get('processing_duration', 0),
                'status': reingest_results['status']
            }
            
            loop_record_path = self.base_path / "loop_records" / f"reingest_{doc_id}_{int(datetime.now().timestamp())}.json"
            with open(loop_record_path, 'w') as f:
                json.dump(loop_record_data, f, indent=2)
            
            return {
                'status': 'completed',
                'hooks_called': notifications_sent,
                'loop_record_saved': True
            }
            
        except Exception as e:
            logger.exception(f"Error sending re-ingestion notifications: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_document_versions(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get version history for a document"""
        return self.version_history.get(doc_id, [])
    
    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get current document information"""
        return self.document_registry.get(doc_id)

# Global re-ingestion manager
reingest_manager = LiveReingestionManager()

# Assistant notification hook example
async def assistant_notification_hook(doc_id: str, reingest_results: Dict[str, Any]):
    """Notify assistant about document updates"""
    try:
        # This would integrate with the Ghost Collective or assistant system
        logger.info(f"📚 Assistant Notification: Document '{doc_id}' has been updated")
        logger.info(f"   Change reason: {reingest_results['change_detection']['reason']}")
        logger.info(f"   Processing status: {reingest_results['status']}")
        
        # In a real implementation, this would:
        # 1. Notify the Ghost Collective about knowledge updates
        # 2. Trigger memory refresh in BraidMemory
        # 3. Update the assistant's context awareness
        
    except Exception as e:
        logger.error(f"Assistant notification failed: {e}")

# Register the assistant notification hook
reingest_manager.register_reingest_hook(assistant_notification_hook)
