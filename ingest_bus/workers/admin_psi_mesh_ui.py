"""
TORI Admin ψMesh Verification UI Backend
Provides REST API for concept verification, moderation, and trust management

Features:
- Concept verification scoring
- Source snippet analysis  
- Manual moderation (verify/reject)
- Trust overlay data
- Audit trail for admin actions
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

# Import TORI components
from psi_mesh_verification_layer import psi_verification, ConceptVerificationResult

logger = logging.getLogger("tori-admin.psi_mesh_ui")

# Request/Response Models
class ConceptModerationRequest(BaseModel):
    action: str  # "verify", "reject", "flag"
    reason: Optional[str] = None
    confidence_override: Optional[float] = None
    notes: Optional[str] = None

class ConceptModerationResponse(BaseModel):
    concept_id: str
    action: str
    previous_status: str
    new_status: str
    moderated_by: str
    moderated_at: str

class ConceptVerificationResponse(BaseModel):
    concept_id: str
    concept_name: str
    current_status: str
    integrity_score: float
    confidence_score: float
    verification_checks: Dict[str, Any]
    source_snippets: List[Dict[str, Any]]
    moderation_history: List[Dict[str, Any]]

class TrustOverlayResponse(BaseModel):
    total_concepts: int
    verified_count: int
    flagged_count: int
    rejected_count: int
    trust_score: float
    concept_heatmap: List[Dict[str, Any]]
    recent_moderations: List[Dict[str, Any]]

# Admin ψMesh API Router
admin_router = APIRouter(prefix="/api/v2/admin/psi-mesh", tags=["admin-psi-mesh"])

class PsiMeshAdminManager:
    """
    Backend manager for ψMesh admin verification UI
    """
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent / "concept-mesh-data"
        self.moderations_path = self.base_path / "moderations"
        self.trust_data_path = self.base_path / "trust_overlay"
        
        # Create directories
        self.moderations_path.mkdir(parents=True, exist_ok=True)
        self.trust_data_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for performance
        self.concept_cache = {}
        self.moderation_cache = {}
        
        logger.info("ψMesh Admin Manager initialized")
    
    def _load_concept_data(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Load concept data from storage"""
        try:
            # Search through ConceptMesh nodes for this concept
            nodes_path = self.base_path / "nodes"
            
            for concept_file in nodes_path.glob("*_concepts.json"):
                with open(concept_file, 'r') as f:
                    data = json.load(f)
                    
                    # Check if this file contains our concept
                    for node in data.get('concept_nodes', []):
                        if node.get('id') == concept_id or concept_id in node.get('name', ''):
                            return {
                                'concept_data': node,
                                'document_data': data.get('document_node', {}),
                                'source_file': str(concept_file)
                            }
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading concept data for {concept_id}: {e}")
            return None
    
    def _load_moderation_history(self, concept_id: str) -> List[Dict[str, Any]]:
        """Load moderation history for a concept"""
        try:
            moderation_file = self.moderations_path / f"{concept_id}_moderations.json"
            
            if moderation_file.exists():
                with open(moderation_file, 'r') as f:
                    return json.load(f).get('moderations', [])
            
            return []
            
        except Exception as e:
            logger.error(f"Error loading moderation history for {concept_id}: {e}")
            return []
    
    def _save_moderation(self, concept_id: str, moderation: Dict[str, Any]):
        """Save moderation action"""
        try:
            moderation_file = self.moderations_path / f"{concept_id}_moderations.json"
            
            # Load existing moderations
            moderations = []
            if moderation_file.exists():
                with open(moderation_file, 'r') as f:
                    moderations = json.load(f).get('moderations', [])
            
            # Add new moderation
            moderations.append(moderation)
            
            # Save updated moderations
            moderation_data = {
                'concept_id': concept_id,
                'moderations': moderations,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(moderation_file, 'w') as f:
                json.dump(moderation_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving moderation for {concept_id}: {e}")
    
    def _find_source_snippets(self, concept_name: str, concept_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find source text snippets for concept verification"""
        snippets = []
        
        try:
            # Get source document information
            source_document = concept_data.get('source_document', '')
            
            # Look for original content in ScholarSphere
            scholar_sphere_path = self.base_path / "scholar_sphere"
            
            for content_file in scholar_sphere_path.glob("*_content.txt"):
                try:
                    with open(content_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Find occurrences of concept in text
                    concept_lower = concept_name.lower()
                    content_lower = content.lower()
                    
                    start = 0
                    while True:
                        pos = content_lower.find(concept_lower, start)
                        if pos == -1:
                            break
                        
                        # Extract context around the match
                        context_start = max(0, pos - 100)
                        context_end = min(len(content), pos + len(concept_name) + 100)
                        context = content[context_start:context_end]
                        
                        snippets.append({
                            'text': context,
                            'match_position': pos - context_start,
                            'match_length': len(concept_name),
                            'source_file': str(content_file),
                            'confidence': 1.0 if concept_lower in context.lower() else 0.8
                        })
                        
                        start = pos + 1
                        
                        # Limit to 10 snippets
                        if len(snippets) >= 10:
                            break
                    
                    if len(snippets) >= 10:
                        break
                        
                except Exception as file_error:
                    logger.warning(f"Error reading content file {content_file}: {file_error}")
                    continue
        
        except Exception as e:
            logger.error(f"Error finding source snippets: {e}")
        
        return snippets[:10]  # Return top 10 snippets
    
    async def get_concept_verification(self, concept_id: str) -> ConceptVerificationResponse:
        """Get comprehensive verification data for a concept"""
        # Load concept data
        concept_data = self._load_concept_data(concept_id)
        if not concept_data:
            raise HTTPException(status_code=404, detail=f"Concept {concept_id} not found")
        
        concept_info = concept_data['concept_data']
        concept_name = concept_info.get('name', 'Unknown')
        
        # Get current verification status
        current_status = concept_info.get('verification_status', 'unverified')
        integrity_score = concept_info.get('integrity_score', 0.0)
        confidence_score = concept_info.get('confidence', 0.0)
        
        # Get verification checks
        verification_checks = concept_info.get('verification_checks', {})
        
        # Find source snippets
        source_snippets = self._find_source_snippets(concept_name, concept_data)
        
        # Load moderation history
        moderation_history = self._load_moderation_history(concept_id)
        
        return ConceptVerificationResponse(
            concept_id=concept_id,
            concept_name=concept_name,
            current_status=current_status,
            integrity_score=integrity_score,
            confidence_score=confidence_score,
            verification_checks=verification_checks,
            source_snippets=source_snippets,
            moderation_history=moderation_history
        )
    
    async def moderate_concept(self, concept_id: str, request: ConceptModerationRequest, 
                             admin_user: str = "admin") -> ConceptModerationResponse:
        """Apply moderation action to a concept"""
        
        # Load concept data
        concept_data = self._load_concept_data(concept_id)
        if not concept_data:
            raise HTTPException(status_code=404, detail=f"Concept {concept_id} not found")
        
        concept_info = concept_data['concept_data']
        previous_status = concept_info.get('verification_status', 'unverified')
        
        # Determine new status based on action
        new_status = previous_status
        if request.action == "verify":
            new_status = "verified"
        elif request.action == "reject":
            new_status = "rejected"
        elif request.action == "flag":
            new_status = "flagged"
        
        # Create moderation record
        moderation_record = {
            'action': request.action,
            'previous_status': previous_status,
            'new_status': new_status,
            'reason': request.reason,
            'confidence_override': request.confidence_override,
            'notes': request.notes,
            'moderated_by': admin_user,
            'moderated_at': datetime.now().isoformat(),
            'concept_name': concept_info.get('name', 'Unknown')
        }
        
        # Save moderation
        self._save_moderation(concept_id, moderation_record)
        
        # Update concept data in original file
        try:
            source_file = concept_data['source_file']
            
            with open(source_file, 'r') as f:
                file_data = json.load(f)
            
            # Update the specific concept node
            for i, node in enumerate(file_data.get('concept_nodes', [])):
                if node.get('id') == concept_id:
                    file_data['concept_nodes'][i]['verification_status'] = new_status
                    file_data['concept_nodes'][i]['last_moderated_at'] = datetime.now().isoformat()
                    file_data['concept_nodes'][i]['moderated_by'] = admin_user
                    
                    if request.confidence_override is not None:
                        file_data['concept_nodes'][i]['confidence'] = request.confidence_override
                    
                    break
            
            # Save updated file
            with open(source_file, 'w') as f:
                json.dump(file_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error updating concept file: {e}")
            # Continue anyway since moderation was saved
        
        return ConceptModerationResponse(
            concept_id=concept_id,
            action=request.action,
            previous_status=previous_status,
            new_status=new_status,
            moderated_by=admin_user,
            moderated_at=moderation_record['moderated_at']
        )
    
    async def get_trust_overlay(self) -> TrustOverlayResponse:
        """Get trust overlay data for admin dashboard"""
        
        # Scan all concepts
        total_concepts = 0
        verified_count = 0
        flagged_count = 0
        rejected_count = 0
        concept_heatmap = []
        
        nodes_path = self.base_path / "nodes"
        
        for concept_file in nodes_path.glob("*_concepts.json"):
            try:
                with open(concept_file, 'r') as f:
                    data = json.load(f)
                
                for node in data.get('concept_nodes', []):
                    total_concepts += 1
                    
                    status = node.get('verification_status', 'unverified')
                    if status == 'verified':
                        verified_count += 1
                    elif status == 'flagged':
                        flagged_count += 1
                    elif status == 'rejected':
                        rejected_count += 1
                    
                    # Add to heatmap
                    concept_heatmap.append({
                        'id': node.get('id', ''),
                        'name': node.get('name', 'Unknown'),
                        'status': status,
                        'integrity_score': node.get('integrity_score', 0.0),
                        'confidence': node.get('confidence', 0.0),
                        'document': data.get('document_node', {}).get('title', 'Unknown')
                    })
                    
            except Exception as e:
                logger.warning(f"Error reading concept file {concept_file}: {e}")
                continue
        
        # Calculate trust score
        if total_concepts > 0:
            trust_score = (verified_count + (flagged_count * 0.5)) / total_concepts
        else:
            trust_score = 0.0
        
        # Get recent moderations
        recent_moderations = []
        for moderation_file in self.moderations_path.glob("*_moderations.json"):
            try:
                with open(moderation_file, 'r') as f:
                    data = json.load(f)
                
                for moderation in data.get('moderations', []):
                    moderation['concept_id'] = data.get('concept_id', '')
                    recent_moderations.append(moderation)
                    
            except Exception as e:
                logger.warning(f"Error reading moderation file {moderation_file}: {e}")
                continue
        
        # Sort by timestamp and take most recent
        recent_moderations.sort(key=lambda x: x.get('moderated_at', ''), reverse=True)
        recent_moderations = recent_moderations[:20]  # Last 20 moderations
        
        return TrustOverlayResponse(
            total_concepts=total_concepts,
            verified_count=verified_count,
            flagged_count=flagged_count,
            rejected_count=rejected_count,
            trust_score=trust_score,
            concept_heatmap=concept_heatmap,
            recent_moderations=recent_moderations
        )

# Global admin manager
admin_manager = PsiMeshAdminManager()

# API Endpoints
@admin_router.get("/concepts/{concept_id}/verify", response_model=ConceptVerificationResponse)
async def verify_concept_endpoint(concept_id: str):
    """Get concept verification data for admin review"""
    return await admin_manager.get_concept_verification(concept_id)

@admin_router.post("/concepts/{concept_id}/moderate", response_model=ConceptModerationResponse)
async def moderate_concept_endpoint(concept_id: str, request: ConceptModerationRequest):
    """Apply moderation action to a concept"""
    # In a real implementation, you'd extract admin_user from authentication
    admin_user = "admin_user"  # TODO: Extract from JWT/session
    return await admin_manager.moderate_concept(concept_id, request, admin_user)

@admin_router.get("/trust-overlay", response_model=TrustOverlayResponse)
async def get_trust_overlay_endpoint():
    """Get trust overlay data for admin dashboard"""
    return await admin_manager.get_trust_overlay()

@admin_router.get("/concepts/search")
async def search_concepts(
    query: str = Query(..., description="Search query for concepts"),
    status: Optional[str] = Query(None, description="Filter by verification status"),
    limit: int = Query(50, description="Maximum number of results")
):
    """Search concepts for admin interface"""
    
    results = []
    nodes_path = admin_manager.base_path / "nodes"
    
    query_lower = query.lower()
    
    for concept_file in nodes_path.glob("*_concepts.json"):
        try:
            with open(concept_file, 'r') as f:
                data = json.load(f)
            
            for node in data.get('concept_nodes', []):
                concept_name = node.get('name', '').lower()
                concept_status = node.get('verification_status', 'unverified')
                
                # Apply filters
                if query_lower in concept_name:
                    if status is None or concept_status == status:
                        results.append({
                            'id': node.get('id', ''),
                            'name': node.get('name', 'Unknown'),
                            'status': concept_status,
                            'integrity_score': node.get('integrity_score', 0.0),
                            'confidence': node.get('confidence', 0.0),
                            'document': data.get('document_node', {}).get('title', 'Unknown')
                        })
                        
                        if len(results) >= limit:
                            break
            
            if len(results) >= limit:
                break
                
        except Exception as e:
            logger.warning(f"Error searching in file {concept_file}: {e}")
            continue
    
    return {
        'query': query,
        'status_filter': status,
        'total_results': len(results),
        'results': results
    }

@admin_router.post("/bulk-moderate")
async def bulk_moderate_concepts(
    concept_ids: List[str],
    action: str,
    reason: Optional[str] = None
):
    """Apply moderation action to multiple concepts"""
    
    results = []
    admin_user = "admin_user"  # TODO: Extract from authentication
    
    for concept_id in concept_ids:
        try:
            request = ConceptModerationRequest(
                action=action,
                reason=reason
            )
            
            result = await admin_manager.moderate_concept(concept_id, request, admin_user)
            results.append({
                'concept_id': concept_id,
                'status': 'success',
                'result': result.dict()
            })
            
        except Exception as e:
            results.append({
                'concept_id': concept_id,
                'status': 'failed',
                'error': str(e)
            })
    
    return {
        'action': action,
        'total_concepts': len(concept_ids),
        'successful': len([r for r in results if r['status'] == 'success']),
        'failed': len([r for r in results if r['status'] == 'failed']),
        'results': results
    }

@admin_router.get("/stats")
async def get_admin_stats():
    """Get comprehensive admin statistics"""
    
    # Get trust overlay data
    trust_data = await admin_manager.get_trust_overlay()
    
    # Calculate additional stats
    total_documents = 0
    total_segments = 0
    
    # Count documents
    scholar_sphere_path = admin_manager.base_path / "scholar_sphere"
    total_documents = len(list(scholar_sphere_path.glob("*_archive.json")))
    
    # Count memory segments
    braid_memory_path = admin_manager.base_path / "braid_memory"
    for memory_file in braid_memory_path.glob("*_memory.json"):
        try:
            with open(memory_file, 'r') as f:
                data = json.load(f)
                total_segments += len(data.get('memory_entries', []))
        except:
            continue
    
    return {
        'concepts': {
            'total': trust_data.total_concepts,
            'verified': trust_data.verified_count,
            'flagged': trust_data.flagged_count,
            'rejected': trust_data.rejected_count,
            'trust_score': trust_data.trust_score
        },
        'documents': {
            'total': total_documents
        },
        'memory_segments': {
            'total': total_segments
        },
        'system_health': {
            'verification_rate': trust_data.verified_count / max(1, trust_data.total_concepts),
            'flagging_rate': trust_data.flagged_count / max(1, trust_data.total_concepts),
            'rejection_rate': trust_data.rejected_count / max(1, trust_data.total_concepts)
        }
    }
