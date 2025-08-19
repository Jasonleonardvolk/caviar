"""
TORI Third-Party Ingestion Gateway
Secure API gateway for external organizations to upload documents

Features:
- Organization-scoped API keys
- Configurable ingestion scopes (memory, archive, both)
- Webhook notifications
- Complete audit trails
- Sandboxed access controls
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, Header, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import hashlib
import uuid
import hmac

# Import TORI components
from live_reingest_manager import reingest_manager
from ingestRouter import route_document_complete

logger = logging.getLogger("tori-gateway.third_party")

# Request/Response Models
class ThirdPartyUploadRequest(BaseModel):
    scope: str = "both"  # "memory", "archive", "both"
    metadata: Optional[Dict[str, Any]] = None
    webhook_url: Optional[str] = None
    tags: Optional[List[str]] = None

class ThirdPartyUploadResponse(BaseModel):
    upload_id: str
    doc_id: str
    org_id: str
    status: str
    processing_started_at: str
    scope: str
    estimated_completion: str

class OrganizationInfo(BaseModel):
    org_id: str
    org_name: str
    api_key_hash: str
    permissions: List[str]
    upload_quota: Dict[str, Any]
    created_at: str
    last_used_at: Optional[str]

class WebhookNotification(BaseModel):
    upload_id: str
    doc_id: str
    org_id: str
    status: str
    processing_results: Dict[str, Any]
    completed_at: str

# Third-Party Gateway Router
gateway_router = APIRouter(prefix="/api/v2/gateway", tags=["third-party-gateway"])

# Security scheme
security = HTTPBearer()

class ThirdPartyGatewayManager:
    """
    Manages third-party document ingestion with security and scoping
    """
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent / "concept-mesh-data"
        self.gateway_path = self.base_path / "third_party_gateway"
        self.organizations_path = self.gateway_path / "organizations"
        self.uploads_path = self.gateway_path / "uploads"
        self.audit_path = self.gateway_path / "audit"
        
        # Create directories
        for path in [self.gateway_path, self.organizations_path, self.uploads_path, self.audit_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Load organizations
        self.organizations = self._load_organizations()
        
        # Upload tracking
        self.active_uploads = {}
        
        logger.info("Third-Party Gateway Manager initialized")
        logger.info(f"Loaded {len(self.organizations)} organizations")
    
    def _load_organizations(self) -> Dict[str, Dict[str, Any]]:
        """Load registered organizations"""
        organizations = {}
        
        try:
            for org_file in self.organizations_path.glob("*.json"):
                with open(org_file, 'r') as f:
                    org_data = json.load(f)
                    org_id = org_data.get('org_id')
                    if org_id:
                        organizations[org_id] = org_data
            
            # Create default test organization if none exist
            if not organizations:
                test_org = self._create_test_organization()
                organizations[test_org['org_id']] = test_org
                
        except Exception as e:
            logger.error(f"Error loading organizations: {e}")
        
        return organizations
    
    def _create_test_organization(self) -> Dict[str, Any]:
        """Create a test organization for development"""
        test_org = {
            'org_id': 'test_org_001',
            'org_name': 'Test Organization',
            'api_key': 'test_api_key_12345',  # In production, this would be securely generated
            'api_key_hash': hashlib.sha256('test_api_key_12345'.encode()).hexdigest(),
            'permissions': ['upload', 'memory', 'archive'],
            'upload_quota': {
                'daily_limit': 100,
                'file_size_limit': 50 * 1024 * 1024,  # 50MB
                'concurrent_uploads': 5
            },
            'webhook_settings': {
                'enabled': True,
                'retry_count': 3,
                'timeout_seconds': 30
            },
            'created_at': datetime.now().isoformat(),
            'last_used_at': None,
            'status': 'active'
        }
        
        # Save test organization
        org_file = self.organizations_path / f"{test_org['org_id']}.json"
        with open(org_file, 'w') as f:
            json.dump(test_org, f, indent=2)
        
        logger.info(f"Created test organization: {test_org['org_id']}")
        return test_org
    
    def _validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return organization info"""
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        for org_id, org_data in self.organizations.items():
            if org_data.get('api_key_hash') == api_key_hash:
                # Update last used timestamp
                org_data['last_used_at'] = datetime.now().isoformat()
                self._save_organization(org_data)
                return org_data
        
        return None
    
    def _save_organization(self, org_data: Dict[str, Any]):
        """Save organization data"""
        try:
            org_file = self.organizations_path / f"{org_data['org_id']}.json"
            with open(org_file, 'w') as f:
                json.dump(org_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving organization data: {e}")
    
    def _check_upload_quota(self, org_data: Dict[str, Any], file_size: int) -> Dict[str, Any]:
        """Check if upload is within quota limits"""
        quota = org_data.get('upload_quota', {})
        org_id = org_data['org_id']
        
        # Check file size limit
        max_file_size = quota.get('file_size_limit', 10 * 1024 * 1024)  # Default 10MB
        if file_size > max_file_size:
            return {
                'allowed': False,
                'reason': f'File size {file_size} exceeds limit {max_file_size}'
            }
        
        # Check daily upload limit
        daily_limit = quota.get('daily_limit', 50)
        today = datetime.now().date()
        
        # Count today's uploads
        today_uploads = 0
        for upload_file in self.uploads_path.glob(f"{org_id}_*.json"):
            try:
                with open(upload_file, 'r') as f:
                    upload_data = json.load(f)
                    upload_date = datetime.fromisoformat(upload_data['started_at']).date()
                    if upload_date == today:
                        today_uploads += 1
            except:
                continue
        
        if today_uploads >= daily_limit:
            return {
                'allowed': False,
                'reason': f'Daily upload limit {daily_limit} exceeded'
            }
        
        # Check concurrent uploads
        concurrent_limit = quota.get('concurrent_uploads', 3)
        active_count = len([u for u in self.active_uploads.values() 
                           if u.get('org_id') == org_id and u.get('status') == 'processing'])
        
        if active_count >= concurrent_limit:
            return {
                'allowed': False,
                'reason': f'Concurrent upload limit {concurrent_limit} exceeded'
            }
        
        return {'allowed': True}
    
    def _generate_scoped_doc_id(self, org_id: str, filename: str) -> str:
        """Generate organization-scoped document ID"""
        timestamp = int(datetime.now().timestamp())
        safe_filename = ''.join(c for c in filename if c.isalnum() or c in '._-')[:50]
        return f"{org_id}::{timestamp}::{safe_filename}"
    
    async def process_third_party_upload(self, 
                                       file: UploadFile,
                                       org_data: Dict[str, Any],
                                       request: ThirdPartyUploadRequest) -> ThirdPartyUploadResponse:
        """Process third-party document upload"""
        
        upload_id = str(uuid.uuid4())
        org_id = org_data['org_id']
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Check quota
        quota_check = self._check_upload_quota(org_data, file_size)
        if not quota_check['allowed']:
            raise HTTPException(status_code=429, detail=quota_check['reason'])
        
        # Generate scoped document ID
        doc_id = self._generate_scoped_doc_id(org_id, file.filename or 'unknown')
        
        # Determine file type
        file_type = self._detect_file_type(file.filename or '')
        
        # Create upload record
        upload_record = {
            'upload_id': upload_id,
            'doc_id': doc_id,
            'org_id': org_id,
            'filename': file.filename,
            'file_type': file_type,
            'file_size': file_size,
            'scope': request.scope,
            'metadata': request.metadata or {},
            'tags': request.tags or [],
            'webhook_url': request.webhook_url,
            'started_at': datetime.now().isoformat(),
            'status': 'processing'
        }
        
        # Save upload record
        upload_file = self.uploads_path / f"{upload_id}.json"
        with open(upload_file, 'w') as f:
            json.dump(upload_record, f, indent=2)
        
        # Add to active uploads
        self.active_uploads[upload_id] = upload_record
        
        # Start async processing
        asyncio.create_task(self._process_upload_async(
            upload_id, file_content, file_type, upload_record
        ))
        
        # Log audit event
        await self._log_audit_event(org_id, 'upload_started', {
            'upload_id': upload_id,
            'doc_id': doc_id,
            'file_size': file_size,
            'scope': request.scope
        })
        
        return ThirdPartyUploadResponse(
            upload_id=upload_id,
            doc_id=doc_id,
            org_id=org_id,
            status='processing',
            processing_started_at=upload_record['started_at'],
            scope=request.scope,
            estimated_completion=(datetime.now() + timedelta(minutes=5)).isoformat()
        )
    
    async def _process_upload_async(self, upload_id: str, file_content: bytes, 
                                  file_type: str, upload_record: Dict[str, Any]):
        """Process the upload asynchronously"""
        try:
            logger.info(f"Starting async processing for upload {upload_id}")
            
            # Prepare metadata for TORI processing
            tori_metadata = {
                'third_party_upload': True,
                'org_id': upload_record['org_id'],
                'upload_id': upload_id,
                'scope': upload_record['scope'],
                'tags': upload_record['tags'],
                **upload_record.get('metadata', {})
            }
            
            # Process through TORI system
            processing_result = await route_document_complete(
                file_content=file_content,
                file_type=file_type,
                filename=upload_record['filename'],
                metadata=tori_metadata
            )
            
            # Apply scope restrictions
            filtered_result = self._apply_scope_restrictions(
                processing_result, upload_record['scope'], upload_record['org_id']
            )
            
            # Update upload record
            upload_record['status'] = 'completed'
            upload_record['completed_at'] = datetime.now().isoformat()
            upload_record['processing_result'] = filtered_result
            upload_record['processing_duration'] = processing_result.get('processing_duration', 0)
            
            # Save updated record
            upload_file = self.uploads_path / f"{upload_id}.json"
            with open(upload_file, 'w') as f:
                json.dump(upload_record, f, indent=2)
            
            # Send webhook notification if configured
            if upload_record.get('webhook_url'):
                await self._send_webhook_notification(upload_record)
            
            # Log completion
            await self._log_audit_event(upload_record['org_id'], 'upload_completed', {
                'upload_id': upload_id,
                'doc_id': upload_record['doc_id'],
                'status': 'completed',
                'processing_duration': upload_record['processing_duration']
            })
            
            logger.info(f"Async processing completed for upload {upload_id}")
            
        except Exception as e:
            logger.exception(f"Error in async processing for upload {upload_id}: {e}")
            
            # Update record with error
            upload_record['status'] = 'failed'
            upload_record['error'] = str(e)
            upload_record['failed_at'] = datetime.now().isoformat()
            
            # Save error record
            upload_file = self.uploads_path / f"{upload_id}.json"
            with open(upload_file, 'w') as f:
                json.dump(upload_record, f, indent=2)
            
            # Log error
            await self._log_audit_event(upload_record['org_id'], 'upload_failed', {
                'upload_id': upload_id,
                'error': str(e)
            })
        
        finally:
            # Remove from active uploads
            if upload_id in self.active_uploads:
                del self.active_uploads[upload_id]
    
    def _apply_scope_restrictions(self, processing_result: Dict[str, Any], 
                                scope: str, org_id: str) -> Dict[str, Any]:
        """Apply scope restrictions to processing results"""
        filtered_result = {
            'routing_id': processing_result.get('routing_id'),
            'status': processing_result.get('status'),
            'processing_duration': processing_result.get('processing_duration'),
            'scope_applied': scope,
            'org_id': org_id
        }
        
        # Include results based on scope
        if scope in ['memory', 'both']:
            filtered_result['memory_integration'] = {
                'concept_mesh': processing_result.get('stages', {}).get('concept_mesh', {}),
                'braid_memory': processing_result.get('stages', {}).get('braid_memory', {}),
                'verification': processing_result.get('stages', {}).get('verification', {})
            }
        
        if scope in ['archive', 'both']:
            filtered_result['archival_integration'] = {
                'scholar_sphere': processing_result.get('stages', {}).get('scholar_sphere', {}),
                'psi_arc': processing_result.get('stages', {}).get('psi_arc', {})
            }
        
        # Always include basic processing info
        filtered_result['document_processing'] = processing_result.get('stages', {}).get('document_processing', {})
        
        return filtered_result
    
    async def _send_webhook_notification(self, upload_record: Dict[str, Any]):
        """Send webhook notification about upload completion"""
        try:
            import aiohttp
            
            webhook_url = upload_record['webhook_url']
            
            notification = WebhookNotification(
                upload_id=upload_record['upload_id'],
                doc_id=upload_record['doc_id'],
                org_id=upload_record['org_id'],
                status=upload_record['status'],
                processing_results=upload_record.get('processing_result', {}),
                completed_at=upload_record.get('completed_at', datetime.now().isoformat())
            )
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=notification.dict(),
                    timeout=30
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook notification sent successfully for upload {upload_record['upload_id']}")
                    else:
                        logger.warning(f"Webhook notification failed: {response.status}")
        
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
    
    def _detect_file_type(self, filename: str) -> str:
        """Detect file type from filename"""
        file_extension = filename.lower().split('.')[-1] if '.' in filename else 'unknown'
        
        type_mapping = {
            'pdf': 'pdf',
            'docx': 'docx',
            'doc': 'docx',
            'csv': 'csv',
            'pptx': 'pptx',
            'ppt': 'pptx',
            'xlsx': 'xlsx',
            'xls': 'xlsx',
            'json': 'json',
            'txt': 'txt',
            'md': 'md',
            'markdown': 'md'
        }
        
        return type_mapping.get(file_extension, 'txt')
    
    async def _log_audit_event(self, org_id: str, event_type: str, event_data: Dict[str, Any]):
        """Log audit event"""
        try:
            audit_record = {
                'org_id': org_id,
                'event_type': event_type,
                'event_data': event_data,
                'timestamp': datetime.now().isoformat(),
                'audit_id': str(uuid.uuid4())
            }
            
            # Save to daily audit file
            today = datetime.now().strftime('%Y-%m-%d')
            audit_file = self.audit_path / f"{org_id}_{today}_audit.json"
            
            # Load existing events
            events = []
            if audit_file.exists():
                with open(audit_file, 'r') as f:
                    data = json.load(f)
                    events = data.get('events', [])
            
            # Add new event
            events.append(audit_record)
            
            # Save updated events
            audit_data = {
                'org_id': org_id,
                'date': today,
                'events': events,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(audit_file, 'w') as f:
                json.dump(audit_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
    
    async def get_upload_status(self, upload_id: str, org_id: str) -> Dict[str, Any]:
        """Get upload status"""
        upload_file = self.uploads_path / f"{upload_id}.json"
        
        if not upload_file.exists():
            raise HTTPException(status_code=404, detail="Upload not found")
        
        with open(upload_file, 'r') as f:
            upload_data = json.load(f)
        
        # Verify organization access
        if upload_data.get('org_id') != org_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return upload_data
    
    async def list_uploads(self, org_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """List uploads for an organization"""
        uploads = []
        
        for upload_file in self.uploads_path.glob("*.json"):
            try:
                with open(upload_file, 'r') as f:
                    upload_data = json.load(f)
                
                if upload_data.get('org_id') == org_id:
                    # Remove sensitive processing details
                    safe_upload = {
                        'upload_id': upload_data['upload_id'],
                        'doc_id': upload_data['doc_id'],
                        'filename': upload_data['filename'],
                        'file_type': upload_data['file_type'],
                        'file_size': upload_data['file_size'],
                        'scope': upload_data['scope'],
                        'status': upload_data['status'],
                        'started_at': upload_data['started_at'],
                        'completed_at': upload_data.get('completed_at'),
                        'processing_duration': upload_data.get('processing_duration')
                    }
                    uploads.append(safe_upload)
                    
                    if len(uploads) >= limit:
                        break
                        
            except Exception as e:
                logger.warning(f"Error reading upload file {upload_file}: {e}")
                continue
        
        # Sort by started_at descending
        uploads.sort(key=lambda x: x['started_at'], reverse=True)
        
        return uploads

# Global gateway manager
gateway_manager = ThirdPartyGatewayManager()

# Authentication dependency
async def authenticate_organization(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Authenticate organization via API key"""
    api_key = credentials.credentials
    
    org_data = gateway_manager._validate_api_key(api_key)
    if not org_data:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if org_data.get('status') != 'active':
        raise HTTPException(status_code=403, detail="Organization account is not active")
    
    return org_data

# API Endpoints
@gateway_router.post("/upload", response_model=ThirdPartyUploadResponse)
async def third_party_upload(
    file: UploadFile = File(...),
    scope: str = Form("both"),
    metadata: str = Form("{}"),
    webhook_url: Optional[str] = Form(None),
    tags: str = Form("[]"),
    org_data: Dict[str, Any] = Depends(authenticate_organization)
):
    """Upload document through third-party gateway"""
    
    # Parse JSON parameters
    try:
        parsed_metadata = json.loads(metadata) if metadata else {}
        parsed_tags = json.loads(tags) if tags else []
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in metadata or tags")
    
    # Validate scope
    if scope not in ['memory', 'archive', 'both']:
        raise HTTPException(status_code=400, detail="Invalid scope. Must be 'memory', 'archive', or 'both'")
    
    # Create request object
    request = ThirdPartyUploadRequest(
        scope=scope,
        metadata=parsed_metadata,
        webhook_url=webhook_url,
        tags=parsed_tags
    )
    
    return await gateway_manager.process_third_party_upload(file, org_data, request)

@gateway_router.get("/uploads/{upload_id}")
async def get_upload_status(
    upload_id: str,
    org_data: Dict[str, Any] = Depends(authenticate_organization)
):
    """Get status of a specific upload"""
    return await gateway_manager.get_upload_status(upload_id, org_data['org_id'])

@gateway_router.get("/uploads")
async def list_uploads(
    limit: int = 50,
    org_data: Dict[str, Any] = Depends(authenticate_organization)
):
    """List uploads for the authenticated organization"""
    uploads = await gateway_manager.list_uploads(org_data['org_id'], limit)
    
    return {
        'org_id': org_data['org_id'],
        'total_uploads': len(uploads),
        'uploads': uploads
    }

@gateway_router.get("/organization/info")
async def get_organization_info(
    org_data: Dict[str, Any] = Depends(authenticate_organization)
):
    """Get organization information and quotas"""
    
    # Calculate current quota usage
    today = datetime.now().date()
    today_uploads = 0
    active_uploads = 0
    
    for upload_file in gateway_manager.uploads_path.glob("*.json"):
        try:
            with open(upload_file, 'r') as f:
                upload_info = json.load(f)
            
            if upload_info.get('org_id') == org_data['org_id']:
                # Count today's uploads
                upload_date = datetime.fromisoformat(upload_info['started_at']).date()
                if upload_date == today:
                    today_uploads += 1
                
                # Count active uploads
                if upload_info.get('status') == 'processing':
                    active_uploads += 1
                    
        except:
            continue
    
    quota = org_data.get('upload_quota', {})
    
    return {
        'org_id': org_data['org_id'],
        'org_name': org_data['org_name'],
        'status': org_data['status'],
        'permissions': org_data['permissions'],
        'quota_usage': {
            'daily_uploads': today_uploads,
            'daily_limit': quota.get('daily_limit', 50),
            'active_uploads': active_uploads,
            'concurrent_limit': quota.get('concurrent_uploads', 3),
            'file_size_limit': quota.get('file_size_limit', 10 * 1024 * 1024)
        },
        'created_at': org_data['created_at'],
        'last_used_at': org_data.get('last_used_at')
    }

@gateway_router.get("/health")
async def gateway_health():
    """Gateway health check"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'organizations': len(gateway_manager.organizations),
        'active_uploads': len(gateway_manager.active_uploads),
        'supported_file_types': ['pdf', 'docx', 'csv', 'pptx', 'xlsx', 'json', 'txt', 'md'],
        'supported_scopes': ['memory', 'archive', 'both']
    }

# Test endpoint for development
@gateway_router.get("/test/credentials")
async def get_test_credentials():
    """Get test credentials for development (remove in production)"""
    test_org = gateway_manager.organizations.get('test_org_001')
    
    if test_org:
        return {
            'org_id': test_org['org_id'],
            'api_key': test_org.get('api_key'),  # Only for development
            'note': 'Use this API key in Authorization header as Bearer token'
        }
    else:
        return {'error': 'No test organization found'}
