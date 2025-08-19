"""
Updated Ingest Bus Routes to Support All File Types
Enhanced queue route with complete TORI integration
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional, Dict, Any
import json
import logging
from datetime import datetime
import uuid

# Import TORI integration components
from workers.integration_coordinator import coordinator, process_document_with_full_integration
from workers.enhanced_extract import DocumentProcessor
from models.schemas import IngestJob, IngestRequest, IngestStatus, DocumentType

logger = logging.getLogger("tori-ingest.routes")
router = APIRouter()

# File type mapping
FILE_TYPE_MAP = {
    '.pdf': 'pdf',
    '.docx': 'docx',
    '.doc': 'docx',
    '.csv': 'csv',
    '.pptx': 'pptx',
    '.xlsx': 'xlsx',
    '.json': 'json',
    '.txt': 'txt',
    '.md': 'md',
    '.markdown': 'md'
}

@router.post("/queue/enhanced")
async def queue_document_enhanced(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),  # JSON string of tags
    metadata: Optional[str] = Form(None),  # JSON string of metadata
    integration_options: Optional[str] = Form(None),  # JSON string of integration options
    callback_url: Optional[str] = Form(None)
):
    """
    Enhanced document queue endpoint supporting all file types
    
    Supports: PDF, DOCX, CSV, PPTX, XLSX, JSON, TXT, MD
    Integrates with: ConceptMesh, ψMesh, Ghost Collective, ScholarSphere
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Determine file type
        file_extension = None
        if file.filename:
            file_extension = '.' + file.filename.split('.')[-1].lower()
        
        document_type = FILE_TYPE_MAP.get(file_extension, 'txt')
        
        # Parse optional parameters
        parsed_tags = []
        if tags:
            try:
                parsed_tags = json.loads(tags)
            except json.JSONDecodeError:
                parsed_tags = [tags]  # Single tag as string
        
        parsed_metadata = {}
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                parsed_metadata = {"raw_metadata": metadata}
        
        parsed_integration_options = {}
        if integration_options:
            try:
                parsed_integration_options = json.loads(integration_options)
            except json.JSONDecodeError:
                logger.warning(f"Invalid integration_options JSON: {integration_options}")
        
        # Default integration options
        default_options = {
            'concept_mesh': True,
            'psi_mesh': True,
            'ghost_collective': True,
            'scholar_sphere': True,
            'integrity_verification': True
        }
        default_options.update(parsed_integration_options)
        
        # Create ingest request
        ingest_request = IngestRequest(
            document_type=document_type,
            title=title or file.filename or "Untitled Document",
            tags=parsed_tags,
            metadata=parsed_metadata,
            callback_url=callback_url
        )
        
        # Create ingest job
        job = IngestJob(
            id=job_id,
            request=ingest_request,
            status=IngestStatus.QUEUED,
            queued_at=datetime.now(),
            percent_complete=0.0,
            chunks_processed=0,
            concepts_mapped=0,
            chunk_ids=[],
            concept_ids=[]
        )
        
        # Read file content
        file_content = await file.read()
        
        logger.info(f"Queued enhanced document processing: {file.filename} ({document_type}) - job {job_id}")
        
        # Process document with full TORI integration
        try:
            processing_result = await process_document_with_full_integration(
                file_path=None,
                file_content=file_content,
                file_type=document_type,
                job=job,
                options=default_options
            )
            
            # Update job with results
            if processing_result.get('status') == 'completed':
                job.status = IngestStatus.COMPLETED
                job.completed_at = datetime.now()
                
                # Extract concept and chunk IDs from results
                if 'stages' in processing_result:
                    if 'parsing' in processing_result['stages']:
                        job.concepts_mapped = processing_result['stages']['parsing'].get('concepts_extracted', 0)
                    
                    if 'concept_mesh' in processing_result['stages']:
                        job.concept_ids = [f"concept_{i}" for i in range(job.concepts_mapped)]
                
                logger.info(f"Enhanced document processing completed: job {job_id}")
            else:
                job.status = IngestStatus.FAILED
                logger.error(f"Enhanced document processing failed: job {job_id} - {processing_result.get('error')}")
            
            return {
                "job_id": job_id,
                "status": job.status.value,
                "message": f"Document {file.filename} processed successfully with TORI integration",
                "document_type": document_type,
                "integration_options": default_options,
                "processing_result": processing_result,
                "file_size": len(file_content),
                "concepts_extracted": job.concepts_mapped,
                "stages_completed": processing_result.get('summary', {}).get('stages_completed', 0),
                "integrity_score": processing_result.get('summary', {}).get('integrity_score', 0.0)
            }
            
        except Exception as processing_error:
            logger.exception(f"Error in enhanced document processing: {processing_error}")
            job.status = IngestStatus.FAILED
            
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(processing_error),
                "message": f"Failed to process document {file.filename}",
                "document_type": document_type
            }
    
    except Exception as e:
        logger.exception(f"Error queueing enhanced document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue document: {str(e)}")

@router.get("/queue/supported_types")
async def get_supported_file_types():
    """
    Get list of supported file types and their capabilities
    """
    return {
        "supported_types": list(FILE_TYPE_MAP.values()),
        "file_extensions": list(FILE_TYPE_MAP.keys()),
        "capabilities": {
            "pdf": {
                "text_extraction": True,
                "structure_detection": True,
                "metadata_extraction": True,
                "concept_extraction": True
            },
            "docx": {
                "text_extraction": True,
                "structure_detection": True,
                "table_extraction": True,
                "heading_detection": True,
                "metadata_extraction": True,
                "concept_extraction": True
            },
            "csv": {
                "data_extraction": True,
                "statistical_analysis": True,
                "column_analysis": True,
                "concept_extraction": True
            },
            "pptx": {
                "text_extraction": True,
                "slide_structure": True,
                "concept_extraction": True
            },
            "xlsx": {
                "data_extraction": True,
                "multi_sheet_support": True,
                "statistical_analysis": True,
                "concept_extraction": True
            },
            "json": {
                "structured_data_extraction": True,
                "schema_analysis": True,
                "nested_structure_support": True,
                "concept_extraction": True
            },
            "txt": {
                "text_extraction": True,
                "concept_extraction": True
            },
            "md": {
                "text_extraction": True,
                "structure_detection": True,
                "heading_detection": True,
                "concept_extraction": True
            }
        },
        "integration_systems": [
            "ConceptMesh",
            "ψMesh",
            "Ghost Collective",
            "ScholarSphere"
        ]
    }

@router.get("/queue/integration_options")
async def get_integration_options():
    """
    Get available integration options for document processing
    """
    return {
        "integration_options": {
            "concept_mesh": {
                "description": "Integrate with ConceptMesh knowledge graph",
                "default": True,
                "required": False
            },
            "psi_mesh": {
                "description": "Create semantic associations in ψMesh",
                "default": True,
                "required": False
            },
            "ghost_collective": {
                "description": "Process with Ghost Collective AI personas",
                "default": True,
                "required": False
            },
            "scholar_sphere": {
                "description": "Archive in ScholarSphere repository",
                "default": True,
                "required": False
            },
            "integrity_verification": {
                "description": "Verify extraction integrity against source",
                "default": True,
                "required": False
            }
        },
        "processing_stages": [
            "Document parsing and concept extraction",
            "ψMesh semantic association creation",
            "Extraction integrity verification",
            "Ghost Collective persona analysis",
            "ConceptMesh knowledge graph integration",
            "ScholarSphere archival"
        ]
    }

@router.post("/queue/batch")
async def queue_batch_documents(
    files: list[UploadFile] = File(...),
    batch_metadata: Optional[str] = Form(None),
    integration_options: Optional[str] = Form(None)
):
    """
    Queue multiple documents for batch processing
    """
    try:
        batch_id = str(uuid.uuid4())
        
        # Parse batch metadata
        parsed_batch_metadata = {}
        if batch_metadata:
            try:
                parsed_batch_metadata = json.loads(batch_metadata)
            except json.JSONDecodeError:
                parsed_batch_metadata = {"raw_metadata": batch_metadata}
        
        # Parse integration options
        parsed_integration_options = {}
        if integration_options:
            try:
                parsed_integration_options = json.loads(integration_options)
            except json.JSONDecodeError:
                logger.warning(f"Invalid batch integration_options JSON: {integration_options}")
        
        job_results = []
        
        for file in files:
            # Process each file individually
            file_content = await file.read()
            
            # Determine file type
            file_extension = None
            if file.filename:
                file_extension = '.' + file.filename.split('.')[-1].lower()
            
            document_type = FILE_TYPE_MAP.get(file_extension, 'txt')
            
            # Create individual job
            job_id = str(uuid.uuid4())
            
            ingest_request = IngestRequest(
                document_type=document_type,
                title=file.filename or "Untitled Document",
                tags=parsed_batch_metadata.get('tags', []),
                metadata={**parsed_batch_metadata, 'batch_id': batch_id},
                callback_url=None
            )
            
            job = IngestJob(
                id=job_id,
                request=ingest_request,
                status=IngestStatus.QUEUED,
                queued_at=datetime.now(),
                percent_complete=0.0,
                chunks_processed=0,
                concepts_mapped=0,
                chunk_ids=[],
                concept_ids=[]
            )
            
            # Process with TORI integration
            try:
                processing_result = await process_document_with_full_integration(
                    file_path=None,
                    file_content=file_content,
                    file_type=document_type,
                    job=job,
                    options=parsed_integration_options
                )
                
                job_results.append({
                    "job_id": job_id,
                    "filename": file.filename,
                    "document_type": document_type,
                    "status": processing_result.get('status', 'unknown'),
                    "concepts_extracted": processing_result.get('summary', {}).get('concepts_extracted', 0),
                    "integrity_score": processing_result.get('summary', {}).get('integrity_score', 0.0)
                })
                
            except Exception as file_error:
                logger.exception(f"Error processing file {file.filename}: {file_error}")
                job_results.append({
                    "job_id": job_id,
                    "filename": file.filename,
                    "document_type": document_type,
                    "status": "failed",
                    "error": str(file_error)
                })
        
        successful_jobs = sum(1 for result in job_results if result.get('status') == 'completed')
        
        return {
            "batch_id": batch_id,
            "total_files": len(files),
            "successful_jobs": successful_jobs,
            "failed_jobs": len(files) - successful_jobs,
            "job_results": job_results,
            "batch_metadata": parsed_batch_metadata,
            "integration_options": parsed_integration_options
        }
        
    except Exception as e:
        logger.exception(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
