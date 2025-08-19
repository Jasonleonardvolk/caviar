"""
ingest-bus/src/services/ingest_service.py

Ingestion microservice for processing documents and storing extracted
content in the Soliton Memory system.
"""

import os
import sys
import json
import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Soliton API client
from clients.soliton_client import SolitonClient
from clients.concept_mesh_client import ConceptMeshClient

# Import extraction modules
from extractors.pdf_extractor import PDFExtractor
from extractors.enhanced_extractor import EnhancedExtractor
from extractors.basic_extractor import BasicExtractor
from models.extracted_concept import ExtractedConcept
from models.ingest_session import IngestSession
from utils.metrics import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ingest_service")

# Constants
SOLITON_API_URL = os.environ.get("SOLITON_API_URL", "http://localhost:8002/api/soliton")
CONCEPT_MESH_URL = os.environ.get("CONCEPT_MESH_URL", "http://localhost:8003/api/mesh")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/uploads")
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "50"))
DEFAULT_MEMORY_STRENGTH = float(os.environ.get("DEFAULT_MEMORY_STRENGTH", "0.75"))

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


class SSEProgressReporter:
    """Server-Sent Events for real-time ingestion progress"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.event_queue = asyncio.Queue()
        
    async def send_progress(self, stage: str, progress: float, details: Dict[str, Any] = None):
        """Send progress update via SSE"""
        event = {
            'id': f"{self.session_id}_{int(time.time() * 1000)}",
            'event': 'ingestion_progress',
            'data': {
                'session_id': self.session_id,
                'stage': stage,
                'progress': progress,  # 0.0 to 1.0
                'details': details or {},
                'timestamp': datetime.now().isoformat()
            }
        }
        
        await self.event_queue.put(event)
        
    async def stream_events(self):
        """Stream events to client"""
        while True:
            event = await self.event_queue.get()
            yield f"id: {event['id']}\n"
            yield f"event: {event['event']}\n"
            yield f"data: {json.dumps(event['data'])}\n\n"


class IngestService:
    """
    Service for ingesting and processing documents, extracting concepts,
    and storing them in the Soliton Memory system.
    """

    def __init__(self):
        """Initialize the ingest service"""
        # Initialize clients
        self.soliton_client = SolitonClient(api_url=SOLITON_API_URL)
        self.concept_mesh_client = ConceptMeshClient(api_url=CONCEPT_MESH_URL)
        
        # Initialize extractors
        self.pdf_extractor = PDFExtractor()
        self.enhanced_extractor = EnhancedExtractor()
        self.basic_extractor = BasicExtractor()
        
        # Initialize metrics collector
        self.metrics = MetricsCollector(prefix="ingest_service")
        
        # Check if Soliton API is available
        self._check_soliton_connectivity()
        
        logger.info("🚀 Ingest Service initialized")
        logger.info(f"🌊 Soliton API URL: {SOLITON_API_URL}")
        logger.info(f"📂 Upload directory: {UPLOAD_DIR}")
    
    def _check_soliton_connectivity(self):
        """Check connectivity to Soliton API"""
        try:
            health = self.soliton_client.check_health()
            if health.get("status") == "operational":
                logger.info(f"✅ Soliton API is operational (engine: {health.get('engine', 'unknown')})")
                self.soliton_available = True
            else:
                logger.warning(f"⚠️ Soliton API health check returned non-operational status: {health}")
                self.soliton_available = False
        except Exception as e:
            logger.error(f"❌ Failed to connect to Soliton API: {str(e)}")
            self.soliton_available = False
            # Don't raise - we'll check again during actual operations
    
    async def process_file(self, 
                          file_path: str, 
                          user_id: str, 
                          session_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None,
                          progress_reporter: Optional[SSEProgressReporter] = None) -> IngestSession:
        """
        Process a file, extract concepts, and store in Soliton Memory
        
        Args:
            file_path: Path to the file to process
            user_id: ID of the user who owns the file
            session_id: Optional session ID for tracking
            metadata: Optional metadata about the file/session
            
        Returns:
            IngestSession object with processing results
        """
        start_time = time.time()
        
        # Create session ID if not provided
        if not session_id:
            session_id = f"ingest_{int(time.time())}_{os.urandom(4).hex()}"
        
        # Create metadata if not provided
        metadata = metadata or {}
        metadata.update({
            "ingestion_time": datetime.now().isoformat(),
            "filename": os.path.basename(file_path),
            "session_id": session_id,
            "user_id": user_id
        })
        
        # Create session object
        session = IngestSession(
            id=session_id,
            user_id=user_id,
            file_path=file_path,
            status="processing",
            metadata=metadata
        )
        
        # Report initial progress
        if progress_reporter:
            await progress_reporter.send_progress('initializing', 0.05, {
                'session_id': session_id,
                'filename': os.path.basename(file_path)
            })
        
        # Validate file
        if progress_reporter:
            await progress_reporter.send_progress('validating', 0.1)
            
        if not os.path.exists(file_path):
            session.status = "failed"
            session.error_message = f"File not found: {file_path}"
            logger.error(f"❌ {session.error_message}")
            if progress_reporter:
                await progress_reporter.send_progress('error', 0.1, {'error': session.error_message})
            return session
        
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            session.status = "failed"
            session.error_message = f"File too large: {file_size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)"
            logger.error(f"❌ {session.error_message}")
            return session
        
        try:
            # Check file type and process accordingly
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                # Extract text from PDF
                if progress_reporter:
                    await progress_reporter.send_progress('extracting_text', 0.2, {
                        'file_type': 'pdf',
                        'method': 'pdf_extractor'
                    })
                    
                pdf_text = self.pdf_extractor.extract_text(file_path)
                
                if not pdf_text or len(pdf_text.strip()) < 10:
                    session.status = "failed"
                    session.error_message = "Extracted text is empty or very short"
                    logger.warning(f"⚠️ {session.error_message} from {os.path.basename(file_path)}")
                    return session
                
                # Store the extracted text for reference
                session.extracted_text = pdf_text
                
                # Extract concepts using enhanced extractor first
                try:
                    logger.info(f"🔍 Using enhanced concept extraction for {os.path.basename(file_path)}")
                    if progress_reporter:
                        await progress_reporter.send_progress('extracting_concepts', 0.4, {
                            'method': 'enhanced',
                            'extractor': 'Jason\'s 4000-hour system'
                        })
                    
                    concepts = self.enhanced_extractor.extract_concepts(pdf_text, metadata=metadata)
                    
                    if concepts and len(concepts) > 0:
                        logger.info(f"✅ SUCCESS: Used Jason's 4000-hour sophisticated system")
                        session.extraction_method = "enhanced"
                        session.concepts = concepts
                        self.metrics.increment("enhanced_extraction_used")
                    else:
                        # Fall back to basic extraction
                        logger.warning("⚠️ Enhanced extraction returned no concepts, falling back to basic")
                        concepts = self.basic_extractor.extract_concepts(pdf_text, metadata=metadata)
                        
                        if concepts and len(concepts) > 0:
                            logger.info(f"🔄 Used fallback extraction: {len(concepts)} concepts found")
                            session.extraction_method = "basic"
                            session.concepts = concepts
                            self.metrics.increment("fallback_extraction_used")
                        else:
                            session.status = "failed"
                            session.error_message = "No concepts extracted from document"
                            logger.warning(f"⚠️ {session.error_message}")
                            return session
                
                except Exception as e:
                    logger.error(f"❌ Enhanced extraction failed: {str(e)}")
                    # Fall back to basic extraction
                    try:
                        logger.info(f"🔄 Using basic extraction after enhanced failed")
                        concepts = self.basic_extractor.extract_concepts(pdf_text, metadata=metadata)
                        
                        if concepts and len(concepts) > 0:
                            logger.info(f"🔄 Used fallback extraction: {len(concepts)} concepts found")
                            session.extraction_method = "basic"
                            session.concepts = concepts
                            self.metrics.increment("fallback_extraction_used")
                        else:
                            session.status = "failed"
                            session.error_message = "No concepts extracted from document"
                            logger.warning(f"⚠️ {session.error_message}")
                            return session
                    except Exception as inner_e:
                        session.status = "failed"
                        session.error_message = f"Both extraction methods failed: {str(e)} / {str(inner_e)}"
                        logger.error(f"❌ {session.error_message}")
                        return session
            
            else:
                session.status = "failed"
                session.error_message = f"Unsupported file type: {file_ext}"
                logger.error(f"❌ {session.error_message}")
                return session
            
            # Add Penrose similarity computation phase
            if progress_reporter:
                await progress_reporter.send_progress('computing_similarity', 0.6, {
                    'algorithm': 'penrose_O(n^2.32)',
                    'concept_count': len(session.concepts)
                })
            
            # Store concepts in Soliton Memory
            if progress_reporter:
                await progress_reporter.send_progress('storing_memories', 0.8, {
                    'target': 'soliton_memory',
                    'concept_count': len(session.concepts)
                })
                
            await self._store_concepts_in_soliton(session, progress_reporter)
            
            # Update session status
            processing_time = time.time() - start_time
            session.processing_time = processing_time
            session.status = "completed"
            
            logger.info(f"✅ Successfully processed {os.path.basename(file_path)} "
                        f"in {processing_time:.2f}s - extracted {len(session.concepts)} concepts, "
                        f"stored {session.stored_memory_count} memories")
            
            if progress_reporter:
                await progress_reporter.send_progress('complete', 1.0, {
                    'session_id': session_id,
                    'processing_time': processing_time,
                    'concepts_extracted': len(session.concepts),
                    'memories_stored': session.stored_memory_count
                })
            
            self.metrics.increment("successful_ingests")
            self.metrics.add("total_concepts_extracted", len(session.concepts))
            self.metrics.add("total_processing_time", processing_time)
            
            return session
            
        except Exception as e:
            session.status = "failed"
            session.error_message = f"Error processing file: {str(e)}"
            logger.error(f"❌ Error processing {os.path.basename(file_path)}: {str(e)}")
            self.metrics.increment("failed_ingests")
            return session
    
    async def _store_concepts_in_soliton(self, session: IngestSession, progress_reporter: Optional[SSEProgressReporter] = None) -> bool:
        """
        Store extracted concepts in Soliton Memory
        
        Args:
            session: IngestSession with concepts to store
            
        Returns:
            True if successful, False otherwise
        """
        if not self.soliton_available:
            # Check again - maybe it's back online
            self._check_soliton_connectivity()
            
            if not self.soliton_available:
                logger.error("❌ Cannot store in Soliton: API is unavailable")
                session.error_message = "Soliton Memory API unavailable - concepts extracted but not stored"
                return False
        
        if not session.concepts or len(session.concepts) == 0:
            logger.warning("⚠️ No concepts to store in Soliton")
            return False
        
        user_id = session.user_id
        
        # Initialize user memory if needed
        try:
            await self.soliton_client.initialize_user(user_id)
        except Exception as e:
            logger.error(f"❌ Failed to initialize Soliton user {user_id}: {str(e)}")
            session.error_message = f"Failed to initialize Soliton memory: {str(e)}"
            return False
        
        stored_count = 0
        failed_count = 0
        
        # Store each concept as a memory
        for concept in session.concepts:
            try:
                # Prepare tags based on metadata and concept properties
                tags = ["ingested", "document"]
                
                if session.extraction_method:
                    tags.append(session.extraction_method)
                
                if hasattr(concept, 'score'):
                    if concept.score > 0.8:
                        tags.append("high_confidence")
                    elif concept.score < 0.4:
                        tags.append("low_confidence")
                
                # Calculate memory strength based on concept score
                strength = DEFAULT_MEMORY_STRENGTH
                if hasattr(concept, 'score'):
                    strength = min(1.0, max(0.3, concept.score * DEFAULT_MEMORY_STRENGTH))
                
                # Generate unique memory ID using unified system
                try:
                    from python.core.unified_id_generator import generate_ingest_id
                    memory_id = generate_ingest_id(content, memory_metadata)
                except ImportError:
                    # Fallback to legacy format
                    memory_id = f"ingested_{concept.id}_{int(time.time())}"
                
                # Get concept text
                content = concept.text if hasattr(concept, 'text') else str(concept)
                
                # Prepare metadata
                concept_metadata = {}
                if hasattr(concept, 'metadata') and concept.metadata:
                    concept_metadata = concept.metadata
                
                memory_metadata = {
                    **session.metadata,
                    **concept_metadata,
                    "extraction_method": session.extraction_method,
                    "session_id": session.id
                }
                
                if hasattr(concept, 'score'):
                    memory_metadata["concept_score"] = concept.score
                
                if hasattr(concept, 'source_page'):
                    memory_metadata["source_page"] = concept.source_page
                
                # Store in Soliton Memory
                success = await self.soliton_client.store_memory(
                    user_id=user_id,
                    memory_id=memory_id,
                    content=content,
                    strength=strength,
                    tags=tags,
                    metadata=memory_metadata
                )
                
                if success:
                    stored_count += 1
                else:
                    logger.warning(f"⚠️ Failed to store concept {concept.id} in Soliton (no error thrown)")
                    failed_count += 1
            
            except Exception as e:
                logger.error(f"❌ Error storing concept {concept.id} in Soliton: {str(e)}")
                failed_count += 1
                # Continue with other concepts even if one fails
        
        # Update session with results
        session.stored_memory_count = stored_count
        session.failed_memory_count = failed_count
        
        if failed_count > 0:
            logger.warning(f"⚠️ Failed to store {failed_count}/{len(session.concepts)} concepts in Soliton")
        
        logger.info(f"🌊 Stored {stored_count}/{len(session.concepts)} concepts in Soliton Memory for user {user_id}")
        self.metrics.add("total_memories_stored", stored_count)
        
        # Track integration metrics
        session.integration_info = {
            "stored_count": stored_count,
            "failed_count": failed_count,
            "advanced_pipeline_used": session.extraction_method == "enhanced",
            "fallback_used": session.extraction_method == "basic",
            "soliton_available": self.soliton_available
        }
        
        return stored_count > 0
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get ingest service metrics"""
        return {
            **self.metrics.get_all(),
            "soliton_available": self.soliton_available
        }
    
    async def process_file_with_sse(self, 
                                  file_path: str, 
                                  user_id: str, 
                                  session_id: Optional[str] = None) -> Tuple[IngestSession, SSEProgressReporter]:
        """Process file with SSE progress reporting"""
        if not session_id:
            session_id = f"ingest_{int(time.time())}_{os.urandom(4).hex()}"
        
        progress_reporter = SSEProgressReporter(session_id)
        
        # Process in background task
        session = await self.process_file(
            file_path=file_path,
            user_id=user_id,
            session_id=session_id,
            progress_reporter=progress_reporter
        )
        
        return session, progress_reporter

# Create singleton instance
ingest_service = IngestService()

# Export for API usage
async def process_file(file_path: str, 
                     user_id: str, 
                     session_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process a file and store its contents in Soliton Memory"""
    session = await ingest_service.process_file(file_path, user_id, session_id, metadata)
    return session.to_dict()

async def get_metrics() -> Dict[str, Any]:
    """Get ingest service metrics"""
    return await ingest_service.get_metrics()
    


# Direct execution test
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ingest_service.py <file_path> <user_id>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    user_id = sys.argv[2]
    
    async def test_ingest():
        result = await process_file(file_path, user_id, metadata={"test": True})
        print(json.dumps(result, indent=2))
        
        metrics = await get_metrics()
        print(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    asyncio.run(test_ingest())
