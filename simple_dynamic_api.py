#!/usr/bin/env python3
"""
🚀 DYNAMIC SIMPLE EXTRACTION API - Auto-port discovery
Automatically finds available port and saves config for SvelteKit discovery
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import socket
import json
import time
import logging
from pathlib import Path

# Add the ingest_pdf directory to path
sys.path.append(str(Path(__file__).parent / "ingest_pdf"))

try:
    from extractConceptsFromDocument import extractConceptsFromDocument
    print("✅ Successfully imported universal extraction module")
except ImportError as e:
    print(f"❌ Failed to import extraction module: {e}")
    print("Make sure you're in the correct directory with ingest_pdf/")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="TORI Simple Concept Extraction API",
    description="Clean, simple Python API for concept extraction",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000", 
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "healthy",
        "service": "TORI Simple Concept Extraction API",
        "version": "2.0.0",
        "type": "simple_clean_api",
        "ready": True
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test extraction
        test_text = "machine learning artificial intelligence neural networks optimization algorithms"
        test_concepts = extractConceptsFromDocument(test_text, threshold=0.0)
        
        return {
            "status": "healthy",
            "extraction_engine": "simple_universal_pipeline",
            "modules_loaded": True,
            "test_extraction": len(test_concepts) > 0,
            "test_concepts_found": len(test_concepts),
            "api_type": "simple_clean",
            "ready": True
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "ready": False
        }

@app.post("/extract")
async def extract_concepts_from_upload(file: UploadFile = File(...)):
    """
    🧬 SIMPLE CONCEPT EXTRACTION
    Clean, direct extraction without complex wrappers
    """
    start_time = time.time()
    
    try:
        logger.info(f"🧬 [SIMPLE] Processing upload: {file.filename}")
        
        # Read file content
        content = await file.read()
        logger.info(f"📁 [SIMPLE] File read: {len(content)} bytes")
        
        # Extract text based on file type
        if file.filename.lower().endswith('.pdf'):
            # Simple PDF text extraction
            text_content = content.decode('utf-8', errors='ignore')
            text = ''.join(char if ord(char) >= 32 or char in '\n\r\t' else ' ' for char in text_content)
        else:
            text = content.decode('utf-8')
        
        logger.info(f"📄 [SIMPLE] Extracted text: {len(text)} characters")
        
        if len(text) < 50:
            raise HTTPException(status_code=400, detail="Text too short")
        
        # 🧬 DIRECT CONCEPT EXTRACTION (same as working test)
        logger.info("🧬 [SIMPLE] Starting direct concept extraction...")
        concepts = extractConceptsFromDocument(text, threshold=0.0)
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ [SIMPLE] Extraction complete: {len(concepts)} concepts in {processing_time:.2f}s")
        
        # Return in format expected by SvelteKit
        response = {
            "success": True,
            "filename": file.filename,
            "extraction_method": "simple_python_pipeline",
            "processing_time_seconds": round(processing_time, 3),
            "text_length": len(text),
            "concept_count": len(concepts),
            "concepts": concepts,  # Full concepts list
            "concept_names": [c.get('name', str(c)) if isinstance(c, dict) else str(c) for c in concepts],  # Names for SvelteKit
            "simple_extraction": True,
            "timestamp": time.time()
        }
        
        logger.info(f"🎯 [SIMPLE] SUCCESS: {len(concepts)} concepts extracted")
        
        # Log first few concepts for debugging
        if concepts:
            logger.info("📋 [SIMPLE] First few concepts:")
            for i, concept in enumerate(concepts[:5]):
                if isinstance(concept, dict):
                    name = concept.get('name', 'unnamed')
                else:
                    name = str(concept)
                logger.info(f"   {i+1}. {name}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [SIMPLE] Extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Simple extraction failed: {str(e)}")

def find_available_port(start_port=8002, max_attempts=10):
    """Find the first available port starting from start_port"""
    for i in range(max_attempts):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    raise Exception(f"No available ports found in range {start_port}-{start_port + max_attempts}")

def save_port_config(port):
    """Save port config for SvelteKit discovery"""
    config = {
        "api_port": port,
        "api_url": f"http://localhost:{port}",
        "api_type": "simple_clean_api",
        "timestamp": time.time(),
        "status": "active"
    }
    
    config_file = Path(__file__).parent / "api_port.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📝 Port config saved: {config_file}")
    return config

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 DYNAMIC SIMPLE EXTRACTION API")
    print("=" * 40)
    
    try:
        # Find available port
        port = find_available_port()
        print(f"✅ Found available port: {port}")
        
        # Save config for SvelteKit
        config = save_port_config(port)
        
        print(f"🚀 Starting simple API server on port {port}")
        print(f"📍 URL: http://localhost:{port}")
        print(f"🔗 Health: http://localhost:{port}/health")
        print(f"📄 Docs: http://localhost:{port}/docs")
        print(f"🎯 Type: Simple Clean API (no complex wrappers)")
        print()
        
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Failed to start simple API: {e}")
