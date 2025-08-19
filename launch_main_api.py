#!/usr/bin/env python3
"""
Launch Main API Server
======================

Standalone script to run the main API server with all routes.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn
import logging
from api import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting TORI Main API Server")
    logger.info("API documentation available at: http://localhost:8001/docs")
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=True  # Enable auto-reload for development
    )
