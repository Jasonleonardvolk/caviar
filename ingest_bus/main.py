#!/usr/bin/env python3
"""
TORI Ingest Bus Service

This FastAPI-based microservice serves as the central ingest queue, status tracker,
and metrics provider for TORI's content ingestion pipeline. It integrates with the
ScholarSphere knowledge system to process and store documents, conversations,
and other content with phase-aligned concept mapping.

Usage:
    uvicorn main:app --reload --port 8080
"""

import os
import sys
import time
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Import ingest-bus modules
from routes import queue, status, metrics
from workers import extract
from models import schemas
from utils import logger, config_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ingest-bus")

# Load configuration
try:
    with open(Path(__file__).parent.parent / "conversation_config.json", "r") as f:
        config = json.load(f)
        logger.info(f"Loaded configuration from conversation_config.json")
except Exception as e:
    logger.warning(f"Could not load conversation_config.json: {str(e)}")
    logger.warning("Using default configuration settings")
    config = {
        "scholar_sphere": {
            "enabled": True,
            "encoder_version": "v2.5.0",
            "chunk_size": 512,
            "chunk_overlap": 128,
            "max_concepts_per_chunk": 12
        },
        "integration": {
            "extraction_timeout_ms": 30000
        }
    }

# Initialize FastAPI app
app = FastAPI(
    title="TORI Ingest Bus",
    description="Central ingestion microservice for TORI's content processing pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers from route modules
app.include_router(queue.router, prefix="/queue", tags=["ingest"])
app.include_router(status.router, prefix="/status", tags=["status"])
app.include_router(metrics.router, prefix="/metrics", tags=["metrics"])

# Add a /health endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint for the Ingest Bus service."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "ingest-bus",
        "version": "1.0.0",
        "scholar_sphere_integration": config.get("scholar_sphere", {}).get("enabled", False)
    }

@app.get("/", tags=["root"])
async def root():
    """Root endpoint for the Ingest Bus service."""
    return {
        "service": "TORI Ingest Bus",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": [
            {"path": "/queue", "description": "Ingest queue operations"},
            {"path": "/status", "description": "Ingest job status"},
            {"path": "/metrics", "description": "Ingest metrics"}
        ]
    }

# Run server if executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
