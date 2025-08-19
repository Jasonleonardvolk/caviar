"""
Main application entry point for ingest-bus service.

This module sets up the FastAPI application, initializes services,
and configures routes and MCP endpoints.
"""

import os
import logging
import asyncio
import json
import argparse
import uuid
import time
from typing import Dict, Any, List, Optional, Set
from fastapi import FastAPI, WebSocket, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import start_http_server, generate_latest, CONTENT_TYPE_LATEST

from .services.ingest_service import IngestService
from .services.metrics_service import get_metrics_service, on_delta_metrics
from .utils.delta_encoder import DeltaEncoder
from .mcp.tools import create_mcp_tools
from .models.job import IngestJob, JobStatus, ProcessingStage, ChunkInfo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Ingest Bus",
    description="Ingest bus service for document processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application state
class AppState:
    """Application state"""
    ingest_service: IngestService = None
    ws_connections: Set[WebSocket] = set()
    delta_encoder: DeltaEncoder = None
    mcp_tools: Dict[str, Any] = None
    build_hash: str = "dev"


app_state = AppState()


# WebSocket connection manager
class ConnectionManager:
    """Manager for WebSocket connections"""
    
    async def connect(self, websocket: WebSocket) -> None:
        """
        Connect a new WebSocket client
        
        Args:
            websocket: The WebSocket connection
        """
        await websocket.accept()
        app_state.ws_connections.add(websocket)
        logger.info(f"WebSocket client connected, total connections: {len(app_state.ws_connections)}")
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """
        Disconnect a WebSocket client
        
        Args:
            websocket: The WebSocket connection
        """
        if websocket in app_state.ws_connections:
            app_state.ws_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected, total connections: {len(app_state.ws_connections)}")
    
    async def send_message(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """
        Send a message to a WebSocket client
        
        Args:
            websocket: The WebSocket connection
            message: The message to send
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message to WebSocket client: {e}")
            await self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """
        Broadcast a message to all connected WebSocket clients
        
        Args:
            message: The message to broadcast
        """
        for connection in list(app_state.ws_connections):
            try:
                await self.send_message(connection, message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket client: {e}")
                await self.disconnect(connection)


# Create connection manager
manager = ConnectionManager()


# WebSocket subscriber for job updates
class WSSubscriber:
    """WebSocket subscriber for job updates"""
    
    def __init__(self, websocket: WebSocket):
        """
        Initialize WebSocket subscriber
        
        Args:
            websocket: The WebSocket connection
        """
        self.websocket = websocket
    
    async def send_update(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Send an update to the WebSocket client
        
        Args:
            topic: The update topic
            data: The update data
        """
        message = {
            "type": "update",
            "topic": topic,
            "data": data
        }
        await manager.send_message(self.websocket, message)


# Pydantic models for API requests/responses
class QueueJobRequest(BaseModel):
    """Request model for queueing a job"""
    file_url: str = Field(..., description="URL to the file to process")
    track: Optional[str] = Field(None, description="Track to assign the document to")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class JobStatusResponse(BaseModel):
    """Response model for job status"""
    success: bool = Field(..., description="Success flag")
    job: Optional[Dict[str, Any]] = Field(None, description="Job data")
    error: Optional[str] = Field(None, description="Error message if not successful")


class JobsListResponse(BaseModel):
    """Response model for listing jobs"""
    success: bool = Field(..., description="Success flag")
    jobs: Optional[List[Dict[str, Any]]] = Field(None, description="List of jobs")
    count: Optional[int] = Field(None, description="Number of jobs returned")
    offset: Optional[int] = Field(None, description="Offset used")
    limit: Optional[int] = Field(None, description="Limit used")
    error: Optional[str] = Field(None, description="Error message if not successful")


class MetricsResponse(BaseModel):
    """Response model for metrics"""
    success: bool = Field(..., description="Success flag")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Metrics data")
    error: Optional[str] = Field(None, description="Error message if not successful")


# MCP Schema
class MCPSchemaOperation(BaseModel):
    """MCP schema operation"""
    name: str = Field(..., description="Operation name")
    description: str = Field(..., description="Operation description")
    parameters: Dict[str, Dict[str, Any]] = Field(..., description="Operation parameters")
    required: List[str] = Field(..., description="Required parameters")


class MCPSchemaV2(BaseModel):
    """MCP schema v2"""
    schema_version: str = Field("2.0", description="Schema version")
    server_name: str = Field(..., description="Server name")
    server_version: str = Field(..., description="Server version")
    tools: Dict[str, MCPSchemaOperation] = Field(..., description="Available tools")


# API routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ingest-bus",
        "version": "1.0.0",
        "build_hash": app_state.build_hash,
        "docs_url": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    # Check if ingest service is initialized
    if app_state.ingest_service is None:
        raise HTTPException(status_code=503, detail="Ingest service not initialized")
    
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """Readiness check endpoint"""
    # Check if all required services are initialized
    if (app_state.ingest_service is None or 
        app_state.mcp_tools is None):
        raise HTTPException(status_code=503, detail="Services not fully initialized")
    
    return {"status": "ready"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/jobs", response_model=JobStatusResponse)
async def queue_job(request: QueueJobRequest):
    """
    Queue a new job
    
    Args:
        request: The queue job request
        
    Returns:
        Job status response
    """
    if app_state.ingest_service is None:
        raise HTTPException(status_code=503, detail="Ingest service not initialized")
    
    # Use ingest.queue MCP tool
    queue_tool = app_state.mcp_tools.get('ingest.queue')
    if queue_tool is None:
        raise HTTPException(status_code=503, detail="Queue tool not available")
    
    result = await queue_tool(
        file_url=request.file_url,
        track=request.track,
        metadata=request.metadata
    )
    
    if not result.get('success', False):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=result
        )
    
    return result


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a job
    
    Args:
        job_id: The job ID
        
    Returns:
        Job status response
    """
    if app_state.ingest_service is None:
        raise HTTPException(status_code=503, detail="Ingest service not initialized")
    
    # Use ingest.status MCP tool
    status_tool = app_state.mcp_tools.get('ingest.status')
    if status_tool is None:
        raise HTTPException(status_code=503, detail="Status tool not available")
    
    result = await status_tool(job_id=job_id)
    
    if not result.get('success', False):
        if result.get('error', '').endswith('not found'):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=result
            )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=result
        )
    
    return result


@app.get("/api/jobs", response_model=JobsListResponse)
async def list_jobs(limit: int = 10, offset: int = 0):
    """
    List jobs
    
    Args:
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip
        
    Returns:
        Jobs list response
    """
    if app_state.ingest_service is None:
        raise HTTPException(status_code=503, detail="Ingest service not initialized")
    
    # Use ingest.status MCP tool without job_id to list jobs
    status_tool = app_state.mcp_tools.get('ingest.status')
    if status_tool is None:
        raise HTTPException(status_code=503, detail="Status tool not available")
    
    result = await status_tool(limit=limit, offset=offset)
    
    if not result.get('success', False):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=result
        )
    
    return result


@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get metrics
    
    Returns:
        Metrics response
    """
    if app_state.ingest_service is None:
        raise HTTPException(status_code=503, detail="Ingest service not initialized")
    
    # Use ingest.metrics MCP tool
    metrics_tool = app_state.mcp_tools.get('ingest.metrics')
    if metrics_tool is None:
        raise HTTPException(status_code=503, detail="Metrics tool not available")
    
    result = await metrics_tool()
    
    if not result.get('success', False):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=result
        )
    
    return result


@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates
    
    Args:
        websocket: The WebSocket connection
    """
    await manager.connect(websocket)
    subscriber = WSSubscriber(websocket)
    
    try:
        # Subscribe to all jobs
        if app_state.ingest_service:
            await app_state.ingest_service.subscribe('jobs', subscriber)
        
        # Keep connection open until client disconnects
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle subscription requests
            if message.get('type') == 'subscribe' and 'topic' in message:
                topic = message['topic']
                
                if app_state.ingest_service:
                    # Subscribe to specific job updates
                    await app_state.ingest_service.subscribe(topic, subscriber)
                    
                    # Send confirmation
                    await manager.send_message(websocket, {
                        'type': 'subscribed',
                        'topic': topic
                    })
            
            # Handle unsubscribe requests
            elif message.get('type') == 'unsubscribe' and 'topic' in message:
                topic = message['topic']
                
                if app_state.ingest_service:
                    # Unsubscribe from specific job updates
                    await app_state.ingest_service.unsubscribe(topic, subscriber)
                    
                    # Send confirmation
                    await manager.send_message(websocket, {
                        'type': 'unsubscribed',
                        'topic': topic
                    })
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Unsubscribe from all topics
        if app_state.ingest_service:
            await app_state.ingest_service.unsubscribe('jobs', subscriber)
        
        # Disconnect WebSocket
        await manager.disconnect(websocket)


@app.get("/mcp/v2/schema")
async def mcp_schema_v2():
    """MCP schema v2 endpoint"""
    if not app_state.mcp_tools:
        raise HTTPException(status_code=503, detail="MCP tools not initialized")
    
    schema = MCPSchemaV2(
        server_name="ingest-bus",
        server_version="1.0.0",
        tools={
            'ingest.queue': MCPSchemaOperation(
                name="ingest.queue",
                description="Queue a document for ingestion",
                parameters={
                    'file_url': {
                        'type': 'string',
                        'description': 'URL to the file to process'
                    },
                    'track': {
                        'type': 'string',
                        'description': 'Track to assign the document to',
                        'enum': ['programming', 'math_physics', 'ai_ml', 'domain', 'ops_sre']
                    },
                    'metadata': {
                        'type': 'object',
                        'description': 'Additional metadata'
                    }
                },
                required=['file_url']
            ),
            'ingest.status': MCPSchemaOperation(
                name="ingest.status",
                description="Get the status of an ingest job or list of jobs",
                parameters={
                    'job_id': {
                        'type': 'string',
                        'description': 'Job ID (optional, if not provided, returns a list of jobs)'
                    },
                    'limit': {
                        'type': 'integer',
                        'description': 'Maximum number of jobs to return',
                        'default': 10
                    },
                    'offset': {
                        'type': 'integer',
                        'description': 'Number of jobs to skip',
                        'default': 0
                    }
                },
                required=[]
            ),
            'ingest.metrics': MCPSchemaOperation(
                name="ingest.metrics",
                description="Get metrics about the ingest process",
                parameters={},
                required=[]
            )
        }
    )
    
    return schema.dict()


@app.post("/mcp/v2/tools/{tool_name}")
async def mcp_tool_v2(tool_name: str, request: Request):
    """
    MCP tool v2 endpoint
    
    Args:
        tool_name: The tool name
        request: The HTTP request
        
    Returns:
        Tool response
    """
    if not app_state.mcp_tools:
        raise HTTPException(status_code=503, detail="MCP tools not initialized")
    
    # Get tool function
    tool_fn = app_state.mcp_tools.get(tool_name)
    if not tool_fn:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    # Parse request body
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    
    # Call tool function with provided arguments
    try:
        result = await tool_fn(**body)
        return result
    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error executing tool: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    # Get build hash from environment or generate a random one for development
    build_hash = os.environ.get('BUILD_HASH', uuid.uuid4().hex[:7])
    app_state.build_hash = build_hash
    
    logger.info(f"Starting ingest-bus service with build hash {build_hash}")
    
    # Initialize metrics service
    metrics_service = get_metrics_service(build_hash)
    
    # Create delta encoder for WebSocket updates
    app_state.delta_encoder = DeltaEncoder(
        require_ack=False,
        on_metrics=on_delta_metrics
    )
    
    # Initialize ingest service
    data_dir = os.environ.get('DATA_DIR', './data/jobs')
    app_state.ingest_service = IngestService(data_dir=data_dir)
    
    # Create MCP tools
    app_state.mcp_tools = create_mcp_tools(app_state.ingest_service, build_hash)
    
    # Start Prometheus metrics HTTP server on a different port
    metrics_port = int(os.environ.get('METRICS_PORT', 8081))
    start_http_server(metrics_port)
    logger.info(f"Prometheus metrics server started on port {metrics_port}")
    
    logger.info("Ingest-bus service startup complete")


def main():
    """Main entry point"""
    import uvicorn
    
    parser = argparse.ArgumentParser(description='Ingest-bus service')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    args = parser.parse_args()
    
    uvicorn.run(
        "ingest-bus.src.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
