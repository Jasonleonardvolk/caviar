"""
Simple client example for ingest-bus service.

This script demonstrates how to interact with the ingest-bus service
as a client, using both the REST API and WebSocket for real-time updates.
"""

import sys
import os
import json
import time
import asyncio
import websockets
import httpx
from typing import Dict, Any, List, Optional

# Base URL of the ingest-bus service
BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/api/ws"


async def queue_document(file_url: str, track: Optional[str] = None) -> Dict[str, Any]:
    """
    Queue a document for processing
    
    Args:
        file_url: URL of the document to process
        track: Optional track to assign
        
    Returns:
        Response from the service
    """
    url = f"{BASE_URL}/api/jobs"
    
    data = {
        "file_url": file_url,
    }
    
    if track:
        data["track"] = track
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)
        return response.json()


async def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get the status of a job
    
    Args:
        job_id: ID of the job
        
    Returns:
        Job status
    """
    url = f"{BASE_URL}/api/jobs/{job_id}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()


async def list_jobs(limit: int = 10, offset: int = 0) -> Dict[str, Any]:
    """
    List jobs
    
    Args:
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip
        
    Returns:
        List of jobs
    """
    url = f"{BASE_URL}/api/jobs?limit={limit}&offset={offset}"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()


async def get_metrics() -> Dict[str, Any]:
    """
    Get ingest metrics
    
    Returns:
        Metrics data
    """
    url = f"{BASE_URL}/api/metrics"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()


async def listen_for_updates(job_id: Optional[str] = None) -> None:
    """
    Listen for real-time updates via WebSocket
    
    Args:
        job_id: Optional job ID to filter updates
    """
    print(f"Connecting to WebSocket at {WS_URL}...")
    
    async with websockets.connect(WS_URL) as websocket:
        print("Connected to WebSocket")
        
        # Subscribe to job updates if job_id is provided
        if job_id:
            await websocket.send(json.dumps({
                "type": "subscribe",
                "topic": job_id
            }))
            print(f"Subscribed to updates for job {job_id}")
        
        # Listen for updates
        print("Listening for updates (Ctrl+C to stop)...")
        try:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data.get("type") == "update":
                    topic = data.get("topic")
                    update_data = data.get("data")
                    
                    print(f"\nReceived update for topic: {topic}")
                    
                    if "status" in update_data:
                        print(f"Status: {update_data['status']}")
                    
                    if "progress" in update_data:
                        print(f"Progress: {update_data['progress']}%")
                    
                    if "chunk_count" in update_data:
                        print(f"Chunks: {update_data['chunk_count']}")
                elif data.get("type") == "subscribed":
                    topic = data.get("topic")
                    print(f"Successfully subscribed to {topic}")
        except KeyboardInterrupt:
            print("\nStopping WebSocket listener")


async def demo() -> None:
    """Run a demonstration of the client"""
    print("=== Ingest Bus Client Demo ===\n")
    
    # Queue a document
    print("Queueing a document...")
    result = await queue_document(
        "https://arxiv.org/pdf/2303.08774.pdf",
        "ai_ml"
    )
    
    if not result.get("success", False):
        print(f"Error: {result.get('error', 'Unknown error')}")
        return
    
    job_id = result.get("job_id")
    print(f"Document queued with job ID: {job_id}")
    
    # Get status
    print("\nGetting job status...")
    status = await get_job_status(job_id)
    
    if status.get("success", False):
        job = status.get("job", {})
        print(f"Status: {job.get('status')}")
        print(f"Progress: {job.get('progress')}%")
    else:
        print(f"Error: {status.get('error', 'Unknown error')}")
    
    # List jobs
    print("\nListing recent jobs...")
    jobs_result = await list_jobs(limit=5)
    
    if jobs_result.get("success", False):
        jobs = jobs_result.get("jobs", [])
        print(f"Found {len(jobs)} jobs:")
        
        for i, job in enumerate(jobs):
            print(f"{i+1}. {job.get('job_id')} - {job.get('status')} - {job.get('file_name')}")
    else:
        print(f"Error: {jobs_result.get('error', 'Unknown error')}")
    
    # Get metrics
    print("\nGetting metrics...")
    metrics_result = await get_metrics()
    
    if metrics_result.get("success", False):
        metrics = metrics_result.get("metrics", {})
        print(f"Total jobs: {metrics.get('total_jobs', 0)}")
        print(f"Status counts: {metrics.get('status_counts', {})}")
    else:
        print(f"Error: {metrics_result.get('error', 'Unknown error')}")
    
    # Listen for updates
    print("\nListening for real-time updates...")
    try:
        await listen_for_updates(job_id)
    except Exception as e:
        print(f"Error connecting to WebSocket: {str(e)}")
        print("Make sure the ingest-bus service is running")


if __name__ == "__main__":
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
