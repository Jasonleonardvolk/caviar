"""
FastAPI endpoints for PsiArchive queries
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncGenerator
import logging
import json
import io
import zipfile
import asyncio
from pathlib import Path
from collections import deque
import time

from core.psi_archive_extended import PSI_ARCHIVER

logger = logging.getLogger(__name__)

# Global event queue for SSE streaming
event_queue: deque = deque(maxlen=1000)
event_listeners: List[asyncio.Queue] = []

# Create router
archive_router = APIRouter(prefix="/api/archive", tags=["archive"])


@archive_router.get("/origin/{concept_id}")
async def get_concept_origin(concept_id: str) -> Dict[str, Any]:
    """
    Find when and where a concept was first learned
    
    Returns:
        - event_id: The event that introduced this concept
        - first_seen: ISO timestamp of first occurrence
        - source_doc_sha: SHA-256 hash of source document
        - source_path: Path to source document
        - parent_event: Parent event ID
        - session_id: Session that ingested this concept
    """
    try:
        origin = PSI_ARCHIVER.find_concept_origin(concept_id)
        
        if not origin:
            raise HTTPException(
                status_code=404,
                detail=f"No origin found for concept '{concept_id}'"
            )
        
        return {
            "status": "success",
            "concept_id": concept_id,
            "origin": origin
        }
        
    except Exception as e:
        logger.error(f"Error finding concept origin: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@archive_router.get("/session/{session_id}")
async def get_session_events(
    session_id: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)")
) -> Dict[str, Any]:
    """
    Get all events for a session (for debugging hallucinations)
    
    Returns chronologically ordered list of all events in the session
    """
    try:
        events = PSI_ARCHIVER.debug_session(session_id, start_date)
        
        if not events:
            raise HTTPException(
                status_code=404,
                detail=f"No events found for session '{session_id}'"
            )
        
        # Convert events to dicts
        event_dicts = [event.to_dict() for event in events]
        
        # Extract key insights
        response_events = [e for e in events if e.event_type == 'RESPONSE_EVENT']
        concept_paths = []
        for event in response_events:
            concept_paths.append({
                'timestamp': event.timestamp.isoformat(),
                'query': event.metadata.get('query', ''),
                'concept_path': event.concept_ids,
                'response_preview': event.metadata.get('response_preview', '')
            })
        
        return {
            "status": "success",
            "session_id": session_id,
            "event_count": len(events),
            "events": event_dicts,
            "response_analysis": {
                "total_responses": len(response_events),
                "concept_paths": concept_paths
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting session events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@archive_router.get("/delta")
async def get_mesh_deltas(
    since: str = Query(..., description="Get deltas since this timestamp (ISO format)")
) -> Dict[str, Any]:
    """
    Get mesh deltas since a timestamp for incremental sync
    
    Returns list of mesh changes (added nodes and edges) since the given time
    """
    try:
        # Parse timestamp
        try:
            since_time = datetime.fromisoformat(since)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid timestamp format. Use ISO format (e.g., 2025-01-10T12:00:00)"
            )
        
        deltas = PSI_ARCHIVER.get_mesh_deltas(since_time)
        
        # Calculate summary stats
        total_nodes = sum(len(d['delta'].get('added_nodes', [])) for d in deltas if d.get('delta'))
        total_edges = sum(len(d['delta'].get('added_edges', [])) for d in deltas if d.get('delta'))
        
        return {
            "status": "success",
            "since": since,
            "delta_count": len(deltas),
            "total_nodes_added": total_nodes,
            "total_edges_added": total_edges,
            "deltas": deltas
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting mesh deltas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@archive_router.get("/query")
async def query_archive(
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    limit: int = Query(100, description="Maximum results to return"),
    offset: int = Query(0, description="Skip this many results")
) -> Dict[str, Any]:
    """
    Query the archive with filters
    
    Supported event types:
    - LEARNING_UPDATE: Concept ingestion events
    - RESPONSE_EVENT: Saigon response generation
    - PLAN_CREATION: Planning/reasoning events
    - CONCEPT_REVISION: Concept modifications
    - MEMORY_STORE: Memory storage events
    - PENROSE_SIM: Penrose similarity computations
    """
    try:
        results = []
        total_scanned = 0
        
        # Scan archive files
        for archive_file in PSI_ARCHIVER._iter_archive_files():
            for event in PSI_ARCHIVER._read_archive_file(archive_file):
                # Apply filters
                if event_type and event.event_type != event_type:
                    continue
                    
                if session_id and event.session_id != session_id:
                    continue
                
                total_scanned += 1
                
                # Apply pagination
                if total_scanned > offset and len(results) < limit:
                    results.append(event.to_dict())
                
                if len(results) >= limit:
                    break
            
            if len(results) >= limit:
                break
        
        # Calculate statistics for specific event types
        stats = {}
        if event_type == "PENROSE_SIM":
            # Calculate Penrose performance stats
            if results:
                total_time = sum(r['metadata'].get('computation_time', 0) for r in results)
                total_concepts = sum(r['metadata'].get('concept_count', 0) for r in results)
                avg_speedup = sum(r['metadata'].get('speedup_factor', 22.7) for r in results) / len(results)
                
                stats = {
                    "total_computations": len(results),
                    "total_computation_time": round(total_time, 3),
                    "total_concepts_processed": total_concepts,
                    "average_speedup": round(avg_speedup, 1)
                }
        
        return {
            "status": "success",
            "query": {
                "event_type": event_type,
                "session_id": session_id,
                "limit": limit,
                "offset": offset
            },
            "total_results": len(results),
            "results": results,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error querying archive: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@archive_router.post("/seal")
async def seal_yesterday_archive() -> Dict[str, Any]:
    """
    Manually trigger sealing of yesterday's archive
    
    This is normally done automatically at day rollover or via cron
    """
    try:
        success = PSI_ARCHIVER.seal_yesterday()
        
        if success:
            return {
                "status": "success",
                "message": "Yesterday's archive sealed successfully"
            }
        else:
            return {
                "status": "info",
                "message": "Yesterday's archive already sealed or not found"
            }
            
    except Exception as e:
        logger.error(f"Error sealing archive: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@archive_router.get("/health")
async def archive_health() -> Dict[str, Any]:
    """Get PsiArchive health status and statistics"""
    try:
        # Get index statistics
        index_stats = PSI_ARCHIVER.index_cache
        
        # Count total events (approximate from index)
        total_events = sum(
            entry.get('event_count', 0) 
            for entry in index_stats.values()
        )
        
        # Get current file info
        current_file_size = 0
        if PSI_ARCHIVER.current_file and PSI_ARCHIVER.current_file.exists():
            current_file_size = PSI_ARCHIVER.current_file.stat().st_size
        
        return {
            "status": "healthy",
            "archive_directory": str(PSI_ARCHIVER.archive_dir),
            "current_date": PSI_ARCHIVER.current_date.isoformat(),
            "current_file": str(PSI_ARCHIVER.current_file),
            "current_file_size_bytes": current_file_size,
            "indexed_days": len(index_stats),
            "estimated_total_events": total_events,
            "index_cache_entries": index_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting archive health: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@archive_router.get("/penrose/stats")
async def get_penrose_stats(
    since: Optional[str] = Query(None, description="Get stats since this date (YYYY-MM-DD)")
) -> StreamingResponse:
    """
    Stream Penrose statistics as JSONL for dashboard visualization
    
    Returns one JSON object per line with Penrose projection stats
    """
    async def generate_stats():
        # Parse since date
        start_date = None
        if since:
            try:
                start_date = datetime.fromisoformat(since).date().isoformat()
            except ValueError:
                yield json.dumps({"error": "Invalid date format. Use YYYY-MM-DD"}) + "\n"
                return
        
        # Scan archive for events with Penrose stats
        stats_count = 0
        total_concepts = 0
        total_relations = 0
        total_time = 0.0
        
        for archive_file in PSI_ARCHIVER._iter_archive_files(start_date):
            for event in PSI_ARCHIVER._read_archive_file(archive_file):
                if event.penrose_stats:
                    stats = {
                        'event_id': event.event_id,
                        'timestamp': event.timestamp.isoformat(),
                        'session_id': event.session_id,
                        'n_concepts': event.penrose_stats.get('n_concepts', 0),
                        'n_edges': event.penrose_stats.get('n_edges', 0),
                        'density_pct': event.penrose_stats.get('density_pct', 0),
                        'speedup': event.penrose_stats.get('speedup_vs_full', 0),
                        'total_time': event.penrose_stats.get('times', {}).get('total', 0),
                        'rank': event.penrose_stats.get('rank', 32),
                        'threshold': event.penrose_stats.get('threshold', 0.7)
                    }
                    
                    # Add relation stats if available
                    if 'relations' in event.penrose_stats:
                        stats['relations_added'] = event.penrose_stats['relations']['relations_added']
                    
                    yield json.dumps(stats) + "\n"
                    
                    stats_count += 1
                    total_concepts += stats['n_concepts']
                    total_relations += stats.get('relations_added', 0)
                    total_time += stats['total_time']
        
        # Final summary line
        if stats_count > 0:
            summary = {
                'summary': True,
                'total_projections': stats_count,
                'total_concepts_processed': total_concepts,
                'total_relations_created': total_relations,
                'total_computation_time': round(total_time, 3),
                'avg_speedup': round(total_time / stats_count, 1) if stats_count > 0 else 0
            }
            yield json.dumps(summary) + "\n"
    
    return StreamingResponse(
        generate_stats(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": "inline; filename=penrose_stats.jsonl"}
    )


@archive_router.post("/replay/penrose")
async def replay_with_penrose(
    until_timestamp: str = Query(..., description="Replay until this timestamp (ISO format)"),
    output_format: str = Query("summary", description="Output format: summary or zip")
) -> Dict[str, Any]:
    """
    Replay archive with Penrose reconstruction
    
    Returns either a summary or a zipped snapshot of the reconstructed mesh
    """
    try:
        # Parse timestamp
        try:
            until_time = datetime.fromisoformat(until_timestamp)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid timestamp format. Use ISO format (e.g., 2025-01-10T12:00:00)"
            )
        
        # Import replay tool
        from tools.psi_replay import PsiReplay
        
        # Create temporary output directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "penrose_replay"
            
            # Run replay
            replay = PsiReplay(PSI_ARCHIVER.archive_dir)
            summary = replay.replay_until(until_time, output_dir, fast_mode=True)
            
            # Count Penrose relations
            mesh_path = output_dir / "concept_mesh"
            penrose_relation_count = 0
            
            if mesh_path.exists():
                # Load mesh to count Penrose relations
                from python.core.concept_mesh import ConceptMesh
                mesh = ConceptMesh({'storage_path': str(mesh_path)})
                
                for relation in mesh.relations:
                    if relation.relation_type == 'similar_penrose':
                        penrose_relation_count += 1
            
            summary['penrose_relations'] = penrose_relation_count
            
            if output_format == "zip":
                # Create zip file
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for file_path in output_dir.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(output_dir)
                            zf.write(file_path, arcname)
                
                zip_buffer.seek(0)
                
                return StreamingResponse(
                    zip_buffer,
                    media_type="application/zip",
                    headers={
                        "Content-Disposition": f"attachment; filename=penrose_replay_{until_time.date()}.zip"
                    }
                )
            else:
                return {
                    "status": "success",
                    "replay_summary": summary,
                    "penrose_stats": {
                        "total_penrose_relations": penrose_relation_count,
                        "replay_timestamp": until_timestamp
                    }
                }
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Penrose replay: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@archive_router.get("/events")
async def event_stream() -> StreamingResponse:
    """
    Server-Sent Events stream for live PsiArchive events
    
    Streams all archive events in real-time, including Penrose stats
    """
    async def generate() -> AsyncGenerator[str, None]:
        # Create a queue for this client
        client_queue = asyncio.Queue()
        event_listeners.append(client_queue)
        
        try:
            # Send recent events first
            for event in list(event_queue):
                yield f"data: {json.dumps(event)}\n\n"
            
            # Then stream live events
            while True:
                try:
                    # Wait for new event with timeout
                    event = await asyncio.wait_for(client_queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield f": heartbeat\n\n"
                    
        except asyncio.CancelledError:
            # Client disconnected
            event_listeners.remove(client_queue)
            raise
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable Nginx buffering
        }
    )


# Function to broadcast events to all SSE clients
async def broadcast_event(event: Dict[str, Any]):
    """Broadcast event to all SSE listeners"""
    # Add to recent events
    event_queue.append(event)
    
    # Send to all active listeners
    for queue in event_listeners:
        try:
            await queue.put(event)
        except:
            # Queue is full or client disconnected
            pass


# Export router and broadcaster for inclusion in main app
__all__ = ['archive_router', 'broadcast_event']
