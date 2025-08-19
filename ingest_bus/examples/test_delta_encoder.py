"""
Sample test script for delta encoder.

This script demonstrates the use of the delta encoder and decoder
for efficient state transfer.
"""

import sys
import os
import json
import time
import asyncio
from typing import Dict, Any, Optional

# Add parent directory to Python path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.delta_encoder import DeltaEncoder, DeltaDecoder


# Sample delta metrics callback
def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Print delta metrics
    
    Args:
        metrics: Metrics data
    """
    ratio = metrics.get('deltaFullRatio', 1.0)
    full_size = metrics.get('fullStateSize', 0)
    delta_size = metrics.get('deltaSize', 0)
    
    print(f"Delta size: {delta_size} bytes")
    print(f"Full size: {full_size} bytes")
    print(f"Ratio: {ratio:.2f}")
    print(f"Savings: {full_size - delta_size} bytes ({(1 - ratio) * 100:.1f}%)")


async def simulate_job_updates() -> None:
    """
    Simulate a series of job updates to demonstrate delta encoding
    """
    # Create encoder and decoder
    encoder = DeltaEncoder(
        require_ack=True,
        on_metrics=print_metrics
    )
    
    decoder = DeltaDecoder(
        on_resync_needed=lambda: print("Resync needed")
    )
    
    # Initial job state
    job = {
        "job_id": "test-job-123",
        "file_url": "https://example.com/document.pdf",
        "file_name": "document.pdf",
        "status": "queued",
        "created_at": "2025-05-19T12:00:00Z",
        "updated_at": "2025-05-19T12:00:00Z",
        "progress": 0.0,
        "chunk_count": 0,
        "chunks": []
    }
    
    # Encode initial state
    packet = encoder.encode(job)
    print("\n=== Initial State ===")
    print(f"Sequence: {packet['sequence']}")
    print(f"Has base state: {'baseState' in packet}")
    print(f"Size: {len(json.dumps(packet))} bytes")
    
    # Decode initial state
    decoded, ack = decoder.decode(packet)
    print(f"Decoded job status: {decoded['status']}")
    
    # Send ack back to encoder
    if ack:
        encoder.handle_ack(ack)
    
    # Simulate progress update (small change)
    await asyncio.sleep(1)
    job["status"] = "processing"
    job["updated_at"] = "2025-05-19T12:00:01Z"
    job["progress"] = 10.0
    
    # Encode update
    packet = encoder.encode(job)
    print("\n=== Small Update ===")
    print(f"Sequence: {packet['sequence']}")
    print(f"Has base state: {'baseState' in packet}")
    print(f"Has deltas: {'deltas' in packet}")
    if 'deltas' in packet:
        print(f"Number of deltas: {len(packet['deltas'])}")
        print(f"Deltas: {json.dumps(packet['deltas'], indent=2)}")
    print(f"Size: {len(json.dumps(packet))} bytes")
    
    # Decode update
    decoded, ack = decoder.decode(packet)
    print(f"Decoded job status: {decoded['status']}")
    print(f"Decoded job progress: {decoded['progress']}")
    
    # Send ack back to encoder
    if ack:
        encoder.handle_ack(ack)
    
    # Simulate adding many chunks (large change)
    await asyncio.sleep(1)
    job["updated_at"] = "2025-05-19T12:00:02Z"
    job["progress"] = 50.0
    
    # Add 20 chunks
    for i in range(20):
        job["chunks"].append({
            "id": f"chunk-{i}",
            "text": f"This is chunk {i} content...",
            "start_offset": i * 1000,
            "end_offset": (i + 1) * 1000 - 1,
            "metadata": {
                "page": i // 5 + 1,
                "section": "introduction" if i < 10 else "methodology"
            }
        })
    
    job["chunk_count"] = len(job["chunks"])
    
    # Encode update
    packet = encoder.encode(job)
    print("\n=== Large Update ===")
    print(f"Sequence: {packet['sequence']}")
    print(f"Has base state: {'baseState' in packet}")
    print(f"Has deltas: {'deltas' in packet}")
    print(f"Size: {len(json.dumps(packet))} bytes")
    
    # Decode update
    decoded, ack = decoder.decode(packet)
    print(f"Decoded job status: {decoded['status']}")
    print(f"Decoded job progress: {decoded['progress']}")
    print(f"Decoded chunk count: {decoded['chunk_count']}")
    
    # Send ack back to encoder
    if ack:
        encoder.handle_ack(ack)
    
    # Simulate completing the job
    await asyncio.sleep(1)
    job["status"] = "completed"
    job["updated_at"] = "2025-05-19T12:00:03Z"
    job["progress"] = 100.0
    job["completed_at"] = "2025-05-19T12:00:03Z"
    
    # Encode final update
    packet = encoder.encode(job)
    print("\n=== Final Update ===")
    print(f"Sequence: {packet['sequence']}")
    print(f"Has base state: {'baseState' in packet}")
    print(f"Has deltas: {'deltas' in packet}")
    print(f"Size: {len(json.dumps(packet))} bytes")
    
    # Decode final update
    decoded, ack = decoder.decode(packet)
    print(f"Decoded job status: {decoded['status']}")
    print(f"Decoded job progress: {decoded['progress']}")
    
    # Send ack back to encoder
    if ack:
        encoder.handle_ack(ack)
    
    print("\n=== Simulation Complete ===")


if __name__ == "__main__":
    print("Testing Delta Encoder/Decoder")
    asyncio.run(simulate_job_updates())
