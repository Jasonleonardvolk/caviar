#!/usr/bin/env python3
"""
Example: Using the Enhanced PsiArchive in Your Pipeline

This demonstrates how to integrate the enhanced archiver with:
- Automatic mesh delta tracking
- Full provenance logging
- Session debugging
- Time-travel replay
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.psi_archive_extended import PSI_ARCHIVER
from core.canonical_ingestion_enhanced import ingest_pdf_file, get_enhanced_ingestion_manager
from tools.psi_replay import PsiReplay


def example_ingestion_with_provenance():
    """Example: Ingest a PDF with full provenance tracking"""
    print("\n📚 Example 1: PDF Ingestion with Provenance Tracking")
    print("=" * 60)
    
    # Create a session ID for this ingestion batch
    session_id = f"example_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Ingest a PDF - mesh delta is automatically captured
    result = ingest_pdf_file(
        file_path="data/pdfs/graphene_electronics.pdf",
        title="Graphene Electronics Review",
        author="Dr. Carbon",
        source_type="paper",
        domain="materials_science",
        session_id=session_id
    )
    
    if result.get('status') == 'success':
        print(f"✅ Ingested {result.get('concepts_extracted', 0)} concepts")
        print(f"📝 PsiArchive Event ID: {result.get('psi_event_id')}")
        
        # Track a specific concept's origin
        if result.get('ingestion_result', {}).get('ingested_concepts'):
            concept_id = result['ingestion_result']['ingested_concepts'][0]
            origin = PSI_ARCHIVER.find_concept_origin(concept_id)
            
            if origin:
                print(f"\n🔍 Concept '{concept_id}' Origin:")
                print(f"   First seen: {origin['first_seen']}")
                print(f"   Source SHA: {origin['source_doc_sha'][:16]}...")
                print(f"   Source: {origin['source_path']}")


def example_response_tracking():
    """Example: Track Saigon response generation"""
    print("\n\n💬 Example 2: Response Generation Tracking")
    print("=" * 60)
    
    # Simulate a response generation event
    query = "What are the electrical properties of graphene?"
    concept_path = ["carbon", "2d_materials", "graphene_001", "conductivity"]
    response = "Graphene exhibits exceptional electrical conductivity due to..."
    
    event_id = PSI_ARCHIVER.log_response_generation(
        query=query,
        concept_path=concept_path,
        response=response,
        session_id="chat_session_123",
        thread_id="reasoning_thread_456"
    )
    
    print(f"✅ Logged response event: {event_id}")
    print(f"📍 Concept path: {' → '.join(concept_path)}")


def example_session_debugging():
    """Example: Debug a session for hallucination analysis"""
    print("\n\n🔍 Example 3: Session Debugging")
    print("=" * 60)
    
    # Get all events for a session
    session_id = "chat_session_123"
    events = PSI_ARCHIVER.debug_session(session_id)
    
    print(f"📊 Found {len(events)} events in session {session_id}")
    
    # Analyze response patterns
    response_events = [e for e in events if e.event_type == 'RESPONSE_EVENT']
    
    for event in response_events[:3]:  # Show first 3
        print(f"\n⏰ {event.timestamp.strftime('%H:%M:%S')}")
        print(f"   Query: {event.metadata.get('query', '')[:50]}...")
        print(f"   Path: {' → '.join(event.concept_ids[:3])}...")
        print(f"   Model: {event.metadata.get('model', 'unknown')}")


def example_incremental_sync():
    """Example: Get mesh deltas for laptop sync"""
    print("\n\n🔄 Example 4: Incremental Sync")
    print("=" * 60)
    
    # Get deltas from 1 hour ago
    since_time = datetime.now() - timedelta(hours=1)
    deltas = PSI_ARCHIVER.get_mesh_deltas(since_time)
    
    print(f"📦 Found {len(deltas)} mesh deltas since {since_time.strftime('%H:%M')}")
    
    total_nodes = 0
    total_edges = 0
    
    for delta_info in deltas[:5]:  # Show first 5
        delta = delta_info['delta']
        if delta:
            nodes = len(delta.get('added_nodes', []))
            edges = len(delta.get('added_edges', []))
            total_nodes += nodes
            total_edges += edges
            
            print(f"\n🕐 {delta_info['timestamp']}")
            print(f"   Added: {nodes} nodes, {edges} edges")
    
    print(f"\n📊 Total changes: {total_nodes} nodes, {total_edges} edges")


def example_time_travel():
    """Example: Replay system state to yesterday"""
    print("\n\n⏰ Example 5: Time Travel Replay")
    print("=" * 60)
    
    # Replay to yesterday at noon
    yesterday_noon = datetime.now().replace(hour=12, minute=0, second=0) - timedelta(days=1)
    
    replay = PsiReplay(
        archive_dir=Path("data/archive"),
        snapshots_dir=Path("data/snapshots")
    )
    
    output_dir = Path("data/replay_demo")
    
    print(f"🔄 Replaying system state to {yesterday_noon}")
    print("   (This is a demo - would actually replay in production)")
    
    # In production, you would run:
    # summary = replay.replay_until(yesterday_noon, output_dir, fast_mode=True)
    
    print("\n📸 Fast mode would use latest snapshot + forward deltas")
    print("🧠 Result: Complete ConceptMesh and MemoryVault at target time")


def example_advanced_queries():
    """Example: Advanced PsiArchive queries"""
    print("\n\n🔬 Example 6: Advanced Archive Queries")
    print("=" * 60)
    
    # Log a Penrose computation event
    PSI_ARCHIVER.append_event(
        PSI_ARCHIVER.PsiEvent(
            event_id=PSI_ARCHIVER._generate_event_id(),
            event_type='PENROSE_SIM',
            timestamp=datetime.now(),
            concept_ids=["graphene_001", "silicon_002", "gaas_003"],
            metadata={
                'computation_time': 0.045,  # 45ms for 1000 concepts
                'concept_count': 1000,
                'speedup_factor': 22.7,
                'similarity_threshold': 0.85
            }
        )
    )
    
    print("✅ Logged Penrose similarity computation")
    print("   1000 concepts processed in 45ms (22.7x speedup)")


def main():
    """Run all examples"""
    print("\n🚀 Enhanced PsiArchive Integration Examples")
    print("=" * 80)
    
    # Run examples (comment out any that require actual files)
    # example_ingestion_with_provenance()  # Requires PDF file
    example_response_tracking()
    example_session_debugging()
    example_incremental_sync()
    example_time_travel()
    example_advanced_queries()
    
    print("\n\n✨ Integration Complete!")
    print("\nKey Integration Points:")
    print("1. Replace 'from core.psi_archive import' → 'from core.psi_archive_extended import'")
    print("2. Use canonical_ingestion_enhanced.py for automatic delta tracking")
    print("3. Schedule cron jobs: daily seal (23:59) + weekly snapshot (Sunday 02:00)")
    print("4. Mount archive_endpoints.py in your FastAPI app")
    print("5. Use psi_replay.py for time-travel debugging")
    
    print("\n📚 Documentation:")
    print("- Archive format: NDJSON with optional gzip + SHA chain")
    print("- Performance: Index keeps scans fast even at 1M+ events")
    print("- Sync: Mesh deltas typically <1KB vs full mesh at 100MB+")
    print("- Safety: File locking prevents corruption from concurrent writes")


if __name__ == "__main__":
    main()
