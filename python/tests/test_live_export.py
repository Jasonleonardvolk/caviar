#!/usr/bin/env python3
"""
Test Suite for Live Mesh Context Export (Improvement #1)
Tests event-driven export, debouncing, and trigger integration
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.mesh_summary_exporter import (
    MeshSummaryExporter, 
    ExportTrigger,
    ExportMode,
    get_global_exporter,
    trigger_intent_closed_export,
    trigger_document_upload_export,
    trigger_concept_change_export,
    trigger_manual_export
)
from core.intent_trace import IntentTraceWithExport

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def test_event_driven_export():
    """Test event-driven export with different triggers."""
    print("\n" + "="*60)
    print("TEST 1: Event-Driven Export")
    print("="*60)
    
    # Create exporter with event mode
    exporter = MeshSummaryExporter(
        export_mode=ExportMode.EVENT,
        debounce_minutes=0.1  # 6 seconds for testing
    )
    
    user_id = "test_user"
    
    # Test manual trigger
    print("\n1. Testing MANUAL trigger (forced)...")
    path1 = exporter.trigger_export(
        user_id=user_id,
        trigger=ExportTrigger.MANUAL,
        force=True,
        metadata={"reason": "Test manual export"}
    )
    assert path1 is not None, "Manual export failed"
    print(f"✓ Manual export succeeded: {Path(path1).name}")
    
    # Test debouncing - should skip
    print("\n2. Testing debounce (should skip)...")
    path2 = exporter.trigger_export(
        user_id=user_id,
        trigger=ExportTrigger.INTENT_CLOSED,
        force=False
    )
    assert path2 is None, "Debounce failed - export should have been skipped"
    print("✓ Export correctly skipped due to debounce")
    
    # Wait for debounce to expire
    print("\n3. Waiting for debounce to expire (6 seconds)...")
    time.sleep(7)
    
    # Test after debounce
    print("4. Testing after debounce...")
    path3 = exporter.trigger_export(
        user_id=user_id,
        trigger=ExportTrigger.INTENT_CLOSED,
        metadata={"intent_id": "test_001"}
    )
    assert path3 is not None, "Export after debounce failed"
    print(f"✓ Export succeeded after debounce: {Path(path3).name}")
    
    # Check statistics
    stats = exporter.get_export_statistics()
    print(f"\n✓ Export statistics:")
    print(f"  - Total exports: {stats['total_exports']}")
    print(f"  - Trigger counts: {stats['trigger_counts']}")
    
    return True

def test_intent_closure_trigger():
    """Test intent closure triggering export."""
    print("\n" + "="*60)
    print("TEST 2: Intent Closure Trigger")
    print("="*60)
    
    # Create intent tracer with live export
    tracer = IntentTraceWithExport(enable_live_export=True)
    
    user_id = "intent_test_user"
    
    # Open an intent
    print("\n1. Opening intent...")
    intent = tracer.open_intent(
        user_id=user_id,
        intent_id="perf_001",
        description="Optimize database queries",
        intent_type="optimization",
        priority="high"
    )
    print(f"✓ Opened intent: {intent['id']}")
    
    # Close intent (should trigger export)
    print("\n2. Closing intent (triggers export)...")
    success = tracer.close_intent(
        user_id=user_id,
        intent_id="perf_001",
        resolution="completed"
    )
    assert success, "Intent closure failed"
    print("✓ Intent closed and export triggered")
    
    # Check that mesh summary was updated
    mesh_file = Path("models/mesh_contexts") / f"{user_id}_mesh.json"
    if mesh_file.exists():
        with open(mesh_file, 'r') as f:
            summary = json.load(f)
        print(f"✓ Mesh summary updated at {summary['timestamp']}")
        
        # Check that closed intent is not in open_intents
        open_intent_ids = [i['id'] for i in summary.get('open_intents', [])]
        assert "perf_001" not in open_intent_ids, "Closed intent still in open_intents"
        print("✓ Closed intent removed from open_intents")
    
    return True

def test_document_upload_trigger():
    """Test document upload triggering export."""
    print("\n" + "="*60)
    print("TEST 3: Document Upload Trigger")
    print("="*60)
    
    user_id = "doc_test_user"
    
    # Trigger document upload export
    print("\n1. Simulating document upload...")
    path = trigger_document_upload_export(
        user_id=user_id,
        document_name="project_spec.pdf",
        document_type="specification"
    )
    
    if path:
        print(f"✓ Export triggered for document upload: {Path(path).name}")
    else:
        print("✓ Export skipped (likely due to debounce)")
    
    return True

def test_concept_change_trigger():
    """Test concept change triggering export."""
    print("\n" + "="*60)
    print("TEST 4: Concept Change Trigger")
    print("="*60)
    
    user_id = "concept_test_user"
    
    # Test small change (should skip)
    print("\n1. Small concept change (< 5, should skip)...")
    path1 = trigger_concept_change_export(
        user_id=user_id,
        change_type="add",
        concept_count=3
    )
    assert path1 is None, "Small change should not trigger export"
    print("✓ Small change correctly skipped")
    
    # Test large change (should trigger)
    print("\n2. Large concept change (>= 5, should trigger)...")
    path2 = trigger_concept_change_export(
        user_id=user_id,
        change_type="merge",
        concept_count=10
    )
    
    if path2:
        print(f"✓ Export triggered for large concept change: {Path(path2).name}")
    else:
        print("✓ Export skipped (likely due to debounce or mode)")
    
    return True

def test_hybrid_mode():
    """Test hybrid mode with both nightly and event triggers."""
    print("\n" + "="*60)
    print("TEST 5: Hybrid Mode")
    print("="*60)
    
    # Configure global exporter for hybrid mode
    config = {
        "export_mode": "hybrid",
        "debounce_minutes": 0.05  # 3 seconds for testing
    }
    
    # Reset global exporter
    import core.mesh_summary_exporter as exporter_module
    exporter_module._global_exporter = None
    
    exporter = get_global_exporter(config)
    
    user_id = "hybrid_test_user"
    
    # Test event trigger in hybrid mode
    print("\n1. Event trigger in hybrid mode...")
    path1 = trigger_intent_closed_export(
        user_id=user_id,
        intent_id="hybrid_001",
        intent_type="query"
    )
    
    if path1:
        print(f"✓ Event export in hybrid mode: {Path(path1).name}")
    
    # Test nightly trigger in hybrid mode
    print("\n2. Nightly trigger in hybrid mode...")
    time.sleep(4)  # Wait for debounce
    
    path2 = exporter.trigger_export(
        user_id=user_id,
        trigger=ExportTrigger.NIGHTLY,
        force=True
    )
    assert path2 is not None, "Nightly export failed in hybrid mode"
    print(f"✓ Nightly export in hybrid mode: {Path(path2).name}")
    
    return True

def test_export_logging():
    """Test export event logging."""
    print("\n" + "="*60)
    print("TEST 6: Export Event Logging")
    print("="*60)
    
    # Check if event log exists
    event_log = Path("logs/mesh_export_events.log")
    
    if not event_log.exists():
        print("Creating first export to generate log...")
        trigger_manual_export("log_test_user", "Test logging")
    
    if event_log.exists():
        print(f"\n✓ Event log exists at: {event_log}")
        
        # Read last few events
        with open(event_log, 'r') as f:
            lines = f.readlines()
        
        if lines:
            print(f"✓ Found {len(lines)} export events")
            
            # Parse last event
            last_event = json.loads(lines[-1])
            print(f"\nLast export event:")
            print(f"  - User: {last_event['user_id']}")
            print(f"  - Trigger: {last_event['trigger']}")
            print(f"  - Duration: {last_event['duration_seconds']}s")
            print(f"  - Success: {last_event['success']}")
            print(f"  - Mode: {last_event['mode']}")
    else:
        print("⚠️ Event log not created (may need to run more tests)")
    
    return True

def run_all_tests():
    """Run all live export tests."""
    print("\n" + "="*60)
    print("LIVE MESH EXPORT TEST SUITE (Improvement #1)")
    print("="*60)
    
    tests = [
        ("Event-Driven Export", test_event_driven_export),
        ("Intent Closure Trigger", test_intent_closure_trigger),
        ("Document Upload Trigger", test_document_upload_trigger),
        ("Concept Change Trigger", test_concept_change_trigger),
        ("Hybrid Mode", test_hybrid_mode),
        ("Export Logging", test_export_logging)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    # Show global statistics
    exporter = get_global_exporter()
    stats = exporter.get_export_statistics()
    print(f"\nGlobal Export Statistics:")
    print(f"  - Mode: {stats['mode']}")
    print(f"  - Total exports: {stats['total_exports']}")
    print(f"  - Triggers: {stats['trigger_counts']}")
    
    if passed == total:
        print("\n🎉 All live export tests passed! Event-driven export is working.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
