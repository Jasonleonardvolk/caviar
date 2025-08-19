"""Test Phase 6-7 integration: diff endpoint and oscillator feed."""
import asyncio
import json
import time
from pathlib import Path
import pytest
import httpx

# Test configuration
API_BASE = "http://localhost:8002"
TEST_RECORD_ID = "test_record_001"
TEST_CONCEPT_ID = "test_concept_001"

@pytest.mark.asyncio
async def test_concept_diff_endpoint():
    """Test Phase 6: Concept mesh diff endpoint."""
    async with httpx.AsyncClient() as client:
        # Submit a diff
        response = await client.post(
            f"{API_BASE}/api/concept-mesh/record_diff",
            json={
                "record_id": TEST_RECORD_ID,
                "concept_id": TEST_CONCEPT_ID,
                "operation": "update",
                "metadata": {"source": "test"}
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert "diff_id" in data
        
        print(f"✅ Phase 6: Diff queued successfully: {data['diff_id']}")
        
        # Verify the diff was written to queue
        psi_archive = Path("data/psi_archive")
        queue_files = list(psi_archive.glob("diff_queue_*.jsonl"))
        assert len(queue_files) > 0
        
        # Read the last line from the most recent queue file
        with open(sorted(queue_files)[-1], "r") as f:
            lines = f.readlines()
            last_diff = json.loads(lines[-1])
            assert last_diff["concept_id"] == TEST_CONCEPT_ID
            
        print(f"✅ Phase 6: Diff written to queue: {last_diff['diff_id']}")

def test_oscillator_feed():
    """Test Phase 7: Oscillator feed integration."""
    # Import and setup
    from python.core.fractal_soliton_events import concept_event_bus, ConceptEvent
    from python.core.lattice_evolution_subscriber import oscillator_count
    from datetime import datetime
    
    initial_count = oscillator_count
    
    # Emit a test event
    test_event = ConceptEvent(
        concept_id="test_oscillator_001",
        phase=0.5,
        operation="add",
        timestamp=datetime.utcnow()
    )
    
    # This should trigger the lattice subscriber
    asyncio.run(concept_event_bus.emit("concept_added", test_event))
    
    # Give it a moment to process
    time.sleep(0.1)
    
    # Check that oscillator count increased
    from python.core.lattice_evolution_subscriber import oscillator_count as new_count
    assert new_count > initial_count
    
    print(f"✅ Phase 7: Oscillator count increased from {initial_count} to {new_count}")

if __name__ == "__main__":
    print("Testing Phase 6-7 Integration...")
    print("=" * 50)
    
    # Test Phase 6
    asyncio.run(test_concept_diff_endpoint())
    
    # Test Phase 7
    test_oscillator_feed()
    
    print("=" * 50)
    print("✅ All integration tests passed!")
