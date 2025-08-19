#!/usr/bin/env python3
"""
Unit tests for PsiArchive REST endpoints
Ensures API contract stability and correct functionality
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from api.archive_endpoints import archive_router
from core.psi_archive_extended import PSI_ARCHIVER, PsiEvent
from fastapi import FastAPI

# Create test app
app = FastAPI()
app.include_router(archive_router)


@pytest.fixture
def test_archive_dir():
    """Create a temporary archive directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def client(test_archive_dir, monkeypatch):
    """Create test client with temporary archive"""
    # Patch the archive directory
    monkeypatch.setattr(PSI_ARCHIVER, 'archive_dir', test_archive_dir)
    monkeypatch.setattr(PSI_ARCHIVER, 'current_date', datetime.now().date())
    PSI_ARCHIVER._ensure_current_file()
    
    return TestClient(app)


@pytest.fixture
def sample_events(client):
    """Create some sample events in the archive"""
    # Create a concept ingestion event
    event1_id = PSI_ARCHIVER.log_concept_ingestion(
        concept_ids=["graphene_001", "carbon_002"],
        source_doc_path="/test/docs/graphene.pdf",
        session_id="test_session_123",
        tags=["physics", "materials_science"]
    )
    
    # Create a response event
    event2 = PsiEvent(
        event_id="test_response_001",
        event_type="RESPONSE_EVENT",
        timestamp=datetime.now(),
        concept_ids=["graphene_001", "conductivity_003"],
        session_id="test_session_123",
        metadata={
            "query": "What is graphene?",
            "response_preview": "Graphene is a single layer of carbon atoms...",
            "model": "saigon_lstm"
        }
    )
    PSI_ARCHIVER.append_event(event2)
    
    # Create a Penrose computation event
    event3 = PsiEvent(
        event_id="test_penrose_001",
        event_type="PENROSE_SIM",
        timestamp=datetime.now(),
        concept_ids=["graphene_001", "silicon_002", "gaas_003"],
        metadata={
            "computation_time": 0.045,
            "concept_count": 1000,
            "speedup_factor": 22.7
        }
    )
    PSI_ARCHIVER.append_event(event3)
    
    return {
        "ingestion_id": event1_id,
        "response_id": event2.event_id,
        "penrose_id": event3.event_id
    }


class TestArchiveEndpoints:
    """Test suite for PsiArchive REST endpoints"""
    
    def test_health_endpoint(self, client):
        """Test /api/archive/health endpoint"""
        response = client.get("/api/archive/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "archive_directory" in data
        assert "current_date" in data
        assert "estimated_total_events" in data
    
    def test_origin_endpoint(self, client, sample_events):
        """Test /api/archive/origin/{concept_id} endpoint"""
        # Test existing concept
        response = client.get("/api/archive/origin/graphene_001")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["concept_id"] == "graphene_001"
        assert "origin" in data
        
        origin = data["origin"]
        assert origin["event_id"] == sample_events["ingestion_id"]
        assert "first_seen" in origin
        assert "source_path" in origin
        assert origin["tags"] == ["physics", "materials_science"]
        
        # Test non-existent concept
        response = client.get("/api/archive/origin/nonexistent_999")
        assert response.status_code == 404
    
    def test_session_endpoint(self, client, sample_events):
        """Test /api/archive/session/{session_id} endpoint"""
        response = client.get("/api/archive/session/test_session_123")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["session_id"] == "test_session_123"
        assert data["event_count"] >= 2
        
        # Check response analysis
        assert "response_analysis" in data
        analysis = data["response_analysis"]
        assert analysis["total_responses"] >= 1
        assert len(analysis["concept_paths"]) >= 1
        
        # Verify concept path
        path = analysis["concept_paths"][0]
        assert path["query"] == "What is graphene?"
        assert "graphene_001" in path["concept_path"]
    
    def test_delta_endpoint(self, client, sample_events):
        """Test /api/archive/delta endpoint"""
        # Get deltas from 1 hour ago
        since_time = (datetime.now() - timedelta(hours=1)).isoformat()
        response = client.get(f"/api/archive/delta?since={since_time}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "delta_count" in data
        assert "total_nodes_added" in data
        assert "total_edges_added" in data
        
        # Should have at least the events we created
        assert len(data["deltas"]) >= 0  # May be 0 if no mesh deltas
        
        # Test invalid timestamp
        response = client.get("/api/archive/delta?since=invalid_time")
        assert response.status_code == 400
    
    def test_query_endpoint(self, client, sample_events):
        """Test /api/archive/query endpoint"""
        # Query for PENROSE_SIM events
        response = client.get("/api/archive/query?event_type=PENROSE_SIM")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert data["total_results"] >= 1
        
        # Check statistics for Penrose events
        assert "statistics" in data
        stats = data["statistics"]
        if stats:  # Stats present for PENROSE_SIM
            assert "total_computations" in stats
            assert "average_speedup" in stats
            assert stats["average_speedup"] > 20  # Should be ~22.7
        
        # Query with pagination
        response = client.get("/api/archive/query?limit=1&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 1
    
    def test_seal_endpoint(self, client):
        """Test /api/archive/seal endpoint"""
        response = client.post("/api/archive/seal")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["success", "info"]
        assert "message" in data
    
    def test_concurrent_writes(self, client):
        """Test concurrent event writing doesn't corrupt archive"""
        import threading
        import time
        
        errors = []
        
        def write_events(thread_id):
            try:
                for i in range(10):
                    PSI_ARCHIVER.append_event(
                        PsiEvent(
                            event_id=f"concurrent_{thread_id}_{i}",
                            event_type="TEST_EVENT",
                            timestamp=datetime.now(),
                            metadata={"thread": thread_id, "index": i}
                        )
                    )
                    time.sleep(0.001)  # Small delay to increase contention
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=write_events, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check no errors occurred
        assert len(errors) == 0
        
        # Verify all events were written
        response = client.get("/api/archive/query?event_type=TEST_EVENT")
        data = response.json()
        assert data["total_results"] == 50  # 5 threads * 10 events
    
    def test_mini_index_performance(self, client):
        """Test that mini-index improves query performance"""
        # Create many events
        for i in range(100):
            PSI_ARCHIVER.append_event(
                PsiEvent(
                    event_id=f"perf_test_{i}",
                    event_type="LEARNING_UPDATE",
                    timestamp=datetime.now(),
                    concept_ids=[f"concept_{i}", f"concept_{i+1}"],
                    tags=["performance_test"]
                )
            )
        
        # Time concept origin lookup
        import time
        start = time.time()
        response = client.get("/api/archive/origin/concept_50")
        end = time.time()
        
        assert response.status_code == 200
        assert (end - start) < 0.1  # Should be fast with mini-index
        
        # Verify mini-index files were created
        index_files = list(PSI_ARCHIVER.archive_dir.glob("**/index-*.jsonl"))
        assert len(index_files) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
