"""
test_mesh_lockdown.py — Verify Mesh Write Lockdown
=================================================
Tests to ensure mesh mutators are properly locked down and only accessible via Prajna API.
"""

import pytest
import requests
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from prajna.memory.concept_mesh_api import ConceptMeshAPI
    MESH_AVAILABLE = True
except ImportError:
    MESH_AVAILABLE = False
    print("⚠️ ConceptMeshAPI not available for testing")

# Test configuration
PRAJNA_API_URL = "http://localhost:8001"
PROPOSAL_ENDPOINT = f"{PRAJNA_API_URL}/api/prajna/propose"
HEALTH_ENDPOINT = f"{PRAJNA_API_URL}/api/health"

class TestMeshLockdown:
    """Test suite for mesh write lockdown."""
    
    @pytest.mark.skipif(not MESH_AVAILABLE, reason="Mesh API not available")
    @pytest.mark.asyncio
    async def test_direct_mesh_write_blocked(self):
        """Test that direct mesh writes are blocked by decorator."""
        mesh = ConceptMeshAPI()
        
        # Attempt direct call to locked method — should raise PermissionError
        with pytest.raises(PermissionError, match="may only be called from prajna_api.py"):
            await mesh._add_node_locked("forbidden_concept", "test context", {"source": "test"})
        
        print("✅ Direct mesh write properly blocked!")

    @pytest.mark.skipif(not MESH_AVAILABLE, reason="Mesh API not available")
    def test_public_mesh_methods_blocked(self):
        """Test that public mesh methods are blocked."""
        mesh = ConceptMeshAPI()
        
        # All these should raise PermissionError
        with pytest.raises(PermissionError, match="Use /api/prajna/propose endpoint"):
            mesh.add_node("bad", "bad", {"bad": "bad"})
        
        with pytest.raises(PermissionError, match="Use /api/prajna/propose endpoint"):
            mesh.add_edge("bad1", "bad2")
            
        with pytest.raises(PermissionError, match="Use /api/prajna/propose endpoint"):
            mesh.update_node("bad", {"bad": "bad"})
            
        print("✅ All public mesh methods properly blocked!")

    def test_prajna_api_health(self):
        """Test that Prajna API is running and reports lockdown status."""
        try:
            response = requests.get(HEALTH_ENDPOINT, timeout=5)
            assert response.status_code == 200
            
            health_data = response.json()
            assert health_data["status"] == "healthy"
            
            # Check if mesh lockdown is reported
            features = health_data.get("features", [])
            mesh_lockdown_enabled = health_data.get("mesh_lockdown_enabled", False)
            
            print(f"✅ Prajna API health check passed!")
            print(f"📊 Mesh lockdown enabled: {mesh_lockdown_enabled}")
            print(f"🎯 Features: {features}")
            
            return True
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Prajna API not running on localhost:8001")
        except Exception as e:
            pytest.fail(f"Health check failed: {e}")

    def test_mesh_proposal_api_success(self):
        """Test that mesh proposals work through the API."""
        proposal = {
            "concept": f"test_concept_{int(datetime.now().timestamp())}",
            "context": "Test context for lockdown verification",
            "provenance": {
                "source": "test_suite",
                "test_type": "lockdown_verification",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        try:
            response = requests.post(
                PROPOSAL_ENDPOINT,
                json=proposal,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 503:
                pytest.skip("Mesh API not available in Prajna")
                
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            
            result = response.json()
            assert result["status"] == "success"
            assert result["lockdown_enforced"] == True
            assert "result" in result
            assert "node_id" in result["result"]
            
            print("✅ Mesh proposal API test passed!")
            print(f"📊 Response: {result}")
            
            return result
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Prajna API not running on localhost:8001")
        except Exception as e:
            pytest.fail(f"Mesh proposal test failed: {e}")

    def test_invalid_proposal_rejected(self):
        """Test that invalid proposals are properly rejected."""
        invalid_proposals = [
            {},  # Empty proposal
            {"concept": ""},  # Empty concept
            {"concept": "test"},  # Missing context and provenance
            {"concept": "test", "context": "test"}  # Missing provenance
        ]
        
        try:
            for proposal in invalid_proposals:
                response = requests.post(
                    PROPOSAL_ENDPOINT,
                    json=proposal,
                    headers={"Content-Type": "application/json"},
                    timeout=5
                )
                
                # Should get validation error (422) or bad request (400)
                assert response.status_code in [400, 422], f"Invalid proposal should be rejected: {proposal}"
            
            print("✅ Invalid proposals properly rejected!")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Prajna API not running on localhost:8001")

    @pytest.mark.skipif(not MESH_AVAILABLE, reason="Mesh API not available")
    def test_read_only_methods_work(self):
        """Test that read-only mesh methods still work."""
        mesh = ConceptMeshAPI()
        
        # These should work fine (read-only)
        stats = mesh.get_mesh_stats()
        assert isinstance(stats, dict)
        assert "total_nodes" in stats
        assert "lockdown_active" in stats
        assert stats["lockdown_active"] == True
        
        # Search should work
        results = mesh.search_concepts("test", limit=5)
        assert isinstance(results, list)
        
        # Get mutations should work
        mutations = mesh.get_recent_mutations(limit=3)
        assert isinstance(mutations, list)
        
        print("✅ Read-only mesh methods work correctly!")
        print(f"📊 Mesh stats: {stats}")

def run_manual_tests():
    """Run tests manually without pytest."""
    print("🧪 Running manual mesh lockdown tests...")
    
    test_instance = TestMeshLockdown()
    
    # Test 1: API Health
    print("\n1️⃣ Testing Prajna API health...")
    try:
        test_instance.test_prajna_api_health()
    except Exception as e:
        print(f"❌ Health test failed: {e}")
    
    # Test 2: Proposal API
    print("\n2️⃣ Testing mesh proposal API...")
    try:
        test_instance.test_mesh_proposal_api_success()
    except Exception as e:
        print(f"❌ Proposal test failed: {e}")
    
    # Test 3: Invalid proposals
    print("\n3️⃣ Testing invalid proposal rejection...")
    try:
        test_instance.test_invalid_proposal_rejected()
    except Exception as e:
        print(f"❌ Invalid proposal test failed: {e}")
    
    # Test 4: Direct writes (if mesh available)
    if MESH_AVAILABLE:
        print("\n4️⃣ Testing direct write blocking...")
        try:
            test_instance.test_public_mesh_methods_blocked()
        except Exception as e:
            print(f"❌ Direct write test failed: {e}")
        
        print("\n5️⃣ Testing read-only methods...")
        try:
            test_instance.test_read_only_methods_work()
        except Exception as e:
            print(f"❌ Read-only test failed: {e}")
    
    print("\n🎉 Manual testing complete!")

if __name__ == "__main__":
    print("🔒 MESH LOCKDOWN TEST SUITE")
    print("=" * 50)
    
    # Check if pytest is available
    try:
        import pytest
        print("Running with pytest...")
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, running manual tests...")
        run_manual_tests()
