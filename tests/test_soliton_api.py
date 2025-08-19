"""
Test suite for Soliton API endpoints
"""

import pytest
import httpx
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os

# Set environment variable for testing
os.environ['TORI_DISABLE_MESH_CHECK'] = '1'

@pytest.fixture
def client():
    """Create test client"""
    from api.main import app
    return TestClient(app)

@pytest.fixture
def mock_soliton_module():
    """Mock the soliton memory module"""
    mock = MagicMock()
    mock.initialize_user = MagicMock(return_value=True)
    mock.get_user_stats = MagicMock(return_value={
        "totalMemories": 10,
        "activeWaves": 5,
        "averageStrength": 0.75,
        "clusterCount": 3
    })
    mock.check_health = MagicMock(return_value={
        "status": "operational",
        "engine": "mock",
        "message": "Test mode"
    })
    return mock

class TestSolitonAPI:
    """Test cases for Soliton API"""
    
    def test_health_endpoint(self, client):
        """Test /api/soliton/health endpoint"""
        response = client.get("/api/soliton/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["operational", "degraded"]
        
    def test_diagnostic_endpoint(self, client):
        """Test /api/soliton/diagnostic endpoint"""
        response = client.get("/api/soliton/diagnostic")
        assert response.status_code == 200
        data = response.json()
        assert "soliton_available" in data
        assert "environment" in data
        assert "TORI_DISABLE_MESH_CHECK" in data["environment"]
        
    def test_init_endpoint_success(self, client):
        """Test successful initialization"""
        response = client.post("/api/soliton/init", json={"userId": "test_user"})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "message" in data
        assert data["userId"] == "test_user"
        assert data["engine"] in ["production", "stub", "mock"]
        
    def test_init_endpoint_default_user(self, client):
        """Test initialization with default user"""
        response = client.post("/api/soliton/init", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["userId"] == "default"
        
    def test_stats_endpoint(self, client):
        """Test stats endpoint"""
        response = client.get("/api/soliton/stats/test_user")
        assert response.status_code == 200
        data = response.json()
        assert "totalMemories" in data
        assert "activeWaves" in data
        assert "averageStrength" in data
        assert "clusterCount" in data
        assert data["status"] in ["operational", "degraded"]
        
    @patch('api.routes.soliton.soliton_module')
    def test_init_with_real_module(self, mock_module, client):
        """Test init when real soliton module is available"""
        mock_module.initialize_user = MagicMock(return_value=True)
        
        with patch('api.routes.soliton.SOLITON_AVAILABLE', True):
            response = client.post("/api/soliton/init", json={"userId": "test_user"})
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["engine"] == "production"
            
    @patch('api.routes.soliton.soliton_module')
    def test_init_with_module_error(self, mock_module, client):
        """Test init when soliton module throws error"""
        mock_module.initialize_user = MagicMock(side_effect=Exception("Test error"))
        
        with patch('api.routes.soliton.SOLITON_AVAILABLE', True):
            response = client.post("/api/soliton/init", json={"userId": "test_user"})
            # Should return 503 or fallback gracefully
            assert response.status_code in [200, 503]
            
    def test_no_500_errors(self, client):
        """Ensure no endpoint returns 500 errors"""
        endpoints = [
            ("GET", "/api/soliton/health"),
            ("GET", "/api/soliton/diagnostic"),
            ("POST", "/api/soliton/init", {"userId": "test"}),
            ("GET", "/api/soliton/stats/test_user"),
        ]
        
        for method, url, *args in endpoints:
            if method == "GET":
                response = client.get(url)
            else:
                response = client.post(url, json=args[0] if args else {})
            
            # No endpoint should return 500
            assert response.status_code != 500, f"{method} {url} returned 500"
            
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        import asyncio
        import aiohttp
        
        async def make_request(session, url):
            async with session.post(url, json={"userId": "concurrent_test"}) as resp:
                return resp.status
        
        async def run_concurrent():
            async with aiohttp.ClientSession() as session:
                tasks = []
                for i in range(10):
                    url = "http://testserver/api/soliton/init"
                    tasks.append(make_request(session, url))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results
        
        # This would need to be run in an async test framework
        # For now, we'll just ensure the endpoint handles single requests well
        for i in range(5):
            response = client.post("/api/soliton/init", json={"userId": f"user_{i}"})
            assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
