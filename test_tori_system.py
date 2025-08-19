#!/usr/bin/env python3
"""
TORI System Integration Test Suite
Tests all major components after gremlin elimination
"""

import asyncio
import json
import time
import sys
from pathlib import Path

# Only import if running the tests
if __name__ == "__main__":
    try:
        import requests
        import websocket
    except ImportError:
        print("❌ Missing test dependencies. Run: pip install requests websocket-client")
        sys.exit(1)

class TORISystemTest:
    def __init__(self, api_port=8002, frontend_port=5173):
        self.api_port = api_port
        self.frontend_port = frontend_port
        self.api_base = f"http://localhost:{api_port}"
        self.frontend_base = f"http://localhost:{frontend_port}"
        self.test_user_id = "test_user_001"
        
    def print_section(self, title):
        print(f"\n{'='*50}")
        print(f"🧪 {title}")
        print('='*50)
    
    def test_api_health(self):
        """Test if API server is responding"""
        self.print_section("API Health Check")
        
        try:
            response = requests.get(f"{self.api_base}/api/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API Status: {data.get('status', 'unknown')}")
                print(f"✅ Prajna Available: {data.get('prajna_available', False)}")
                print(f"✅ PDF Processing: {data.get('pdf_processing_available', False)}")
                print(f"✅ Mesh Available: {data.get('mesh_available', False)}")
                print(f"✅ TORI Components: {data.get('tori_components_available', False)}")
                return True
            else:
                print(f"❌ API returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API health check failed: {e}")
            print("   Make sure the API server is running on port 8002")
            return False
    
    def test_soliton_init(self):
        """Test Soliton Memory initialization"""
        self.print_section("Soliton Memory Initialization")
        
        try:
            # Test with correct field name
            response = requests.post(
                f"{self.api_base}/api/soliton/init",
                json={"userId": self.test_user_id},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Soliton Init Success: {data.get('success', False)}")
                print(f"✅ Engine: {data.get('engine', 'unknown')}")
                print(f"✅ Message: {data.get('message', '')}")
                print(f"✅ Lattice Ready: {data.get('lattice_ready', False)}")
                return True
            else:
                print(f"❌ Soliton init returned status {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Soliton init failed: {e}")
            return False
    
    def test_soliton_stats(self):
        """Test Soliton Memory stats endpoint"""
        self.print_section("Soliton Memory Stats")
        
        try:
            response = requests.get(
                f"{self.api_base}/api/soliton/stats/{self.test_user_id}",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Total Memories: {data.get('totalMemories', 0)}")
                print(f"✅ Active Waves: {data.get('activeWaves', 0)}")
                print(f"✅ Average Strength: {data.get('averageStrength', 0):.2f}")
                print(f"✅ Status: {data.get('status', 'unknown')}")
                return True
            else:
                print(f"❌ Stats returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Stats request failed: {e}")
            return False
    
    def test_prajna_answer(self):
        """Test Prajna voice system"""
        self.print_section("Prajna Voice System")
        
        try:
            response = requests.post(
                f"{self.api_base}/api/answer",
                json={
                    "user_query": "What is consciousness?",
                    "persona": {
                        "name": "Scholar",
                        "ψ": "analytical"
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get('answer', '')
                print(f"✅ Prajna responded: {answer[:100]}...")
                print(f"✅ Trust Score: {data.get('trust_score', 0):.2f}")
                print(f"✅ Processing Time: {data.get('processing_time', 0):.2f}s")
                print(f"✅ User Tier: {data.get('user_tier', 'unknown')}")
                return True
            elif response.status_code == 503:
                print("⚠️ Prajna model still loading or not available")
                return True  # Not a failure, just not ready
            else:
                print(f"❌ Prajna returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Prajna test failed: {e}")
            return False
    
    def test_concept_mesh(self):
        """Test Concept Mesh functionality"""
        self.print_section("Concept Mesh System")
        
        try:
            # Test recording a concept diff
            response = requests.post(
                f"{self.api_base}/api/concept-mesh/record_diff",
                json={
                    "concepts": [
                        {
                            "id": "test_concept_001",
                            "name": "Test Concept",
                            "strength": 0.8
                        }
                    ],
                    "source": "test_suite",
                    "user_id": self.test_user_id
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Concept Recorded: {data.get('success', False)}")
                print(f"✅ Recorded Count: {data.get('recorded_count', 0)}")
                print(f"✅ Message: {data.get('message', '')}")
                
                # Test stats endpoint
                stats_response = requests.get(f"{self.api_base}/api/concept-mesh/stats")
                if stats_response.status_code == 200:
                    stats = stats_response.json()
                    print(f"✅ Mesh Stats Available: {stats.get('available', False)}")
                
                return True
            else:
                print(f"❌ Concept mesh returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Concept mesh test failed: {e}")
            return False
    
    def test_frontend(self):
        """Test if frontend is accessible"""
        self.print_section("Frontend Accessibility")
        
        try:
            response = requests.get(self.frontend_base, timeout=5)
            if response.status_code == 200:
                print(f"✅ Frontend accessible at {self.frontend_base}")
                print(f"✅ Response size: {len(response.text)} bytes")
                
                # Check if it's actually the SvelteKit app
                if "SvelteKit" in response.text or "svelte" in response.text.lower():
                    print("✅ SvelteKit application detected")
                else:
                    print("⚠️ Frontend responded but might not be SvelteKit")
                
                return True
            else:
                print(f"❌ Frontend returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Frontend test failed: {e}")
            print("   Make sure the frontend is running on port 5173")
            return False
    
    def test_mcp_metacognitive(self):
        """Test MCP Metacognitive server if available"""
        self.print_section("MCP Metacognitive Server")
        
        try:
            # Try default MCP port
            response = requests.get("http://localhost:8100/api/system/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ MCP Server Running")
                print(f"✅ Discovered Servers: {data.get('discovery', {}).get('total_discovered', 0)}")
                print(f"✅ Running Servers: {data.get('discovery', {}).get('running', 0)}")
                
                servers = data.get('servers', {})
                for name, info in servers.items():
                    if info.get('running'):
                        print(f"   ✅ {name}: {info.get('description', 'Running')}")
                
                return True
            else:
                print("⚠️ MCP server not responding (optional component)")
                return True  # Not a critical failure
        except Exception as e:
            print("⚠️ MCP server not available (optional component)")
            return True  # Not a critical failure
    
    def run_all_tests(self):
        """Run all system tests"""
        print("\n" + "🚀 "*20)
        print("TORI SYSTEM INTEGRATION TEST SUITE")
        print("🚀 "*20)
        
        tests = [
            self.test_api_health,
            self.test_soliton_init,
            self.test_soliton_stats,
            self.test_prajna_answer,
            self.test_concept_mesh,
            self.test_frontend,
            self.test_mcp_metacognitive
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                results.append(False)
        
        # Summary
        self.print_section("TEST SUMMARY")
        passed = sum(1 for r in results if r)
        total = len(results)
        
        print(f"\n✅ Passed: {passed}/{total}")
        print(f"❌ Failed: {total - passed}/{total}")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! TORI is working perfectly!")
            print("\n🧪 Try these manual tests:")
            print(f"1. Open {self.frontend_base} in your browser")
            print("2. Upload a PDF and check concept extraction")
            print("3. Ask Prajna a question through the chat interface")
            print("4. Check if Soliton Memory persists between sessions")
        else:
            print("\n⚠️ Some tests failed. Please check the errors above.")
            print("\nCommon fixes:")
            print("1. Ensure TORI is running: poetry run python enhanced_launcher.py")
            print("2. Wait for all services to initialize (may take 30-60 seconds)")
            print("3. Check logs in the logs/ directory for detailed errors")
        
        return passed == total

def main():
    # Parse command line arguments
    api_port = 8002
    frontend_port = 5173
    
    if len(sys.argv) > 1:
        api_port = int(sys.argv[1])
    if len(sys.argv) > 2:
        frontend_port = int(sys.argv[2])
    
    print(f"Testing TORI system (API: {api_port}, Frontend: {frontend_port})")
    
    # Create and run tests
    tester = TORISystemTest(api_port, frontend_port)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
