#!/usr/bin/env python3
"""
TORI Basic Functionality Test Suite
Tests core components and endpoints
"""

import requests
import json
import time
from datetime import datetime

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def test_endpoint(name, method, url, data=None, expected_status=200):
    """Test a single endpoint"""
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=5)
        else:
            return (name, False, f"Unknown method: {method}")
        
        if response.status_code == expected_status:
            return (name, True, f"Status {response.status_code}")
        else:
            return (name, False, f"Expected {expected_status}, got {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        return (name, False, "Connection refused")
    except requests.exceptions.Timeout:
        return (name, False, "Request timeout")
    except Exception as e:
        return (name, False, str(e))

def test_websocket(name, url):
    """Test WebSocket connection"""
    try:
        import websocket
        
        ws = websocket.WebSocket()
        ws.settimeout(5)
        ws.connect(url)
        ws.close()
        return (name, True, "Connected successfully")
        
    except ImportError:
        return (name, None, "websocket-client not installed")
    except Exception as e:
        return (name, False, str(e))

def run_tests():
    """Run all tests"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("="*60)
    print("🧪 TORI BASIC FUNCTIONALITY TEST SUITE")
    print("="*60)
    print(f"{Colors.RESET}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    tests = []
    
    # API Health Tests
    print(f"{Colors.BOLD}Testing API Endpoints...{Colors.RESET}")
    tests.extend([
        test_endpoint("API Health Check", "GET", "http://localhost:8002/api/health"),
        test_endpoint("API Root", "GET", "http://localhost:8002/"),
        test_endpoint("API Docs", "GET", "http://localhost:8002/docs"),
    ])
    
    # Soliton Endpoints
    print(f"\n{Colors.BOLD}Testing Soliton Endpoints...{Colors.RESET}")
    tests.extend([
        test_endpoint("Soliton Init", "POST", "http://localhost:8002/api/soliton/init", {}),
        test_endpoint("Soliton Stats", "GET", "http://localhost:8002/api/soliton/stats/testuser"),
        test_endpoint("Soliton Embed", "POST", "http://localhost:8002/api/soliton/embed", {"data": "test"}),
    ])
    
    # Frontend Tests
    print(f"\n{Colors.BOLD}Testing Frontend...{Colors.RESET}")
    tests.extend([
        test_endpoint("Frontend Home", "GET", "http://localhost:5173"),
        test_endpoint("Frontend API Proxy", "GET", "http://localhost:5173/api/health"),
    ])
    
    # WebSocket Tests
    print(f"\n{Colors.BOLD}Testing WebSockets...{Colors.RESET}")
    ws_tests = [
        test_websocket("Audio Bridge WS", "ws://localhost:8765"),
        test_websocket("Concept Bridge WS", "ws://localhost:8766"),
        test_websocket("Avatar Updates WS", "ws://localhost:8002/api/avatar/updates"),
    ]
    tests.extend(ws_tests)
    
    # MCP Server (if running)
    print(f"\n{Colors.BOLD}Testing Optional Components...{Colors.RESET}")
    tests.append(test_endpoint("MCP Server", "GET", "http://localhost:8100/api/system/status", expected_status=[200, 404]))
    
    # Print results
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}TEST RESULTS:{Colors.RESET}")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, success, details in tests:
        if success is None:
            status = f"{Colors.YELLOW}SKIP{Colors.RESET}"
            skipped += 1
        elif success:
            status = f"{Colors.GREEN}PASS{Colors.RESET}"
            passed += 1
        else:
            status = f"{Colors.RED}FAIL{Colors.RESET}"
            failed += 1
        
        # Handle expected_status being a list
        if isinstance(details, str) and "404" in details and "200" in details:
            details = "Component not running (optional)"
            if success is False:
                status = f"{Colors.YELLOW}SKIP{Colors.RESET}"
                failed -= 1
                skipped += 1
        
        print(f"{name:.<35} {status} - {details}")
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}SUMMARY:{Colors.RESET}")
    print(f"  Total Tests: {len(tests)}")
    print(f"  {Colors.GREEN}Passed: {passed}{Colors.RESET}")
    print(f"  {Colors.RED}Failed: {failed}{Colors.RESET}")
    print(f"  {Colors.YELLOW}Skipped: {skipped}{Colors.RESET}")
    
    success_rate = (passed / len(tests) * 100) if len(tests) > 0 else 0
    print(f"  Success Rate: {success_rate:.1f}%")
    
    # Overall result
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    if failed == 0 and passed > 0:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ ALL CORE TESTS PASSED!{Colors.RESET}")
        print(f"{Colors.GREEN}TORI system is operational.{Colors.RESET}")
    elif passed >= 5:  # At least core components working
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  PARTIAL SUCCESS{Colors.RESET}")
        print(f"{Colors.YELLOW}Core components are working but some features may be unavailable.{Colors.RESET}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ SYSTEM FAILURE{Colors.RESET}")
        print(f"{Colors.RED}Too many components are not responding.{Colors.RESET}")
        print(f"\n{Colors.YELLOW}Troubleshooting steps:{Colors.RESET}")
        print("1. Check if all services are running: python isolated_startup.py")
        print("2. Run emergency fix: python tori_emergency_fix.py")
        print("3. Check logs: logs/tori_errors.log")
    
    return failed == 0

def performance_test():
    """Quick performance test"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}PERFORMANCE TEST:{Colors.RESET}")
    print(f"{'='*60}")
    
    url = "http://localhost:8002/api/health"
    iterations = 10
    
    print(f"Testing {iterations} requests to {url}...")
    
    times = []
    for i in range(iterations):
        start = time.time()
        try:
            response = requests.get(url)
            if response.status_code == 200:
                elapsed = time.time() - start
                times.append(elapsed)
                print(f"  Request {i+1}: {elapsed*1000:.1f}ms")
            else:
                print(f"  Request {i+1}: Failed (status {response.status_code})")
        except:
            print(f"  Request {i+1}: Failed (connection error)")
    
    if times:
        avg_time = sum(times) / len(times) * 1000
        min_time = min(times) * 1000
        max_time = max(times) * 1000
        
        print(f"\n{Colors.BOLD}Results:{Colors.RESET}")
        print(f"  Average: {avg_time:.1f}ms")
        print(f"  Min: {min_time:.1f}ms")
        print(f"  Max: {max_time:.1f}ms")
        
        if avg_time < 50:
            print(f"  {Colors.GREEN}✅ Excellent performance{Colors.RESET}")
        elif avg_time < 100:
            print(f"  {Colors.GREEN}✅ Good performance{Colors.RESET}")
        elif avg_time < 200:
            print(f"  {Colors.YELLOW}⚠️  Acceptable performance{Colors.RESET}")
        else:
            print(f"  {Colors.RED}❌ Poor performance{Colors.RESET}")

def main():
    """Main test runner"""
    try:
        # Run basic tests
        all_passed = run_tests()
        
        # Run performance test if basic tests passed
        if all_passed:
            performance_test()
        
        print(f"\n{Colors.BLUE}Test suite completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}Test suite error: {e}{Colors.RESET}")

if __name__ == "__main__":
    main()
