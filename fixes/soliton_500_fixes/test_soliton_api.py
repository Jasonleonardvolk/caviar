#!/usr/bin/env python3
"""
Test Soliton API endpoints
"""

import requests
import json
import sys
from datetime import datetime

# Colors for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

def test_endpoint(method, url, data=None, expected_status=200):
    """Test a single endpoint"""
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        status_color = Colors.GREEN if response.status_code == expected_status else Colors.RED
        print(f"{status_color}  Status: {response.status_code}{Colors.END}")
        
        if response.status_code == 200:
            print(f"  Response: {json.dumps(response.json(), indent=2)}")
        elif response.status_code == 500:
            print(f"{Colors.RED}  500 ERROR - This is what we're fixing!{Colors.END}")
            print(f"  Error preview: {response.text[:200]}...")
        
        return response.status_code == expected_status
        
    except requests.exceptions.ConnectionError:
        print(f"{Colors.RED}  CONNECTION FAILED - Is the API running?{Colors.END}")
        print(f"  Start it with: uvicorn api.main:app --port 5173 --reload")
        return False
    except Exception as e:
        print(f"{Colors.RED}  ERROR: {e}{Colors.END}")
        return False

def main():
    base_url = "http://localhost:5173"
    
    print(f"\n{Colors.CYAN}{'='*50}")
    print("SOLITON API ENDPOINT TESTS")
    print(f"{'='*50}{Colors.END}\n")
    
    print(f"Testing at: {base_url}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Health Check
    print(f"{Colors.YELLOW}1. Testing /api/soliton/health...{Colors.END}")
    tests_total += 1
    if test_endpoint("GET", f"{base_url}/api/soliton/health"):
        tests_passed += 1
    
    # Test 2: Diagnostic
    print(f"\n{Colors.YELLOW}2. Testing /api/soliton/diagnostic...{Colors.END}")
    tests_total += 1
    if test_endpoint("GET", f"{base_url}/api/soliton/diagnostic", expected_status=[200, 404]):
        tests_passed += 1
        print(f"  (404 is OK if fix not applied yet)")
    
    # Test 3: Init
    print(f"\n{Colors.YELLOW}3. Testing /api/soliton/init...{Colors.END}")
    tests_total += 1
    test_data = {"userId": f"test_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}
    if test_endpoint("POST", f"{base_url}/api/soliton/init", data=test_data):
        tests_passed += 1
    
    # Test 4: Stats
    print(f"\n{Colors.YELLOW}4. Testing /api/soliton/stats/test_user...{Colors.END}")
    tests_total += 1
    if test_endpoint("GET", f"{base_url}/api/soliton/stats/test_user"):
        tests_passed += 1
    
    # Summary
    print(f"\n{Colors.CYAN}{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}{Colors.END}\n")
    
    if tests_passed == tests_total:
        print(f"{Colors.GREEN}✓ All tests passed! ({tests_passed}/{tests_total}){Colors.END}")
    else:
        print(f"{Colors.RED}✗ Some tests failed: {tests_passed}/{tests_total} passed{Colors.END}")
        
        print(f"\n{Colors.YELLOW}If you see 500 errors:{Colors.END}")
        print("1. Run the main fix script:")
        print(f"   {Colors.CYAN}python fix_soliton_main.py{Colors.END}")
        print("2. Apply the API fix:")
        print(f"   {Colors.CYAN}python apply_soliton_api_fix.py{Colors.END}")
        print("3. Restart the API server")
    
    return tests_passed == tests_total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
