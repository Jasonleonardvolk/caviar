#!/usr/bin/env python3
"""Test all TORI components after fixes"""

import requests
import json
import time
import sys

def test_component(name, test_func):
    """Test a component and report results"""
    print(f"\n🧪 Testing {name}...", end="", flush=True)
    try:
        result = test_func()
        if result:
            print(f" ✅ PASS")
            return True
        else:
            print(f" ❌ FAIL")
            return False
    except Exception as e:
        print(f" ❌ ERROR: {e}")
        return False

def test_api_health():
    """Test main API health"""
    try:
        r = requests.get("http://localhost:8002/api/health", timeout=5)
        return r.status_code == 200
    except:
        return False

def test_soliton_health():
    """Test Soliton API health"""
    try:
        r = requests.get("http://localhost:8002/api/soliton/health", timeout=5)
        return r.status_code == 200
    except:
        return False

def test_soliton_stats():
    """Test Soliton stats endpoint"""
    try:
        r = requests.get("http://localhost:8002/api/soliton/stats/test_user", timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"     Stats: {json.dumps(data, indent=2)}")
            return True
        return False
    except:
        return False

def test_frontend():
    """Test frontend availability"""
    try:
        r = requests.get("http://localhost:5173", timeout=5)
        return r.status_code == 200
    except:
        return False

def main():
    print("🔬 TORI Component Test Suite")
    print("=" * 50)
    
    # Check if API is running
    if not test_component("API Connection", test_api_health):
        print("\n⚠️  API server not running!")
        print("Start it with: python enhanced_launcher.py")
        return
    
    # Test all components
    tests = [
        ("Soliton Health", test_soliton_health),
        ("Soliton Stats", test_soliton_stats),
        ("Frontend", test_frontend),
    ]
    
    passed = 0
    total = len(tests) + 1  # +1 for API connection
    
    for name, test_func in tests:
        if test_component(name, test_func):
            passed += 1
    
    print(f"\n📊 Results: {passed+1}/{total} tests passed")
    
    if passed + 1 == total:
        print("🎉 All components operational!")
    else:
        print("⚠️  Some components need attention")

if __name__ == "__main__":
    main()
