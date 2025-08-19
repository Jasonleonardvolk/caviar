#!/usr/bin/env python3
"""Test script to verify the health endpoint"""

import requests
import time

print("Testing SvelteKit health endpoint...")
print("-" * 50)

# Test different endpoints
endpoints = [
    "http://localhost:5173/health",
    "http://localhost:5173/",
    "http://127.0.0.1:5173/health",
    "http://127.0.0.1:5173/"
]

for endpoint in endpoints:
    try:
        print(f"\nTesting: {endpoint}")
        response = requests.get(endpoint, timeout=5)
        print(f"  Status: {response.status_code}")
        print(f"  Content: {response.text[:100]}")
        print(f"  Headers: {dict(response.headers)}")
    except requests.exceptions.ConnectionError:
        print(f"  ERROR: Connection refused - is the dev server running?")
    except requests.exceptions.Timeout:
        print(f"  ERROR: Request timed out")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
    
    time.sleep(0.5)

print("\n" + "-" * 50)
print("Test complete. If /health returns 404, try:")
print("1. Stop the dev server (Ctrl+C)")
print("2. Delete .svelte-kit folder: rm -rf .svelte-kit")
print("3. Run: npm run dev")
print("4. Wait for 'ready' message")
print("5. Run this test again")