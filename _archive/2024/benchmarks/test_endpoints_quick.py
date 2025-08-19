#!/usr/bin/env python3
import requests

# Test embed endpoint
try:
    r = requests.post("http://localhost:8002/api/soliton/embed", 
                     json={"text": "test embedding"})
    print(f"Embed endpoint: {r.status_code}")
    if r.status_code == 200:
        print("SUCCESS: Got embedding: {} dimensions".format(len(r.json()['embedding'])))
    else:
        print("ERROR: Embed endpoint not working")
except:
    print("ERROR: Could not reach API")

# Test stats endpoint  
try:
    r = requests.get("http://localhost:8002/api/soliton/stats/adminuser")
    print(f"\nStats endpoint: {r.status_code}")
    if r.status_code == 200:
        print("SUCCESS: Stats: {}".format(r.json()))
except:
    print("ERROR: Stats endpoint error")
