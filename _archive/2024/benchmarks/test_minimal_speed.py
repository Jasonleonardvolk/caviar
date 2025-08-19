# test_minimal_speed.py - Test absolute minimal embedding speed
import time
import requests

print("🏃 MINIMAL SPEED TEST - No formatting, no extras")
print("=" * 50)

# Test the absolute simplest case
url = "http://localhost:8080/embed"

# Single word
print("\n1. Single word:")
start = time.time()
response = requests.post(url, json={"texts": ["Hello"]}, timeout=30)
elapsed = time.time() - start
print(f"   Time: {elapsed:.3f}s")
if response.status_code == 200:
    data = response.json()
    print(f"   Server time: {data.get('processing_time_ms', 0):.1f}ms")

# Short sentence
print("\n2. Short sentence:")
start = time.time()
response = requests.post(url, json={"texts": ["The cat sat."]}, timeout=30)
elapsed = time.time() - start
print(f"   Time: {elapsed:.3f}s")

# Batch of 5 short texts
print("\n3. Batch of 5 short texts:")
texts = ["One", "Two", "Three", "Four", "Five"]
start = time.time()
response = requests.post(url, json={"texts": texts}, timeout=30)
elapsed = time.time() - start
print(f"   Time: {elapsed:.3f}s")
print(f"   Per text: {elapsed/5:.3f}s")

print("\n" + "=" * 50)
print("If these are still >1s, the issue is NOT the instruction formatting")
print("but something more fundamental with the GPU/model setup.")
