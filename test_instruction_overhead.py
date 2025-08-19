# test_instruction_overhead.py - Test if instruction formatting is the issue
import requests
import time

url = "http://127.0.0.1:8080/embed"

print("🔍 TESTING INSTRUCTION FORMATTING OVERHEAD")
print("=" * 50)

# Test 1: Without instruction formatting (if your service allows)
print("\n1️⃣ Raw text (no instruction):")
texts = ["Test one", "Test two", "Test three", "Test four", "Test five"]
start = time.time()
response = requests.post(url, json={"texts": texts, "instruction": ""}, timeout=30)
time1 = time.time() - start
if response.status_code == 200:
    print(f"   Time: {time1:.3f}s")
    print(f"   Server: {response.json()['processing_time_ms']:.1f}ms")

# Test 2: With default instruction
print("\n2️⃣ With instruction formatting:")
start = time.time()
response = requests.post(url, json={"texts": texts}, timeout=30)
time2 = time.time() - start
if response.status_code == 200:
    print(f"   Time: {time2:.3f}s")
    print(f"   Server: {response.json()['processing_time_ms']:.1f}ms")

# Test 3: Very short texts
print("\n3️⃣ Single words (minimal processing):")
words = ["cat", "dog", "fish", "bird", "mouse"]
start = time.time()
response = requests.post(url, json={"texts": words}, timeout=30)
time3 = time.time() - start
if response.status_code == 200:
    data = response.json()
    print(f"   Time: {time3:.3f}s")
    print(f"   Server: {data['processing_time_ms']:.1f}ms")
    print(f"   Per word: {data['processing_time_ms']/5:.1f}ms")

print("\n" + "=" * 50)
print("If single words still take >200ms each, the issue is deeper than formatting.")
