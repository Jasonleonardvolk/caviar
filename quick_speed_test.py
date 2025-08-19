# quick_speed_test.py - Verify batching is working
import requests
import time

url = "http://127.0.0.1:8080/embed"

print("🏃 QUICK SPEED TEST - With proper batching")
print("=" * 50)

# Test 10 sentences
texts = [f"This is test sentence number {i}." for i in range(10)]
payload = {"texts": texts}

print("Testing 10 sentences...")
t0 = time.perf_counter()
response = requests.post(url, json=payload, timeout=30)
elapsed = time.perf_counter() - t0

if response.status_code == 200:
    data = response.json()
    server_ms = data.get('processing_time_ms', 0)
    
    print(f"\n✅ SUCCESS!")
    print(f"⏱️  Total time: {elapsed:.3f}s")
    print(f"⚡ Server time: {server_ms:.1f}ms")
    print(f"📊 Per text: {server_ms/10:.1f}ms")
    print(f"🚀 Speed: {10/(server_ms/1000):.1f} texts/second")
    
    print("\n🎯 EXPECTED vs ACTUAL:")
    print(f"   Before: 18,000ms (1,800ms per text)")
    print(f"   Target: <700ms (<70ms per text)")
    print(f"   Actual: {server_ms:.0f}ms ({server_ms/10:.0f}ms per text)")
    
    if server_ms < 1000:
        print("\n🎉 SUCCESS! That's a " + f"{18000/server_ms:.0f}x speedup!")
    else:
        print("\n⚠️  Still slow - check if using ultra_fast service")
else:
    print(f"❌ Error: {response.status_code}")
