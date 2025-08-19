import os
import sys
from pathlib import Path

# Setup environment
os.environ["TORI_ENV"] = "test"
sys.path.insert(0, str(Path(__file__).parent))

output = []

output.append("Testing route mounting...\n")

# Test 1: Direct router import
try:
    from api.routes.soliton import router as soliton_router
    output.append(f"✅ Soliton router imported: {soliton_router.prefix}")
    output.append(f"   Routes: {[r.path for r in soliton_router.routes]}")
except Exception as e:
    output.append(f"❌ Router import failed: {e}")

# Test 2: Check if it mounts to a test app
output.append("\nTesting FastAPI mounting...")
try:
    from fastapi import FastAPI
    test_app = FastAPI()
    test_app.include_router(soliton_router)
    
    routes = [r.path for r in test_app.routes if hasattr(r, 'path') and '/soliton' in r.path]
    output.append(f"✅ Mounted to test app: {len(routes)} soliton routes")
    for r in routes:
        output.append(f"   - {r}")
except Exception as e:
    output.append(f"❌ Mount test failed: {e}")

# Write output to file
with open("test_output/route_test_results.txt", "w") as f:
    f.write("\n".join(output))

print("Test complete! Check test_output/route_test_results.txt")
