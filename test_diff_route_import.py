"""
Quick test to check if diff_route can be imported
"""
import sys
from pathlib import Path

# Add paths like prajna_api does
sys.path.insert(0, str(Path(__file__).parent))

try:
    from api.diff_route import router as concept_mesh_router
    print("✅ SUCCESS: diff_route imported successfully!")
    print(f"   Router prefix: {concept_mesh_router.prefix}")
    print(f"   Routes: {[r.path for r in concept_mesh_router.routes]}")
except Exception as e:
    print(f"❌ FAILED to import diff_route: {e}")
    import traceback
    traceback.print_exc()

# Also test deepdiff
try:
    import deepdiff
    print("✅ deepdiff is available")
except ImportError:
    print("❌ deepdiff is NOT installed")
