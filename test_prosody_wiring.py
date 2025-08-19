"""
Test if prosody engine is properly wired up
"""
import sys
from pathlib import Path

print("🔍 TESTING PROSODY ENGINE WIRING...")
print("-" * 50)

# Test 1: Can we import the prosody engine?
try:
    sys.path.append(str(Path(__file__).parent))
    from prosody_engine.core import NetflixKillerProsodyEngine, get_prosody_engine
    print("✅ TEST 1 PASSED: Can import prosody engine core")
except Exception as e:
    print(f"❌ TEST 1 FAILED: Cannot import prosody engine core: {e}")

# Test 2: Can we import the API routes?
try:
    from prosody_engine.api import prosody_router
    print("✅ TEST 2 PASSED: Can import prosody API router")
except Exception as e:
    print(f"❌ TEST 2 FAILED: Cannot import prosody API router: {e}")

# Test 3: Can we create an engine instance?
try:
    engine = get_prosody_engine()
    emotion_count = len(engine.emotion_categories)
    print(f"✅ TEST 3 PASSED: Engine created with {emotion_count} emotions")
except Exception as e:
    print(f"❌ TEST 3 FAILED: Cannot create engine instance: {e}")

# Test 4: Check integration module
try:
    from prajna.api.prosody_integration import integrate_prosody_complete, PROSODY_ENGINE_AVAILABLE
    print(f"✅ TEST 4 PASSED: Integration module loaded (Available: {PROSODY_ENGINE_AVAILABLE})")
except Exception as e:
    print(f"❌ TEST 4 FAILED: Cannot import integration module: {e}")

# Test 5: Check if Prajna imports it
try:
    sys.path.insert(0, str(Path(__file__).parent))
    import prajna.api.prajna_api as prajna_api
    print(f"✅ TEST 5 PASSED: Prajna API imports prosody (Available: {prajna_api.PROSODY_ENGINE_AVAILABLE})")
except Exception as e:
    print(f"❌ TEST 5 FAILED: Cannot check Prajna import: {e}")

print("-" * 50)
print("🎯 WIRING TEST COMPLETE!")
