# test_basic_flow.py - Test the most basic flow possible
import asyncio
import httpx
import numpy as np

async def test_basic():
    print("🧪 Testing basic embedding flow...\n")
    
    # Step 1: Direct HTTP test
    print("1️⃣ Direct HTTP embedding test...")
    # Configure proper timeout
    TIMEOUT = httpx.Timeout(
        30.0,     # total
        connect=5.0,
        read=30.0,
        write=30.0
    )
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            "http://localhost:8080/embed",
            json={"texts": ["Test sentence one.", "Test sentence two."]}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Got {len(data['embeddings'])} embeddings")
            print(f"   Dimensions: {len(data['embeddings'][0])}")
        else:
            print(f"❌ Failed: {response.status_code}")
            return
    
    # Step 2: Test with our wrapper
    print("\n2️⃣ Testing with our embedding client...")
    from core.embedding_client_noauth import embed_concepts
    
    result = await embed_concepts(["Quantum computing", "Machine learning"])
    print(f"✅ Got embeddings: shape {result.shape}")
    
    # Step 3: Test Penrose with real embeddings
    print("\n3️⃣ Testing Penrose verification...")
    from core.penrose_verifier_test import get_penrose_verifier
    
    verifier = get_penrose_verifier()
    penrose_result = verifier.verify_tessera(
        result,
        ["Quantum computing", "Machine learning"]
    )
    print(f"✅ Penrose status: {penrose_result.status}")
    
    # Step 4: Test JSON serialization
    print("\n4️⃣ Testing JSON serialization...")
    from core.json_encoder_fix import safe_json_dumps
    
    test_data = {
        "vectors": result.tolist(),
        "penrose": penrose_result.status,
        "numpy_bool": np.bool_(True),
        "numpy_float": np.float32(3.14)
    }
    
    json_str = safe_json_dumps(test_data)
    print(f"✅ JSON serialization worked: {len(json_str)} chars")
    
    print("\n✅ All basic tests passed!")

if __name__ == "__main__":
    asyncio.run(test_basic())
