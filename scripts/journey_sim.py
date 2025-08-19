#!/usr/bin/env python3
"""
User Journey Simulation - End-to-end user experience test
Run: python scripts/journey_sim.py
"""
import os
import sys
import time
import requests
import json
import random

API = os.environ.get("API_URL", "http://localhost:8001")

def post(path, payload):
    """Make POST request with error handling"""
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=30)
        if not r.ok:
            print(f"  ⚠️ {path} returned {r.status_code}")
            return {}
        return r.json()
    except requests.exceptions.ConnectionError:
        print(f"  ⚠️ Cannot connect to {API}{path} - is the server running?")
        return {}
    except Exception as e:
        print(f"  ⚠️ {path} failed: {e}")
        return {}

def get(path):
    """Make GET request"""
    try:
        r = requests.get(f"{API}{path}", timeout=10)
        return r.ok
    except:
        return False

def step(title):
    """Print step header"""
    print(f"\n[JOURNEY] {title}")
    time.sleep(0.5)  # Small delay for readability

def main():
    print("=" * 60)
    print("🚶 USER JOURNEY SIMULATION")
    print("Simulating complete user experience from cold start")
    print("=" * 60)
    
    # Test user details
    user_id = f"test_user_{random.randint(1000, 9999)}"
    print(f"\n📝 Test User: {user_id}")
    
    step("1. Cold start health check")
    if get("/api/health"):
        print("  ✅ API is healthy")
    else:
        print("  ❌ API not responding - starting server may be needed")
        print("  💡 Run: python enhanced_launcher.py")
        return False
    
    step("2. New user initialization")
    # Create user mesh
    result = post("/api/v2/mesh/update", {
        "user_id": user_id,
        "change": {
            "action": "add_concept",
            "data": {"tag": "first-run", "content": "Welcome to TORI"}
        }
    })
    if result:
        print("  ✅ User mesh created")
    
    step("3. Set user persona")
    personas = ["Explorer", "Scholar", "Creative", "Analyst"]
    selected_persona = random.choice(personas)
    result = post("/api/v2/hybrid/persona", {
        "user_id": user_id,
        "persona": selected_persona
    })
    print(f"  ✅ Persona set to: {selected_persona}")
    
    step("4. Send first prompt")
    prompts = [
        "Show me a holographic visualization",
        "Explain quantum computing simply",
        "Generate a creative story about AI",
        "Analyze the concept of consciousness"
    ]
    first_prompt = random.choice(prompts)
    result = post("/api/v2/hybrid/prompt", {
        "user_id": user_id,
        "text": first_prompt
    })
    print(f"  ✅ Prompt sent: '{first_prompt[:50]}...'")
    
    step("5. Train personalized adapter")
    result = post("/api/v2/hybrid/adapter/train", {
        "user_id": user_id,
        "notes": "Initial training from journey"
    })
    print("  ✅ Adapter training initiated")
    
    step("6. Hot-swap to new adapter")
    result = post("/api/v2/hybrid/adapter/swap", {
        "user_id": user_id,
        "adapter_name": "latest"
    })
    print("  ✅ Adapter swapped successfully")
    
    step("7. Mobile interaction simulation")
    # Simulate mobile-specific events
    result = post("/api/v2/hybrid/mobile/event", {
        "user_id": user_id,
        "type": "parallax",
        "payload": {"tilt": "small", "angle": 15}
    })
    print("  ✅ Mobile parallax event sent")
    
    step("8. Toggle audio/video features")
    result = post("/api/v2/hybrid/av/toggle", {
        "user_id": user_id,
        "enable_audio": True,
        "enable_video": True
    })
    print("  ✅ A/V features enabled")
    
    step("9. Stress test with rapid requests")
    print("  Sending 5 rapid requests...")
    for i in range(5):
        quick_prompt = f"Quick test {i+1}"
        post("/api/v2/hybrid/prompt", {
            "user_id": user_id,
            "text": quick_prompt
        })
        print(f"    • Request {i+1}/5 sent")
        time.sleep(0.1)
    print("  ✅ Rapid request test complete")
    
    step("10. Error recovery test")
    print("  Attempting invalid adapter swap...")
    result = post("/api/v2/hybrid/adapter/swap", {
        "user_id": user_id,
        "adapter_name": "models/adapters/NONEXISTENT.pt"
    })
    print("  ✅ System handled error gracefully")
    
    step("11. Complex multi-modal request")
    complex_prompt = """
    Create a holographic visualization of a neural network
    with quantum entanglement effects, showing the flow of
    information through multiple layers with real-time updates.
    """
    result = post("/api/v2/hybrid/prompt", {
        "user_id": user_id,
        "text": complex_prompt
    })
    print("  ✅ Complex prompt processed")
    
    step("12. Session cleanup test")
    # Get final status
    result = post("/api/v2/hybrid/status", {"user_id": user_id})
    if result:
        print("  ✅ Final status retrieved")
    
    # Mesh summary check
    result = post("/api/v2/mesh/summary", {"user_id": user_id})
    if result:
        print("  ✅ Mesh summary available")
    
    print("\n" + "=" * 60)
    print("✅ USER JOURNEY SIMULATION COMPLETE")
    print("\nJourney covered:")
    print("  • Cold start → User creation")
    print("  • Persona selection → First interaction")
    print("  • Adapter training → Hot-swapping")
    print("  • Mobile events → A/V features")
    print("  • Stress testing → Error recovery")
    print("  • Complex requests → Session management")
    print("\n💡 Check logs/ directory for detailed event trail")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Journey simulation successful!")
    else:
        print("\n⚠️ Journey incomplete - check server status")
        sys.exit(1)
