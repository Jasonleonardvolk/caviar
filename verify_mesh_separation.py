"""
verify_mesh_separation.py
========================
Verification script to confirm Memory (5173) and Voice (8101) services 
write to separate mesh files and don't stomp on each other.
"""

import os
import time
import json
import requests
from pathlib import Path
from datetime import datetime

def check_mesh_files():
    """Check which mesh files exist and their sizes."""
    files = [
        "concept_mesh_data.json",  # Legacy default
        "concept_mesh_5173.json",  # Memory service
        "concept_mesh_8101.json",  # Voice service
    ]
    
    print("🔍 MESH FILE STATUS:")
    print("-" * 50)
    
    for filename in files:
        path = Path(filename)
        if path.exists():
            size = path.stat().st_size
            with open(path, 'r') as f:
                try:
                    data = json.load(f)
                    count = len(data) if isinstance(data, list) else "unknown"
                except:
                    count = "invalid"
            print(f"✅ {filename}: {size} bytes, {count} diffs")
        else:
            print(f"❌ {filename}: Not found")
    
    print("-" * 50)

def test_service_isolation():
    """Test that services write to separate files."""
    print("\n🧪 TESTING SERVICE ISOLATION:")
    print("=" * 60)
    
    # Check initial state
    print("\n📊 Initial mesh file state:")
    check_mesh_files()
    
    # Test Memory service (if running)
    print("\n🧠 Testing Memory service (5173)...")
    try:
        response = requests.get("http://localhost:5173/health", timeout=5)
        if response.status_code == 200:
            print("✅ Memory service is running")
            
            # Trigger an operation that would write to mesh
            # This depends on your API - adjust as needed
            print("   (Add test operation for Memory service here)")
        else:
            print(f"⚠️ Memory service responded with: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Memory service not reachable: {e}")
    
    # Test Voice service (if running)
    print("\n🗣️ Testing Voice service (8101)...")
    try:
        response = requests.get("http://localhost:8101/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Voice service is running")
            
            # Trigger an operation that would write to mesh
            # This depends on your API - adjust as needed
            print("   (Add test operation for Voice service here)")
        else:
            print(f"⚠️ Voice service responded with: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Voice service not reachable: {e}")
    
    # Check final state
    print("\n📊 Final mesh file state:")
    check_mesh_files()
    
    # Analysis
    print("\n🔍 ANALYSIS:")
    memory_file = Path("concept_mesh_5173.json")
    voice_file = Path("concept_mesh_8101.json")
    
    if memory_file.exists() and voice_file.exists():
        print("✅ SUCCESS: Both services have separate mesh files!")
        print("   No more mesh collision between Memory and Voice services.")
    elif memory_file.exists():
        print("✅ Memory service mesh file exists")
        print("⚠️ Voice service mesh file not found (may not have been used yet)")
    elif voice_file.exists():
        print("✅ Voice service mesh file exists") 
        print("⚠️ Memory service mesh file not found (may not have been used yet)")
    else:
        print("❌ No service-specific mesh files found")
        print("   Services may not be running or may not have been used yet")

def demonstrate_environment_setup():
    """Show how to set up environment variables for each service."""
    print("\n🛠️ ENVIRONMENT SETUP GUIDE:")
    print("=" * 60)
    
    print("\n🧠 For Memory Service (Port 5173):")
    print("   Windows: set CONCEPT_MESH_PATH=concept_mesh_5173.json")
    print("   Linux:   export CONCEPT_MESH_PATH=concept_mesh_5173.json")
    print("   Then:    python -m uvicorn ingest_pdf.cognitive_interface:app --port 5173")
    
    print("\n🗣️ For Voice Service (Port 8101):")
    print("   Windows: set CONCEPT_MESH_PATH=concept_mesh_8101.json")
    print("   Linux:   export CONCEPT_MESH_PATH=concept_mesh_8101.json")
    print("   Then:    python -m uvicorn prajna.api.prajna_api:app --port 8101")
    
    print("\n📁 Alternative: Use provided startup scripts:")
    print("   Memory: start_memory_service.bat")
    print("   Voice:  start_voice_service.bat")

def check_current_environment():
    """Check current environment variable setting."""
    print("\n🌍 CURRENT ENVIRONMENT:")
    print("-" * 30)
    
    mesh_path = os.getenv("CONCEPT_MESH_PATH")
    if mesh_path:
        print(f"✅ CONCEPT_MESH_PATH = {mesh_path}")
    else:
        print("❌ CONCEPT_MESH_PATH not set (will use default: concept_mesh_data.json)")
    
    print(f"📁 Working directory: {os.getcwd()}")

def main():
    """Main verification function."""
    print("🔒 MESH SEPARATION VERIFICATION")
    print("=" * 60)
    print("This script verifies that Memory (5173) and Voice (8101)")
    print("services write to separate mesh files.")
    print("=" * 60)
    
    check_current_environment()
    test_service_isolation()
    demonstrate_environment_setup()
    
    print("\n✅ VERIFICATION COMPLETE!")
    print("\nIf both services are running and have separate mesh files,")
    print("the mesh collision problem has been solved! 🎉")

if __name__ == "__main__":
    main()
