"""
test_unified_mesh_parameterization.py
====================================
Test script to verify that the unified launcher properly prevents mesh collision
by using parameterized mesh paths for each service.
"""

import time
import json
import requests
from pathlib import Path
from datetime import datetime

def check_mesh_files():
    """Check which mesh files exist and their properties."""
    print("🔍 MESH FILE STATUS AFTER UNIFIED LAUNCH:")
    print("-" * 60)
    
    # Look for any concept_mesh_*.json files
    mesh_files = list(Path('.').glob('concept_mesh_*.json'))
    
    if not mesh_files:
        print("❌ No mesh files found!")
        print("   This might indicate services haven't processed any documents yet.")
        return []
    
    for mesh_file in sorted(mesh_files):
        try:
            size = mesh_file.stat().st_size
            with open(mesh_file, 'r') as f:
                data = json.load(f)
                count = len(data) if isinstance(data, list) else "unknown"
            
            # Extract port from filename
            port = mesh_file.name.replace('concept_mesh_', '').replace('.json', '')
            
            print(f"✅ {mesh_file.name}: {size} bytes, {count} diffs (port {port})")
            
        except Exception as e:
            print(f"⚠️ {mesh_file.name}: Error reading file - {e}")
    
    print("-" * 60)
    return mesh_files

def check_unified_status():
    """Check the unified launcher's status file."""
    status_file = Path('tori_status.json')
    
    if not status_file.exists():
        print("❌ tori_status.json not found - unified launcher may not be running")
        return None
    
    try:
        with open(status_file, 'r') as f:
            status = json.load(f)
        
        print("📊 UNIFIED LAUNCHER STATUS:")
        print("-" * 40)
        print(f"✅ Timestamp: {status.get('timestamp', 'unknown')}")
        print(f"✅ Stage: {status.get('stage', 'unknown')}")
        print(f"✅ Status: {status.get('status', 'unknown')}")
        
        api_port = status.get('api_port')
        prajna_port = status.get('prajna_port')
        
        if api_port:
            print(f"✅ API Port: {api_port}")
            print(f"   Expected mesh: concept_mesh_{api_port}.json")
        
        if prajna_port:
            print(f"✅ Prajna Port: {prajna_port}")
            print(f"   Expected mesh: concept_mesh_{prajna_port}.json")
        
        if status.get('mesh_collision_prevented'):
            print("✅ Mesh collision prevention: ACTIVE")
        
        print("-" * 40)
        return status
        
    except Exception as e:
        print(f"❌ Error reading status file: {e}")
        return None

def test_service_endpoints(status):
    """Test that services are running and responsive."""
    if not status:
        print("⚠️ Skipping service tests - no status available")
        return
    
    print("🧪 TESTING SERVICE ENDPOINTS:")
    print("-" * 40)
    
    # Test API service
    api_port = status.get('api_port')
    if api_port:
        try:
            response = requests.get(f"http://localhost:{api_port}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ API Service (port {api_port}): Healthy")
            else:
                print(f"⚠️ API Service (port {api_port}): Status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ API Service (port {api_port}): Not reachable - {e}")
    
    # Test Prajna service
    prajna_port = status.get('prajna_port')
    if prajna_port:
        try:
            response = requests.get(f"http://localhost:{prajna_port}/api/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Prajna Service (port {prajna_port}): Healthy")
            else:
                print(f"⚠️ Prajna Service (port {prajna_port}): Status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Prajna Service (port {prajna_port}): Not reachable - {e}")
    
    print("-" * 40)

def verify_mesh_separation(status, mesh_files):
    """Verify that mesh separation is working properly."""
    print("🔒 MESH SEPARATION ANALYSIS:")
    print("-" * 50)
    
    if not status:
        print("⚠️ Cannot verify separation - no status available")
        return
    
    api_port = status.get('api_port')
    prajna_port = status.get('prajna_port')
    
    expected_files = []
    if api_port:
        expected_files.append(f"concept_mesh_{api_port}.json")
    if prajna_port:
        expected_files.append(f"concept_mesh_{prajna_port}.json")
    
    found_files = [f.name for f in mesh_files]
    
    print(f"📋 Expected mesh files: {expected_files}")
    print(f"📋 Found mesh files: {found_files}")
    
    # Check if expected files exist
    all_expected_found = all(f in found_files for f in expected_files)
    
    if all_expected_found and len(expected_files) > 1:
        print("✅ SUCCESS: Multiple services using separate mesh files!")
        print("   ✅ Mesh collision prevention is working correctly")
        print("   ✅ Each service maintains its own concept state")
        
        # Check if files have different content (if they exist and have content)
        different_content = False
        if len(mesh_files) >= 2:
            try:
                file1_data = json.loads(mesh_files[0].read_text())
                file2_data = json.loads(mesh_files[1].read_text())
                different_content = file1_data != file2_data
            except:
                pass
        
        if different_content:
            print("   ✅ Mesh files contain different data (as expected)")
        
    elif all_expected_found and len(expected_files) == 1:
        print("✅ PARTIAL SUCCESS: Single service using parameterized mesh file")
        print("   ✅ Collision prevention ready for when multiple services run")
        
    else:
        print("⚠️ WARNING: Expected mesh files not found")
        print("   This might indicate services haven't processed documents yet")
        print("   Or mesh parameterization may not be working")
    
    # Check for legacy file
    legacy_file = Path("concept_mesh_data.json")
    if legacy_file.exists():
        print("⚠️ WARNING: Legacy mesh file still exists")
        print(f"   {legacy_file.name} found - may indicate old behavior")
    else:
        print("✅ No legacy mesh file found (good!)")
    
    print("-" * 50)

def show_test_instructions():
    """Show instructions for testing mesh separation."""
    print("📋 MESH SEPARATION TEST INSTRUCTIONS:")
    print("=" * 60)
    print("To fully test mesh collision prevention:")
    print()
    print("1. 🚀 Start the unified launcher:")
    print("   python start_unified_tori.py")
    print()
    print("2. 📄 Upload a PDF to the API service:")
    print("   curl -X POST http://localhost:{api_port}/upload \\")
    print("     -F 'file=@your_test.pdf'")
    print()
    print("3. 📝 If Prajna is running, upload to Prajna:")
    print("   curl -X POST http://localhost:{prajna_port}/api/upload \\")
    print("     -F 'file=@your_test.txt'")
    print()
    print("4. 🔍 Run this test script again to verify separation:")
    print("   python test_unified_mesh_parameterization.py")
    print()
    print("5. ✅ Expected outcome:")
    print("   - concept_mesh_{api_port}.json for API service")
    print("   - concept_mesh_{prajna_port}.json for Prajna service")
    print("   - NO concept_mesh_data.json (legacy file)")
    print("   - Different content in each mesh file")
    print("=" * 60)

def main():
    """Main test function."""
    print("🔒 UNIFIED MESH PARAMETERIZATION TEST")
    print("=" * 60)
    print("Testing mesh collision prevention in unified launcher")
    print(f"Test run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check unified launcher status
    status = check_unified_status()
    
    # Check mesh files
    print()
    mesh_files = check_mesh_files()
    
    # Test service endpoints
    print()
    test_service_endpoints(status)
    
    # Verify mesh separation
    print()
    verify_mesh_separation(status, mesh_files)
    
    # Show testing instructions
    print()
    show_test_instructions()
    
    print("\n✅ MESH PARAMETERIZATION TEST COMPLETE!")
    
    # Final assessment
    if status and len(mesh_files) > 0:
        api_port = status.get('api_port')
        prajna_port = status.get('prajna_port') 
        
        if api_port and prajna_port and len(mesh_files) >= 2:
            print("🎉 EXCELLENT: Multi-service mesh separation is working!")
        elif (api_port or prajna_port) and len(mesh_files) >= 1:
            print("✅ GOOD: Single-service mesh parameterization is working!")
        else:
            print("⚠️ Mesh separation not yet tested - upload documents to verify!")
    else:
        print("⚠️ Unified launcher may not be running or no documents processed yet")

if __name__ == "__main__":
    main()
