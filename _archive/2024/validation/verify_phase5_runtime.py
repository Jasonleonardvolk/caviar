#!/usr/bin/env python3
"""
Phase 5: Runtime Health - Final System Verification
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime

def verify_runtime_health():
    """Verify runtime health and system readiness"""
    
    print("🏥 PHASE 5: RUNTIME HEALTH VERIFICATION")
    print("=" * 60)
    
    all_good = True
    
    # 1. Canonical References Check
    print("\n📍 Checking Canonical References...")
    
    canonical_path = Path("concept_mesh/data.json")
    if canonical_path.exists():
        print(f"   ✅ Canonical concept file: {canonical_path}")
        
        # Check no other concept files exist
        old_files = [
            Path("concept_mesh_data.json"),
            Path("concepts.json"),
            Path("concept_mesh/concepts.json"),
            Path("prajna/concept_mesh_data.json")
        ]
        
        for old_file in old_files:
            if old_file.exists():
                print(f"   ⚠️  Deprecated file still exists: {old_file}")
                all_good = False
        
        # Check code references
        files_to_check = [
            ("enhanced_launcher.py", "concept_mesh/data.json"),
            ("concept_mesh/loader.py", "data.json"),
            ("api/diff_route.py", "data.json")
        ]
        
        for file_name, expected_ref in files_to_check:
            file_path = Path(file_name)
            if file_path.exists():
                content = file_path.read_text()
                if expected_ref in content:
                    print(f"   ✅ {file_name} uses canonical path")
                else:
                    print(f"   ❌ {file_name} not using canonical path")
                    all_good = False
    else:
        print("   ❌ Canonical file missing!")
        all_good = False
    
    # 2. API Endpoint Audit
    print("\n🌐 API Endpoint Audit...")
    
    if not check_api_running():
        print("   ⚠️  API not running - start TORI first")
        return False
    
    endpoints_to_check = [
        # Core endpoints
        ("/api/health", "GET", "Health check"),
        ("/docs", "GET", "API documentation"),
        
        # Upload endpoints
        ("/api/upload", "POST", "PDF upload"),
        ("/upload", "POST", "Upload fallback"),
        
        # Prajna endpoints
        ("/api/answer", "POST", "Prajna answer"),
        ("/api/prajna/stats", "GET", "Prajna stats"),
        
        # Concept mesh endpoints
        ("/api/concept-mesh/record_diff", "POST", "Record diff"),
        ("/api/concept-mesh/sync_to_scholarsphere", "POST", "Sync to ScholarSphere"),
        
        # Lattice endpoints
        ("/api/lattice/snapshot", "GET", "Lattice snapshot"),
        ("/api/lattice/rebuild", "POST", "Lattice rebuild"),
        
        # Soliton endpoints
        ("/api/soliton/health", "GET", "Soliton health")
    ]
    
    working_endpoints = 0
    for endpoint, method, description in endpoints_to_check:
        try:
            if method == "GET":
                response = requests.get(f"http://localhost:8002{endpoint}", timeout=2)
            else:
                # For POST endpoints, just check OPTIONS
                response = requests.options(f"http://localhost:8002{endpoint}", timeout=2)
            
            if response.status_code in [200, 204, 405]:  # 405 is OK for OPTIONS
                print(f"   ✅ {endpoint} - {description}")
                working_endpoints += 1
            else:
                print(f"   ⚠️  {endpoint} - Status: {response.status_code}")
        except Exception as e:
            print(f"   ❌ {endpoint} - Error: {type(e).__name__}")
    
    print(f"\n   📊 {working_endpoints}/{len(endpoints_to_check)} endpoints working")
    
    # 3. Frontend Proxy Check
    print("\n🔀 Frontend Proxy Configuration...")
    
    vite_config = Path("tori_ui_svelte/vite.config.js")
    if vite_config.exists():
        content = vite_config.read_text()
        if "proxy" in content and "/api" in content:
            print("   ✅ API proxy configured in Vite")
        else:
            print("   ⚠️  API proxy not configured")
            print("      Add proxy configuration to vite.config.js")
            all_good = False
    
    # 4. Cleanup Check
    print("\n🧹 Redundancy and Cleanup Check...")
    
    # Check for fix scripts
    fix_scripts = list(Path(".").glob("fix_*.py"))
    if fix_scripts:
        print(f"   ℹ️  Found {len(fix_scripts)} fix scripts")
        print("      Consider moving to fixes/ directory after applying")
    
    # Check for backup files
    backup_files = list(Path(".").glob("**/*.backup*", recursive=True))
    if backup_files:
        print(f"   ℹ️  Found {len(backup_files)} backup files")
        print("      Consider archiving or removing old backups")
    
    # 5. System Resources
    print("\n💻 System Resources...")
    
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        print(f"   CPU: {cpu_percent}%")
        print(f"   Memory: {memory.percent}% used ({memory.available / (1024**3):.1f}GB available)")
        print(f"   Disk: {disk.percent}% used ({disk.free / (1024**3):.1f}GB free)")
        
        if cpu_percent > 80 or memory.percent > 80:
            print("   ⚠️  System under high load")
            all_good = False
        else:
            print("   ✅ System resources healthy")
    except ImportError:
        print("   ℹ️  psutil not installed - can't check resources")
    
    # 6. Final Smoke Test
    print("\n🧪 Final Smoke Test...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8002/api/health")
        if response.status_code == 200:
            health_data = response.json()
            
            print("   ✅ API Health Check:")
            print(f"      Status: {health_data.get('status', 'unknown')}")
            print(f"      Prajna: {health_data.get('prajna_loaded', False)}")
            print(f"      PDF Processing: {health_data.get('pdf_processing_available', False)}")
            print(f"      Upload Dir: {health_data.get('upload_directory_exists', False)}")
            
            # Test concept mesh
            response = requests.get("http://localhost:8002/api/lattice/snapshot")
            if response.status_code == 200:
                lattice_data = response.json()
                print(f"\n   ✅ Lattice Status:")
                print(f"      Oscillators: {lattice_data.get('oscillator_count', 0)}")
                print(f"      Energy: {lattice_data.get('total_energy', 0):.3f}")
    except Exception as e:
        print(f"   ❌ Smoke test failed: {e}")
        all_good = False
    
    # 7. Create launch checklist
    create_launch_checklist()
    
    # Summary
    print("\n" + "=" * 60)
    
    if all_good:
        print("✅ SYSTEM READY FOR LAUNCH!")
        print("\n🚀 Launch command:")
        print("   poetry run python enhanced_launcher.py")
        print("\n📋 Post-launch checklist:")
        print("   1. Upload a test PDF")
        print("   2. Verify concepts appear in mesh")
        print("   3. Check oscillator lattice populates")
        print("   4. Confirm Enola avatar displays")
        print("   5. Test ScholarSphere sync")
    else:
        print("⚠️  SOME ISSUES REMAIN")
        print("\n📝 Review the output above and fix any ❌ items")
        print("   Then run this verification again")
    
    return all_good

def check_api_running():
    """Check if the API is running"""
    try:
        response = requests.get("http://localhost:8002/api/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def create_launch_checklist():
    """Create a final launch checklist"""
    
    checklist = '''# TORI FINAL LAUNCH CHECKLIST

## Pre-Launch Verification
- [ ] Run: `python verify_canonical_concept_mesh.py`
- [ ] Run: `python verify_phase2_scholarsphere.py`
- [ ] Run: `python verify_phase3_memory.py`
- [ ] Run: `python verify_phase4_persona.py`
- [ ] Run: `python verify_phase5_runtime.py`

## Launch Steps
1. Set environment variables:
   ```bash
   export TORI_ENABLE_ENTROPY_PRUNING=1
   ```

2. Start TORI:
   ```bash
   poetry run python enhanced_launcher.py
   ```

3. Wait for all services to initialize:
   - API Server ready
   - Frontend started
   - MCP Cognitive Engine running
   - Oscillator lattice active

## Post-Launch Tests
1. **Upload Test**
   - Upload a PDF file
   - Verify "success" method (not fallback)
   - Check concepts in concept_mesh/data.json

2. **Memory Wiring**
   - Run: `python test_lattice_rebuild.py`
   - Check oscillator count > 0
   - Verify lattice energy > 0

3. **Persona Display**
   - Open frontend
   - Verify Enola in persona selector
   - Check hologram display active

4. **ScholarSphere Sync**
   - Upload creates files in data/scholarsphere/uploaded/
   - Manual sync via API works

## Success Indicators
- ✅ All endpoints return 200/204
- ✅ Concepts propagate to lattice
- ✅ Enola avatar visible
- ✅ No "fallback" processing
- ✅ System resources < 80%

## Troubleshooting
- If uploads fail: Check tmp/ directory permissions
- If lattice empty: Run /api/lattice/rebuild
- If Enola missing: Clear browser cache
- If sync fails: Check data/scholarsphere/ permissions

---
Generated: {}
'''.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    checklist_path = Path("TORI_LAUNCH_CHECKLIST.md")
    checklist_path.write_text(checklist)
    print("\n   ✅ Created TORI_LAUNCH_CHECKLIST.md")

if __name__ == "__main__":
    verify_runtime_health()
