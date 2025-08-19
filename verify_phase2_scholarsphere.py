#!/usr/bin/env python3
"""
Phase 2: ScholarSphere Integration Verification
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime

def verify_scholarsphere_setup():
    """Verify ScholarSphere integration is properly configured"""
    
    print("🔍 PHASE 2: SCHOLARSPHERE INTEGRATION VERIFICATION")
    print("=" * 60)
    
    issues = []
    
    # 1. Check directory structure
    print("\n📁 Checking directory structure...")
    required_dirs = [
        "data/scholarsphere/pending",
        "data/scholarsphere/uploaded",
        "data/psi_archive/diffs",
        "data/mesh_snapshots"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"   ✅ {dir_path} exists")
        else:
            print(f"   ❌ {dir_path} missing - creating...")
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {dir_path} created")
    
    # 2. Check API key requirement removed
    print("\n🔑 Checking API key configuration...")
    upload_file = Path("api/scholarsphere_upload.py")
    if upload_file.exists():
        content = upload_file.read_text()
        if "No ScholarSphere API key configured" in content:
            print("   ⚠️  API key check still present in code")
            issues.append("Remove API key requirement from scholarsphere_upload.py")
        else:
            print("   ✅ API key requirement removed")
    
    # 3. Check concept mesh sync endpoint
    print("\n🔄 Checking sync endpoint...")
    diff_route_file = Path("api/diff_route.py")
    if diff_route_file.exists():
        content = diff_route_file.read_text()
        if "sync_to_scholarsphere" in content:
            print("   ✅ sync_to_scholarsphere endpoint exists")
        else:
            print("   ❌ sync_to_scholarsphere endpoint missing")
            issues.append("Add sync_to_scholarsphere endpoint to diff_route.py")
            
        if "Auto-upload to ScholarSphere" in content:
            print("   ✅ Auto-upload implemented in record_diff")
        else:
            print("   ⚠️  Auto-upload not implemented in record_diff")
            issues.append("Add auto-upload to record_diff endpoint")
    
    # 4. Check canonical concept mesh path
    print("\n📍 Checking canonical concept mesh...")
    canonical_path = Path("concept_mesh/data.json")
    if canonical_path.exists():
        print(f"   ✅ Canonical file exists: {canonical_path}")
        
        # Check if it's being used in diff_route
        if diff_route_file.exists():
            content = diff_route_file.read_text()
            if 'concept_mesh" / "data.json' in content or 'concept_mesh/data.json' in content:
                print("   ✅ diff_route.py uses canonical path")
            else:
                print("   ⚠️  diff_route.py not using canonical path")
                issues.append("Update diff_route.py to use concept_mesh/data.json")
    else:
        print(f"   ❌ Canonical file missing: {canonical_path}")
        issues.append("Create canonical concept mesh file")
    
    # 5. Test local file movement
    print("\n🧪 Testing local file operations...")
    test_file = Path("data/scholarsphere/pending/test.jsonl")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create test file
        test_data = {"test": True, "timestamp": datetime.now().isoformat()}
        test_file.write_text(json.dumps(test_data))
        print("   ✅ Created test file in pending")
        
        # Move to uploaded
        import shutil
        uploaded_file = Path("data/scholarsphere/uploaded/test.jsonl")
        shutil.move(str(test_file), str(uploaded_file))
        print("   ✅ Moved file to uploaded")
        
        # Cleanup
        uploaded_file.unlink()
        print("   ✅ Cleanup successful")
        
    except Exception as e:
        print(f"   ❌ File operations failed: {e}")
        issues.append("Fix file permissions for ScholarSphere directories")
    
    # Summary
    print("\n" + "=" * 60)
    if issues:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n📝 Run the following to fix:")
        print("   python update_scholarsphere_integration.py")
    else:
        print("✅ PHASE 2 COMPLETE: ScholarSphere integration ready!")
        print("\n📋 Next steps:")
        print("   1. Upload a PDF to test the pipeline")
        print("   2. Check data/scholarsphere/uploaded/ for synced files")
        print("   3. Verify concepts appear in concept_mesh/data.json")
    
    return len(issues) == 0

if __name__ == "__main__":
    verify_scholarsphere_setup()
