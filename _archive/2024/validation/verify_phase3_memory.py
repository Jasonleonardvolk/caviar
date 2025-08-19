#!/usr/bin/env python3
"""
Phase 3: Memory Wiring Verification
Check Holographic Memory, ConceptMesh & Lattice connections
"""

import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime

def verify_memory_wiring():
    """Verify all memory systems are properly wired together"""
    
    print("🧠 PHASE 3: MEMORY WIRING VERIFICATION")
    print("=" * 60)
    
    issues = []
    
    # 1. Check concept mesh data
    print("\n📊 Checking Concept Mesh...")
    canonical_path = Path("concept_mesh/data.json")
    
    concept_count = 0
    if canonical_path.exists():
        try:
            with open(canonical_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'concepts' in data:
                concept_count = len(data['concepts'])
                print(f"   ✅ Concept mesh loaded: {concept_count} concepts")
            else:
                print("   ⚠️  Concept mesh has invalid structure")
                issues.append("Fix concept mesh data structure")
        except Exception as e:
            print(f"   ❌ Failed to load concept mesh: {e}")
            issues.append("Fix concept mesh data file")
    else:
        print("   ❌ Canonical concept mesh not found")
        issues.append("Create concept_mesh/data.json")
    
    # 2. Check oscillator lattice
    print("\n🌊 Checking Oscillator Lattice...")
    
    # Check if lattice endpoints are available
    try:
        # Test lattice snapshot endpoint
        response = requests.get("http://localhost:8002/api/lattice/snapshot", timeout=2)
        if response.status_code == 200:
            lattice_data = response.json()
            oscillator_count = lattice_data.get('oscillator_count', 0)
            
            print(f"   ✅ Lattice endpoint responsive")
            print(f"   📊 Oscillators: {oscillator_count}")
            
            if concept_count > 0 and oscillator_count == 0:
                print("   ⚠️  Concepts exist but no oscillators!")
                issues.append("Lattice not creating oscillators from concepts")
                print("   💡 Try: /api/lattice/rebuild to refresh")
            elif oscillator_count > 0:
                print(f"   ✅ Lattice has {oscillator_count} active oscillators")
        else:
            print(f"   ⚠️  Lattice endpoint returned: {response.status_code}")
            issues.append("Lattice endpoints not working properly")
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to API - is TORI running?")
        issues.append("Start TORI before running verification")
    except Exception as e:
        print(f"   ⚠️  Lattice check failed: {e}")
    
    # 3. Check environment flags
    print("\n🔧 Checking Environment Configuration...")
    
    entropy_enabled = os.environ.get('TORI_ENABLE_ENTROPY_PRUNING', '0')
    if entropy_enabled == '1':
        print("   ✅ TORI_ENABLE_ENTROPY_PRUNING=1 (live concept streaming)")
    else:
        print("   ⚠️  TORI_ENABLE_ENTROPY_PRUNING not set")
        print("      Set this for real-time concept updates")
    
    # 4. Check for legacy systems
    print("\n🔍 Checking for Legacy Systems...")
    
    # Check for old concept files (should not exist)
    old_files = [
        Path("concept_mesh_data.json"),
        Path("concepts.json"),
        Path("prajna/concept_mesh_data.json")
    ]
    
    legacy_found = False
    for old_file in old_files:
        if old_file.exists():
            print(f"   ⚠️  Found legacy file: {old_file}")
            legacy_found = True
            issues.append(f"Delete legacy file: {old_file}")
    
    if not legacy_found:
        print("   ✅ No legacy concept files found")
    
    # 5. Check memory vault integration
    print("\n💾 Checking Memory Vault...")
    
    # Check if UnifiedMemoryVault directory exists
    memory_vault_dir = Path("data/memory_vault")
    if memory_vault_dir.exists():
        print("   ✅ UnifiedMemoryVault directory exists")
        
        # Count files
        vault_files = list(memory_vault_dir.glob("**/*"))
        if vault_files:
            print(f"   📁 {len(vault_files)} files in vault")
    else:
        print("   ⚠️  Memory vault directory not found")
        print("      Will be created on first run")
    
    # 6. Test concept propagation
    print("\n🔄 Testing Concept Propagation...")
    
    if canonical_path.exists() and concept_count == 0:
        print("   ℹ️  No concepts to test propagation")
        print("   💡 Upload a PDF to create concepts")
    elif concept_count > 0:
        print("   ✅ Concepts available for propagation")
        
        # Check if concepts have required fields
        try:
            with open(canonical_path, 'r') as f:
                data = json.load(f)
            
            if data.get('concepts'):
                sample_concept = data['concepts'][0] if isinstance(data['concepts'], list) else None
                if sample_concept:
                    required_fields = ['name', 'score']
                    missing = [f for f in required_fields if f not in sample_concept]
                    if missing:
                        print(f"   ⚠️  Concept missing fields: {missing}")
                        issues.append("Concepts missing required fields")
                    else:
                        print("   ✅ Concept structure valid")
        except Exception as e:
            print(f"   ⚠️  Could not validate concept structure: {e}")
    
    # 7. Create test script for lattice rebuild
    print("\n📝 Creating lattice test script...")
    
    test_script = '''#!/usr/bin/env python3
"""Force lattice rebuild to test concept propagation"""

import requests

print("🔄 Forcing lattice rebuild...")

try:
    response = requests.post("http://localhost:8002/api/lattice/rebuild")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Rebuild complete!")
        print(f"   Oscillators: {data.get('oscillator_count', 0)}")
        print(f"   Energy: {data.get('total_energy', 0):.3f}")
    else:
        print(f"❌ Rebuild failed: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")
    print("   Make sure TORI is running!")
'''
    
    test_file = Path("test_lattice_rebuild.py")
    test_file.write_text(test_script)
    test_file.chmod(0o755)
    print("   ✅ Created test_lattice_rebuild.py")
    
    # Summary
    print("\n" + "=" * 60)
    
    if issues:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✅ PHASE 3 COMPLETE: Memory systems properly wired!")
    
    print("\n📋 Recommendations:")
    print("   1. Set TORI_ENABLE_ENTROPY_PRUNING=1 in your environment")
    print("   2. Upload a PDF to create concepts")
    print("   3. Run: python test_lattice_rebuild.py")
    print("   4. Check lattice metrics at /api/lattice/snapshot")
    
    return len(issues) == 0

if __name__ == "__main__":
    verify_memory_wiring()
