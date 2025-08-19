#!/usr/bin/env python3
"""
Test ScholarSphere Integration
"""

import asyncio
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from api.scholarsphere_upload import upload_concepts_to_scholarsphere

async def test_scholarsphere_upload():
    """Test the ScholarSphere upload functionality"""
    
    print("🧪 TESTING SCHOLARSPHERE INTEGRATION")
    print("=" * 60)
    
    # Create test concepts
    test_concepts = [
        {
            "name": "Quantum Computing",
            "score": 0.95,
            "source": "test_document.pdf",
            "timestamp": "2025-01-19T12:00:00Z"
        },
        {
            "name": "Machine Learning",
            "score": 0.87,
            "source": "test_document.pdf",
            "timestamp": "2025-01-19T12:00:00Z"
        }
    ]
    
    print("📤 Uploading test concepts...")
    print(f"   Concepts: {[c['name'] for c in test_concepts]}")
    
    try:
        # Test upload
        diff_id = await upload_concepts_to_scholarsphere(test_concepts, source="test_upload")
        
        if diff_id:
            print(f"\n✅ Upload successful!")
            print(f"   Diff ID: {diff_id}")
            
            # Check if file exists in uploaded directory
            uploaded_dir = Path("data/scholarsphere/uploaded")
            uploaded_files = list(uploaded_dir.glob(f"{diff_id}_*.jsonl"))
            
            if uploaded_files:
                print(f"   File: {uploaded_files[0].name}")
                
                # Read and verify content
                with open(uploaded_files[0], 'r') as f:
                    lines = f.readlines()
                    print(f"   Lines written: {len(lines)}")
                    
                    # Check first line
                    first_entry = json.loads(lines[0])
                    print(f"   First concept: {first_entry['concept']['name']}")
            else:
                print("   ⚠️  File not found in uploaded directory")
        else:
            print("\n❌ Upload failed - no diff_id returned")
            
    except Exception as e:
        print(f"\n❌ Upload failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Check directory structure
    print("\n📁 Directory Structure:")
    dirs_to_check = [
        "data/scholarsphere/pending",
        "data/scholarsphere/uploaded"
    ]
    
    for dir_path in dirs_to_check:
        path = Path(dir_path)
        if path.exists():
            files = list(path.glob("*.jsonl"))
            print(f"   {dir_path}: {len(files)} files")
        else:
            print(f"   {dir_path}: ❌ Does not exist")
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(test_scholarsphere_upload())
