"""
Fix ScholarSphere Fallback Mode
===============================

Complete integration fix for ScholarSphere upload functionality.
"""

import os
import sys
from pathlib import Path

# Step 1: Set ScholarSphere credentials
def set_scholarsphere_env():
    """Set ScholarSphere environment variables"""
    # Use placeholder values - replace with actual credentials
    os.environ['SCHOLARSPHERE_API_KEY'] = 'your-scholarsphere-api-key-here'
    os.environ['SCHOLARSPHERE_API_URL'] = 'https://api.scholarsphere.org'
    os.environ['SCHOLARSPHERE_BUCKET'] = 'concept-diffs'
    
    print("✅ ScholarSphere environment variables set")
    print("   NOTE: Replace 'your-scholarsphere-api-key-here' with actual API key")

# Step 2: Fix the sync_to_scholarsphere endpoint
def fix_concept_mesh_api():
    """Update concept_mesh.py to use real uploader"""
    
    api_file = Path('api/concept_mesh.py')
    if not api_file.exists():
        print(f"❌ File not found: {api_file}")
        return False
    
    # Read current content
    with open(api_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already fixed
    if 'upload_concepts_to_scholarsphere' in content:
        print("✅ API already imports ScholarSphere uploader")
        return True
    
    # Add import at the top
    import_line = "from scholarsphere_upload import upload_concepts_to_scholarsphere\n"
    
    # Find where to insert import (after other imports)
    lines = content.split('\n')
    import_index = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_index = i + 1
    
    # Insert import
    lines.insert(import_index, import_line)
    
    # Find and replace sync_to_scholarsphere function
    new_function = '''@router.post("/sync_to_scholarsphere")
async def sync_to_scholarsphere():
    """Sync concepts to ScholarSphere"""
    try:
        # Convert in-memory concepts to list for uploader
        concept_list = list(_concepts.values())
        
        if not concept_list:
            return {
                "success": False,
                "message": "No concepts to sync",
                "conceptCount": 0
            }
        
        # Use the real uploader
        diff_id = await upload_concepts_to_scholarsphere(concept_list, source="concept_mesh_sync")
        
        if diff_id:
            logger.info(f"Successfully synced to ScholarSphere: {diff_id}")
            return {
                "success": True,
                "diffId": diff_id,
                "conceptCount": len(concept_list),
                "message": "Uploaded to ScholarSphere"
            }
        else:
            logger.warning("ScholarSphere sync failed - check API key")
            return {
                "success": False,
                "conceptCount": len(concept_list),
                "message": "Upload failed - check logs and API key"
            }
    
    except Exception as e:
        logger.error(f"Sync error: {e}")
        raise HTTPException(status_code=500, detail=str(e))'''
    
    # Replace the function
    start_marker = '@router.post("/sync_to_scholarsphere")'
    end_marker = 'logger.info("Concept Mesh routes initialized")'
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print("❌ Could not find sync_to_scholarsphere function")
        return False
    
    # Find the end of the current function
    func_end = content.find('\n\n', start_idx)
    if func_end == -1 or func_end > end_idx:
        func_end = end_idx - 1
    
    # Replace the function
    new_content = content[:start_idx] + new_function + '\n\n' + content[func_end:]
    
    # Write updated content
    with open(api_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Updated concept_mesh.py with real ScholarSphere integration")
    return True

# Step 3: Create a test script
def create_test_script():
    """Create test script to verify ScholarSphere integration"""
    
    test_script = '''#!/usr/bin/env python3
"""Test ScholarSphere Integration"""

import asyncio
import aiohttp
import json

async def test_scholarsphere():
    """Test the ScholarSphere sync endpoint"""
    
    async with aiohttp.ClientSession() as session:
        # First check if we have concepts
        async with session.get('http://localhost:8002/api/concept_mesh/stats') as resp:
            stats = await resp.json()
            print(f"Current concepts: {stats.get('totalConcepts', 0)}")
        
        # Try to sync
        async with session.post('http://localhost:8002/api/concept_mesh/sync_to_scholarsphere') as resp:
            result = await resp.json()
            print(f"\\nSync result: {json.dumps(result, indent=2)}")
            
            if result.get('success'):
                print("\\n✅ ScholarSphere sync successful!")
                print(f"   Diff ID: {result.get('diffId')}")
                print(f"   Concepts: {result.get('conceptCount')}")
            else:
                print("\\n❌ ScholarSphere sync failed")
                print(f"   Message: {result.get('message')}")

if __name__ == "__main__":
    asyncio.run(test_scholarsphere())
'''
    
    with open('test_scholarsphere_sync.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("✅ Created test_scholarsphere_sync.py")

# Step 4: Create environment setup script
def create_env_setup():
    """Create script to set environment variables"""
    
    setup_script = '''@echo off
echo Setting ScholarSphere environment variables...

REM Replace with your actual API key
set SCHOLARSPHERE_API_KEY=your-scholarsphere-api-key-here
set SCHOLARSPHERE_API_URL=https://api.scholarsphere.org
set SCHOLARSPHERE_BUCKET=concept-diffs

REM Enable entropy pruning
set TORI_ENABLE_ENTROPY_PRUNING=1
set TORI_DISABLE_ENTROPY_PRUNE=

echo.
echo Environment variables set:
echo   SCHOLARSPHERE_API_KEY = %SCHOLARSPHERE_API_KEY%
echo   SCHOLARSPHERE_API_URL = %SCHOLARSPHERE_API_URL%
echo   SCHOLARSPHERE_BUCKET = %SCHOLARSPHERE_BUCKET%
echo   TORI_ENABLE_ENTROPY_PRUNING = %TORI_ENABLE_ENTROPY_PRUNING%
echo.
echo IMPORTANT: Replace 'your-scholarsphere-api-key-here' with your actual API key!
echo.
echo Now run: poetry run python enhanced_launcher.py
pause
'''
    
    with open('setup_scholarsphere_env.bat', 'w', encoding='utf-8') as f:
        f.write(setup_script)
    
    print("✅ Created setup_scholarsphere_env.bat")

def main():
    """Main fix process"""
    
    print("🔧 FIXING SCHOLARSPHERE FALLBACK MODE")
    print("=" * 50)
    
    # Step 1: Set environment (for current process)
    print("\n📋 Step 1: Setting environment variables...")
    set_scholarsphere_env()
    
    # Step 2: Fix the API
    print("\n🔌 Step 2: Updating concept mesh API...")
    fix_concept_mesh_api()
    
    # Step 3: Create test script
    print("\n🧪 Step 3: Creating test script...")
    create_test_script()
    
    # Step 4: Create environment setup
    print("\n🔧 Step 4: Creating environment setup script...")
    create_env_setup()
    
    print("\n✅ FIX COMPLETE!")
    print("=" * 50)
    print("\nNEXT STEPS:")
    print("1. Edit setup_scholarsphere_env.bat and add your real API key")
    print("2. Run: .\\setup_scholarsphere_env.bat")
    print("3. Restart launcher in the same window: poetry run python enhanced_launcher.py")
    print("4. Test the sync: python test_scholarsphere_sync.py")
    print("\nThe fallback mode should now be resolved!")

if __name__ == "__main__":
    main()
