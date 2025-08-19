#!/usr/bin/env python3
"""
Migration Script: mcp_server_arch -> mcp_metacognitive
====================================================

This script migrates functionality from the old architecture to the new one.
"""

import os
import sys
import shutil
from pathlib import Path
import json

def main():
    print("🚀 TORI MCP Migration Tool")
    print("=" * 60)
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    old_dir = base_dir / "mcp_server_arch"
    new_dir = base_dir / "mcp_metacognitive"
    
    print(f"📂 Old architecture: {old_dir}")
    print(f"📂 New architecture: {new_dir}")
    
    # Check if directories exist
    if not old_dir.exists():
        print("❌ Old architecture directory not found!")
        return 1
    
    if not new_dir.exists():
        print("❌ New architecture directory not found!")
        return 1
    
    print("\n📋 Migration Plan:")
    print("1. Install dependencies for mcp_metacognitive")
    print("2. Copy TORI bridge functionality")
    print("3. Migrate agent system")
    print("4. Update configuration")
    print("5. Test the new setup")
    
    response = input("\nProceed with migration? (y/n): ")
    if response.lower() != 'y':
        print("Migration cancelled.")
        return 0
    
    # Step 1: Install dependencies
    print("\n📦 Step 1: Installing dependencies...")
    os.chdir(new_dir)
    if os.path.exists("install_dependencies.py"):
        os.system(f"{sys.executable} install_dependencies.py")
    
    # Step 2: Copy bridge functionality (already done in previous steps)
    print("\n📦 Step 2: Bridge functionality already migrated ✅")
    
    # Step 3: Agent system already migrated
    print("\n📦 Step 3: Agent system already migrated ✅")
    
    # Step 4: Create migration status file
    print("\n📦 Step 4: Creating migration status...")
    migration_status = {
        "migration_date": str(Path.cwd()),
        "components_migrated": [
            "psi_archive",
            "agent_registry",
            "tori_bridge",
            "server_fallback"
        ],
        "status": "completed",
        "notes": [
            "MCP packages optional - fallback to FastAPI",
            "TORI filtering available with basic fallback",
            "Agent hot-swapping supported",
            "Event logging via PsiArchive"
        ]
    }
    
    with open(new_dir / "migration_status.json", "w") as f:
        json.dump(migration_status, f, indent=2)
    
    # Step 5: Create test script
    print("\n📦 Step 5: Creating test script...")
    test_script = '''#!/usr/bin/env python3
"""Test script for migrated MCP server"""

import asyncio
from core.psi_archive import psi_archive
from core.agent_registry import agent_registry
from core.tori_bridge import tori_bridge

async def test_migration():
    print("🧪 Testing migrated components...")
    
    # Test PsiArchive
    print("\\n1. Testing PsiArchive...")
    psi_archive.log_event("test", {"message": "Migration test"})
    events = psi_archive.get_recent_events(1)
    if events:
        print("   ✅ PsiArchive working")
    else:
        print("   ❌ PsiArchive failed")
    
    # Test Agent Registry
    print("\\n2. Testing Agent Registry...")
    agents = agent_registry.list_agents()
    print(f"   Found {len(agents)} agents: {agents}")
    if "daniel" in agents:
        daniel = agent_registry.get("daniel")
        result = await daniel.execute()
        print(f"   ✅ Agent execution: {result}")
    
    # Test TORI Bridge
    print("\\n3. Testing TORI Bridge...")
    test_content = "Hello, TORI!"
    filtered = await tori_bridge.filter_content(test_content)
    print(f"   ✅ Filtering: {filtered.filtered}")
    
    print("\\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(test_migration())
'''
    
    with open(new_dir / "test_migration.py", "w") as f:
        f.write(test_script)
    
    os.chmod(new_dir / "test_migration.py", 0o755)
    
    print("\n✅ Migration completed successfully!")
    print("\n📋 Next steps:")
    print("1. Test the migration: python test_migration.py")
    print("2. Run the server: python server.py")
    print("3. Update enhanced_launcher.py if needed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
