#!/usr/bin/env python3
"""
TORI HOLOGRAM SYSTEM - AUTOMATIC WIRING FIX
This script automatically fixes the critical wiring issue by replacing
all imports of the placeholder ghostEngine.js with the real implementation.
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime

class TORIWiringFixer:
    def __init__(self, base_path="C:\\Users\\jason\\Desktop\\tori\\kha"):
        self.base_path = Path(base_path)
        self.backup_dir = self.base_path / f"backups_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.changes_made = []
        
    def create_backup(self, file_path):
        """Create a backup of the file before modifying"""
        relative_path = file_path.relative_to(self.base_path)
        backup_path = self.backup_dir / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_path)
        
    def fix_imports(self, file_path):
        """Fix imports in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            original_content = content
            
            # Pattern 1: import GhostEngine from './ghostEngine'
            content = re.sub(
                r"import\s+GhostEngine\s+from\s+['\"]\.\/ghostEngine(?:\.js)?['\"]",
                "import { RealGhostEngine as GhostEngine } from './realGhostEngine.js'",
                content
            )
            
            # Pattern 2: import { GhostEngine } from './ghostEngine'
            content = re.sub(
                r"import\s+\{\s*GhostEngine\s*\}\s+from\s+['\"]\.\/ghostEngine(?:\.js)?['\"]",
                "import { RealGhostEngine as GhostEngine } from './realGhostEngine.js'",
                content
            )
            
            # Pattern 3: const GhostEngine = require('./ghostEngine')
            content = re.sub(
                r"const\s+GhostEngine\s*=\s*require\s*\(['\"]\.\/ghostEngine(?:\.js)?['\"]\)",
                "const { RealGhostEngine: GhostEngine } = require('./realGhostEngine.js')",
                content
            )
            
            # Pattern 4: import GhostEngine from '$lib/ghostEngine'
            content = re.sub(
                r"import\s+GhostEngine\s+from\s+['\"]\\$lib\/ghostEngine(?:\.js)?['\"]",
                "import { RealGhostEngine as GhostEngine } from '$lib/realGhostEngine.js'",
                content
            )
            
            if content != original_content:
                self.create_backup(file_path)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.changes_made.append(str(file_path))
                return True
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
        return False
    
    def find_and_fix_all(self):
        """Find all files that need fixing and fix them"""
        print("🔍 Searching for files to fix...")
        
        # Directories to search
        search_dirs = [
            self.base_path / "tori_ui_svelte",
            self.base_path / "frontend",
            self.base_path / "src"
        ]
        
        # File extensions to check
        extensions = ['.js', '.ts', '.svelte', '.jsx', '.tsx']
        
        files_to_check = []
        for search_dir in search_dirs:
            if search_dir.exists():
                for ext in extensions:
                    files_to_check.extend(search_dir.rglob(f"*{ext}"))
        
        print(f"📋 Found {len(files_to_check)} files to check")
        
        fixed_count = 0
        for file_path in files_to_check:
            if 'node_modules' in str(file_path) or 'backup' in str(file_path):
                continue
                
            if self.fix_imports(file_path):
                fixed_count += 1
                print(f"✅ Fixed: {file_path.relative_to(self.base_path)}")
        
        return fixed_count
    
    def create_systemd_services(self):
        """Create systemd service files for all TORI components"""
        services_dir = self.base_path / "TORI_IMPLEMENTATION" / "systemd"
        services_dir.mkdir(parents=True, exist_ok=True)
        
        # TORI Backend Service
        backend_service = """[Unit]
Description=TORI Backend API Service
After=network.target

[Service]
Type=simple
User=tori
WorkingDirectory=/opt/tori/backend
Environment="PATH=/opt/tori/.venv/bin:/usr/local/bin:/usr/bin"
ExecStart=/opt/tori/.venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        (services_dir / "tori-backend.service").write_text(backend_service, encoding='utf-8')
        
        # Audio Bridge Service
        audio_service = """[Unit]
Description=TORI Audio Bridge Service
After=network.target tori-backend.service

[Service]
Type=simple
User=tori
WorkingDirectory=/opt/tori
Environment="PATH=/opt/tori/.venv/bin:/usr/local/bin:/usr/bin"
ExecStart=/opt/tori/.venv/bin/python audio_hologram_bridge.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        (services_dir / "tori-audio-bridge.service").write_text(audio_service, encoding='utf-8')
        
        # Hologram Renderer Service
        hologram_service = """[Unit]
Description=TORI Hologram Renderer Service
After=network.target

[Service]
Type=simple
User=tori
WorkingDirectory=/opt/tori/hologram-service
Environment="HOLOGRAM_MODE=desktop"
Environment="NUM_VIEWS=45"
Environment="WEBGPU_BACKEND=nvidia"
ExecStart=/usr/bin/node dist/hologram-desktop.js
Restart=always
RestartSec=10
# Grant real-time priority
AmbientCapabilities=CAP_SYS_NICE

[Install]
WantedBy=multi-user.target
"""
        (services_dir / "tori-hologram.service").write_text(hologram_service, encoding='utf-8')
        
        print("✅ Created systemd service files")
    
    def create_launch_script(self):
        """Create a master launch script"""
        script_content = """#!/bin/bash
# TORI System Launch Script

echo "🚀 Starting TORI Hologram System..."

# Check if running as root for systemd
if [ "$EUID" -eq 0 ]; then 
    echo "⚠️  Please run as normal user, not root"
    exit 1
fi

# Function to check if service is running
check_service() {
    if systemctl is-active --quiet $1; then
        echo "✅ $1 is running"
    else
        echo "❌ $1 is not running"
        return 1
    fi
}

# Start all services
echo "Starting backend services..."
sudo systemctl start tori-backend
sleep 3

sudo systemctl start tori-audio-bridge
sleep 2

sudo systemctl start tori-hologram
sleep 2

# Check all services
echo ""
echo "Service Status:"
check_service tori-backend
check_service tori-audio-bridge
check_service tori-hologram

# Start the frontend
echo ""
echo "Starting frontend..."
cd tori_ui_svelte
npm run dev &

echo ""
echo "✅ TORI System started!"
echo "🌐 Frontend: http://localhost:5173"
echo "🔌 Backend API: http://localhost:8001"
echo "🎵 Audio Bridge: ws://localhost:6789"
echo "📺 Hologram Status: http://localhost:7690/health"
"""
        
        script_path = self.base_path / "TORI_IMPLEMENTATION" / "launch_tori.sh"
        script_path.write_text(script_content, encoding='utf-8')
        script_path.chmod(0o755)
        print("✅ Created launch script")
    
    def create_test_script(self):
        """Create a test script to verify the system is working"""
        test_content = """#!/usr/bin/env python3
import asyncio
import aiohttp
import json

async def test_tori_system():
    print("🧪 Testing TORI System Components...")
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Backend API Health
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8001/api/health') as resp:
                if resp.status == 200:
                    print("✅ Backend API is healthy")
                    tests_passed += 1
                else:
                    print("❌ Backend API health check failed")
                    tests_failed += 1
    except Exception as e:
        print(f"❌ Backend API error: {e}")
        tests_failed += 1
    
    # Test 2: WebSocket Connection
    try:
        import websockets
        async with websockets.connect('ws://localhost:6789') as ws:
            await ws.send(json.dumps({"type": "ping"}))
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            print("✅ Audio WebSocket is responsive")
            tests_passed += 1
    except Exception as e:
        print(f"❌ Audio WebSocket error: {e}")
        tests_failed += 1
    
    # Test 3: Hologram Service Health
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:7690/health') as resp:
                if resp.status == 200:
                    print("✅ Hologram service is healthy")
                    tests_passed += 1
                else:
                    print("❌ Hologram service health check failed")
                    tests_failed += 1
    except Exception as e:
        print(f"❌ Hologram service error: {e}")
        tests_failed += 1
    
    # Test 4: Concept Mesh Query
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"filter": {"type": "test"}}
            async with session.post('http://localhost:8001/api/soliton/query', 
                                   json=payload) as resp:
                if resp.status == 200:
                    print("✅ Concept Mesh is accessible")
                    tests_passed += 1
                else:
                    print("❌ Concept Mesh query failed")
                    tests_failed += 1
    except Exception as e:
        print(f"❌ Concept Mesh error: {e}")
        tests_failed += 1
    
    print(f"\n📊 Test Results: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0

if __name__ == "__main__":
    asyncio.run(test_tori_system())
"""
        
        test_path = self.base_path / "TORI_IMPLEMENTATION" / "test_tori_system.py"
        test_path.write_text(test_content, encoding='utf-8')
        test_path.chmod(0o755)
        print("✅ Created test script")
    
    def run(self):
        """Run the complete wiring fix"""
        print("🔧 TORI Hologram System - Automatic Wiring Fix")
        print("=" * 50)
        
        # Step 1: Fix all imports
        fixed_count = self.find_and_fix_all()
        
        if fixed_count > 0:
            print(f"\n✅ Fixed {fixed_count} files!")
            print(f"📁 Backups saved to: {self.backup_dir}")
        else:
            print("\n✅ No files needed fixing - system may already be wired correctly")
        
        # Step 2: Create service files
        self.create_systemd_services()
        
        # Step 3: Create launch script
        self.create_launch_script()
        
        # Step 4: Create test script
        self.create_test_script()
        
        # Print summary
        print("\n" + "=" * 50)
        print("🎉 TORI Wiring Fix Complete!")
        print("\nNext steps:")
        print("1. Install systemd services:")
        print("   sudo cp TORI_IMPLEMENTATION/systemd/*.service /etc/systemd/system/")
        print("   sudo systemctl daemon-reload")
        print("2. Launch TORI:")
        print("   ./TORI_IMPLEMENTATION/launch_tori.sh")
        print("3. Test the system:")
        print("   python TORI_IMPLEMENTATION/test_tori_system.py")
        
        if self.changes_made:
            print(f"\nFiles modified:")
            for file in self.changes_made[:5]:  # Show first 5
                print(f"  - {Path(file).relative_to(self.base_path)}")
            if len(self.changes_made) > 5:
                print(f"  ... and {len(self.changes_made) - 5} more")

if __name__ == "__main__":
    fixer = TORIWiringFixer()
    fixer.run()
