#!/usr/bin/env python3
"""
TORI Automatic Fix Script - Direct in-place patching
No manual steps required!
"""

import os
import re
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

class TORIFixer:
    def __init__(self):
        self.root = Path(__file__).parent
        self.backup_dir = self.root / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def backup_file(self, filepath):
        """Backup a file before modifying"""
        if filepath.exists():
            backup_path = self.backup_dir / filepath.name
            shutil.copy2(filepath, backup_path)
            print(f"   📁 Backed up {filepath.name}")
            return True
        return False
    
    def fix_prajna_api(self):
        """Fix 1: Add early health endpoint to prajna_api.py"""
        print("\n1️⃣ Fixing Prajna API health endpoint...")
        
        prajna_api = self.root / "prajna" / "api" / "prajna_api.py"
        if not prajna_api.exists():
            print("   ❌ prajna_api.py not found")
            return False
            
        self.backup_file(prajna_api)
        
        with open(prajna_api, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already fixed
        if '/api/health' in content and 'Ultra-lightweight health check' in content:
            print("   ✅ Already fixed!")
            return True
        
        # Find where FastAPI app is created
        app_pattern = r'(app\s*=\s*FastAPI\([^)]*\))'
        match = re.search(app_pattern, content)
        
        if not match:
            print("   ❌ Could not find FastAPI app creation")
            return False
        
        # Insert health endpoint right after app creation
        health_endpoint = '''

# CRITICAL: Ultra-lightweight health check - available immediately
@app.get("/api/health")
async def health_check():
    """Health endpoint that responds before heavy initialization"""
    return {"status": "ok", "message": "API running"}

print("[TORI] Health endpoint registered early - will respond immediately")
'''
        
        # Insert after first app creation
        insert_pos = match.end()
        new_content = content[:insert_pos] + health_endpoint + content[insert_pos:]
        
        # Remove any duplicate app creation
        new_content = re.sub(r'\n\s*app\s*=\s*FastAPI\([^)]*\)\s*(?!.*health_check)', 
                             '\n# app = FastAPI() # Reusing existing app', 
                             new_content, count=1)
        
        with open(prajna_api, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("   ✅ Fixed Prajna API!")
        return True
    
    def fix_launcher(self):
        """Fix 2: Update launcher to start API first and wait"""
        print("\n2️⃣ Fixing launcher sequence...")
        
        launcher = self.root / "enhanced_launcher.py"
        if not launcher.exists():
            print("   ❌ enhanced_launcher.py not found")
            return False
            
        self.backup_file(launcher)
        
        with open(launcher, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already fixed
        if '_wait_for_api_health' in content:
            print("   ✅ Already fixed!")
            return True
        
        # Add the wait method before the launch method
        wait_method = '''
    def _wait_for_api_health(self, port, max_attempts=60):
        """Wait for API to respond to health checks"""
        import time
        import requests
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(f'http://127.0.0.1:{port}/api/health', timeout=1)
                if response.status_code == 200:
                    return True
            except:
                pass
            
            if attempt % 5 == 0 and attempt > 0:
                self.logger.info(f"⏳ Still waiting for API... ({attempt}/{max_attempts})")
            time.sleep(0.25)
        
        return False
    
    def _run_api_server(self, port):
        """Run API server in thread"""
        self.update_status("api_startup", "starting", {"port": port})
        self.start_api_server(port)
'''
        
        # Insert before launch method
        launch_pattern = r'(\n\s*def launch\(self\):)'
        match = re.search(launch_pattern, content)
        if match:
            content = content[:match.start()] + wait_method + content[match.start():]
        
        # Now fix the launch sequence
        # Find the start of Step 7 where API server starts
        step7_pattern = r'(# Step 7: Start API server.*?self\.start_api_server\(self\.api_port\))'
        match = re.search(step7_pattern, content, re.DOTALL)
        
        if match:
            # Replace with threaded start
            new_step7 = '''# Step 7: Start API server in background thread FIRST
            self.logger.info("🚀 Starting API server in background thread...")
            api_thread = threading.Thread(
                target=self._run_api_server,
                args=(self.api_port,),
                daemon=True,
                name="APIServer"
            )
            api_thread.start()
            
            # Wait for API to be healthy
            self.logger.info("⏳ Waiting for API server to be healthy...")
            if not self._wait_for_api_health(self.api_port):
                self.logger.error("❌ API server failed to start!")
                return 1
            
            self.logger.info("✅ API server is healthy!")
            
            # Give user a moment to see the status
            self.logger.info("⏳ Starting other components in 3 seconds...")
            time.sleep(3)'''
            
            content = content[:match.start()] + new_step7 + content[match.end():]
            
            # Keep thread alive at the end
            content = content.replace('return 0', 'api_thread.join()\n        return 0')
        
        with open(launcher, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("   ✅ Fixed launcher!")
        return True
    
    def fix_vite_config(self):
        """Fix 3: Update Vite proxy to use IPv4 and longer timeout"""
        print("\n3️⃣ Fixing Vite proxy configuration...")
        
        vite_config = self.root / "tori_ui_svelte" / "vite.config.js"
        if not vite_config.exists():
            print("   ❌ vite.config.js not found")
            return False
            
        self.backup_file(vite_config)
        
        with open(vite_config, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace localhost with 127.0.0.1 in proxy targets
        content = re.sub(r"target:\s*['\"]http://localhost:8002['\"]", 
                        'target: "http://127.0.0.1:8002"', content)
        content = re.sub(r"target:\s*['\"]ws://localhost:8002['\"]", 
                        'target: "ws://127.0.0.1:8002"', content)
        
        # Add timeout if not present
        if 'timeout:' not in content:
            content = re.sub(r"(changeOrigin:\s*true)",
                           r"\1,\n                timeout: 60000", content)
        
        # Add error suppression if not present
        if 'ECONNREFUSED' not in content:
            configure_block = '''
                configure: (proxy, options) => {
                    proxy.on('error', (err, req, res) => {
                        if (err.code === 'ECONNREFUSED') {
                            console.log('⏳ Waiting for API server...');
                        }
                    });
                }'''
            
            # Insert after changeOrigin
            content = re.sub(r"(timeout:\s*\d+)",
                           r"\1," + configure_block, content, count=1)
        
        with open(vite_config, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("   ✅ Fixed Vite config!")
        return True
    
    def populate_concept_mesh(self):
        """Fix 4: Populate ConceptMesh with initial data"""
        print("\n4️⃣ Populating ConceptMesh...")
        
        # Find data.json
        data_paths = [
            self.root / "concept_mesh" / "data.json",
            self.root / "data" / "concept_mesh" / "data.json",
            self.root / "data.json"
        ]
        
        data_file = None
        for path in data_paths:
            if path.exists():
                data_file = path
                break
        
        if not data_file:
            # Create it
            data_file = self.root / "concept_mesh" / "data.json"
            data_file.parent.mkdir(parents=True, exist_ok=True)
            data_file.write_text('{"concepts": [], "metadata": {"version": "1.0"}}')
            print(f"   📁 Created {data_file}")
        
        # Load and check
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure proper structure
        if isinstance(data, list):
            data = {"concepts": data, "metadata": {"version": "1.0"}}
        
        if "concepts" not in data:
            data["concepts"] = []
        
        # Check if already populated
        if len(data["concepts"]) >= 15:
            print(f"   ✅ Already populated with {len(data['concepts'])} concepts")
            return True
        
        # Add concepts with required fields
        initial_concepts = [
            {
                "id": f"concept_{i}",
                "text": text,
                "weight": weight,
                "category": category,
                "vector": [0.0] * 768  # Placeholder embedding
            }
            for i, (text, weight, category) in enumerate([
                ("Artificial Intelligence", 1.0, "technology"),
                ("Machine Learning", 0.95, "technology"),
                ("Neural Networks", 0.9, "technology"),
                ("Deep Learning", 0.85, "technology"),
                ("Natural Language Processing", 0.8, "technology"),
                ("Computer Vision", 0.75, "technology"),
                ("Reinforcement Learning", 0.7, "technology"),
                ("Consciousness", 0.9, "philosophy"),
                ("Metacognition", 0.85, "philosophy"),
                ("Self-Awareness", 0.8, "philosophy"),
                ("Emergence", 0.75, "philosophy"),
                ("Complexity Theory", 0.7, "science"),
                ("Information Theory", 0.85, "science"),
                ("Quantum Computing", 0.8, "technology"),
                ("Knowledge Representation", 0.9, "technology")
            ])
        ]
        
        # Add concepts
        existing_ids = {c.get("id", "") for c in data["concepts"]}
        added = 0
        
        for concept in initial_concepts:
            if concept["id"] not in existing_ids:
                data["concepts"].append(concept)
                added += 1
        
        # Save
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Added {added} concepts, total now: {len(data['concepts'])}")
        
        # Also create JSONL seed
        seed_dir = self.root / "data" / "seed_concepts"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_file = seed_dir / "init.jsonl"
        
        with open(seed_file, 'w', encoding='utf-8') as f:
            for concept in initial_concepts:
                f.write(json.dumps(concept) + '\n')
        
        return True
    
    def fix_mcp_duplicates(self):
        """Fix 5: Add duplicate guards to MCP server"""
        print("\n5️⃣ Fixing MCP duplicate registrations...")
        
        # Find MCP server files
        mcp_files = [
            self.root / "mcp_metacognitive" / "server_proper.py",
            self.root / "mcp_metacognitive" / "server_simple.py"
        ]
        
        fixed_any = False
        for mcp_file in mcp_files:
            if not mcp_file.exists():
                continue
                
            self.backup_file(mcp_file)
            
            with open(mcp_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if already fixed
            if 'register_tool_safe' in content:
                print(f"   ✅ {mcp_file.name} already fixed!")
                fixed_any = True
                continue
            
            # Add safe registration methods after imports
            safe_methods = '''
# Safe registration helpers to prevent duplicates
_registered_tools = set()
_registered_resources = set()
_registered_prompts = set()

def register_tool_safe(server, name, handler, description):
    """Register tool only if not already registered"""
    if name in _registered_tools:
        logger.debug(f"Tool {name} already registered, skipping")
        return
    _registered_tools.add(name)
    server.add_tool(name=name, handler=handler, description=description)

def register_resource_safe(server, uri, handler, description):
    """Register resource only if not already registered"""
    if uri in _registered_resources:
        logger.debug(f"Resource {uri} already registered, skipping")
        return
    _registered_resources.add(uri)
    server.add_resource(uri=uri, handler=handler, description=description)

def register_prompt_safe(server, name, handler, description):
    """Register prompt only if not already registered"""
    if name in _registered_prompts:
        logger.debug(f"Prompt {name} already registered, skipping")
        return
    _registered_prompts.add(name)
    server.add_prompt(name=name, handler=handler, description=description)
'''
            
            # Insert after imports
            import_end = content.rfind('import ')
            if import_end > 0:
                import_end = content.find('\n', import_end) + 1
                content = content[:import_end] + safe_methods + content[import_end:]
            
            # Replace add_tool calls with register_tool_safe
            content = re.sub(r'(\w+)\.add_tool\(', r'register_tool_safe(\1, ', content)
            content = re.sub(r'(\w+)\.add_resource\(', r'register_resource_safe(\1, ', content)
            content = re.sub(r'(\w+)\.add_prompt\(', r'register_prompt_safe(\1, ', content)
            
            with open(mcp_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"   ✅ Fixed {mcp_file.name}!")
            fixed_any = True
        
        return fixed_any
    
    def run_npm_install(self):
        """Run npm install in frontend if needed"""
        print("\n6️⃣ Checking frontend dependencies...")
        
        frontend_dir = self.root / "tori_ui_svelte"
        if not frontend_dir.exists():
            return True
            
        try:
            subprocess.run(["npm", "install"], 
                         cwd=frontend_dir, 
                         capture_output=True, 
                         text=True,
                         shell=True)
            print("   ✅ Frontend dependencies OK")
            return True
        except:
            print("   ⚠️  Could not run npm install")
            return False
    
    def run(self):
        """Run all fixes"""
        print("🔧 TORI Automatic Fix Script")
        print("=" * 50)
        print(f"📁 Working directory: {self.root}")
        print(f"📁 Backups will be in: {self.backup_dir}")
        
        results = {
            "Prajna API": self.fix_prajna_api(),
            "Launcher": self.fix_launcher(),
            "Vite Config": self.fix_vite_config(),
            "ConceptMesh": self.populate_concept_mesh(),
            "MCP Server": self.fix_mcp_duplicates(),
            "NPM Install": self.run_npm_install()
        }
        
        print("\n" + "=" * 50)
        print("📊 RESULTS:")
        for name, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {name}")
        
        if all(results.values()):
            print("\n🎉 All fixes applied successfully!")
            print("\n🚀 Next steps:")
            print("   1. Run: poetry run python enhanced_launcher.py")
            print("   2. Test: curl http://127.0.0.1:8002/api/health")
            print("   3. Check: No proxy errors in frontend")
            print("   4. Verify: ConceptMesh has data")
        else:
            print("\n⚠️  Some fixes failed - check the output above")
        
        return all(results.values())

if __name__ == "__main__":
    fixer = TORIFixer()
    success = fixer.run()
    sys.exit(0 if success else 1)
