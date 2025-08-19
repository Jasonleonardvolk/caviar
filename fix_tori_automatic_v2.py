#!/usr/bin/env python3
"""
TORI Automatic Fix Script v2 - Production-ready with AST parsing and rollback
"""

import ast
import os
import re
import sys
import json
import uuid
import shutil
import hashlib
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlunparse
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class BackupManager:
    """Manages backups with rollback capability"""
    
    def __init__(self, root: Path):
        self.root = root
        self.backup_dir = root / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backed_up_files: Dict[Path, Path] = {}
        self.prune_old_backups()
    
    def backup(self, filepath: Path) -> Optional[Path]:
        """Backup a file and track it for rollback"""
        if not filepath.exists():
            return None
            
        backup_path = self.backup_dir / filepath.name
        shutil.copy2(filepath, backup_path)
        self.backed_up_files[filepath] = backup_path
        logger.info(f"   📁 Backed up {filepath.name}")
        return backup_path
    
    def rollback(self):
        """Restore all backed up files"""
        logger.warning("🔄 Rolling back all changes...")
        for original, backup in self.backed_up_files.items():
            try:
                shutil.copy2(backup, original)
                logger.info(f"   ↩️  Restored {original.name}")
            except Exception as e:
                logger.error(f"   ❌ Failed to restore {original.name}: {e}")
    
    def prune_old_backups(self, days=30):
        """Remove backups older than N days"""
        cutoff = datetime.now() - timedelta(days=days)
        backups_root = self.root / "backups"
        
        if not backups_root.exists():
            return
            
        for backup_dir in backups_root.iterdir():
            if backup_dir.is_dir():
                try:
                    # Parse timestamp from dirname
                    timestamp = datetime.strptime(backup_dir.name, "%Y%m%d_%H%M%S")
                    if timestamp < cutoff:
                        shutil.rmtree(backup_dir)
                        logger.debug(f"Pruned old backup: {backup_dir.name}")
                except:
                    pass

class ASTHelper:
    """AST-based code manipulation for Python files"""
    
    @staticmethod
    def add_method_to_class(source: str, class_name: str, method_code: str) -> str:
        """Add a method to a class using AST parsing"""
        tree = ast.parse(source)
        
        # Find the class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Parse the new method
                method_ast = ast.parse(method_code).body[0]
                
                # Find insertion point (before launch method if exists)
                insert_idx = len(node.body)
                for i, item in enumerate(node.body):
                    if isinstance(item, ast.FunctionDef) and item.name == "launch":
                        insert_idx = i
                        break
                
                # Insert the method
                node.body.insert(insert_idx, method_ast)
                
                # Convert back to source
                return ast.unparse(tree)
        
        # Fallback if class not found
        return source
    
    @staticmethod
    def modify_function_body(source: str, func_name: str, 
                           old_pattern: str, new_code: str) -> str:
        """Modify a function's body using pattern matching"""
        # This is still regex-based but more targeted
        func_pattern = rf'(def {func_name}\([^)]*\):.*?)({re.escape(old_pattern)})'
        
        def replacer(match):
            return match.group(1) + new_code
        
        return re.sub(func_pattern, replacer, source, flags=re.DOTALL)

class TORIFixerV2:
    """Enhanced TORI fixer with better error handling and AST parsing"""
    
    def __init__(self):
        self.root = Path(__file__).parent
        self.backup_manager = BackupManager(self.root)
        self.patch_metadata_file = self.root / ".tori_patch_version"
        self.fixes_applied = []
        
    def get_api_port(self) -> int:
        """Get API port from env or default"""
        env_file = self.root / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("API_PORT="):
                        return int(line.split("=")[1].strip())
        return 8002
    
    def record_patch_metadata(self):
        """Record patch version and timestamp"""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "git_commit": self._get_git_commit(),
            "fixes_applied": self.fixes_applied,
            "api_port": self.get_api_port()
        }
        
        with open(self.patch_metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📝 Recorded patch metadata to {self.patch_metadata_file.name}")
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.root
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None
    
    def fix_prajna_api(self) -> bool:
        """Fix 1: Add early health endpoint using smarter approach"""
        logger.info("\n1️⃣ Fixing Prajna API health endpoint...")
        
        prajna_api = self.root / "prajna" / "api" / "prajna_api.py"
        if not prajna_api.exists():
            logger.error("   ❌ prajna_api.py not found")
            return False
        
        try:
            self.backup_manager.backup(prajna_api)
            
            with open(prajna_api, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if already fixed
            if '@app.get("/api/health")' in content:
                logger.info("   ✅ Already fixed!")
                return True
            
            # Parse to find all app creations
            app_creations = list(re.finditer(r'^app\s*=\s*FastAPI\([^)]*\)', content, re.MULTILINE))
            
            if not app_creations:
                logger.error("   ❌ No FastAPI app creation found")
                return False
            
            # Insert health endpoint after FIRST app creation
            first_app = app_creations[0]
            insert_pos = first_app.end()
            
            health_endpoint = '''

# CRITICAL: Ultra-lightweight health check - available immediately
@app.get("/api/health")
async def health_check():
    """Health endpoint that responds before heavy initialization"""
    return {"status": "ok", "message": "API running"}

logger.info("[TORI] Health endpoint registered early - will respond immediately")
'''
            
            new_content = content[:insert_pos] + health_endpoint + content[insert_pos:]
            
            # Convert subsequent app creations to reuse first app
            if len(app_creations) > 1:
                for app_match in reversed(app_creations[1:]):  # Reverse to maintain positions
                    start, end = app_match.span()
                    new_content = new_content[:start] + "# Reusing existing app instance\n# " + new_content[start:end] + "\napp = app  # Reuse" + new_content[end:]
            
            with open(prajna_api, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            logger.info("   ✅ Fixed Prajna API!")
            self.fixes_applied.append("prajna_api")
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Failed to fix Prajna API: {e}")
            return False
    
    def fix_launcher(self) -> bool:
        """Fix 2: Update launcher using AST parsing for reliability"""
        logger.info("\n2️⃣ Fixing launcher sequence...")
        
        launcher = self.root / "enhanced_launcher.py"
        if not launcher.exists():
            logger.error("   ❌ enhanced_launcher.py not found")
            return False
        
        try:
            self.backup_manager.backup(launcher)
            
            with open(launcher, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if already fixed
            if '_wait_for_api_health' in content:
                logger.info("   ✅ Already fixed!")
                return True
            
            # Add helper methods
            wait_methods = '''
    def _wait_for_api_health(self, port, max_attempts=60):
        """Wait for API to respond to health checks"""
        import time
        import requests
        
        wait_times = [0.25, 0.5, 1.0, 2.0, 5.0]  # Exponential backoff
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(f'http://127.0.0.1:{port}/api/health', timeout=1)
                if response.status_code == 200:
                    return True
            except:
                pass
            
            wait_time = wait_times[min(attempt, len(wait_times)-1)]
            if attempt % 5 == 0 and attempt > 0:
                self.logger.info(f"⏳ Still waiting for API... ({attempt}/{max_attempts})")
            time.sleep(wait_time)
        
        return False
    
    def _run_api_server(self, port):
        """Run API server in thread"""
        self.update_status("api_startup", "starting", {"port": port})
        try:
            self.start_api_server(port)
        except Exception as e:
            self.logger.error(f"API server crashed: {e}")
'''
            
            # Try AST approach first
            try:
                new_content = ASTHelper.add_method_to_class(
                    content, 
                    "EnhancedUnifiedToriLauncher",
                    wait_methods
                )
            except:
                # Fallback to regex
                class_pattern = r'(class EnhancedUnifiedToriLauncher.*?:)'
                match = re.search(class_pattern, content)
                if match:
                    # Find the next method definition
                    next_method = content.find('\n    def ', match.end())
                    if next_method > 0:
                        new_content = content[:next_method] + wait_methods + content[next_method:]
                    else:
                        new_content = content + wait_methods
                else:
                    logger.error("   ❌ Could not find launcher class")
                    return False
            
            # Now fix the launch sequence - find where API server starts
            api_start_pattern = r'# Step \d+: Start API server.*?self\.start_api_server\(self\.api_port\)'
            
            replacement = '''# Step 7: Start API server in background thread FIRST
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
                sys.exit(1)
            
            self.logger.info("✅ API server is healthy!")'''
            
            new_content = re.sub(api_start_pattern, replacement, new_content, flags=re.DOTALL)
            
            # Make sure we keep the thread alive
            if 'api_thread.join()' not in new_content:
                new_content = new_content.replace(
                    'return 0',
                    'api_thread.join()\n        return 0'
                )
            
            with open(launcher, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            logger.info("   ✅ Fixed launcher!")
            self.fixes_applied.append("launcher")
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Failed to fix launcher: {e}")
            return False
    
    def fix_vite_config(self) -> bool:
        """Fix 3: Update Vite proxy with better URL parsing"""
        logger.info("\n3️⃣ Fixing Vite proxy configuration...")
        
        vite_config = self.root / "tori_ui_svelte" / "vite.config.js"
        if not vite_config.exists():
            logger.error("   ❌ vite.config.js not found")
            return False
        
        try:
            self.backup_manager.backup(vite_config)
            
            with open(vite_config, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Function to replace URLs safely
            def replace_url(match):
                url = match.group(1)
                parsed = urlparse(url)
                if parsed.hostname == 'localhost':
                    new_parsed = parsed._replace(netloc=f'127.0.0.1:{parsed.port or 8002}')
                    return f'target: "{urlunparse(new_parsed)}"'
                return match.group(0)
            
            # Replace all proxy targets
            content = re.sub(r'target:\s*["\']([^"\']+)["\']', replace_url, content)
            
            # Add timeout if not present
            if 'timeout:' not in content:
                content = re.sub(
                    r'(changeOrigin:\s*true)(?!.*timeout)',
                    r'\1,\n                timeout: 60000',
                    content
                )
            
            # Add error handler if not present
            if 'ECONNREFUSED' not in content:
                error_handler = '''
                configure: (proxy, options) => {
                    proxy.on('error', (err, req, res) => {
                        if (err.code === 'ECONNREFUSED') {
                            console.log('⏳ Waiting for API server...');
                        } else {
                            console.error('Proxy error:', err.code);
                        }
                    });
                },'''
                
                # Insert after changeOrigin or timeout
                content = re.sub(
                    r'(timeout:\s*\d+)',
                    r'\1,' + error_handler,
                    content,
                    count=1
                )
            
            with open(vite_config, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("   ✅ Fixed Vite config!")
            self.fixes_applied.append("vite_config")
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Failed to fix Vite config: {e}")
            return False
    
    def populate_concept_mesh(self) -> bool:
        """Fix 4: Populate ConceptMesh with UUID-based concepts"""
        logger.info("\n4️⃣ Populating ConceptMesh...")
        
        try:
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
                data_file = self.root / "concept_mesh" / "data.json"
                data_file.parent.mkdir(parents=True, exist_ok=True)
                data_file.write_text('{"concepts": [], "metadata": {"version": "1.0"}}')
                logger.info(f"   📁 Created {data_file}")
            
            self.backup_manager.backup(data_file)
            
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure structure
            if isinstance(data, list):
                data = {"concepts": data, "metadata": {"version": "1.0"}}
            
            if "concepts" not in data:
                data["concepts"] = []
            
            # Generate stable IDs based on text hash
            def generate_concept_id(text):
                return f"concept_{hashlib.md5(text.encode()).hexdigest()[:8]}"
            
            # Initial concepts
            concept_specs = [
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
            ]
            
            # Check existing
            existing_ids = {c.get("id", "") for c in data["concepts"]}
            added = 0
            
            for text, weight, category in concept_specs:
                concept_id = generate_concept_id(text)
                
                if concept_id not in existing_ids:
                    concept = {
                        "id": concept_id,
                        "text": text,
                        "weight": weight,
                        "category": category,
                        "vector": [0.0] * 768,  # Placeholder
                        "created_at": datetime.now().isoformat()
                    }
                    data["concepts"].append(concept)
                    added += 1
            
            # Save
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"   ✅ Added {added} concepts, total now: {len(data['concepts'])}")
            
            # Create JSONL seed
            seed_dir = self.root / "data" / "seed_concepts"
            seed_dir.mkdir(parents=True, exist_ok=True)
            seed_file = seed_dir / "init.jsonl"
            
            with open(seed_file, 'w', encoding='utf-8') as f:
                for concept in data["concepts"][-added:]:  # Only new ones
                    f.write(json.dumps(concept) + '\n')
            
            self.fixes_applied.append("concept_mesh")
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Failed to populate ConceptMesh: {e}")
            return False
    
    def fix_mcp_duplicates(self) -> bool:
        """Fix 5: Add duplicate guards to MCP server"""
        logger.info("\n5️⃣ Fixing MCP duplicate registrations...")
        
        mcp_files = [
            self.root / "mcp_metacognitive" / "server_proper.py",
            self.root / "mcp_metacognitive" / "server_simple.py"
        ]
        
        fixed_any = False
        
        for mcp_file in mcp_files:
            if not mcp_file.exists():
                continue
            
            try:
                self.backup_manager.backup(mcp_file)
                
                with open(mcp_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if already fixed
                if 'register_tool_safe' in content:
                    logger.info(f"   ✅ {mcp_file.name} already fixed!")
                    fixed_any = True
                    continue
                
                # Add safe registration as a class-level solution
                safe_registration = '''
# Global registration tracking to prevent duplicates
_REGISTRATION_REGISTRY = {
    'tools': set(),
    'resources': set(), 
    'prompts': set()
}

def register_tool_safe(server, name, handler, description):
    """Register tool only if not already registered"""
    if name in _REGISTRATION_REGISTRY['tools']:
        logger.debug(f"Tool {name} already registered, skipping")
        return
    _REGISTRATION_REGISTRY['tools'].add(name)
    server.add_tool(name=name, handler=handler, description=description)

def register_resource_safe(server, uri, handler, description):
    """Register resource only if not already registered"""
    if uri in _REGISTRATION_REGISTRY['resources']:
        logger.debug(f"Resource {uri} already registered, skipping")
        return
    _REGISTRATION_REGISTRY['resources'].add(uri)
    server.add_resource(uri=uri, handler=handler, description=description)

def register_prompt_safe(server, name, handler, description):
    """Register prompt only if not already registered"""
    if name in _REGISTRATION_REGISTRY['prompts']:
        logger.debug(f"Prompt {name} already registered, skipping")
        return
    _REGISTRATION_REGISTRY['prompts'].add(name)
    server.add_prompt(name=name, handler=handler, description=description)
'''
                
                # Insert after imports
                import_section_end = 0
                for match in re.finditer(r'^(import|from)\s+', content, re.MULTILINE):
                    import_section_end = match.end()
                
                # Find end of import section
                next_line = content.find('\n', import_section_end)
                if next_line > 0:
                    # Skip any blank lines
                    while next_line < len(content) and content[next_line:next_line+1] in '\n\r':
                        next_line += 1
                    
                    content = content[:next_line] + safe_registration + '\n' + content[next_line:]
                
                # Replace registration calls
                content = re.sub(r'(\w+)\.add_tool\(', r'register_tool_safe(\1, ', content)
                content = re.sub(r'(\w+)\.add_resource\(', r'register_resource_safe(\1, ', content)
                content = re.sub(r'(\w+)\.add_prompt\(', r'register_prompt_safe(\1, ', content)
                
                with open(mcp_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"   ✅ Fixed {mcp_file.name}!")
                fixed_any = True
                
            except Exception as e:
                logger.error(f"   ❌ Failed to fix {mcp_file.name}: {e}")
        
        if fixed_any:
            self.fixes_applied.append("mcp_duplicates")
        
        return fixed_any
    
    def check_npm_updates_needed(self) -> bool:
        """Check if npm install is needed"""
        frontend_dir = self.root / "tori_ui_svelte"
        if not frontend_dir.exists():
            return False
        
        vite_config = frontend_dir / "vite.config.js"
        node_modules = frontend_dir / "node_modules"
        lock_file = frontend_dir / "package-lock.json"
        
        if not node_modules.exists():
            return True
        
        if not lock_file.exists():
            return True
        
        # Check if vite.config.js is newer than node_modules
        if vite_config.exists():
            vite_mtime = vite_config.stat().st_mtime
            modules_mtime = node_modules.stat().st_mtime
            return vite_mtime > modules_mtime
        
        return False
    
    def run_npm_install(self) -> bool:
        """Run npm install only if needed"""
        logger.info("\n6️⃣ Checking frontend dependencies...")
        
        frontend_dir = self.root / "tori_ui_svelte"
        if not frontend_dir.exists():
            return True
        
        if not self.check_npm_updates_needed():
            logger.info("   ✅ Frontend dependencies up to date")
            return True
        
        try:
            logger.info("   📦 Running npm install...")
            result = subprocess.run(
                ["npm", "install"],
                cwd=frontend_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("   ✅ Frontend dependencies installed")
                return True
            else:
                logger.error(f"   ❌ npm install failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"   ❌ Could not run npm install: {e}")
            return False
    
    def run(self) -> bool:
        """Run all fixes with rollback on failure"""
        logger.info("🔧 TORI Automatic Fix Script v2")
        logger.info("=" * 50)
        logger.info(f"📁 Working directory: {self.root}")
        logger.info(f"📁 Backups: {self.backup_manager.backup_dir}")
        
        # Check if already patched
        if self.patch_metadata_file.exists():
            with open(self.patch_metadata_file) as f:
                metadata = json.load(f)
            logger.info(f"ℹ️  Previous patch applied at: {metadata['timestamp']}")
        
        fixes = [
            ("Prajna API", self.fix_prajna_api),
            ("Launcher", self.fix_launcher),
            ("Vite Config", self.fix_vite_config),
            ("ConceptMesh", self.populate_concept_mesh),
            ("MCP Server", self.fix_mcp_duplicates),
            ("NPM Install", self.run_npm_install)
        ]
        
        results = {}
        
        for name, fix_func in fixes:
            try:
                results[name] = fix_func()
            except Exception as e:
                logger.error(f"   💥 Unexpected error in {name}: {e}")
                results[name] = False
                
                # Rollback on critical failures
                if name in ["Prajna API", "Launcher"]:
                    self.backup_manager.rollback()
                    logger.error("🚨 Critical fix failed - rolled back all changes")
                    return False
        
        # Record metadata
        self.record_patch_metadata()
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("📊 RESULTS:")
        for name, success in results.items():
            status = "✅" if success else "❌"
            logger.info(f"   {status} {name}")
        
        if all(results.values()):
            logger.info("\n🎉 All fixes applied successfully!")
            logger.info("\n🚀 Next steps:")
            logger.info("   1. Run: poetry run python enhanced_launcher.py")
            logger.info("   2. Test: python verify_tori_fixes.py")
        else:
            logger.warning("\n⚠️  Some fixes failed - check the logs above")
        
        return all(results.values())

if __name__ == "__main__":
    try:
        fixer = TORIFixerV2()
        success = fixer.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n💥 Fatal error: {e}")
        sys.exit(1)
