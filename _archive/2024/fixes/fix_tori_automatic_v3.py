#!/usr/bin/env python3
"""
TORI Automatic Fix Script v3 - Addresses all critical gaps
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
from contextlib import contextmanager
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
        self.file_digests: Dict[str, str] = {}
        self.prune_old_backups()
    
    def backup(self, filepath: Path) -> Optional[Path]:
        """Backup a file and track it for rollback"""
        if not filepath.exists():
            return None
            
        # Calculate digest
        with open(filepath, 'rb') as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        self.file_digests[str(filepath)] = digest
            
        backup_path = self.backup_dir / filepath.name
        shutil.copy2(filepath, backup_path)
        self.backed_up_files[filepath] = backup_path
        logger.info(f"   📁 Backed up {filepath.name} (SHA256: {digest[:8]}...)")
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
                    timestamp = datetime.strptime(backup_dir.name, "%Y%m%d_%H%M%S")
                    if timestamp < cutoff:
                        shutil.rmtree(backup_dir)
                        logger.debug(f"Pruned old backup: {backup_dir.name}")
                except:
                    pass

@contextmanager
def undo_on_failure(backup_manager: BackupManager, operation_name: str):
    """Context manager for atomic operations with rollback"""
    try:
        yield
    except Exception as e:
        logger.error(f"❌ {operation_name} failed: {e}")
        backup_manager.rollback()
        raise

class TORIFixerV3:
    """Enhanced TORI fixer addressing all critical gaps"""
    
    def __init__(self):
        self.root = Path(__file__).parent
        self.backup_manager = BackupManager(self.root)
        self.patch_metadata_file = self.root / ".tori_patch_version"
        self.fixes_applied = []
        
    def get_api_port(self) -> int:
        """Get API port from api_port.json, env, or default"""
        # Check api_port.json first (runtime config)
        api_port_file = self.root / "api_port.json"
        if api_port_file.exists():
            try:
                with open(api_port_file) as f:
                    data = json.load(f)
                    return data.get("api_port", 8002)
            except:
                pass
        
        # Check .env file
        env_file = self.root / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("API_PORT="):
                        return int(line.split("=")[1].strip())
        
        return 8002
    
    def record_patch_metadata(self):
        """Record patch version, timestamp, and file digests"""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "git_commit": self._get_git_commit(),
            "fixes_applied": self.fixes_applied,
            "api_port": self.get_api_port(),
            "file_digests": self.backup_manager.file_digests
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
        """Fix 1: Add early health endpoint using AST approach"""
        logger.info("\n1️⃣ Fixing Prajna API health endpoint...")
        
        prajna_api = self.root / "prajna" / "api" / "prajna_api.py"
        if not prajna_api.exists():
            logger.error("   ❌ prajna_api.py not found")
            return False
        
        with undo_on_failure(self.backup_manager, "Prajna API fix"):
            self.backup_manager.backup(prajna_api)
            
            with open(prajna_api, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if already fixed
            if '@app.get("/api/health")' in content:
                logger.info("   ✅ Already fixed!")
                return True
            
            # Find ALL app creations
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
            
            # Comment out ALL subsequent app creations to ensure single instance
            if len(app_creations) > 1:
                for app_match in reversed(app_creations[1:]):
                    start, end = app_match.span()
                    app_line = new_content[start:end]
                    new_content = new_content[:start] + f"# {app_line}  # Disabled - reusing first app instance" + new_content[end:]
            
            with open(prajna_api, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            logger.info("   ✅ Fixed Prajna API!")
            self.fixes_applied.append("prajna_api")
            return True
    
    def fix_launcher(self) -> bool:
        """Fix 2: PROPERLY replace launch method to fix sequence"""
        logger.info("\n2️⃣ Fixing launcher sequence (overwriting method)...")
        
        launcher = self.root / "enhanced_launcher.py"
        if not launcher.exists():
            logger.error("   ❌ enhanced_launcher.py not found")
            return False
        
        with undo_on_failure(self.backup_manager, "Launcher fix"):
            self.backup_manager.backup(launcher)
            
            with open(launcher, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # First add helper methods if not present
            if '_wait_for_api_health' not in content:
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
                
                # Find class definition and add methods before launch
                class_pattern = r'(class\s+EnhancedUnifiedToriLauncher[^:]*:.*?)(\n\s*def\s+launch\s*\()'
                match = re.search(class_pattern, content, re.DOTALL)
                if match:
                    content = match.group(1) + wait_methods + match.group(2) + content[match.end():]
            
            # Now REPLACE the entire launch method
            # Find the start and end of the launch method
            launch_start = re.search(r'def\s+launch\s*\(self\)\s*:', content)
            if not launch_start:
                logger.error("   ❌ Could not find launch method")
                return False
            
            # Find the next method or end of class
            next_method = re.search(r'\n    def\s+\w+\s*\(', content[launch_start.end():])
            if next_method:
                launch_end = launch_start.end() + next_method.start()
            else:
                # Find end of class (dedent or EOF)
                class_end = re.search(r'\n(?!    )', content[launch_start.end():])
                launch_end = launch_start.end() + (class_end.start() if class_end else len(content) - launch_start.end())
            
            # New launch method with correct sequence
            new_launch = '''def launch(self):
        """Bulletproof launch sequence: API first, wait, then everything else"""
        try:
            self.print_banner()
            
            # Step 1: Find and secure API port
            self.update_status("startup", "port_search", {"message": "Finding available port"})
            port = self.find_available_port(service_name="API")
            secured_port = self.secure_port_aggressively(port, "API")
            self.api_port = secured_port
            
            # Step 2: Start API server FIRST in background thread
            self.logger.info("\\n" + "=" * 50)
            self.logger.info("🚀 STARTING API SERVER FIRST...")
            self.logger.info("=" * 50)
            
            api_thread = threading.Thread(
                target=self._run_api_server,
                args=(self.api_port,),
                daemon=True,
                name="APIServer"
            )
            api_thread.start()
            
            # Step 3: WAIT for API to be healthy before anything else
            self.logger.info("⏳ Waiting for API server to be healthy...")
            if not self._wait_for_api_health(self.api_port):
                self.logger.error("❌ API server failed to start!")
                return 1
            
            self.logger.info("✅ API server is healthy and ready!")
            
            # NOW start other components
            # Step 4: MCP Metacognitive server
            self.logger.info("\\n" + "=" * 50)
            self.logger.info("🧠 STARTING MCP METACOGNITIVE SERVER...")
            self.logger.info("=" * 50)
            mcp_started = self.start_mcp_metacognitive_server()
            
            # Step 5: Configure Prajna
            self.logger.info("\\n" + "=" * 50)
            self.logger.info("🧠 CONFIGURING PRAJNA...")
            self.logger.info("=" * 50)
            prajna_configured = self.configure_prajna_integration_enhanced()
            
            # Step 6: Start core components
            self.logger.info("\\n" + "=" * 50)
            self.logger.info("🧠 STARTING CORE COMPONENTS...")
            self.logger.info("=" * 50)
            core_components_started = self.start_core_python_components()
            stability_components_started = self.start_stability_components()
            
            # Step 7: Start async components
            try:
                from python.core.lattice_evolution_runner import run_forever as run_lattice
                
                def run_lattice_thread():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(run_lattice())
                    except Exception as e:
                        self.logger.error(f"Lattice runner error: {e}")
                    finally:
                        loop.close()
                
                lattice_thread = threading.Thread(
                    target=run_lattice_thread,
                    daemon=True,
                    name="OscillatorLattice"
                )
                lattice_thread.start()
                self.logger.info("✅ Oscillator lattice started in background")
                
            except ImportError:
                self.logger.warning("⚠️ Oscillator lattice not available")
            
            # Step 8: FINALLY start frontend (API is already up!)
            self.logger.info("\\n" + "=" * 50)
            self.logger.info("🎨 STARTING FRONTEND...")
            self.logger.info("=" * 50)
            frontend_started = self.start_frontend_services_enhanced()
            
            # Save config and print status
            config = self.save_port_config(self.api_port, self.prajna_port, self.frontend_port, self.mcp_metacognitive_port)
            self.print_complete_system_ready(self.api_port, prajna_configured, frontend_started, mcp_started, core_components_started, stability_components_started)
            
            # Keep main thread alive
            self.logger.info("\\n🎯 System ready! Press Ctrl+C to shutdown.")
            api_thread.join()
            
        except KeyboardInterrupt:
            self.logger.info("\\n👋 Shutdown requested by user")
            self.update_status("shutdown", "user_requested", {"message": "Ctrl+C pressed"})
        except Exception as e:
            self.logger.error(f"❌ Launch failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.update_status("startup", "failed", {"error": str(e), "traceback": traceback.format_exc()})
            return 1
        
        return 0
    '''
            
            # Replace the entire launch method
            new_content = content[:launch_start.start()] + new_launch + content[launch_end:]
            
            with open(launcher, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            logger.info("   ✅ Fixed launcher (method overwritten)!")
            self.fixes_applied.append("launcher")
            return True
    
    def fix_vite_config(self) -> bool:
        """Fix 3: Update Vite proxy with IPv4 and proper error handling"""
        logger.info("\n3️⃣ Fixing Vite proxy configuration...")
        
        vite_config = self.root / "tori_ui_svelte" / "vite.config.js"
        if not vite_config.exists():
            logger.error("   ❌ vite.config.js not found")
            return False
        
        with undo_on_failure(self.backup_manager, "Vite config fix"):
            self.backup_manager.backup(vite_config)
            
            with open(vite_config, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace all localhost with 127.0.0.1
            content = re.sub(r'localhost:8002', '127.0.0.1:8002', content)
            content = re.sub(r'localhost:(\d+)', r'127.0.0.1:\1', content)
            
            # Ensure timeout is set
            if 'timeout:' not in content:
                content = re.sub(
                    r'(changeOrigin:\s*true)',
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
                
                content = re.sub(
                    r'(timeout:\s*\d+)(?![\s\S]*configure:)',
                    r'\1,' + error_handler,
                    content,
                    count=1
                )
            
            with open(vite_config, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("   ✅ Fixed Vite config!")
            self.fixes_applied.append("vite_config")
            return True
    
    def populate_concept_mesh(self) -> bool:
        """Fix 4: Populate ConceptMesh at CORRECT path with proper structure"""
        logger.info("\n4️⃣ Populating ConceptMesh at correct path...")
        
        with undo_on_failure(self.backup_manager, "ConceptMesh population"):
            # Use the CORRECT path that the loader expects
            mesh_path = self.root / "data" / "concept_mesh" / "data.json"
            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"   📁 Using path: {mesh_path}")
            
            # Create if doesn't exist
            if not mesh_path.exists():
                mesh_path.write_text('{"concepts": [], "metadata": {"version": "1.0"}}')
                logger.info(f"   📁 Created {mesh_path}")
            
            self.backup_manager.backup(mesh_path)
            
            with open(mesh_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure structure
            if isinstance(data, list):
                data = {"concepts": data, "metadata": {"version": "1.0"}}
            
            if "concepts" not in data:
                data["concepts"] = []
            
            # Initial concepts with PROPER structure including id and embedding
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
            existing_texts = {c.get("text", "") for c in data["concepts"]}
            added = 0
            
            for text, weight, category in concept_specs:
                if text not in existing_texts:
                    concept = {
                        "id": str(uuid.uuid4()),
                        "text": text,
                        "embedding": [0.0] * 768,  # Required placeholder
                        "weight": weight,
                        "category": category,
                        "created_at": datetime.now().isoformat()
                    }
                    data["concepts"].append(concept)
                    added += 1
            
            # Save at correct path
            with open(mesh_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"   ✅ Added {added} concepts, total now: {len(data['concepts'])}")
            
            # Also set environment variable if needed
            os.environ['TORI_CONCEPT_MESH_PATH'] = str(mesh_path)
            
            self.fixes_applied.append("concept_mesh")
            return True
    
    def fix_mcp_duplicates(self) -> bool:
        """Fix 5: PROPERLY replace all add_* calls with safe wrappers"""
        logger.info("\n5️⃣ Fixing MCP duplicate registrations (replacing all calls)...")
        
        mcp_files = [
            self.root / "mcp_metacognitive" / "server_proper.py",
            self.root / "mcp_metacognitive" / "server_simple.py"
        ]
        
        fixed_any = False
        
        for mcp_file in mcp_files:
            if not mcp_file.exists():
                continue
            
            with undo_on_failure(self.backup_manager, f"MCP {mcp_file.name} fix"):
                self.backup_manager.backup(mcp_file)
                
                with open(mcp_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add safe registration functions at module level
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
                
                # Insert after imports only if not already present
                if 'register_tool_safe' not in content:
                    import_section_end = 0
                    for match in re.finditer(r'^(import|from)\s+', content, re.MULTILINE):
                        import_section_end = match.end()
                    
                    next_line = content.find('\n', import_section_end)
                    if next_line > 0:
                        while next_line < len(content) and content[next_line:next_line+1] in '\n\r':
                            next_line += 1
                        content = content[:next_line] + safe_registration + '\n' + content[next_line:]
                
                # Count replacements for verification
                original_add_tool_count = len(re.findall(r'\.add_tool\(', content))
                original_add_resource_count = len(re.findall(r'\.add_resource\(', content))
                original_add_prompt_count = len(re.findall(r'\.add_prompt\(', content))
                
                # Replace ALL registration calls with safe versions
                content = re.sub(r'(\w+)\.add_tool\(', r'register_tool_safe(\1, ', content)
                content = re.sub(r'(\w+)\.add_resource\(', r'register_resource_safe(\1, ', content)
                content = re.sub(r'(\w+)\.add_prompt\(', r'register_prompt_safe(\1, ', content)
                
                # Verify replacements
                safe_tool_count = len(re.findall(r'register_tool_safe\(', content))
                safe_resource_count = len(re.findall(r'register_resource_safe\(', content))
                safe_prompt_count = len(re.findall(r'register_prompt_safe\(', content))
                
                logger.info(f"   📊 {mcp_file.name} replacements:")
                logger.info(f"      Tools: {original_add_tool_count} → {safe_tool_count}")
                logger.info(f"      Resources: {original_add_resource_count} → {safe_resource_count}")
                logger.info(f"      Prompts: {original_add_prompt_count} → {safe_prompt_count}")
                
                with open(mcp_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"   ✅ Fixed {mcp_file.name}!")
                fixed_any = True
        
        if fixed_any:
            self.fixes_applied.append("mcp_duplicates")
        
        return fixed_any
    
    def ensure_node_deps(self) -> bool:
        """Fix 6: Portable npm install with better fallbacks"""
        logger.info("\n6️⃣ Checking frontend dependencies...")
        
        frontend_dir = self.root / "tori_ui_svelte"
        if not frontend_dir.exists():
            logger.warning("   ⚠️  Frontend directory not found")
            return True  # Not a failure
        
        node_modules = frontend_dir / "node_modules"
        if node_modules.exists():
            logger.info("   ✅ node_modules exists - skipping install")
            return True
        
        # Find npm executable
        npm_cmd = None
        for cmd in ["npm", "npm.cmd", "pnpm", "yarn"]:
            npm_path = shutil.which(cmd)
            if npm_path:
                npm_cmd = [npm_path]
                logger.info(f"   📦 Found package manager: {cmd}")
                break
        
        if not npm_cmd:
            logger.warning("   ⚠️  Node.js not found - skipping frontend deps")
            logger.warning("   💡 Install Node.js from https://nodejs.org/")
            return False  # Downgrade to warning, not error
        
        try:
            logger.info("   📦 Running npm install...")
            result = subprocess.run(
                npm_cmd + ["install"],
                cwd=frontend_dir,
                capture_output=True,
                text=True,
                shell=(sys.platform == "win32")  # Use shell on Windows
            )
            
            if result.returncode == 0:
                logger.info("   ✅ Frontend dependencies installed")
                return True
            else:
                logger.warning(f"   ⚠️  npm install had issues: {result.stderr}")
                return False
                
        except Exception as e:
            logger.warning(f"   ⚠️  Could not run npm install: {e}")
            return False
    
    def run(self) -> bool:
        """Run all fixes with atomic rollback"""
        logger.info("🔧 TORI Automatic Fix Script v3")
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
            ("NPM Install", self.ensure_node_deps)
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
        
        # Record metadata with file digests
        self.record_patch_metadata()
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("📊 RESULTS:")
        for name, success in results.items():
            if name == "NPM Install" and not success:
                logger.warning(f"   ⚠️  {name} (non-critical)")
            else:
                status = "✅" if success else "❌"
                logger.info(f"   {status} {name}")
        
        # Don't fail overall if only npm failed
        critical_success = all(results[k] for k in results if k != "NPM Install")
        
        if critical_success:
            logger.info("\n🎉 All critical fixes applied successfully!")
            logger.info("\n🚀 Next steps:")
            logger.info("   1. Run: poetry run python enhanced_launcher.py")
            logger.info("   2. Test: python verify_tori_fixes_v3.py")
        else:
            logger.warning("\n⚠️  Some critical fixes failed - check the logs above")
        
        return critical_success

if __name__ == "__main__":
    try:
        fixer = TORIFixerV3()
        success = fixer.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n💥 Fatal error: {e}")
        sys.exit(1)
