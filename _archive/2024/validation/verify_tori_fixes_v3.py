#!/usr/bin/env python3
"""
TORI Fix Verification Script v3 - Enhanced with all recommended checks
"""

import os
import re
import sys
import json
import time
import logging
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class TORIVerifierV3:
    """Enhanced verifier with comprehensive checks"""
    
    def __init__(self):
        self.root = Path(__file__).parent
        self.patch_metadata_file = self.root / ".tori_patch_version"
        self.api_port = self._get_api_port()
        
    def _get_api_port(self) -> int:
        """Get API port from api_port.json first, then other sources"""
        # Check api_port.json (runtime config)
        api_port_file = self.root / "api_port.json"
        if api_port_file.exists():
            try:
                with open(api_port_file) as f:
                    data = json.load(f)
                    port = data.get("api_port", 8002)
                    logger.info(f"📡 Using API port from api_port.json: {port}")
                    return port
            except:
                pass
        
        # Check patch metadata
        if self.patch_metadata_file.exists():
            with open(self.patch_metadata_file) as f:
                metadata = json.load(f)
                if 'api_port' in metadata:
                    return metadata['api_port']
        
        # Check .env file
        env_file = self.root / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("API_PORT="):
                        return int(line.split("=")[1].strip())
        
        # Default
        return 8002
    
    def show_patch_info(self):
        """Display patch metadata and verify file integrity"""
        if self.patch_metadata_file.exists():
            with open(self.patch_metadata_file) as f:
                metadata = json.load(f)
            
            logger.info(f"📋 Patch Information:")
            logger.info(f"   Applied: {metadata['timestamp']}")
            logger.info(f"   Fixes: {', '.join(metadata['fixes_applied'])}")
            
            if metadata.get('git_commit'):
                logger.info(f"   Commit: {metadata['git_commit'][:8]}")
            
            # Check file digests for drift
            if 'file_digests' in metadata:
                logger.info("   Checking file integrity...")
                drift_detected = False
                
                for filepath, expected_digest in metadata['file_digests'].items():
                    file_path = Path(filepath)
                    if file_path.exists():
                        import hashlib
                        with open(file_path, 'rb') as f:
                            actual_digest = hashlib.sha256(f.read()).hexdigest()
                        
                        if actual_digest != expected_digest:
                            logger.warning(f"   ⚠️  {file_path.name} has been modified since patch!")
                            drift_detected = True
                
                if not drift_detected:
                    logger.info("   ✅ All patched files intact")
    
    def test_api_health(self, max_retries=10) -> Tuple[bool, float]:
        """Test API health - should respond on first try if launch order fixed"""
        logger.info("🔍 Testing API health endpoint...")
        
        start_time = time.time()
        
        try:
            response = requests.get(
                f"http://127.0.0.1:{self.api_port}/api/health", 
                timeout=2
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"   ✅ API health: {data}")
                logger.info(f"   ⏱️  Response time: {elapsed:.2f}s")
                
                if elapsed > 1.0:
                    logger.warning("   ⚠️  API response time is slow!")
                
                return True, elapsed
                
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"   ❌ API health check failed: {e}")
        
        return False, elapsed
    
    def test_concept_mesh(self) -> Dict[str, any]:
        """Test ConceptMesh - should have > 0 concepts"""
        logger.info("\n🔍 Testing ConceptMesh...")
        
        try:
            response = requests.get(
                f"http://127.0.0.1:{self.api_port}/api/concept_mesh/stats",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                total = data.get('totalConcepts', 0)
                relations = data.get('totalRelations', 0)
                
                logger.info(f"   ✅ ConceptMesh stats:")
                logger.info(f"      Concepts: {total}")
                logger.info(f"      Relations: {relations}")
                
                if total == 0:
                    logger.error("   ❌ ConceptMesh is EMPTY!")
                    return {'success': False, 'concepts': 0, 'relations': 0}
                
                return {
                    'success': True,
                    'concepts': total,
                    'relations': relations
                }
                
        except Exception as e:
            logger.error(f"   ❌ ConceptMesh test failed: {e}")
        
        return {'success': False, 'concepts': 0, 'relations': 0}
    
    def test_lattice(self) -> Dict[str, any]:
        """Test lattice - rebuild if needed and verify oscillators > 0"""
        logger.info("\n🔍 Testing oscillator lattice...")
        
        try:
            # Check current state
            response = requests.get(
                f"http://127.0.0.1:{self.api_port}/api/lattice/snapshot",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                summary = data.get('summary', {})
                oscillators = summary.get('oscillators', 0)
                
                if oscillators == 0:
                    logger.info("   🔄 No oscillators, rebuilding lattice...")
                    
                    # Rebuild
                    rebuild_response = requests.post(
                        f"http://127.0.0.1:{self.api_port}/api/lattice/rebuild",
                        timeout=10
                    )
                    
                    if rebuild_response.status_code == 200:
                        time.sleep(2)  # Wait for rebuild
                        
                        # Check again
                        response = requests.get(
                            f"http://127.0.0.1:{self.api_port}/api/lattice/snapshot",
                            timeout=5
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            summary = data.get('summary', {})
                            oscillators = summary.get('oscillators', 0)
                
                logger.info(f"   📊 Lattice stats:")
                logger.info(f"      Oscillators: {oscillators}")
                logger.info(f"      R (sync): {summary.get('R', 0):.3f}")
                logger.info(f"      H (entropy): {summary.get('H', 0):.3f}")
                
                if oscillators == 0:
                    logger.error("   ❌ Lattice has NO oscillators!")
                    return {'success': False, 'oscillators': 0}
                
                return {
                    'success': True,
                    'oscillators': oscillators,
                    'R': summary.get('R', 0),
                    'H': summary.get('H', 0)
                }
                
        except Exception as e:
            logger.error(f"   ❌ Lattice test failed: {e}")
        
        return {'success': False, 'oscillators': 0, 'R': 0, 'H': 0}
    
    def check_file_modifications(self) -> Dict[str, bool]:
        """Check if key fixes are in place"""
        logger.info("\n🔍 Checking file modifications...")
        
        checks = []
        
        # Check Prajna API
        prajna_api = self.root / "prajna" / "api" / "prajna_api.py"
        if prajna_api.exists():
            content = prajna_api.read_text(encoding='utf-8')
            
            # Check for health endpoint
            has_health = bool(re.search(
                r'@app\.get\(["\']\/api\/health["\']\]\s*\)\s*\n\s*(async\s+)?def\s+health_check',
                content,
                re.MULTILINE
            ))
            
            # Check for single app instance
            app_count = len(re.findall(r'^app\s*=\s*FastAPI\(', content, re.MULTILINE))
            single_app = app_count == 1
            
            checks.append(("prajna_api.py", has_health and single_app, 
                          f"health endpoint {'✓' if has_health else '✗'}, single app {'✓' if single_app else '✗'}"))
        
        # Check launcher
        launcher = self.root / "enhanced_launcher.py"
        if launcher.exists():
            content = launcher.read_text(encoding='utf-8')
            
            # Check for wait method
            has_wait = bool(re.search(r'def\s+_wait_for_api_health\s*\(', content))
            
            # Check launch order - API should start before frontend
            api_start = content.find('Starting API server in background thread')
            frontend_start = content.find('STARTING FRONTEND...')
            correct_order = api_start > 0 and frontend_start > 0 and api_start < frontend_start
            
            checks.append(("enhanced_launcher.py", has_wait and correct_order,
                          f"wait method {'✓' if has_wait else '✗'}, correct order {'✓' if correct_order else '✗'}"))
        
        # Check Vite config
        vite_config = self.root / "tori_ui_svelte" / "vite.config.js"
        if vite_config.exists():
            content = vite_config.read_text(encoding='utf-8')
            has_ipv4 = '127.0.0.1' in content
            has_timeout = 'timeout:' in content
            has_error_handler = 'ECONNREFUSED' in content
            
            checks.append(("vite.config.js", all([has_ipv4, has_timeout, has_error_handler]),
                          f"IPv4 {'✓' if has_ipv4 else '✗'}, timeout {'✓' if has_timeout else '✗'}, error handler {'✓' if has_error_handler else '✗'}"))
        
        # Check concept mesh data at CORRECT path
        mesh_path = self.root / "data" / "concept_mesh" / "data.json"
        if mesh_path.exists():
            try:
                with open(mesh_path) as f:
                    data = json.load(f)
                concepts = data.get('concepts', [])
                has_concepts = len(concepts) > 0
                has_embeddings = all(c.get('embedding') for c in concepts) if concepts else False
                
                checks.append(("data/concept_mesh/data.json", has_concepts and has_embeddings,
                              f"{len(concepts)} concepts, embeddings {'✓' if has_embeddings else '✗'}"))
            except:
                checks.append(("data/concept_mesh/data.json", False, "parse error"))
        else:
            checks.append(("data/concept_mesh/data.json", False, "not found"))
        
        # Check MCP files
        for mcp_name in ["server_proper.py", "server_simple.py"]:
            mcp_file = self.root / "mcp_metacognitive" / mcp_name
            if mcp_file.exists():
                content = mcp_file.read_text(encoding='utf-8')
                has_safe_funcs = 'register_tool_safe' in content
                no_add_tool = len(re.findall(r'\.add_tool\(', content)) == 0
                
                checks.append((f"mcp_metacognitive/{mcp_name}", has_safe_funcs and no_add_tool,
                              f"safe funcs {'✓' if has_safe_funcs else '✗'}, no add_tool {'✓' if no_add_tool else '✗'}"))
        
        # Display results
        all_good = True
        for filepath, status, details in checks:
            icon = "✅" if status else "❌"
            logger.info(f"   {icon} {filepath} - {details}")
            all_good = all_good and status
        
        return {check[0]: check[1] for check in checks}
    
    def check_frontend_logs(self, wait_time=10) -> bool:
        """Check frontend logs for ECONNREFUSED after startup"""
        logger.info(f"\n🔍 Checking frontend logs (waiting {wait_time}s for startup)...")
        
        # Wait for frontend to fully start
        time.sleep(wait_time)
        
        # Check multiple possible log locations
        log_paths = [
            list(self.root.glob("logs/session_*/frontend.log")),
            [self.root / "tori_ui_svelte" / ".vite.log"],
            [self.root / "tori_ui_svelte" / "npm-debug.log"]
        ]
        
        found_log = False
        proxy_errors_found = False
        
        for path_list in log_paths:
            for log_file in path_list:
                if log_file.exists():
                    found_log = True
                    try:
                        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                        
                        # Check last 1000 chars for recent errors
                        recent_content = content[-1000:]
                        
                        if 'ECONNREFUSED' in recent_content:
                            logger.error(f"   ❌ Found ECONNREFUSED in {log_file.name}")
                            proxy_errors_found = True
                        
                        if 'proxy error' in recent_content.lower():
                            logger.error(f"   ❌ Found proxy errors in {log_file.name}")
                            proxy_errors_found = True
                            
                    except Exception as e:
                        logger.error(f"   ❌ Could not read {log_file.name}: {e}")
        
        if not found_log:
            logger.info("   ℹ️  No frontend logs found yet")
            return True
        
        if not proxy_errors_found:
            logger.info("   ✅ No proxy errors found in frontend logs")
            return True
        
        return False
    
    def check_processes(self) -> Dict[str, bool]:
        """Check if key processes are running"""
        logger.info("\n🔍 Checking running processes...")
        
        import psutil
        
        processes = {
            'python': False,
            'node': False,
            'npm': False
        }
        
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                name = proc.info['name'].lower()
                cmdline = ' '.join(proc.info['cmdline'] or []).lower()
                
                if 'python' in name and 'enhanced_launcher' in cmdline:
                    processes['python'] = True
                
                if 'node' in name:
                    processes['node'] = True
                    
                if 'npm' in name:
                    processes['npm'] = True
                    
            except:
                pass
        
        for proc, running in processes.items():
            icon = "✅" if running else "⚠️"
            logger.info(f"   {icon} {proc} process: {'running' if running else 'not found'}")
        
        return processes
    
    def run_all_tests(self) -> bool:
        """Run all verification tests"""
        logger.info("🧪 TORI Fix Verification v3")
        logger.info("=" * 50)
        logger.info(f"📡 API Port: {self.api_port}")
        
        # Show patch info
        self.show_patch_info()
        
        # Check if TORI is running
        api_healthy, response_time = self.test_api_health()
        
        if not api_healthy:
            logger.warning("\n⚠️  TORI doesn't seem to be running!")
            logger.info("   Run: poetry run python enhanced_launcher.py")
            
            # Still check files even if not running
            file_checks_pass = all(self.check_file_modifications().values())
            
            if file_checks_pass:
                logger.info("\n✅ Files are properly patched - ready to start TORI")
            else:
                logger.error("\n❌ Files need patching - run: python fix_tori_automatic_v3.py")
            
            return False
        
        # Run all tests
        concept_mesh_result = self.test_concept_mesh()
        lattice_result = self.test_lattice()
        
        results = {
            "API Health": api_healthy,
            "API Response < 1s": response_time < 1.0,
            "ConceptMesh > 0": concept_mesh_result['success'],
            "Oscillators > 0": lattice_result['success'],
            "File Patches": all(self.check_file_modifications().values()),
            "No Proxy Errors": self.check_frontend_logs(),
            "Processes Running": self.check_processes()['python']
        }
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("📊 VERIFICATION RESULTS:")
        
        for test, passed in results.items():
            status = "✅" if passed else "❌"
            logger.info(f"   {status} {test}")
        
        success = all(results.values())
        
        if success:
            logger.info("\n🎉 All tests passed! TORI is bulletproof.")
            logger.info("\n🎯 Quick commands:")
            logger.info(f"   Chat: http://localhost:5173")
            logger.info(f"   API: http://localhost:{self.api_port}/docs")
            logger.info(f"   Health: curl http://127.0.0.1:{self.api_port}/api/health")
        else:
            logger.warning("\n⚠️  Some tests failed.")
            
            if not results["ConceptMesh > 0"]:
                logger.error("   🔧 ConceptMesh empty - check data/concept_mesh/data.json")
            
            if not results["No Proxy Errors"]:
                logger.error("   🔧 Proxy errors detected - API not starting first")
            
            logger.info("\n   Run: python fix_tori_automatic_v3.py")
        
        return success

def main():
    """Main entry point"""
    try:
        verifier = TORIVerifierV3()
        success = verifier.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
