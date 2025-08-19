#!/usr/bin/env python3
"""
TORI Fix Verification Script v2 - Enhanced with better checks
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

class TORIVerifier:
    """Enhanced verifier with smarter checks"""
    
    def __init__(self):
        self.root = Path(__file__).parent
        self.patch_metadata_file = self.root / ".tori_patch_version"
        self.api_port = self._get_api_port()
        
    def _get_api_port(self) -> int:
        """Get API port from patch metadata, env, or default"""
        # Try patch metadata first
        if self.patch_metadata_file.exists():
            with open(self.patch_metadata_file) as f:
                metadata = json.load(f)
                if 'api_port' in metadata:
                    return metadata['api_port']
        
        # Try .env file
        env_file = self.root / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("API_PORT="):
                        return int(line.split("=")[1].strip())
        
        # Default
        return 8002
    
    def show_patch_info(self):
        """Display patch metadata if available"""
        if self.patch_metadata_file.exists():
            with open(self.patch_metadata_file) as f:
                metadata = json.load(f)
            
            logger.info(f"📋 Patch Information:")
            logger.info(f"   Applied: {metadata['timestamp']}")
            logger.info(f"   Fixes: {', '.join(metadata['fixes_applied'])}")
            if metadata.get('git_commit'):
                logger.info(f"   Commit: {metadata['git_commit'][:8]}")
    
    def test_api_health(self, max_retries=10) -> Tuple[bool, float]:
        """Test API health with exponential backoff"""
        logger.info("🔍 Testing API health endpoint...")
        
        wait_times = [0.25, 0.5, 1.0, 2.0, 5.0]
        total_time = 0
        
        for i in range(max_retries):
            start = time.time()
            try:
                response = requests.get(
                    f"http://127.0.0.1:{self.api_port}/api/health", 
                    timeout=2
                )
                elapsed = time.time() - start
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"   ✅ API health: {data}")
                    logger.info(f"   ⏱️  Response time: {elapsed:.2f}s")
                    return True, elapsed
                    
            except Exception as e:
                elapsed = time.time() - start
                
            total_time += elapsed
            wait = wait_times[min(i, len(wait_times)-1)]
            
            if i < max_retries - 1:
                logger.debug(f"   ⏳ Retry {i+1}/{max_retries} in {wait}s...")
                time.sleep(wait)
                total_time += wait
        
        logger.error(f"   ❌ API health check failed after {total_time:.1f}s")
        return False, total_time
    
    def test_concept_mesh(self) -> Dict[str, any]:
        """Test ConceptMesh with detailed stats"""
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
                
                return {
                    'success': total > 0,
                    'concepts': total,
                    'relations': relations
                }
                
        except Exception as e:
            logger.error(f"   ❌ ConceptMesh test failed: {e}")
        
        return {'success': False, 'concepts': 0, 'relations': 0}
    
    def test_lattice(self) -> Dict[str, any]:
        """Test lattice, only rebuild if needed"""
        logger.info("\n🔍 Testing oscillator lattice...")
        
        try:
            # First check current state
            response = requests.get(
                f"http://127.0.0.1:{self.api_port}/api/lattice/snapshot",
                timeout=5
            )
            
            rebuild_needed = False
            
            if response.status_code == 200:
                data = response.json()
                summary = data.get('summary', {})
                oscillators = summary.get('oscillators', 0)
                
                if oscillators == 0:
                    logger.info("   🔄 No oscillators, triggering rebuild...")
                    rebuild_needed = True
            else:
                rebuild_needed = True
            
            # Rebuild if needed
            if rebuild_needed:
                requests.post(
                    f"http://127.0.0.1:{self.api_port}/api/lattice/rebuild",
                    timeout=10
                )
                time.sleep(2)  # Give it time to rebuild
                
                # Check again
                response = requests.get(
                    f"http://127.0.0.1:{self.api_port}/api/lattice/snapshot",
                    timeout=5
                )
            
            if response.status_code == 200:
                data = response.json()
                summary = data.get('summary', {})
                
                logger.info(f"   ✅ Lattice stats:")
                logger.info(f"      Oscillators: {summary.get('oscillators', 0)}")
                logger.info(f"      R (sync): {summary.get('R', 0):.3f}")
                logger.info(f"      H (entropy): {summary.get('H', 0):.3f}")
                
                return {
                    'success': summary.get('oscillators', 0) > 0,
                    'oscillators': summary.get('oscillators', 0),
                    'R': summary.get('R', 0),
                    'H': summary.get('H', 0)
                }
                
        except Exception as e:
            logger.error(f"   ❌ Lattice test failed: {e}")
        
        return {'success': False, 'oscillators': 0, 'R': 0, 'H': 0}
    
    def check_file_modifications(self) -> Dict[str, bool]:
        """Check files using AST/regex for accuracy"""
        logger.info("\n🔍 Checking file modifications...")
        
        checks = []
        
        # Check Prajna API for health endpoint function
        prajna_api = self.root / "prajna" / "api" / "prajna_api.py"
        if prajna_api.exists():
            content = prajna_api.read_text(encoding='utf-8')
            # Look for the actual function definition
            has_health = bool(re.search(
                r'@app\.get\(["\']\/api\/health["\']\]\s*\)\s*\n\s*(async\s+)?def\s+health_check',
                content,
                re.MULTILINE
            ))
            checks.append((
                "prajna/api/prajna_api.py", 
                has_health,
                "health endpoint function"
            ))
        
        # Check launcher for wait method
        launcher = self.root / "enhanced_launcher.py"
        if launcher.exists():
            content = launcher.read_text(encoding='utf-8')
            has_wait = bool(re.search(
                r'def\s+_wait_for_api_health\s*\(',
                content
            ))
            checks.append((
                "enhanced_launcher.py",
                has_wait,
                "wait_for_api_health method"
            ))
        
        # Check Vite config for IPv4
        vite_config = self.root / "tori_ui_svelte" / "vite.config.js"
        if vite_config.exists():
            content = vite_config.read_text(encoding='utf-8')
            has_ipv4 = '127.0.0.1' in content and 'localhost' not in content
            checks.append((
                "tori_ui_svelte/vite.config.js",
                has_ipv4,
                "IPv4 targets"
            ))
        
        # Check concept mesh for data
        for path in ["concept_mesh/data.json", "data/concept_mesh/data.json", "data.json"]:
            full_path = self.root / path
            if full_path.exists():
                try:
                    with open(full_path) as f:
                        data = json.load(f)
                    has_concepts = len(data.get('concepts', [])) > 0
                    checks.append((
                        path,
                        has_concepts,
                        f"{len(data.get('concepts', []))} concepts"
                    ))
                    break
                except:
                    pass
        
        # Display results
        all_good = True
        for filepath, status, what in checks:
            icon = "✅" if status else "❌"
            logger.info(f"   {icon} {filepath} - {what}")
            all_good = all_good and status
        
        return {check[0]: check[1] for check in checks}
    
    def check_frontend_logs(self, tail_lines=50) -> bool:
        """Check frontend logs for proxy errors"""
        logger.info("\n🔍 Checking frontend logs for proxy errors...")
        
        log_paths = [
            self.root / "logs" / "session_*" / "frontend.log",
            self.root / "tori_ui_svelte" / "npm-debug.log"
        ]
        
        found_log = False
        has_errors = False
        
        for log_pattern in log_paths:
            for log_file in self.root.glob(str(log_pattern)):
                if log_file.exists():
                    found_log = True
                    try:
                        # Read last N lines
                        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                            lines = f.readlines()
                            recent_lines = lines[-tail_lines:]
                        
                        # Check for proxy errors
                        error_count = 0
                        for line in recent_lines:
                            if 'ECONNREFUSED' in line or 'proxy error' in line:
                                error_count += 1
                        
                        if error_count > 0:
                            logger.warning(f"   ⚠️  Found {error_count} proxy errors in {log_file.name}")
                            has_errors = True
                        else:
                            logger.info(f"   ✅ No proxy errors in recent {log_file.name}")
                            
                    except Exception as e:
                        logger.error(f"   ❌ Could not read {log_file.name}: {e}")
        
        if not found_log:
            logger.info("   ℹ️  No frontend logs found (may not be running yet)")
            return True  # Not an error if no logs exist
        
        return not has_errors
    
    def run_all_tests(self) -> bool:
        """Run all verification tests"""
        logger.info("🧪 TORI Fix Verification v2")
        logger.info("=" * 50)
        logger.info(f"📡 API Port: {self.api_port}")
        
        # Show patch info if available
        self.show_patch_info()
        
        # Check if TORI is running
        api_healthy, response_time = self.test_api_health()
        
        if not api_healthy:
            logger.warning("\n⚠️  TORI doesn't seem to be running!")
            logger.info("   Run: poetry run python enhanced_launcher.py")
            return False
        
        # Run all tests
        results = {
            "API Health": api_healthy,
            "API Response Time": response_time < 1.0,  # Should be fast
            "ConceptMesh Data": self.test_concept_mesh()['success'],
            "Oscillator Lattice": self.test_lattice()['success'],
            "File Modifications": all(self.check_file_modifications().values()),
            "Frontend Logs Clean": self.check_frontend_logs()
        }
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("📊 VERIFICATION RESULTS:")
        
        for test, passed in results.items():
            if test == "API Response Time" and not passed:
                logger.warning(f"   ⚠️  {test} - {response_time:.2f}s (slow)")
            else:
                status = "✅" if passed else "❌"
                logger.info(f"   {status} {test}")
        
        success = all(results.values())
        
        if success:
            logger.info("\n🎉 All tests passed! TORI is working correctly.")
            logger.info("\n🎯 Quick commands:")
            logger.info("   Chat: Open http://localhost:5173")
            logger.info("   API Docs: http://localhost:8002/docs")
        else:
            logger.warning("\n⚠️  Some tests failed.")
            logger.info("   Run: python fix_tori_automatic_v2.py")
        
        return success

def main():
    """Main entry point"""
    try:
        verifier = TORIVerifier()
        success = verifier.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
